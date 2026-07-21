"""Run the registered checkpoint splice and delayed-recovery audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from lsa.kappa_probe import estimate_kappa
from research_gym.analysis.lsa_kappa_transient_mechanism import decompose_estimate
from research_gym.integrity import canonical_file_sha256, verify_file_sha256
from research_gym.scripts.bench_loop_schedule_algebra_v0_1 import (
    ROOT,
    _append_event,
    _build_model,
    _cleanup,
    _device,
    _git_head,
    _save_checkpoint,
    _set_vram_fraction,
    _task_batch,
    _write_json,
    _write_records,
    measurement_seed,
)
from research_gym.scripts.bench_loop_schedule_kappa_surface import _interval_summary
from research_gym.scripts.bench_loop_schedule_kappa_transient_v0_2_recovery import (
    _exact_equal,
)


DEFAULT_CONFIG = ROOT / "configs" / "loop_schedule_kappa_splice_v0_2_2.json"
REGISTRATION = ROOT / "configs" / "loop_schedule_kappa_splice_v0_2_2_registration.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "loop_schedule_kappa_splice_v0_2_2"
FINAL_RECEIPT = ROOT / "data" / "benchmarks" / "lsa_kappa_splice_v0_2_2_receipt.json"


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if path != ROOT.resolve() and ROOT.resolve() not in path.parents:
        raise RuntimeError(f"registered path escapes repository: {value}")
    return path


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def continuation_streams(start_exposure: int, end_exposure: int, *, quantum: int) -> list[int]:
    if start_exposure < 0 or end_exposure <= start_exposure or quantum <= 0:
        raise ValueError("continuation bounds and quantum must be positive and ordered")
    if start_exposure % quantum or end_exposure % quantum:
        raise ValueError("continuation exposures must be exactly reachable")
    return list(range(start_exposure // quantum, end_exposure // quantum))


def classify_splice(
    forward_223: bool,
    forward_227: bool,
    reciprocal_211: bool,
) -> str:
    if forward_223 and forward_227 and not reciprocal_211:
        return "suffix_controlled_exit"
    if not forward_223 and not forward_227 and reciprocal_211:
        return "prehistory_controlled_exit"
    return "mixed_state_suffix_control"


def _source_kappa(records: Iterable[Mapping[str, Any]], order_seed: int, exposure: int) -> float:
    matches = [
        float(row["kappa"])
        for row in records
        if int(row["seed_channels"]["data_order_seed"]) == order_seed
        and int(row["exposure"]) == exposure
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one source kappa for D{order_seed} E{exposure}")
    return matches[0]


def _load_checkpoint(
    trainer: dict[str, Any],
    config: Mapping[str, Any],
    path: Path,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, Any]]:
    frozen = config["frozen_training"]
    torch.manual_seed(int(frozen["construction_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(frozen["construction_seed"]))
    device = _device(trainer)
    model = _build_model(
        trainer,
        "primary_mlp",
        rounds=64,
        tied=True,
        alpha=float(frozen["alpha"]),
        beta=float(frozen["beta"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        float(frozen["learning_rate"]),
        weight_decay=float(frozen["weight_decay"]),
    )
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    optimizer.load_state_dict(payload["optimizer"])
    return model, optimizer, payload


def _measure(
    config: Mapping[str, Any],
    trainer: dict[str, Any],
    model: torch.nn.Module,
    *,
    cell_id: str,
    state_seed: int,
    suffix_seed: int,
    exposure: int,
    interval_gradients: list[float],
    interval_start: int,
    latest_loss: float,
) -> dict[str, Any]:
    frozen = config["frozen_training"]
    device = _device(trainer)
    inputs, targets = _task_batch(
        trainer,
        "primary_mlp",
        seed=int(frozen["construction_seed"]),
        batch_size=int(frozen["measurement_batch_size"]),
        device=device,
        stream=999_001,
        data_seed=int(frozen["measurement_data_seed"]),
    )
    estimate = estimate_kappa(
        model,
        inputs,
        targets,
        power_iterations=int(frozen["power_iterations"]),
        seed=measurement_seed(int(frozen["probe_seed"]), exposure),
    )
    del inputs, targets
    decomposition = decompose_estimate(estimate.to_dict())
    return {
        "record_id": f"{cell_id}_E{exposure}",
        "kind": "kappa_splice",
        "cell_id": cell_id,
        "state_order_seed": state_seed,
        "suffix_order_seed": suffix_seed,
        "rounds": 64,
        "exposure": exposure,
        "kappa": estimate.kappa,
        "estimate": estimate.to_dict(),
        "sensitivity_interference_ratio": decomposition[
            "sensitivity_interference_ratio"
        ],
        "gradient_interference_ratio": decomposition["gradient_interference_ratio"],
        "interval_gradient": _interval_summary(
            interval_gradients, interval_start, exposure
        ),
        "latest_training_loss": latest_loss,
    }


def _continue_arm(
    config: Mapping[str, Any],
    trainer: dict[str, Any],
    *,
    state_seed: int,
    suffix_seed: int,
    checkpoint_path: Path,
    end_exposure: int,
    measurement_exposures: Iterable[int],
    output_dir: Path,
    event_path: Path,
    cell_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frozen = config["frozen_training"]
    model, optimizer, payload = _load_checkpoint(trainer, config, checkpoint_path)
    start_exposure = int(payload["exposure"])
    start_step = int(payload["step"])
    quantum = int(frozen["batch_size"]) * 64
    streams = continuation_streams(start_exposure, end_exposure, quantum=quantum)
    targets = {int(value) for value in measurement_exposures}
    if not targets or min(targets) <= start_exposure or max(targets) > end_exposure:
        raise RuntimeError("splice measurements must lie after the checkpoint and within the arm")
    records: list[dict[str, Any]] = []
    interval_gradients: list[float] = []
    interval_start = start_exposure
    latest_loss = math.nan
    maximum_gradient = 0.0
    nonfinite = False
    device = _device(trainer)
    _append_event(
        event_path,
        {
            "event": "splice_start",
            "cell_id": cell_id,
            "state_order_seed": state_seed,
            "suffix_order_seed": suffix_seed,
            "source_checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
            "start_exposure": start_exposure,
            "end_exposure": end_exposure,
            "streams": streams,
            "device": str(device),
        },
    )
    completed_step = start_step
    for stream in streams:
        step = stream + 1
        inputs, expected = _task_batch(
            trainer,
            "primary_mlp",
            seed=int(frozen["construction_seed"]),
            batch_size=int(frozen["batch_size"]),
            device=device,
            stream=stream,
            data_seed=suffix_seed,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), expected)
        if not torch.isfinite(loss):
            nonfinite = True
            break
        loss.backward()
        gradient_squared = sum(
            float(parameter.grad.detach().float().square().sum().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        gradient = math.sqrt(gradient_squared)
        if not math.isfinite(gradient):
            nonfinite = True
            break
        interval_gradients.append(gradient)
        maximum_gradient = max(maximum_gradient, gradient)
        optimizer.step()
        latest_loss = float(loss.detach().item())
        completed_step = step
        exposure = step * quantum
        if exposure in targets:
            record = _measure(
                config,
                trainer,
                model,
                cell_id=cell_id,
                state_seed=state_seed,
                suffix_seed=suffix_seed,
                exposure=exposure,
                interval_gradients=interval_gradients,
                interval_start=interval_start,
                latest_loss=latest_loss,
            )
            records.append(record)
            _append_event(event_path, {"event": "measurement", **record})
            interval_gradients.clear()
            interval_start = exposure
        del inputs, expected, loss
    terminal_exposure = completed_step * quantum
    terminal_checkpoint = _save_checkpoint(
        model,
        optimizer,
        output_dir=output_dir,
        cell_id=cell_id,
        step=completed_step,
        exposure=terminal_exposure,
    )
    summary = {
        "cell_id": cell_id,
        "state_order_seed": state_seed,
        "suffix_order_seed": suffix_seed,
        "source_exposure": start_exposure,
        "terminal_exposure": terminal_exposure,
        "steps_completed": completed_step - start_step,
        "streams": streams,
        "nonfinite": nonfinite,
        "maximum_gradient_norm": maximum_gradient,
        "latest_training_loss": latest_loss,
        "terminal_checkpoint": terminal_checkpoint,
    }
    _append_event(event_path, {"event": "splice_complete", **summary})
    del model, optimizer, payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records, summary


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _recovery_summary(
    records: Iterable[Mapping[str, Any]],
    *,
    start_kappa: float,
    threshold: float,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: int(row["exposure"]))
    changes = [
        (int(row["exposure"]), math.log(float(row["kappa"]) / start_kappa))
        for row in ordered
    ]
    recovered = [exposure for exposure, change in changes if change >= threshold]
    return {
        "start_kappa": start_kappa,
        "endpoint_kappa": float(ordered[-1]["kappa"]),
        "endpoint_log_change": changes[-1][1],
        "recovered_by_endpoint": bool(recovered),
        "first_measured_recovery_exposure": recovered[0] if recovered else None,
        "log_changes": {str(exposure): change for exposure, change in changes},
    }


def load_registered_config(path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    config_hash = canonical_file_sha256(path)
    if config_hash != registration["config_sha256"]:
        raise RuntimeError("splice config does not match frozen registration")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["protocol_id"] != registration["protocol_id"]:
        raise RuntimeError("splice protocol id does not match registration")
    for parent in config["parents"].values():
        source = _root_path(parent["path"])
        if not verify_file_sha256(source, parent["sha256"]):
            raise RuntimeError(f"parent hash mismatch: {source.name}")
    trainer_path = _root_path(config["parents"]["trainer_config"]["path"])
    return config, config_hash, json.loads(trainer_path.read_text(encoding="utf-8"))


def validate(config: Mapping[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    frozen = config["frozen_training"]
    if int(frozen["rounds"]) != 64 or int(frozen["construction_seed"]) != 103:
        raise RuntimeError("registered splice construction changed")
    if config["splice_arms"] != [
        {"state_seed": 211, "suffix_seed": 211, "role": "exact_replay"},
        {"state_seed": 211, "suffix_seed": 223, "role": "forward_transfer"},
        {"state_seed": 211, "suffix_seed": 227, "role": "forward_replication"},
        {"state_seed": 223, "suffix_seed": 211, "role": "reciprocal_transfer"},
    ]:
        raise RuntimeError("registered splice arms changed")
    for key in ("d211_e2048", "d211_e4096", "d223_e2048"):
        payload = _checkpoint_payload(_root_path(config["parents"][key]["path"]))
        expected_exposure = 4096 if key == "d211_e4096" else 2048
        if int(payload["exposure"]) != expected_exposure:
            raise RuntimeError(f"checkpoint metadata changed: {key}")
    outcomes = (
        output_dir / "splice_records.jsonl",
        output_dir / "splice_result.json",
        output_dir / "result_receipt.json",
        FINAL_RECEIPT,
    )
    existing = [str(path) for path in outcomes if path.exists()]
    if existing:
        raise RuntimeError(f"registered splice outcome already exists: {existing}")
    return {
        "status": "valid",
        "config_sha256": config_hash,
        "new_continuation_arms": 5,
        "new_prefix_training": 0,
        "outcomes_present": False,
    }


def run(
    config: Mapping[str, Any],
    config_hash: str,
    trainer: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    records_path = output_dir / "splice_records.jsonl"
    result_path = output_dir / "splice_result.json"
    event_path = output_dir / "splice_events.jsonl"
    if any(path.exists() for path in (records_path, result_path, event_path)):
        raise RuntimeError("refusing to overwrite splice outcomes")
    source_records = _read_records(_root_path(config["parents"]["source_records"]["path"]))
    start_211 = _source_kappa(source_records, 211, 2048)
    start_223 = _source_kappa(source_records, 223, 2048)
    threshold = float(config["registered_endpoints"]["practical_log_change"])
    all_records: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []

    replay_records, replay_cell = _continue_arm(
        config,
        trainer,
        state_seed=211,
        suffix_seed=211,
        checkpoint_path=_root_path(config["parents"]["d211_e2048"]["path"]),
        end_exposure=4096,
        measurement_exposures=[4096],
        output_dir=output_dir,
        event_path=event_path,
        cell_id="splice_S211_D211_replay",
    )
    all_records.extend(replay_records)
    cells.append(replay_cell)
    replay_terminal = _root_path(replay_cell["terminal_checkpoint"]["path"])
    parent_terminal = _root_path(config["parents"]["d211_e4096"]["path"])
    replay_payload = _checkpoint_payload(replay_terminal)
    parent_payload = _checkpoint_payload(parent_terminal)
    _exact_equal(replay_payload["state_dict"], parent_payload["state_dict"], "model")
    _exact_equal(replay_payload["optimizer"], parent_payload["optimizer"], "optimizer")
    source_endpoint = _source_kappa(source_records, 211, 4096)
    replay_difference = abs(float(replay_records[-1]["kappa"]) - source_endpoint)
    replay_passed = replay_difference <= float(
        config["integrity_gate"]["absolute_kappa_tolerance"]
    )
    if not replay_passed:
        raise RuntimeError("splice replay kappa failed exact tolerance")

    arm_specs = (
        (211, 223, "splice_S211_D223", "d211_e2048"),
        (211, 227, "splice_S211_D227", "d211_e2048"),
        (223, 211, "splice_S223_D211", "d223_e2048"),
    )
    arm_records: dict[str, list[dict[str, Any]]] = {}
    for state_seed, suffix_seed, cell_id, checkpoint_key in arm_specs:
        records, cell = _continue_arm(
            config,
            trainer,
            state_seed=state_seed,
            suffix_seed=suffix_seed,
            checkpoint_path=_root_path(config["parents"][checkpoint_key]["path"]),
            end_exposure=4096,
            measurement_exposures=[2560, 3072, 3584, 4096],
            output_dir=output_dir,
            event_path=event_path,
            cell_id=cell_id,
        )
        all_records.extend(records)
        cells.append(cell)
        arm_records[cell_id] = records

    extension_records, extension_cell = _continue_arm(
        config,
        trainer,
        state_seed=211,
        suffix_seed=211,
        checkpoint_path=parent_terminal,
        end_exposure=8192,
        measurement_exposures=[5120, 6144, 8192],
        output_dir=output_dir,
        event_path=event_path,
        cell_id="extension_S211_D211",
    )
    all_records.extend(extension_records)
    cells.append(extension_cell)
    _write_records(records_path, all_records)
    failures = [cell for cell in cells if cell["nonfinite"]]

    recovery = {
        "S211_D223": _recovery_summary(
            arm_records["splice_S211_D223"], start_kappa=start_211, threshold=threshold
        ),
        "S211_D227": _recovery_summary(
            arm_records["splice_S211_D227"], start_kappa=start_211, threshold=threshold
        ),
        "S223_D211": _recovery_summary(
            arm_records["splice_S223_D211"], start_kappa=start_223, threshold=threshold
        ),
        "S211_D211_extended": _recovery_summary(
            [
                {
                    "exposure": 4096,
                    "kappa": source_endpoint,
                },
                *extension_records,
            ],
            start_kappa=start_211,
            threshold=threshold,
        ),
    }
    if failures:
        scientific_status = "resource_or_training_failure"
        splice_classification = None
    else:
        splice_classification = classify_splice(
            recovery["S211_D223"]["recovered_by_endpoint"],
            recovery["S211_D227"]["recovered_by_endpoint"],
            recovery["S223_D211"]["recovered_by_endpoint"],
        )
        scientific_status = "complete"
    extension = recovery["S211_D211_extended"]
    censoring_status = (
        f"delayed_recovery_by_E{extension['first_measured_recovery_exposure']}"
        if extension["recovered_by_endpoint"]
        else "right_censored_at_E8192"
    )
    result = {
        "protocol_id": config["protocol_id"],
        "status": "complete",
        "phase": "run",
        "config_sha256": config_hash,
        "git_head": _git_head(),
        "records_path": records_path.relative_to(ROOT).as_posix(),
        "records_sha256": canonical_file_sha256(records_path),
        "record_count": len(all_records),
        "events_path": event_path.relative_to(ROOT).as_posix(),
        "events_sha256": canonical_file_sha256(event_path),
        "summary": {
            "scientific_status": scientific_status,
            "integrity_replay": {
                "passed": replay_passed,
                "tensor_exact_model_and_optimizer": True,
                "endpoint_kappa_absolute_difference": replay_difference,
            },
            "splice_classification": splice_classification,
            "censoring_status": censoring_status,
            "recovery": recovery,
            "cells": cells,
            "failed_cells": failures,
        },
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(result_path, result)
    return result


def finalize(config: Mapping[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    result_path = output_dir / "splice_result.json"
    records_path = output_dir / "splice_records.jsonl"
    resource_path = output_dir / "run.resource_receipt.json"
    if not all(path.exists() for path in (result_path, records_path, resource_path)):
        raise RuntimeError("splice run is incomplete")
    if (output_dir / "result_receipt.json").exists() or FINAL_RECEIPT.exists():
        raise RuntimeError("splice result already sealed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    if result["config_sha256"] != config_hash or result["status"] != "complete":
        raise RuntimeError("splice result does not match registered config")
    if not verify_file_sha256(records_path, result["records_sha256"]):
        raise RuntimeError("splice records failed re-verification")
    if resource["status"] != "completed" or not resource["cleanup_passed"]:
        raise RuntimeError("splice resource receipt did not pass")
    receipt = {
        "protocol_id": config["protocol_id"],
        "status": "sealed",
        "config_path": DEFAULT_CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": config_hash,
        "result_path": result_path.relative_to(ROOT).as_posix(),
        "result_sha256": canonical_file_sha256(result_path),
        "records_path": records_path.relative_to(ROOT).as_posix(),
        "records_sha256": canonical_file_sha256(records_path),
        "record_count": result["record_count"],
        "resource_path": resource_path.relative_to(ROOT).as_posix(),
        "resource_sha256": canonical_file_sha256(resource_path),
        "cleanup_passed": True,
        "scientific_status": result["summary"]["scientific_status"],
        "splice_classification": result["summary"]["splice_classification"],
        "censoring_status": result["summary"]["censoring_status"],
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(output_dir / "result_receipt.json", receipt)
    _write_json(FINAL_RECEIPT, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=("validate", "run", "finalize"), required=True)
    parser.add_argument("--vram-fraction", type=float)
    args = parser.parse_args()
    config, config_hash, trainer = load_registered_config(args.config.resolve())
    _set_vram_fraction(config, args.vram_fraction)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        if args.phase == "validate":
            payload = validate(config, config_hash, output)
        elif args.phase == "run":
            payload = run(config, config_hash, trainer, output)
        else:
            payload = finalize(config, config_hash, output)
        print(json.dumps(payload, sort_keys=True))
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
