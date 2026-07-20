"""Deterministically complete the registered R=128 transient run after timeout."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from lsa.kappa_probe import estimate_kappa
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
from research_gym.scripts.bench_loop_schedule_kappa_surface import _train_surface_cell
from research_gym.scripts.bench_loop_schedule_kappa_transient_v0_2 import (
    _grouped_kappa,
    _power_exponent,
    _read_records,
    _root_path,
    _stress_ratio,
    classify_timing,
)


DEFAULT_CONFIG = ROOT / "configs" / "loop_schedule_kappa_transient_v0_2_recovery1.json"
REGISTRATION = (
    ROOT / "configs" / "loop_schedule_kappa_transient_v0_2_recovery1_registration.json"
)
DEFAULT_OUTPUT = ROOT / "experiments" / "loop_schedule_kappa_transient_v0_2_recovery1"
FINAL_RECEIPT = (
    ROOT / "data" / "benchmarks" / "lsa_kappa_transient_v0_2_recovery1_receipt.json"
)


def _exact_equal(left: Any, right: Any, path: str = "state") -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise RuntimeError(f"tensor-exact replay failed at {path}")
        return
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise RuntimeError(f"mapping keys differ at {path}")
        for key in left:
            _exact_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise RuntimeError(f"sequence lengths differ at {path}")
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            _exact_equal(left_value, right_value, f"{path}[{index}]")
        return
    if left != right:
        raise RuntimeError(f"values differ at {path}: {left!r} != {right!r}")


def _measurement_record(
    config: Mapping[str, Any],
    trainer: dict[str, Any],
    model: torch.nn.Module,
    *,
    seed: int,
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
        seed=seed,
        batch_size=int(frozen["measurement_batch_size"]),
        device=device,
        stream=999_001,
    )
    estimate = estimate_kappa(
        model,
        inputs,
        targets,
        power_iterations=int(frozen["power_iterations"]),
        seed=measurement_seed(seed, exposure),
    )
    interval = {
        "start_exposure": interval_start,
        "end_exposure": exposure,
        "steps": len(interval_gradients),
        "maximum": max(interval_gradients),
        "root_mean_square": math.sqrt(
            sum(value * value for value in interval_gradients) / len(interval_gradients)
        ),
        "terminal": interval_gradients[-1],
    }
    del inputs, targets
    return {
        "record_id": f"surface_tied_R128_S{seed}_E{exposure}",
        "kind": "kappa_surface",
        "regime": "tied",
        "rounds": 128,
        "seed": seed,
        "exposure": exposure,
        "interval_gradient": interval,
        "latest_training_loss": latest_loss,
        "kappa": estimate.kappa,
        "estimate": estimate.to_dict(),
    }


def _gradient_norm(model: torch.nn.Module) -> float:
    return math.sqrt(
        sum(
            float(parameter.grad.detach().float().square().sum().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )


def _resume_seed_107(
    config: dict[str, Any], trainer: dict[str, Any], output_dir: Path, event_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frozen = config["frozen_training"]
    seed = 107
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(trainer)
    model = _build_model(
        trainer,
        "primary_mlp",
        rounds=128,
        tied=True,
        alpha=float(frozen["alpha"]),
        beta=float(frozen["beta"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        float(frozen["learning_rate"]),
        weight_decay=float(frozen["weight_decay"]),
    )
    all_gradients: list[float] = []
    step_four_gradients: list[float] = []
    initial_loss: float | None = None
    latest_loss = math.nan
    for step in range(1, 5):
        inputs, targets = _task_batch(
            trainer,
            "primary_mlp",
            seed=seed,
            batch_size=int(frozen["batch_size"]),
            device=device,
            stream=step - 1,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        if initial_loss is None:
            initial_loss = float(loss.detach().item())
        loss.backward()
        gradient = _gradient_norm(model)
        if not math.isfinite(gradient):
            raise RuntimeError("nonfinite gradient during seed-107 replay")
        all_gradients.append(gradient)
        if step >= 3:
            step_four_gradients.append(gradient)
        optimizer.step()
        latest_loss = float(loss.detach().item())
        del inputs, targets, loss
    checkpoint_path = _root_path(config["parent_attempt"]["resume_checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    expected_identity = ("surface_tied_R128_S107", 4, 4096)
    actual_identity = (checkpoint["cell_id"], int(checkpoint["step"]), int(checkpoint["exposure"]))
    if actual_identity != expected_identity:
        raise RuntimeError(f"resume checkpoint identity mismatch: {actual_identity}")
    _exact_equal(model.state_dict(), checkpoint["state_dict"], "model")
    _exact_equal(optimizer.state_dict(), checkpoint["optimizer"], "optimizer")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    _append_event(
        event_path,
        {
            "event": "replay_gate",
            "cell_id": expected_identity[0],
            "model_tensor_exact": True,
            "optimizer_tensor_exact": True,
            "checkpoint_sha256": config["parent_attempt"]["resume_checkpoint_sha256"],
        },
    )
    records = [
        _measurement_record(
            config,
            trainer,
            model,
            seed=seed,
            exposure=4096,
            interval_gradients=step_four_gradients,
            interval_start=2048,
            latest_loss=latest_loss,
        )
    ]
    _append_event(event_path, {"event": "measurement", **records[-1]})
    continuation_gradients: list[float] = []
    for step in range(5, 9):
        inputs, targets = _task_batch(
            trainer,
            "primary_mlp",
            seed=seed,
            batch_size=int(frozen["batch_size"]),
            device=device,
            stream=step - 1,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        gradient = _gradient_norm(model)
        if not math.isfinite(gradient):
            raise RuntimeError("nonfinite gradient during seed-107 continuation")
        all_gradients.append(gradient)
        continuation_gradients.append(gradient)
        optimizer.step()
        latest_loss = float(loss.detach().item())
        del inputs, targets, loss
    checkpoint_receipt = _save_checkpoint(
        model,
        optimizer,
        output_dir=output_dir,
        cell_id=expected_identity[0],
        step=8,
        exposure=8192,
    )
    _append_event(
        event_path,
        {"event": "checkpoint", "cell_id": expected_identity[0], "exposure": 8192, **checkpoint_receipt},
    )
    records.append(
        _measurement_record(
            config,
            trainer,
            model,
            seed=seed,
            exposure=8192,
            interval_gradients=continuation_gradients,
            interval_start=4096,
            latest_loss=latest_loss,
        )
    )
    _append_event(event_path, {"event": "measurement", **records[-1]})
    training = {
        "cell_id": expected_identity[0],
        "regime": "tied",
        "rounds": 128,
        "seed": seed,
        "steps": 8,
        "state_visit_exposures": 8192,
        "initial_loss": initial_loss,
        "final_loss": latest_loss,
        "max_gradient_norm": max(all_gradients),
        "nonfinite": False,
        "checkpoints": [checkpoint_receipt],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "replay_gate_passed": True,
    }
    _append_event(event_path, {"event": "cell_complete", **training})
    del model, optimizer, checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records, training


def _admitted_parent_records(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    events = _read_records(_root_path(config["parent_attempt"]["events_path"]))
    admitted = []
    for row in events:
        if row.get("event") != "measurement" or row.get("regime") != "tied":
            continue
        seed = int(row["seed"])
        exposure = int(row["exposure"])
        if seed in {101, 103} or (seed == 107 and exposure in {1024, 2048}):
            record = {key: value for key, value in row.items() if key not in {"event", "ts"}}
            admitted.append(record)
    return admitted


def load_registered_config(path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    config_hash = canonical_file_sha256(path)
    if config_hash != registration["config_sha256"]:
        raise RuntimeError("recovery config does not match registration")
    config = json.loads(path.read_text(encoding="utf-8"))
    parent = config["parent_attempt"]
    for name in ("config", "events", "resource", "resume_checkpoint"):
        source = _root_path(parent[f"{name}_path"])
        if not verify_file_sha256(source, parent[f"{name}_sha256"]):
            raise RuntimeError(f"recovery parent hash mismatch: {name}")
    parent_config = json.loads(_root_path(parent["config_path"]).read_text(encoding="utf-8"))
    trainer_path = _root_path(parent_config["parents"]["trainer_config"]["path"])
    trainer = json.loads(trainer_path.read_text(encoding="utf-8"))
    return config, config_hash, trainer


def validate(config: Mapping[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    resource = json.loads(_root_path(config["parent_attempt"]["resource_path"]).read_text())
    if resource["status"] != "aborted" or resource["abort_reason"] != "phase_timeout":
        raise RuntimeError("recovery parent is not the registered timeout")
    if not resource["cleanup_passed"]:
        raise RuntimeError("recovery parent cleanup failed")
    admitted = _admitted_parent_records(config)
    if len(admitted) != int(config["admission"]["parent_measurement_count"]):
        raise RuntimeError("admitted parent measurement count changed")
    identities = {(int(row["seed"]), int(row["exposure"])) for row in admitted}
    expected = {
        *((seed, exposure) for seed in (101, 103) for exposure in (1024, 2048, 4096, 8192)),
        (107, 1024),
        (107, 2048),
    }
    if identities != expected:
        raise RuntimeError("admitted parent identities changed")
    outcomes = (
        output_dir / "combined_records.jsonl",
        output_dir / "recovery_result.json",
        output_dir / "result_receipt.json",
        FINAL_RECEIPT,
    )
    existing = [str(path) for path in outcomes if path.exists()]
    if existing:
        raise RuntimeError(f"recovery outcome already exists: {existing}")
    return {
        "status": "valid",
        "config_sha256": config_hash,
        "admitted_parent_measurements": len(admitted),
        "resume_seed": 107,
        "untied_cells_to_run": 3,
        "debit_before_recovery_seconds": config["resource_accounting"][
            "debit_before_recovery_seconds"
        ],
    }


def run_recovery(
    config: dict[str, Any], config_hash: str, trainer: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    result_path = output_dir / "recovery_result.json"
    records_path = output_dir / "combined_records.jsonl"
    event_path = output_dir / "recovery_events.jsonl"
    if any(path.exists() for path in (result_path, records_path, event_path)):
        raise RuntimeError("refusing to overwrite recovery outcomes")
    parent_records = _admitted_parent_records(config)
    resumed_records, resumed_training = _resume_seed_107(config, trainer, output_dir, event_path)
    untied_records: list[dict[str, Any]] = []
    untied_training: list[dict[str, Any]] = []
    frozen = config["frozen_training"]
    for seed in frozen["seeds"]:
        records, training = _train_surface_cell(
            config,
            trainer,
            rounds=128,
            regime="untied",
            seed=int(seed),
            output_dir=output_dir,
            event_path=event_path,
            kappa_exposures=[],
            gradient_exposures=frozen["gradient_exposures"],
            checkpoint_exposures=[],
            checkpoint_pacing_seconds=0.0,
        )
        untied_records.extend(records)
        untied_training.append(training)
    all_records = parent_records + resumed_records + untied_records
    if len({row["record_id"] for row in all_records}) != len(all_records):
        raise RuntimeError("duplicate record identity in recovery")
    _write_records(records_path, all_records)
    failures = [
        row
        for row in [resumed_training, *untied_training]
        if row["nonfinite"] or int(row["state_visit_exposures"]) != 8192
    ]
    if failures:
        summary: dict[str, Any] = {
            "scientific_status": "nonfinite_boundary",
            "timing": None,
            "recovery": "nonfinite_boundary",
            "failed_cells": failures,
        }
    else:
        kappa = _grouped_kappa(all_records)
        threshold = float(config["registered_predictions"]["practical_log_change"])
        timing = classify_timing(kappa, threshold)
        if timing["classification"] in {"exposure_pinned", "step_pinned"}:
            recovery = "survived_recovered_excursion"
        else:
            changes = timing["log_changes"]
            has_drop = min(changes["E1024_to_E2048"], changes["E2048_to_E4096"]) <= -threshold
            recovery = "survived_unrecovered_excursion" if has_drop else "no_registered_excursion"
        parent_protocol = json.loads(
            _root_path(config["parent_attempt"]["config_path"]).read_text(encoding="utf-8")
        )
        lower_records = _read_records(
            _root_path(
                parent_protocol["parents"]["surface_records"]["path"]
            )
        )
        ratios = {
            32: _stress_ratio(lower_records, 32, 1024),
            64: _stress_ratio(lower_records, 64, 2048),
            128: _stress_ratio(all_records, 128, 4096),
        }
        summary = {
            "scientific_status": "complete",
            "timing": timing,
            "recovery": recovery,
            "tied_geometric_kappa": {str(key): value for key, value in sorted(kappa.items())},
            "step_four_stress_ratios": {str(key): value for key, value in sorted(ratios.items())},
            "three_depth_stress_exponent": _power_exponent(ratios),
            "replay_gate_passed": resumed_training["replay_gate_passed"],
            "new_training_cells": [resumed_training, *untied_training],
        }
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
        "summary": summary,
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(result_path, result)
    return result


def finalize(config: Mapping[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    result_path = output_dir / "recovery_result.json"
    records_path = output_dir / "combined_records.jsonl"
    resource_path = output_dir / "run.resource_receipt.json"
    if not all(path.exists() for path in (result_path, records_path, resource_path)):
        raise RuntimeError("recovery run is incomplete")
    if (output_dir / "result_receipt.json").exists() or FINAL_RECEIPT.exists():
        raise RuntimeError("recovery result already sealed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    if result["config_sha256"] != config_hash or result["status"] != "complete":
        raise RuntimeError("recovery result does not match registration")
    if not verify_file_sha256(records_path, result["records_sha256"]):
        raise RuntimeError("recovery records failed re-verification")
    if resource["status"] != "completed" or not resource["cleanup_passed"]:
        raise RuntimeError("recovery resource receipt did not pass")
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
        "timing_classification": (
            None if result["summary"]["timing"] is None else result["summary"]["timing"]["classification"]
        ),
        "recovery_classification": result["summary"]["recovery"],
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
            payload = run_recovery(config, config_hash, trainer, output)
        else:
            payload = finalize(config, config_hash, output)
        print(json.dumps(payload, sort_keys=True))
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
