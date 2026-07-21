"""Run the registered model-weight x AdamW-moment x suffix decomposition."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from research_gym.analysis.lsa_kappa_transient_mechanism import decompose_estimate
from research_gym.integrity import canonical_file_sha256, verify_file_sha256
from research_gym.scripts.bench_loop_schedule_algebra_v0_1 import (
    ROOT,
    _append_event,
    _cleanup,
    _device,
    _git_head,
    _save_checkpoint,
    _set_vram_fraction,
    _task_batch,
    _write_json,
    _write_records,
)
from research_gym.scripts.bench_loop_schedule_kappa_splice_v0_2_2 import (
    _load_checkpoint,
    _measure,
    _read_records,
    _source_kappa,
    continuation_streams,
)
from research_gym.scripts.bench_loop_schedule_kappa_transient_v0_2_recovery import (
    _exact_equal,
)


DEFAULT_CONFIG = ROOT / "configs" / "loop_schedule_kappa_state_moment_v0_2_3.json"
REGISTRATION = ROOT / "configs" / "loop_schedule_kappa_state_moment_v0_2_3_registration.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "loop_schedule_kappa_state_moment_v0_2_3"
FINAL_RECEIPT = ROOT / "data" / "benchmarks" / "lsa_kappa_state_moment_v0_2_3_receipt.json"

TERM_FACTORS: dict[str, tuple[str, ...]] = {
    "model_weight": ("model_weight_seed",),
    "optimizer_moment": ("optimizer_moment_seed",),
    "suffix": ("suffix_seed",),
    "model_weight_x_optimizer_moment": (
        "model_weight_seed",
        "optimizer_moment_seed",
    ),
    "model_weight_x_suffix": ("model_weight_seed", "suffix_seed"),
    "optimizer_moment_x_suffix": ("optimizer_moment_seed", "suffix_seed"),
    "three_way_interaction": (
        "model_weight_seed",
        "optimizer_moment_seed",
        "suffix_seed",
    ),
}


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if path != ROOT.resolve() and ROOT.resolve() not in path.parents:
        raise RuntimeError(f"registered path escapes repository: {value}")
    return path


def factorial_effects(cells: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    rows = list(cells)
    combinations = {
        (
            int(row["model_weight_seed"]),
            int(row["optimizer_moment_seed"]),
            int(row["suffix_seed"]),
        )
        for row in rows
    }
    expected = {
        (model_seed, optimizer_seed, suffix_seed)
        for model_seed in (211, 223)
        for optimizer_seed in (211, 223)
        for suffix_seed in (211, 223)
    }
    if len(rows) != 8 or combinations != expected:
        raise ValueError("factorial analysis requires each registered cell exactly once")
    levels = {211: -1.0, 223: 1.0}
    effects: dict[str, float] = {}
    for name, factors in TERM_FACTORS.items():
        contrast = 0.0
        for row in rows:
            sign = math.prod(levels[int(row[factor])] for factor in factors)
            contrast += sign * float(row["endpoint_log_change"])
        effects[name] = contrast / 4.0
    return effects


def classify_factorial(
    effects: Mapping[str, float], *, dominance_ratio: float = 1.5
) -> dict[str, Any]:
    if set(effects) != set(TERM_FACTORS):
        raise ValueError("factorial classification requires all registered effects")
    ranking = sorted(
        ((name, value, abs(value)) for name, value in effects.items()),
        key=lambda item: (-item[2], item[0]),
    )
    largest, runner_up = ranking[0], ranking[1]
    ratio = math.inf if runner_up[2] == 0.0 and largest[2] > 0.0 else (
        largest[2] / runner_up[2] if runner_up[2] else 1.0
    )
    classification = (
        f"{largest[0]}_dominant"
        if largest[2] > 0.0 and ratio >= dominance_ratio
        else "factorial_unresolved"
    )
    return {
        "classification": classification,
        "dominant_term": largest[0] if classification != "factorial_unresolved" else None,
        "largest_to_runner_up_ratio": ratio,
        "absolute_effect_ranking": [
            {"term": name, "effect": value, "absolute_effect": absolute}
            for name, value, absolute in ranking
        ],
    }


def _checkpoint_payload(path: Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def _find_source_record(
    records: Iterable[Mapping[str, Any]], order_seed: int, exposure: int
) -> Mapping[str, Any]:
    matches = [
        row
        for row in records
        if int(row["seed_channels"]["data_order_seed"]) == order_seed
        and int(row["exposure"]) == exposure
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one source record for D{order_seed} E{exposure}")
    return matches[0]


def _find_splice_record(
    records: Iterable[Mapping[str, Any]], cell_id: str, exposure: int
) -> Mapping[str, Any]:
    matches = [
        row
        for row in records
        if str(row["cell_id"]) == cell_id and int(row["exposure"]) == exposure
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one splice record for {cell_id} E{exposure}")
    return matches[0]


def _component_checkpoint(config: Mapping[str, Any], seed: int) -> Path:
    return _root_path(config["parents"][f"d{seed}_e2048"]["path"])


def _hybrid_checkpoint(
    trainer: dict[str, Any],
    config: Mapping[str, Any],
    *,
    model_seed: int,
    optimizer_seed: int,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, Any]]:
    model_path = _component_checkpoint(config, model_seed)
    optimizer_path = _component_checkpoint(config, optimizer_seed)
    model, optimizer, model_payload = _load_checkpoint(trainer, config, model_path)
    optimizer_payload = _checkpoint_payload(optimizer_path, _device(trainer))
    if int(model_payload["exposure"]) != 2048 or int(optimizer_payload["exposure"]) != 2048:
        raise RuntimeError("component checkpoint exposure is not E2048")
    if int(model_payload["step"]) != 4 or int(optimizer_payload["step"]) != 4:
        raise RuntimeError("component checkpoint optimizer step is not 4")
    optimizer.load_state_dict(optimizer_payload["optimizer"])
    _exact_equal(model.state_dict(), model_payload["state_dict"], "model_component")
    _exact_equal(
        optimizer.state_dict(), optimizer_payload["optimizer"], "optimizer_component"
    )
    return model, optimizer, {
        "model_source_path": model_path.relative_to(ROOT).as_posix(),
        "model_source_sha256": canonical_file_sha256(model_path),
        "optimizer_source_path": optimizer_path.relative_to(ROOT).as_posix(),
        "optimizer_source_sha256": canonical_file_sha256(optimizer_path),
        "model_tensor_exact": True,
        "optimizer_tensor_exact": True,
        "source_exposure": 2048,
        "source_step": 4,
    }


def _continue_hybrid_cell(
    config: Mapping[str, Any],
    trainer: dict[str, Any],
    *,
    model_seed: int,
    optimizer_seed: int,
    suffix_seed: int,
    output_dir: Path,
    event_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frozen = config["frozen_training"]
    cell_id = f"state_moment_M{model_seed}_O{optimizer_seed}_D{suffix_seed}"
    model, optimizer, gate = _hybrid_checkpoint(
        trainer,
        config,
        model_seed=model_seed,
        optimizer_seed=optimizer_seed,
    )
    start_exposure = int(frozen["source_exposure"])
    end_exposure = int(frozen["terminal_exposure"])
    quantum = int(frozen["batch_size"]) * int(frozen["rounds"])
    streams = continuation_streams(start_exposure, end_exposure, quantum=quantum)
    targets = {int(value) for value in frozen["measurement_exposures"]}
    if targets != {2560, 3072, 3584, 4096}:
        raise RuntimeError("registered measurement grid changed")
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
            "event": "hybrid_component_gate_passed",
            "cell_id": cell_id,
            "model_weight_seed": model_seed,
            "optimizer_moment_seed": optimizer_seed,
            "suffix_seed": suffix_seed,
            "component_integrity": gate,
            "streams": streams,
            "device": str(device),
        },
    )
    completed_step = 4
    try:
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
                del inputs, expected, loss
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
                del inputs, expected, loss
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
                    state_seed=model_seed,
                    suffix_seed=suffix_seed,
                    exposure=exposure,
                    interval_gradients=interval_gradients,
                    interval_start=interval_start,
                    latest_loss=latest_loss,
                )
                record["kind"] = "kappa_state_moment"
                record["model_weight_seed"] = model_seed
                record["optimizer_moment_seed"] = optimizer_seed
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
            "model_weight_seed": model_seed,
            "optimizer_moment_seed": optimizer_seed,
            "suffix_seed": suffix_seed,
            "component_integrity": gate,
            "source_exposure": start_exposure,
            "terminal_exposure": terminal_exposure,
            "steps_completed": completed_step - 4,
            "streams": streams,
            "nonfinite": nonfinite,
            "maximum_gradient_norm": maximum_gradient,
            "latest_training_loss": latest_loss,
            "terminal_checkpoint": terminal_checkpoint,
        }
        _append_event(event_path, {"event": "hybrid_cell_complete", **summary})
        return records, summary
    finally:
        del model, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _standardized_cell(
    *,
    model_seed: int,
    optimizer_seed: int,
    suffix_seed: int,
    start_kappa: float,
    endpoint_record: Mapping[str, Any],
    origin: str,
    practical_threshold: float,
) -> dict[str, Any]:
    endpoint_kappa = float(endpoint_record["kappa"])
    change = math.log(endpoint_kappa / start_kappa)
    if "sensitivity_interference_ratio" in endpoint_record:
        sensitivity = float(endpoint_record["sensitivity_interference_ratio"])
        gradient = float(endpoint_record["gradient_interference_ratio"])
    else:
        decomposition = decompose_estimate(endpoint_record["estimate"])
        sensitivity = float(decomposition["sensitivity_interference_ratio"])
        gradient = float(decomposition["gradient_interference_ratio"])
    return {
        "cell_id": f"M{model_seed}_O{optimizer_seed}_D{suffix_seed}",
        "model_weight_seed": model_seed,
        "optimizer_moment_seed": optimizer_seed,
        "suffix_seed": suffix_seed,
        "origin": origin,
        "start_kappa": start_kappa,
        "endpoint_kappa": endpoint_kappa,
        "endpoint_log_change": change,
        "recovered_by_endpoint": change >= practical_threshold,
        "endpoint_sensitivity_interference_ratio": sensitivity,
        "endpoint_gradient_interference_ratio": gradient,
    }


def load_registered_config(path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    config_hash = canonical_file_sha256(path)
    if config_hash != registration["config_sha256"]:
        raise RuntimeError("state-moment config does not match frozen registration")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["protocol_id"] != registration["protocol_id"]:
        raise RuntimeError("state-moment protocol id does not match registration")
    for parent in config["parents"].values():
        source = _root_path(parent["path"])
        if not verify_file_sha256(source, parent["sha256"]):
            raise RuntimeError(f"parent hash mismatch: {source.name}")
    trainer_path = _root_path(config["parents"]["trainer_config"]["path"])
    return config, config_hash, json.loads(trainer_path.read_text(encoding="utf-8"))


def validate(config: Mapping[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    frozen = config["frozen_training"]
    if int(frozen["rounds"]) != 64 or int(frozen["construction_seed"]) != 103:
        raise RuntimeError("registered state-moment construction changed")
    expected_new = {
        (211, 223, 211),
        (211, 223, 223),
        (223, 211, 211),
        (223, 211, 223),
    }
    observed_new = {
        (
            int(cell["model_weight_seed"]),
            int(cell["optimizer_moment_seed"]),
            int(cell["suffix_seed"]),
        )
        for cell in config["new_swap_cells"]
    }
    if observed_new != expected_new or len(config["new_swap_cells"]) != 4:
        raise RuntimeError("registered new swap cells changed")
    for seed in (211, 223):
        payload = _checkpoint_payload(_component_checkpoint(config, seed))
        if int(payload["exposure"]) != 2048 or int(payload["step"]) != 4:
            raise RuntimeError(f"checkpoint metadata changed for D{seed}")
    outcomes = (
        output_dir / "state_moment_records.jsonl",
        output_dir / "state_moment_result.json",
        output_dir / "result_receipt.json",
        FINAL_RECEIPT,
    )
    existing = [str(path) for path in outcomes if path.exists()]
    if existing:
        raise RuntimeError(f"registered state-moment outcome already exists: {existing}")
    return {
        "status": "valid",
        "config_sha256": config_hash,
        "new_continuation_cells": 4,
        "imported_sealed_controls": 4,
        "new_prefix_training": 0,
        "outcomes_present": False,
    }


def run(
    config: Mapping[str, Any],
    config_hash: str,
    trainer: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    records_path = output_dir / "state_moment_records.jsonl"
    result_path = output_dir / "state_moment_result.json"
    event_path = output_dir / "state_moment_events.jsonl"
    if any(path.exists() for path in (records_path, result_path, event_path)):
        raise RuntimeError("refusing to overwrite state-moment outcomes")
    source_records = _read_records(_root_path(config["parents"]["source_records"]["path"]))
    splice_records = _read_records(_root_path(config["parents"]["splice_records"]["path"]))
    starts = {seed: _source_kappa(source_records, seed, 2048) for seed in (211, 223)}
    threshold = float(config["registered_endpoints"]["practical_log_change"])

    all_new_records: list[dict[str, Any]] = []
    new_summaries: list[dict[str, Any]] = []
    new_endpoint_records: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    for spec in config["new_swap_cells"]:
        model_seed = int(spec["model_weight_seed"])
        optimizer_seed = int(spec["optimizer_moment_seed"])
        suffix_seed = int(spec["suffix_seed"])
        records, summary = _continue_hybrid_cell(
            config,
            trainer,
            model_seed=model_seed,
            optimizer_seed=optimizer_seed,
            suffix_seed=suffix_seed,
            output_dir=output_dir,
            event_path=event_path,
        )
        if not summary["nonfinite"] and len(records) != 4:
            raise RuntimeError(f"incomplete measurement grid for {summary['cell_id']}")
        all_new_records.extend(records)
        new_summaries.append(summary)
        if records:
            new_endpoint_records[(model_seed, optimizer_seed, suffix_seed)] = records[-1]
    _write_records(records_path, all_new_records)
    failures = [cell for cell in new_summaries if cell["nonfinite"]]

    cells = [
        _standardized_cell(
            model_seed=211,
            optimizer_seed=211,
            suffix_seed=211,
            start_kappa=starts[211],
            endpoint_record=_find_source_record(source_records, 211, 4096),
            origin="sealed_v0_2_1",
            practical_threshold=threshold,
        ),
        _standardized_cell(
            model_seed=211,
            optimizer_seed=211,
            suffix_seed=223,
            start_kappa=starts[211],
            endpoint_record=_find_splice_record(splice_records, "splice_S211_D223", 4096),
            origin="sealed_v0_2_2",
            practical_threshold=threshold,
        ),
        _standardized_cell(
            model_seed=223,
            optimizer_seed=223,
            suffix_seed=211,
            start_kappa=starts[223],
            endpoint_record=_find_splice_record(splice_records, "splice_S223_D211", 4096),
            origin="sealed_v0_2_2",
            practical_threshold=threshold,
        ),
        _standardized_cell(
            model_seed=223,
            optimizer_seed=223,
            suffix_seed=223,
            start_kappa=starts[223],
            endpoint_record=_find_source_record(source_records, 223, 4096),
            origin="sealed_v0_2_1",
            practical_threshold=threshold,
        ),
    ]
    if not failures:
        for spec in config["new_swap_cells"]:
            key = (
                int(spec["model_weight_seed"]),
                int(spec["optimizer_moment_seed"]),
                int(spec["suffix_seed"]),
            )
            cells.append(
                _standardized_cell(
                    model_seed=key[0],
                    optimizer_seed=key[1],
                    suffix_seed=key[2],
                    start_kappa=starts[key[0]],
                    endpoint_record=new_endpoint_records[key],
                    origin="new_v0_2_3_swap",
                    practical_threshold=threshold,
                )
            )
        effects = factorial_effects(cells)
        classification = classify_factorial(
            effects,
            dominance_ratio=float(config["registered_endpoints"]["dominance_ratio"]),
        )
        scientific_status = "complete"
    else:
        effects = None
        classification = None
        scientific_status = "resource_or_training_failure"
    result = {
        "protocol_id": config["protocol_id"],
        "status": "complete",
        "phase": "run",
        "config_sha256": config_hash,
        "git_head": _git_head(),
        "records_path": records_path.relative_to(ROOT).as_posix(),
        "records_sha256": canonical_file_sha256(records_path),
        "record_count": len(all_new_records),
        "events_path": event_path.relative_to(ROOT).as_posix(),
        "events_sha256": canonical_file_sha256(event_path),
        "summary": {
            "scientific_status": scientific_status,
            "factorial_effects": effects,
            "factorial_classification": classification,
            "zero_residual_degrees_of_freedom": True,
            "cells": sorted(
                cells,
                key=lambda cell: (
                    cell["model_weight_seed"],
                    cell["optimizer_moment_seed"],
                    cell["suffix_seed"],
                ),
            ),
            "new_cell_execution": new_summaries,
            "failed_cells": failures,
        },
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(result_path, result)
    return result


def finalize(config: Mapping[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    result_path = output_dir / "state_moment_result.json"
    records_path = output_dir / "state_moment_records.jsonl"
    resource_path = output_dir / "run.resource_receipt.json"
    if not all(path.exists() for path in (result_path, records_path, resource_path)):
        raise RuntimeError("state-moment run is incomplete")
    if (output_dir / "result_receipt.json").exists() or FINAL_RECEIPT.exists():
        raise RuntimeError("state-moment result already sealed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    if result["config_sha256"] != config_hash or result["status"] != "complete":
        raise RuntimeError("state-moment result does not match registered config")
    if not verify_file_sha256(records_path, result["records_sha256"]):
        raise RuntimeError("state-moment records failed re-verification")
    if resource["status"] != "completed" or not resource["cleanup_passed"]:
        raise RuntimeError("state-moment resource receipt did not pass")
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
        "factorial_classification": result["summary"]["factorial_classification"],
        "zero_residual_degrees_of_freedom": True,
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
