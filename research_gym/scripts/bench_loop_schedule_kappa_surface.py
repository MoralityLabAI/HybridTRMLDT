"""Run the registered kappa-by-training surface under the frozen LSA construction."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from lsa.kappa_probe import estimate_kappa, spread_ratio
from research_gym.analysis.lsa_kappa_surface import (
    classify_surface,
    compare_surface_models,
    curvature_intervals,
    curvature_onset,
    geometric_surface,
    score_gradient_coupling,
    score_terminal_replication,
    score_untied_control,
)
from research_gym.scripts.bench_loop_schedule_algebra_v0_1 import (
    ROOT,
    _append_event,
    _build_model,
    _cleanup,
    _device,
    _git_head,
    _save_checkpoint,
    _set_vram_fraction,
    _sha256,
    _task_batch,
    _write_json,
    _write_records,
    measurement_seed,
    steps_for_exposure_budget,
)


DEFAULT_CONFIG = ROOT / "configs" / "loop_schedule_kappa_surface_v1.json"
REGISTRATION = ROOT / "configs" / "loop_schedule_kappa_surface_v1_registration.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "loop_schedule_kappa_surface_v1"
FINAL_RECEIPT = ROOT / "data" / "benchmarks" / "lsa_kappa_surface_v1_receipt.json"


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if path != ROOT.resolve() and ROOT.resolve() not in path.parents:
        raise RuntimeError(f"registered path escapes repository: {value}")
    return path


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_registered_config(path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    actual_hash = _sha256(path)
    if actual_hash != registration["config_sha256"]:
        raise RuntimeError(
            f"registered config hash mismatch: expected {registration['config_sha256']}, got {actual_hash}"
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["protocol_id"] != registration["protocol_id"]:
        raise RuntimeError("protocol id does not match registration")
    for parent in config["parents"].values():
        for artifact in ("config", "receipt"):
            artifact_path = _root_path(parent[f"{artifact}_path"])
            if _sha256(artifact_path) != parent[f"{artifact}_sha256"]:
                raise RuntimeError(f"parent {artifact} hash mismatch: {artifact_path.name}")
        for artifact in ("r32_records", "r64_records"):
            path_key = f"{artifact}_path"
            if path_key not in parent:
                continue
            artifact_path = _root_path(parent[path_key])
            if _sha256(artifact_path) != parent[f"{artifact}_sha256"]:
                raise RuntimeError(f"parent {artifact} hash mismatch")
    parent_path = _root_path(config["parents"]["v0_1"]["config_path"])
    parent_config = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_receipt_path = _root_path(config["parents"]["v0_1"]["receipt_path"])
    parent_receipt = json.loads(parent_receipt_path.read_text(encoding="utf-8"))
    primary = parent_receipt["phases"]["primary"]
    primary_records = _root_path(primary["records_path"])
    if _sha256(primary_records) != primary["records_sha256"]:
        raise RuntimeError("v0.1 primary records do not match the sealed parent receipt")
    return config, actual_hash, parent_config


def _outcome_paths(output_dir: Path) -> tuple[Path, ...]:
    return (
        output_dir / "surface_records.jsonl",
        output_dir / "surface_result.json",
        output_dir / "result_receipt.json",
        FINAL_RECEIPT,
    )


def validate(config: Mapping[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    frozen = config["frozen_training"]
    unreachable = []
    for rounds in frozen["rounds"]:
        quantum = int(frozen["batch_size"]) * int(rounds)
        for exposure in frozen["measurement_exposures"]:
            if int(exposure) % quantum:
                unreachable.append({"rounds": rounds, "exposure": exposure, "quantum": quantum})
    if unreachable:
        raise RuntimeError(f"registered exposure grid contains unreachable cells: {unreachable}")
    existing = [str(path) for path in _outcome_paths(output_dir) if path.exists()]
    if existing:
        raise RuntimeError(f"registered outcome already exists: {existing}")
    return {
        "status": "valid",
        "config_sha256": config_hash,
        "outcomes_present": False,
        "cell_count": len(frozen["rounds"]) * len(frozen["regimes"]) * len(frozen["seeds"]),
        "measurement_count": (
            len(frozen["rounds"])
            * len(frozen["regimes"])
            * len(frozen["seeds"])
            * len(frozen["measurement_exposures"])
        ),
    }


def _interval_summary(gradients: list[float], start: int, end: int) -> dict[str, Any]:
    if not gradients:
        return {
            "start_exposure": start,
            "end_exposure": end,
            "steps": 0,
            "maximum": None,
            "root_mean_square": None,
            "terminal": None,
        }
    return {
        "start_exposure": start,
        "end_exposure": end,
        "steps": len(gradients),
        "maximum": max(gradients),
        "root_mean_square": math.sqrt(sum(value * value for value in gradients) / len(gradients)),
        "terminal": gradients[-1],
    }


def _train_surface_cell(
    config: Mapping[str, Any],
    parent: dict[str, Any],
    *,
    rounds: int,
    regime: str,
    seed: int,
    output_dir: Path,
    event_path: Path,
    kappa_exposures: Iterable[int] | None = None,
    gradient_exposures: Iterable[int] | None = None,
    checkpoint_exposures: Iterable[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frozen = config["frozen_training"]
    tied = regime == "tied"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(parent)
    model = _build_model(
        parent,
        "primary_mlp",
        rounds=rounds,
        tied=tied,
        alpha=float(frozen["alpha"]),
        beta=float(frozen["beta"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        float(frozen["learning_rate"]),
        weight_decay=float(frozen["weight_decay"]),
    )
    batch_size = int(frozen["batch_size"])
    exposures_per_step = batch_size * rounds
    target = int(frozen["progress_target"])
    steps = steps_for_exposure_budget(target, batch_size, rounds)
    cell_id = f"surface_{regime}_R{rounds}_S{seed}"
    default_kappa = frozen.get("measurement_exposures", frozen.get("surface_exposures", ()))
    kappa_targets = {
        int(value) for value in (default_kappa if kappa_exposures is None else kappa_exposures)
    }
    gradient_targets = {
        int(value)
        for value in (kappa_targets if gradient_exposures is None else gradient_exposures)
    }
    checkpoint_targets = {
        int(value)
        for value in (
            frozen["checkpoint_exposures"]
            if checkpoint_exposures is None
            else checkpoint_exposures
        )
    }
    observation_targets = kappa_targets | gradient_targets
    all_targets = observation_targets | checkpoint_targets
    if any(value < 0 or value > target or value % exposures_per_step for value in all_targets):
        raise RuntimeError(f"{cell_id} has an unreachable registered exposure")
    if steps * exposures_per_step != target:
        raise RuntimeError("surface target must be exactly reachable")
    records: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    interval_gradients: list[float] = []
    interval_start = 0
    initial_loss: float | None = None
    final_loss = math.nan
    max_gradient_norm = 0.0
    nonfinite = False

    def observe(step: int, exposure: int) -> None:
        nonlocal interval_start
        if exposure in checkpoint_targets:
            checkpoint = _save_checkpoint(
                model,
                optimizer,
                output_dir=output_dir,
                cell_id=cell_id,
                step=step,
                exposure=exposure,
            )
            checkpoints.append({"exposure": exposure, **checkpoint})
            _append_event(
                event_path,
                {"event": "checkpoint", "cell_id": cell_id, "exposure": exposure, **checkpoint},
            )
        if exposure not in observation_targets:
            return
        interval = _interval_summary(interval_gradients, interval_start, exposure)
        record = {
            "record_id": f"{cell_id}_E{exposure}",
            "kind": "kappa_surface" if exposure in kappa_targets else "gradient_surface",
            "regime": regime,
            "rounds": rounds,
            "seed": seed,
            "exposure": exposure,
            "interval_gradient": interval,
            "latest_training_loss": None if exposure == 0 else final_loss,
        }
        if exposure in kappa_targets:
            inputs, targets = _task_batch(
                parent,
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
            record.update({"kappa": estimate.kappa, "estimate": estimate.to_dict()})
            del inputs, targets
        records.append(record)
        _append_event(event_path, {"event": "measurement", **record})
        interval_gradients.clear()
        interval_start = exposure

    _append_event(
        event_path,
        {
            "event": "cell_start",
            "cell_id": cell_id,
            "regime": regime,
            "rounds": rounds,
            "seed": seed,
            "steps": steps,
            "device": str(device),
        },
    )
    if 0 in all_targets:
        observe(0, 0)
    completed_steps = 0
    for step in range(1, steps + 1):
        inputs, targets = _task_batch(
            parent,
            "primary_mlp",
            seed=seed,
            batch_size=batch_size,
            device=device,
            stream=step - 1,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        if initial_loss is None:
            initial_loss = float(loss.detach().item())
        if not torch.isfinite(loss):
            nonfinite = True
            break
        loss.backward()
        gradient_squared = sum(
            float(parameter.grad.detach().float().square().sum().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        gradient_norm = math.sqrt(gradient_squared)
        if not math.isfinite(gradient_norm):
            nonfinite = True
            break
        interval_gradients.append(gradient_norm)
        max_gradient_norm = max(max_gradient_norm, gradient_norm)
        optimizer.step()
        final_loss = float(loss.detach().item())
        completed_steps = step
        exposure = step * exposures_per_step
        if exposure in all_targets:
            observe(step, exposure)
        del inputs, targets, loss
    training = {
        "cell_id": cell_id,
        "regime": regime,
        "rounds": rounds,
        "seed": seed,
        "steps": completed_steps,
        "state_visit_exposures": completed_steps * exposures_per_step,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "max_gradient_norm": max_gradient_norm,
        "nonfinite": nonfinite,
        "checkpoints": checkpoints,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
    }
    _append_event(event_path, {"event": "cell_complete", **training})
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records, training


def _prior_terminal_values(config: Mapping[str, Any]) -> dict[tuple[str, int, int], float]:
    paths = [
        ROOT / "experiments" / "loop_schedule_algebra_v0_1" / "primary_records.jsonl",
        _root_path(config["parents"]["high_loop_addendum"]["r32_records_path"]),
        _root_path(config["parents"]["high_loop_addendum"]["r64_records_path"]),
    ]
    prior: dict[tuple[str, int, int], float] = {}
    for path in paths:
        for row in _read_records(path):
            rounds = int(row["rounds"])
            if rounds not in {16, 32, 64} or int(row["exposure"]) != 4096:
                continue
            key = (str(row["regime"]), rounds, int(row["seed"]))
            prior[key] = float(row["kappa"])
    if len(prior) != 18:
        raise RuntimeError(f"expected 18 prior terminal cells, found {len(prior)}")
    return prior


def _spread_check(
    values: Mapping[tuple[int, int], Mapping[int, float]], maximum: float
) -> dict[str, Any]:
    spreads = {
        f"R{rounds}_E{exposure}": spread_ratio(list(seed_values.values()))
        for (rounds, exposure), seed_values in sorted(values.items())
    }
    return {"passed": max(spreads.values()) <= maximum, "maximum": max(spreads.values()), "cells": spreads}


def run_surface(
    config: dict[str, Any], config_hash: str, parent: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    existing = [path for path in _outcome_paths(output_dir) if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite registered outcomes: {[path.name for path in existing]}")
    event_path = output_dir / "surface_events.jsonl"
    if event_path.exists():
        raise RuntimeError("partial surface event stream exists; preserve it and use a new registered attempt")
    records: list[dict[str, Any]] = []
    training_cells: list[dict[str, Any]] = []
    for regime in config["frozen_training"]["regimes"]:
        for rounds in config["frozen_training"]["rounds"]:
            for seed in config["frozen_training"]["seeds"]:
                cell_records, training = _train_surface_cell(
                    config,
                    parent,
                    rounds=int(rounds),
                    regime=str(regime),
                    seed=int(seed),
                    output_dir=output_dir,
                    event_path=event_path,
                )
                records.extend(cell_records)
                training_cells.append(training)
    failures = [row for row in training_cells if row["nonfinite"] or row["state_visit_exposures"] != 4096]
    records_path = output_dir / "surface_records.jsonl"
    _write_records(records_path, records)
    if failures:
        summary: dict[str, Any] = {
            "status": "resource_or_training_failure",
            "classification": "resource_or_training_failure",
            "failed_cells": failures,
            "training_cells": training_cells,
        }
    else:
        tied_cells, tied_values = geometric_surface(records, regime="tied")
        untied_cells, untied_values = geometric_surface(records, regime="untied")
        maximum_spread = float(config["stop_branches"]["maximum_across_seed_kappa_spread"])
        tied_spread = _spread_check(tied_values, maximum_spread)
        untied_spread = _spread_check(untied_values, maximum_spread)
        if not tied_spread["passed"] or not untied_spread["passed"]:
            raise RuntimeError("registered estimator-spread stop fired")
        models = compare_surface_models(tied_cells, config)
        curvatures = curvature_intervals(tied_values, config)
        onset = curvature_onset(curvatures, config)
        gradient = score_gradient_coupling(records, onset, config)
        replication = score_terminal_replication(records, _prior_terminal_values(config), config)
        untied = score_untied_control(untied_cells, config)
        classification = classify_surface(
            winner=models["winner"],
            onset=onset,
            gradient_supported=gradient["co_localized"],
            replication_passed=replication["passed"],
            untied_passed=untied["passed"],
        )
        summary = {
            "status": "complete",
            "classification": classification,
            "surface_model_comparison": models,
            "tied_geometric_surface": tied_cells,
            "untied_geometric_surface": untied_cells,
            "depth_curvature": curvatures,
            "preterminal_curvature_onset": onset,
            "gradient_coupling": gradient,
            "terminal_replication": replication,
            "untied_control": untied,
            "tied_spread": tied_spread,
            "untied_spread": untied_spread,
            "training_cells": training_cells,
        }
    result = {
        "protocol_id": config["protocol_id"],
        "phase": "run",
        "config_sha256": config_hash,
        "git_head": _git_head(),
        "records_path": str(records_path.relative_to(ROOT)),
        "records_sha256": _sha256(records_path),
        "record_count": len(records),
        "events_path": str(event_path.relative_to(ROOT)),
        "events_sha256": _sha256(event_path),
        "summary": summary,
    }
    _write_json(output_dir / "surface_result.json", result)
    return result


def finalize(config: dict[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    result_path = output_dir / "surface_result.json"
    records_path = output_dir / "surface_records.jsonl"
    if not result_path.exists() or not records_path.exists():
        raise RuntimeError("surface phase is incomplete")
    if (output_dir / "result_receipt.json").exists() or FINAL_RECEIPT.exists():
        raise RuntimeError("final receipt already exists")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["config_sha256"] != config_hash or _sha256(records_path) != result["records_sha256"]:
        raise RuntimeError("surface integrity verification failed")
    resources = {}
    for phase in ("validate", "run"):
        path = output_dir / f"{phase}.resource_receipt.json"
        if not path.exists():
            raise RuntimeError(f"missing resource receipt: {path.name}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["status"] != "completed" or not value["cleanup_passed"]:
            raise RuntimeError(f"resource phase did not pass: {phase}")
        resources[phase] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "elapsed_seconds": value["elapsed_seconds"],
            "peak_ram_mb": value["peak_ram_mb"],
            "peak_io_mb_s": value["peak_io_mb_s"],
            "peak_vram_mb": value["peak_vram_mb"],
            "cleanup_passed": value["cleanup_passed"],
        }
    receipt = {
        "protocol_id": config["protocol_id"],
        "status": result["summary"]["status"],
        "classification": result["summary"]["classification"],
        "claim_scope": config["claim_scope"],
        "claim_boundary": config["claim_boundary"],
        "config": {
            "path": str(DEFAULT_CONFIG.relative_to(ROOT)),
            "sha256": config_hash,
            "registration_path": str(REGISTRATION.relative_to(ROOT)),
        },
        "execution_git_head": result["git_head"],
        "finalization_git_head": _git_head(),
        "surface_result": {
            "path": str(result_path.relative_to(ROOT)),
            "sha256": _sha256(result_path),
            "records_path": result["records_path"],
            "records_sha256": result["records_sha256"],
            "record_count": result["record_count"],
            "events_path": result["events_path"],
            "events_sha256": result["events_sha256"],
        },
        "primary_findings": {
            "surface_model_winner": result["summary"].get("surface_model_comparison", {}).get("winner"),
            "preterminal_curvature_onset": result["summary"].get("preterminal_curvature_onset"),
            "gradient_co_localized": result["summary"].get("gradient_coupling", {}).get("co_localized"),
            "terminal_replication_passed": result["summary"].get("terminal_replication", {}).get("passed"),
            "untied_control_passed": result["summary"].get("untied_control", {}).get("passed"),
        },
        "resources": resources,
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
    config, config_hash, parent = load_registered_config(args.config.resolve())
    _set_vram_fraction(parent, args.vram_fraction)
    output = args.output.resolve()
    try:
        if args.phase == "validate":
            print(json.dumps(validate(config, config_hash, output), sort_keys=True))
        elif args.phase == "run":
            print(json.dumps(run_surface(config, config_hash, parent, output), sort_keys=True))
        else:
            print(json.dumps(finalize(config, config_hash, output), sort_keys=True))
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
