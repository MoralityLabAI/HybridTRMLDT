"""Execute the registered post-partial recovery of the kappa surface study."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from lsa.kappa_probe import fit_power_law, geometric_mean
from research_gym.analysis.lsa_kappa_surface import (
    compare_surface_models,
    curvature_intervals,
    curvature_onset,
    geometric_surface,
)
from research_gym.scripts.bench_loop_schedule_algebra_v0_1 import (
    ROOT,
    _cleanup,
    _git_head,
    _set_vram_fraction,
    _sha256,
    _write_json,
    _write_records,
)
from research_gym.scripts.bench_loop_schedule_kappa_surface import (
    _prior_terminal_values,
    _read_records,
    _root_path,
    _train_surface_cell,
)


DEFAULT_CONFIG = ROOT / "configs" / "loop_schedule_kappa_surface_v1_recovery1.json"
REGISTRATION = (
    ROOT / "configs" / "loop_schedule_kappa_surface_v1_recovery1_registration.json"
)
ORIGINAL_CONFIG = ROOT / "configs" / "loop_schedule_kappa_surface_v1.json"
ORIGINAL_REGISTRATION = ROOT / "configs" / "loop_schedule_kappa_surface_v1_registration.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "loop_schedule_kappa_surface_v1_recovery1"
FINAL_RECEIPT = ROOT / "data" / "benchmarks" / "lsa_kappa_surface_v1_recovery1_receipt.json"


def load_registered_config(path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    config_hash = _sha256(path)
    if config_hash != registration["config_sha256"]:
        raise RuntimeError("recovery config hash does not match registration")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["protocol_id"] != registration["protocol_id"]:
        raise RuntimeError("recovery protocol id does not match registration")
    observed = config["observed_before_recovery"]
    for name in ("failure_receipt", "partial_ledger"):
        source = _root_path(observed[f"{name}_path"])
        if _sha256(source) != observed[f"{name}_sha256"]:
            raise RuntimeError(f"recovery source hash mismatch: {name}")
    original_registration = json.loads(ORIGINAL_REGISTRATION.read_text(encoding="utf-8"))
    if _sha256(ORIGINAL_CONFIG) != original_registration["config_sha256"]:
        raise RuntimeError("original surface config no longer matches registration")
    original = json.loads(ORIGINAL_CONFIG.read_text(encoding="utf-8"))
    parent_path = _root_path(original["parents"]["v0_1"]["config_path"])
    if _sha256(parent_path) != original["parents"]["v0_1"]["config_sha256"]:
        raise RuntimeError("frozen trainer parent hash mismatch")
    return config, config_hash, json.loads(parent_path.read_text(encoding="utf-8"))


def _admitted_partial_records(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    observed = config["observed_before_recovery"]
    path = _root_path(observed["partial_ledger_path"])
    if _sha256(path) != observed["partial_ledger_sha256"]:
        raise RuntimeError("partial ledger changed before admission")
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    completed = {
        str(row["cell_id"]): row
        for row in events
        if row.get("event") == "cell_complete"
        and not row.get("nonfinite", True)
        and int(row.get("state_visit_exposures", -1)) == 4096
    }
    expected = set(observed["fully_completed_cells"])
    if set(completed) != expected:
        raise RuntimeError("partial-ledger complete-cell inventory changed")
    records_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("event") != "measurement":
            continue
        cell_id = str(event["record_id"]).rsplit("_E", 1)[0]
        if cell_id not in expected:
            continue
        record = {key: value for key, value in event.items() if key not in {"event", "ts"}}
        records_by_cell[cell_id].append(record)
    required = set(config["admission_rule"]["required_measurements"])
    for cell_id in sorted(expected):
        exposures = {int(row["exposure"]) for row in records_by_cell[cell_id]}
        if exposures != required:
            raise RuntimeError(f"partial cell does not have its complete measurement set: {cell_id}")
    return [row for cell_id in sorted(records_by_cell) for row in records_by_cell[cell_id]]


def _outcome_paths(output_dir: Path) -> tuple[Path, ...]:
    return (
        output_dir / "r64_tied_result.json",
        output_dir / "untied_gradients_result.json",
        output_dir / "result_receipt.json",
        FINAL_RECEIPT,
    )


def validate(config: Mapping[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    admitted = _admitted_partial_records(config)
    existing = [str(path) for path in _outcome_paths(output_dir) if path.exists()]
    if existing:
        raise RuntimeError(f"recovery outcome already exists: {existing}")
    return {
        "status": "valid",
        "config_sha256": config_hash,
        "admitted_measurements": len(admitted),
        "admitted_cells": len(config["observed_before_recovery"]["fully_completed_cells"]),
        "incomplete_r64_cell_admitted": False,
        "attempt_1_debit_seconds": config["resource_accounting"]["attempt_1_debit_seconds"],
    }


def _phase_result(
    config: Mapping[str, Any],
    config_hash: str,
    output_dir: Path,
    phase: str,
    records: Iterable[dict[str, Any]],
    training_cells: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(records)
    records_path = output_dir / f"{phase}_records.jsonl"
    _write_records(records_path, rows)
    failures = [
        row
        for row in training_cells
        if row["nonfinite"] or int(row["state_visit_exposures"]) != 4096
    ]
    result = {
        "protocol_id": config["protocol_id"],
        "phase": phase,
        "status": "complete" if not failures else "resource_or_training_failure",
        "config_sha256": config_hash,
        "git_head": _git_head(),
        "records_path": str(records_path.relative_to(ROOT)),
        "records_sha256": _sha256(records_path),
        "record_count": len(rows),
        "training_cells": training_cells,
        "failed_cells": failures,
    }
    _write_json(output_dir / f"{phase}_result.json", result)
    return result


def run_r64_tied(
    config: dict[str, Any], config_hash: str, parent: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    phase = "r64_tied"
    result_path = output_dir / f"{phase}_result.json"
    if result_path.exists() or (output_dir / f"{phase}_events.jsonl").exists():
        raise RuntimeError("refusing to overwrite R64 recovery outcomes")
    specification = config["new_execution"][phase]
    records: list[dict[str, Any]] = []
    training_cells = []
    event_path = output_dir / f"{phase}_events.jsonl"
    for seed in specification["seeds"]:
        cell_records, training = _train_surface_cell(
            config,
            parent,
            rounds=64,
            regime="tied",
            seed=int(seed),
            output_dir=output_dir,
            event_path=event_path,
            kappa_exposures=specification["kappa_exposures"],
            gradient_exposures=specification["gradient_exposures"],
            checkpoint_exposures=config["frozen_training"]["checkpoint_exposures"],
        )
        records.extend(cell_records)
        training_cells.append(training)
    return _phase_result(config, config_hash, output_dir, phase, records, training_cells)


def run_untied_gradients(
    config: dict[str, Any], config_hash: str, parent: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    phase = "untied_gradients"
    result_path = output_dir / f"{phase}_result.json"
    if result_path.exists() or (output_dir / f"{phase}_events.jsonl").exists():
        raise RuntimeError("refusing to overwrite untied recovery outcomes")
    specification = config["new_execution"][phase]
    records: list[dict[str, Any]] = []
    training_cells = []
    event_path = output_dir / f"{phase}_events.jsonl"
    for rounds in specification["rounds"]:
        for seed in specification["seeds"]:
            cell_records, training = _train_surface_cell(
                config,
                parent,
                rounds=int(rounds),
                regime="untied",
                seed=int(seed),
                output_dir=output_dir,
                event_path=event_path,
                kappa_exposures=specification["kappa_exposures"],
                gradient_exposures=specification["gradient_exposures"],
                checkpoint_exposures=config["frozen_training"]["checkpoint_exposures"],
            )
            records.extend(cell_records)
            training_cells.append(training)
    return _phase_result(config, config_hash, output_dir, phase, records, training_cells)


def _require_phase(output_dir: Path, phase: str, config_hash: str) -> dict[str, Any]:
    result_path = output_dir / f"{phase}_result.json"
    resource_path = output_dir / f"{phase}.resource_receipt.json"
    if not result_path.exists() or not resource_path.exists():
        raise RuntimeError(f"recovery phase is incomplete: {phase}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    if result["config_sha256"] != config_hash or result["status"] != "complete":
        raise RuntimeError(f"recovery result did not pass: {phase}")
    if resource["status"] != "completed" or not resource["cleanup_passed"]:
        raise RuntimeError(f"recovery resource receipt did not pass: {phase}")
    records_path = _root_path(result["records_path"])
    if _sha256(records_path) != result["records_sha256"]:
        raise RuntimeError(f"recovery records changed: {phase}")
    return {"result": result, "resource": resource}


def _prior_tied_replication(
    records: Iterable[Mapping[str, Any]], original: Mapping[str, Any]
) -> dict[str, Any]:
    prior = _prior_terminal_values(original)
    comparisons = []
    for row in records:
        if row.get("regime") != "tied" or int(row["exposure"]) != 4096:
            continue
        key = ("tied", int(row["rounds"]), int(row["seed"]))
        difference = abs(math.log(float(row["kappa"]) / prior[key]))
        comparisons.append(
            {
                "rounds": key[1],
                "seed": key[2],
                "prior_kappa": prior[key],
                "new_kappa": float(row["kappa"]),
                "absolute_log_ratio": difference,
                "passed": difference <= 0.02,
            }
        )
    return {
        "passed": len(comparisons) == 9 and all(row["passed"] for row in comparisons),
        "maximum_absolute_log_ratio": max(row["absolute_log_ratio"] for row in comparisons),
        "comparisons": comparisons,
    }


def _sealed_untied_terminal_control(original: Mapping[str, Any]) -> dict[str, Any]:
    source_paths = [
        ROOT / "experiments" / "loop_schedule_algebra_v0_1" / "primary_records.jsonl",
        _root_path(original["parents"]["high_loop_addendum"]["r32_records_path"]),
        _root_path(original["parents"]["high_loop_addendum"]["r64_records_path"]),
    ]
    values: dict[int, list[float]] = defaultdict(list)
    for path in source_paths:
        for row in _read_records(path):
            if row["regime"] == "untied" and int(row["exposure"]) == 4096:
                values[int(row["rounds"])].append(float(row["kappa"]))
    rounds = [2, 4, 8, 16, 32, 64]
    if any(len(values[rounds_value]) != 3 for rounds_value in rounds):
        raise RuntimeError("sealed untied terminal control is incomplete")
    kappas = [geometric_mean(values[rounds_value]) for rounds_value in rounds]
    fit = fit_power_law(rounds, kappas)
    return {
        "passed": abs(fit.gamma) <= 0.1,
        "equivalence_region": [-0.1, 0.1],
        "fit": fit.to_dict(),
        "geometric_mean_kappa": dict(zip(map(str, rounds), kappas)),
        "source": "sealed parent terminal records; no recovery untied-kappa outcome",
    }


def _gradient_coupling(
    records: Iterable[Mapping[str, Any]], onset: int | None, config: Mapping[str, Any]
) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in records:
        exposure = int(row["exposure"])
        if exposure == 0:
            continue
        grouped[(str(row["regime"]), int(row["rounds"]), exposure)].append(
            float(row["interval_gradient"]["maximum"])
        )
    ratios = {}
    for rounds in (16, 32, 64):
        for exposure in (512, 1024, 2048, 4096):
            tied = grouped[("tied", rounds, exposure)]
            untied = grouped[("untied", rounds, exposure)]
            if len(tied) != 3 or len(untied) != 3:
                raise RuntimeError(f"incomplete gradient control at R={rounds}, E={exposure}")
            ratios[(rounds, exposure)] = geometric_mean(tied) / geometric_mean(untied)
    onset_ratios = None
    passed = False
    if onset is not None:
        onset_ratios = {str(rounds): ratios[(rounds, onset)] for rounds in (16, 32, 64)}
        passed = (
            onset_ratios["64"] >= float(config["gradient_coupling"]["support_threshold"])
            and onset_ratios["64"] > onset_ratios["16"]
            and onset_ratios["64"] > onset_ratios["32"]
        )
    return {
        "co_localized": passed,
        "onset_ratios": onset_ratios,
        "stress_ratios": {
            f"R{rounds}_E{exposure}": value
            for (rounds, exposure), value in sorted(ratios.items())
        },
        "association_diagnostic": "omitted_by_registered_recovery",
    }


def _classify(
    winner: str,
    onset: int | None,
    gradient: bool,
    replication: bool,
    untied: bool,
) -> str:
    if not replication or not untied:
        return "integrity_failure"
    if winner == "r64_change_point" and onset is not None:
        return (
            "post_partial_depth_training_transition"
            if gradient
            else "post_partial_geometric_transition_without_gradient_support"
        )
    if winner in {"separable", "smooth_interaction"} and onset is None:
        return "post_partial_smooth_surface"
    return "form_unresolved"


def finalize(config: dict[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    if (output_dir / "result_receipt.json").exists() or FINAL_RECEIPT.exists():
        raise RuntimeError("recovery receipt already exists")
    phases = {
        phase: _require_phase(output_dir, phase, config_hash)
        for phase in ("r64_tied", "untied_gradients")
    }
    original = json.loads(ORIGINAL_CONFIG.read_text(encoding="utf-8"))
    partial = _admitted_partial_records(config)
    r64 = _read_records(_root_path(phases["r64_tied"]["result"]["records_path"]))
    untied_gradients = _read_records(
        _root_path(phases["untied_gradients"]["result"]["records_path"])
    )
    combined = partial + r64 + untied_gradients
    combined_path = output_dir / "combined_records.jsonl"
    _write_records(combined_path, combined)
    surface_exposures = set(config["frozen_training"]["surface_exposures"])
    tied_kappa = [
        row
        for row in partial + r64
        if row["regime"] == "tied"
        and "kappa" in row
        and int(row["exposure"]) in surface_exposures
    ]
    tied_cells, tied_values = geometric_surface(tied_kappa, regime="tied")
    if len(tied_cells) != int(config["surface_models"]["fit_cells"]):
        raise RuntimeError("recovery surface does not contain 12 registered cells")
    models = compare_surface_models(tied_cells, config)
    curvatures = curvature_intervals(tied_values, config)
    onset = curvature_onset(curvatures, config)
    gradients = _gradient_coupling(partial + r64 + untied_gradients, onset, config)
    replication = _prior_tied_replication(tied_kappa, original)
    untied = _sealed_untied_terminal_control(original)
    classification = _classify(
        models["winner"], onset, gradients["co_localized"], replication["passed"], untied["passed"]
    )
    resource_elapsed = sum(
        float(value["resource"]["elapsed_seconds"]) for value in phases.values()
    )
    aggregate_elapsed = float(config["resource_accounting"]["attempt_1_debit_seconds"]) + resource_elapsed
    if aggregate_elapsed > float(config["resource_accounting"]["aggregate_limit_seconds"]):
        raise RuntimeError("aggregate one-GPU-hour cap exceeded before finalization")
    summary = {
        "classification": classification,
        "status_label": config["status_label"],
        "surface_model_comparison": models,
        "tied_geometric_surface": tied_cells,
        "depth_curvature": curvatures,
        "preterminal_curvature_onset": onset,
        "gradient_coupling": gradients,
        "tied_terminal_replication": replication,
        "sealed_untied_terminal_control": untied,
        "admitted_attempt_1_measurements": len(partial),
        "excluded_incomplete_r64_cell": config["observed_before_recovery"]["incomplete_cell"],
    }
    result_path = output_dir / "recovery_result.json"
    _write_json(result_path, summary)
    receipt = {
        "protocol_id": config["protocol_id"],
        "status": "complete",
        "classification": classification,
        "status_label": config["status_label"],
        "claim_scope": config["claim_scope"],
        "claim_boundary": config["claim_boundary"],
        "config": {
            "path": str(DEFAULT_CONFIG.relative_to(ROOT)),
            "sha256": config_hash,
            "registration_path": str(REGISTRATION.relative_to(ROOT)),
        },
        "execution_git_heads": {
            phase: value["result"]["git_head"] for phase, value in phases.items()
        },
        "finalization_git_head": _git_head(),
        "artifacts": {
            "combined_records_path": str(combined_path.relative_to(ROOT)),
            "combined_records_sha256": _sha256(combined_path),
            "combined_record_count": len(combined),
            "result_path": str(result_path.relative_to(ROOT)),
            "result_sha256": _sha256(result_path),
        },
        "primary_findings": {
            "surface_model_winner": models["winner"],
            "preterminal_curvature_onset": onset,
            "gradient_co_localized": gradients["co_localized"],
            "tied_terminal_replication_passed": replication["passed"],
            "sealed_untied_terminal_point_equivalence_passed": untied["passed"],
        },
        "resources": {
            "attempt_1_debit_seconds": config["resource_accounting"]["attempt_1_debit_seconds"],
            "recovery_phase_elapsed_seconds": resource_elapsed,
            "aggregate_elapsed_seconds": aggregate_elapsed,
            "aggregate_limit_seconds": config["resource_accounting"]["aggregate_limit_seconds"],
            "phases": {
                phase: {
                    "receipt_path": str(
                        (output_dir / f"{phase}.resource_receipt.json").relative_to(ROOT)
                    ),
                    "receipt_sha256": _sha256(output_dir / f"{phase}.resource_receipt.json"),
                    "elapsed_seconds": value["resource"]["elapsed_seconds"],
                    "peak_ram_mb": value["resource"]["peak_ram_mb"],
                    "peak_io_mb_s": value["resource"]["peak_io_mb_s"],
                    "peak_vram_mb": value["resource"]["peak_vram_mb"],
                    "cleanup_passed": value["resource"]["cleanup_passed"],
                }
                for phase, value in phases.items()
            },
        },
    }
    _write_json(output_dir / "result_receipt.json", receipt)
    _write_json(FINAL_RECEIPT, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--phase", choices=("validate", "r64_tied", "untied_gradients", "finalize"), required=True
    )
    parser.add_argument("--vram-fraction", type=float)
    args = parser.parse_args()
    config, config_hash, parent = load_registered_config(args.config.resolve())
    _set_vram_fraction(parent, args.vram_fraction)
    output = args.output.resolve()
    try:
        if args.phase == "validate":
            print(json.dumps(validate(config, config_hash, output), sort_keys=True))
        elif args.phase == "r64_tied":
            print(json.dumps(run_r64_tied(config, config_hash, parent, output), sort_keys=True))
        elif args.phase == "untied_gradients":
            print(json.dumps(run_untied_gradients(config, config_hash, parent, output), sort_keys=True))
        else:
            print(json.dumps(finalize(config, config_hash, output), sort_keys=True))
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
