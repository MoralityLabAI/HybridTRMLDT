"""Pacing-only recovery of the untied gradient-control phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

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
    _read_records,
    _root_path,
    _train_surface_cell,
)
from research_gym.scripts.bench_loop_schedule_kappa_surface_recovery import (
    ORIGINAL_CONFIG,
    _admitted_partial_records,
    _classify,
    _gradient_coupling,
    _phase_result,
    _prior_tied_replication,
    _sealed_untied_terminal_control,
    load_registered_config as load_recovery1_config,
)


DEFAULT_CONFIG = ROOT / "configs" / "loop_schedule_kappa_surface_v1_recovery2.json"
REGISTRATION = (
    ROOT / "configs" / "loop_schedule_kappa_surface_v1_recovery2_registration.json"
)
DEFAULT_OUTPUT = ROOT / "experiments" / "loop_schedule_kappa_surface_v1_recovery2"
FINAL_RECEIPT = ROOT / "data" / "benchmarks" / "lsa_kappa_surface_v1_recovery2_receipt.json"


def load_registered_config(
    path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    config_hash = _sha256(path)
    if config_hash != registration["config_sha256"]:
        raise RuntimeError("Recovery 2 config hash does not match registration")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["protocol_id"] != registration["protocol_id"]:
        raise RuntimeError("Recovery 2 protocol id does not match registration")
    parent = config["parent_recovery"]
    for name in ("config", "r64_result", "r64_records", "r64_resource"):
        source = _root_path(parent[f"{name}_path"])
        if _sha256(source) != parent[f"{name}_sha256"]:
            raise RuntimeError(f"Recovery 2 parent hash mismatch: {name}")
    failed = config["failed_untied_attempt"]
    for name in ("event", "resource"):
        source = _root_path(failed[f"{name}_path"])
        if _sha256(source) != failed[f"{name}_sha256"]:
            raise RuntimeError(f"Recovery 2 failed-attempt hash mismatch: {name}")
    recovery1, _, trainer_parent = load_recovery1_config(
        _root_path(parent["config_path"])
    )
    return config, config_hash, recovery1, trainer_parent


def validate(
    config: Mapping[str, Any],
    config_hash: str,
    recovery1: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    admitted = _admitted_partial_records(recovery1)
    parent = config["parent_recovery"]
    r64_result = json.loads(_root_path(parent["r64_result_path"]).read_text(encoding="utf-8"))
    r64_resource = json.loads(
        _root_path(parent["r64_resource_path"]).read_text(encoding="utf-8")
    )
    if r64_result["status"] != "complete":
        raise RuntimeError("parent R64 phase did not complete")
    if r64_resource["status"] != "completed" or not r64_resource["cleanup_passed"]:
        raise RuntimeError("parent R64 resource receipt did not pass")
    failed = json.loads(
        _root_path(config["failed_untied_attempt"]["resource_path"]).read_text(encoding="utf-8")
    )
    if failed["status"] != "aborted" or failed["abort_reason"] != "sustained_io_cap_exceeded":
        raise RuntimeError("Recovery 2 source is not the registered I/O abort")
    outcomes = (
        output_dir / "untied_gradients_paced_result.json",
        output_dir / "result_receipt.json",
        FINAL_RECEIPT,
    )
    existing = [str(value) for value in outcomes if value.exists()]
    if existing:
        raise RuntimeError(f"Recovery 2 outcome already exists: {existing}")
    return {
        "status": "valid",
        "config_sha256": config_hash,
        "admitted_partial_measurements": len(admitted),
        "parent_r64_records_sha256": parent["r64_records_sha256"],
        "failed_untied_records_admitted": 0,
        "checkpoint_pacing_seconds": config["paced_untied_execution"][
            "checkpoint_pacing_seconds"
        ],
        "debit_before_recovery2_seconds": config["resource_accounting"][
            "debit_before_recovery2_seconds"
        ],
    }


def run_untied_gradients_paced(
    config: dict[str, Any],
    config_hash: str,
    trainer_parent: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    phase = "untied_gradients_paced"
    result_path = output_dir / f"{phase}_result.json"
    event_path = output_dir / f"{phase}_events.jsonl"
    if result_path.exists() or event_path.exists():
        raise RuntimeError("refusing to overwrite paced untied outcomes")
    specification = config["paced_untied_execution"]
    records = []
    training_cells = []
    for rounds in specification["rounds"]:
        for seed in specification["seeds"]:
            cell_records, training = _train_surface_cell(
                config,
                trainer_parent,
                rounds=int(rounds),
                regime="untied",
                seed=int(seed),
                output_dir=output_dir,
                event_path=event_path,
                kappa_exposures=specification["kappa_exposures"],
                gradient_exposures=specification["gradient_exposures"],
                checkpoint_exposures=specification["checkpoint_exposures"],
                checkpoint_pacing_seconds=float(specification["checkpoint_pacing_seconds"]),
            )
            records.extend(cell_records)
            training_cells.append(training)
    return _phase_result(
        config, config_hash, output_dir, phase, records, training_cells
    )


def _require_paced_phase(
    output_dir: Path, config_hash: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    phase = "untied_gradients_paced"
    result_path = output_dir / f"{phase}_result.json"
    resource_path = output_dir / f"{phase}.resource_receipt.json"
    if not result_path.exists() or not resource_path.exists():
        raise RuntimeError("paced untied phase is incomplete")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    if result["config_sha256"] != config_hash or result["status"] != "complete":
        raise RuntimeError("paced untied result did not pass")
    if resource["status"] != "completed" or not resource["cleanup_passed"]:
        raise RuntimeError("paced untied resource receipt did not pass")
    records_path = _root_path(result["records_path"])
    if _sha256(records_path) != result["records_sha256"]:
        raise RuntimeError("paced untied records changed")
    return result, resource


def finalize(
    config: dict[str, Any],
    config_hash: str,
    recovery1: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    if (output_dir / "result_receipt.json").exists() or FINAL_RECEIPT.exists():
        raise RuntimeError("Recovery 2 final receipt already exists")
    paced_result, paced_resource = _require_paced_phase(output_dir, config_hash)
    partial = _admitted_partial_records(recovery1)
    r64 = _read_records(_root_path(config["parent_recovery"]["r64_records_path"]))
    untied = _read_records(_root_path(paced_result["records_path"]))
    combined = partial + r64 + untied
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
        raise RuntimeError("Recovery 2 surface does not contain 12 registered cells")
    models = compare_surface_models(tied_cells, config)
    curvatures = curvature_intervals(tied_values, config)
    onset = curvature_onset(curvatures, config)
    gradients = _gradient_coupling(partial + r64 + untied, onset, config)
    original = json.loads(ORIGINAL_CONFIG.read_text(encoding="utf-8"))
    replication = _prior_tied_replication(tied_kappa, original)
    untied_control = _sealed_untied_terminal_control(original)
    classification = _classify(
        models["winner"],
        onset,
        gradients["co_localized"],
        replication["passed"],
        untied_control["passed"],
    )
    aggregate_elapsed = (
        float(config["resource_accounting"]["debit_before_recovery2_seconds"])
        + float(paced_resource["elapsed_seconds"])
    )
    if aggregate_elapsed > float(config["resource_accounting"]["aggregate_limit_seconds"]):
        raise RuntimeError("aggregate one-GPU-hour cap exceeded")
    summary = {
        "classification": classification,
        "status_label": config["status_label"],
        "surface_model_comparison": models,
        "tied_geometric_surface": tied_cells,
        "depth_curvature": curvatures,
        "preterminal_curvature_onset": onset,
        "gradient_coupling": gradients,
        "tied_terminal_replication": replication,
        "sealed_untied_terminal_control": untied_control,
        "failed_untied_records_admitted": 0,
        "checkpoint_pacing_seconds": config["paced_untied_execution"][
            "checkpoint_pacing_seconds"
        ],
    }
    result_path = output_dir / "recovery2_result.json"
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
        "execution_git_head": paced_result["git_head"],
        "finalization_git_head": _git_head(),
        "artifacts": {
            "combined_records_path": str(combined_path.relative_to(ROOT)),
            "combined_records_sha256": _sha256(combined_path),
            "combined_record_count": len(combined),
            "result_path": str(result_path.relative_to(ROOT)),
            "result_sha256": _sha256(result_path),
            "parent_r64_records_sha256": config["parent_recovery"]["r64_records_sha256"],
        },
        "primary_findings": {
            "surface_model_winner": models["winner"],
            "preterminal_curvature_onset": onset,
            "gradient_co_localized": gradients["co_localized"],
            "tied_terminal_replication_passed": replication["passed"],
            "sealed_untied_terminal_point_equivalence_passed": untied_control["passed"],
        },
        "resources": {
            "debit_before_recovery2_seconds": config["resource_accounting"][
                "debit_before_recovery2_seconds"
            ],
            "paced_phase_elapsed_seconds": paced_resource["elapsed_seconds"],
            "aggregate_elapsed_seconds": aggregate_elapsed,
            "aggregate_limit_seconds": config["resource_accounting"]["aggregate_limit_seconds"],
            "paced_receipt_path": str(
                (output_dir / "untied_gradients_paced.resource_receipt.json").relative_to(ROOT)
            ),
            "paced_receipt_sha256": _sha256(
                output_dir / "untied_gradients_paced.resource_receipt.json"
            ),
            "peak_ram_mb": paced_resource["peak_ram_mb"],
            "peak_io_mb_s": paced_resource["peak_io_mb_s"],
            "peak_vram_mb": paced_resource["peak_vram_mb"],
            "cleanup_passed": paced_resource["cleanup_passed"],
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
        "--phase", choices=("validate", "untied_gradients_paced", "finalize"), required=True
    )
    parser.add_argument("--vram-fraction", type=float)
    args = parser.parse_args()
    config, config_hash, recovery1, trainer_parent = load_registered_config(
        args.config.resolve()
    )
    _set_vram_fraction(trainer_parent, args.vram_fraction)
    output = args.output.resolve()
    try:
        if args.phase == "validate":
            print(json.dumps(validate(config, config_hash, recovery1, output), sort_keys=True))
        elif args.phase == "untied_gradients_paced":
            print(
                json.dumps(
                    run_untied_gradients_paced(
                        config, config_hash, trainer_parent, output
                    ),
                    sort_keys=True,
                )
            )
        else:
            print(json.dumps(finalize(config, config_hash, recovery1, output), sort_keys=True))
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
