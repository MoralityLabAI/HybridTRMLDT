"""Run the registered data-order-only intervention on the R64 kappa trough."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from research_gym.analysis.lsa_kappa_transient_mechanism import decompose_estimate
from research_gym.integrity import canonical_file_sha256, verify_file_sha256
from research_gym.scripts.bench_loop_schedule_algebra_v0_1 import (
    ROOT,
    _cleanup,
    _git_head,
    _set_vram_fraction,
    _write_json,
    _write_records,
)
from research_gym.scripts.bench_loop_schedule_kappa_surface import _train_surface_cell


DEFAULT_CONFIG = ROOT / "configs" / "loop_schedule_kappa_data_order_v0_2_1.json"
REGISTRATION = ROOT / "configs" / "loop_schedule_kappa_data_order_v0_2_1_registration.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "loop_schedule_kappa_data_order_v0_2_1"
FINAL_RECEIPT = ROOT / "data" / "benchmarks" / "lsa_kappa_data_order_v0_2_1_receipt.json"


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if path != ROOT.resolve() and ROOT.resolve() not in path.parents:
        raise RuntimeError(f"registered path escapes repository: {value}")
    return path


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _records_by_order(records: Iterable[Mapping[str, Any]]) -> dict[int, list[Mapping[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[int(row["seed_channels"]["data_order_seed"])].append(row)
    return {seed: sorted(rows, key=lambda row: int(row["exposure"])) for seed, rows in grouped.items()}


def classify_order_trace(records: Iterable[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    kappa = {int(row["exposure"]): float(row["kappa"]) for row in records}
    required = {0, 512, 1024, 2048, 4096}
    if set(kappa) != required:
        raise ValueError(f"order trace requires exposures {sorted(required)}")
    drop = math.log(kappa[2048] / kappa[1024])
    rebound = math.log(kappa[4096] / kappa[2048])
    return {
        "classification": (
            "e2048_trough_and_recovery"
            if drop <= -threshold and rebound >= threshold
            else "registered_shape_absent"
        ),
        "log_kappa_E1024_to_E2048": drop,
        "log_kappa_E2048_to_E4096": rebound,
        "minimum_checkpoint_exposure": min((1024, 2048, 4096), key=kappa.__getitem__),
        "kappa": {str(exposure): value for exposure, value in sorted(kappa.items())},
    }


def baseline_replay_gate(
    new_records: Iterable[Mapping[str, Any]],
    parent_records: Iterable[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    new = {int(row["exposure"]): row for row in new_records}
    parent = {
        int(row["exposure"]): row
        for row in parent_records
        if row.get("regime") == "tied"
        and int(row.get("rounds", 0)) == 64
        and int(row.get("seed", -1)) == 103
        and "kappa" in row
    }
    if set(new) != {0, 512, 1024, 2048, 4096}:
        raise ValueError("baseline replay has an unexpected exposure grid")
    if set(parent) != {0, 1024, 2048, 4096}:
        raise ValueError("sealed parent has an unexpected kappa exposure grid")
    differences = {
        str(exposure): abs(float(new[exposure]["kappa"]) - float(parent[exposure]["kappa"]))
        for exposure in sorted(parent)
    }
    maximum = max(differences.values())
    return {
        "passed": maximum <= tolerance,
        "absolute_tolerance": tolerance,
        "maximum_absolute_kappa_difference": maximum,
        "per_exposure_absolute_kappa_difference": differences,
    }


def load_registered_config(path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    config_hash = canonical_file_sha256(path)
    if config_hash != registration["config_sha256"]:
        raise RuntimeError("data-order config does not match frozen registration")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["protocol_id"] != registration["protocol_id"]:
        raise RuntimeError("data-order protocol id does not match registration")
    for parent in config["parents"].values():
        source = _root_path(parent["path"])
        if not verify_file_sha256(source, parent["sha256"]):
            raise RuntimeError(f"parent hash mismatch: {source.name}")
    trainer_path = _root_path(config["parents"]["trainer_config"]["path"])
    return config, config_hash, json.loads(trainer_path.read_text(encoding="utf-8"))


def validate(config: Mapping[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    frozen = config["frozen_training"]
    channels = config["seed_channels"]
    if frozen["rounds"] != [64] or frozen["regimes"] != ["tied"]:
        raise RuntimeError("registered intervention must contain only tied R64 cells")
    if channels["construction_seed"] != 103:
        raise RuntimeError("registered construction seed changed")
    if channels["data_order_seeds"] != [103, 211, 223, 227]:
        raise RuntimeError("registered order-seed ensemble changed")
    if frozen["measurement_exposures"] != [0, 512, 1024, 2048, 4096]:
        raise RuntimeError("registered exposure grid changed")
    quantum = int(frozen["batch_size"]) * 64
    if any(int(exposure) % quantum for exposure in frozen["measurement_exposures"]):
        raise RuntimeError("registered exposure is unreachable")
    outcomes = (
        output_dir / "data_order_records.jsonl",
        output_dir / "data_order_result.json",
        output_dir / "result_receipt.json",
        FINAL_RECEIPT,
    )
    existing = [str(path) for path in outcomes if path.exists()]
    if existing:
        raise RuntimeError(f"registered outcome already exists: {existing}")
    return {
        "status": "valid",
        "config_sha256": config_hash,
        "new_training_cells": 4,
        "fresh_data_order_cells": 3,
        "outcomes_present": False,
    }


def run(
    config: dict[str, Any], config_hash: str, trainer_config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    records_path = output_dir / "data_order_records.jsonl"
    result_path = output_dir / "data_order_result.json"
    event_path = output_dir / "data_order_events.jsonl"
    if any(path.exists() for path in (records_path, result_path, event_path)):
        raise RuntimeError("refusing to overwrite data-order outcomes")
    frozen = config["frozen_training"]
    channels = config["seed_channels"]
    construction_seed = int(channels["construction_seed"])
    records: list[dict[str, Any]] = []
    training_cells: list[dict[str, Any]] = []
    for order_seed in channels["data_order_seeds"]:
        cell_records, training = _train_surface_cell(
            config,
            trainer_config,
            rounds=64,
            regime="tied",
            seed=int(order_seed),
            output_dir=output_dir,
            event_path=event_path,
            kappa_exposures=frozen["measurement_exposures"],
            gradient_exposures=frozen["gradient_exposures"],
            checkpoint_exposures=frozen["checkpoint_exposures"],
            checkpoint_pacing_seconds=float(frozen["checkpoint_pacing_seconds"]),
            model_seed=construction_seed,
            task_seed=construction_seed,
            data_order_seed=int(order_seed),
            measurement_data_seed=construction_seed,
            probe_seed=construction_seed,
            cell_id_override=f"data_order_tied_R64_C{construction_seed}_D{order_seed}",
        )
        records.extend(cell_records)
        training_cells.append(training)
    _write_records(records_path, records)

    grouped = _records_by_order(records)
    parent_records = _read_records(_root_path(config["parents"]["surface_records"]["path"]))
    replay = baseline_replay_gate(
        grouped[construction_seed],
        parent_records,
        tolerance=float(config["baseline_replay"]["absolute_kappa_tolerance"]),
    )
    threshold = float(config["registered_predictions"]["practical_log_change"])
    traces = {
        str(seed): classify_order_trace(seed_records, threshold)
        for seed, seed_records in sorted(grouped.items())
    }
    fresh = [int(seed) for seed in channels["fresh_data_order_seeds"]]
    persistence_count = sum(
        traces[str(seed)]["classification"] == "e2048_trough_and_recovery"
        for seed in fresh
    )
    nonfinite = any(
        cell["nonfinite"]
        or int(cell["state_visit_exposures"]) != int(frozen["progress_target"])
        for cell in training_cells
    )
    if nonfinite:
        scientific_status = "resource_or_training_failure"
    elif not replay["passed"]:
        scientific_status = "instrument_drift"
    elif persistence_count >= int(config["registered_predictions"]["minimum_persistent_fresh_orders"]):
        scientific_status = "persists_under_data_order_reseed"
    else:
        scientific_status = "data_order_sensitive"

    cancellation = {}
    for seed, seed_records in sorted(grouped.items()):
        trough = next(row for row in seed_records if int(row["exposure"]) == 2048)
        decomposition = decompose_estimate(trough["estimate"])
        cancellation[str(seed)] = {
            "sensitivity_interference_ratio": decomposition[
                "sensitivity_interference_ratio"
            ],
            "sensitivity_interference_class": decomposition[
                "sensitivity_interference_class"
            ],
            "gradient_interference_ratio": decomposition["gradient_interference_ratio"],
            "gradient_interference_class": decomposition["gradient_interference_class"],
        }

    result = {
        "protocol_id": config["protocol_id"],
        "status": "complete",
        "phase": "run",
        "config_sha256": config_hash,
        "git_head": _git_head(),
        "records_path": records_path.relative_to(ROOT).as_posix(),
        "records_sha256": canonical_file_sha256(records_path),
        "record_count": len(records),
        "events_path": event_path.relative_to(ROOT).as_posix(),
        "events_sha256": canonical_file_sha256(event_path),
        "summary": {
            "scientific_status": scientific_status,
            "baseline_replay": replay,
            "per_order_trace": traces,
            "fresh_persistence_count": persistence_count,
            "fresh_order_count": len(fresh),
            "e2048_cancellation": cancellation,
            "training_cells": training_cells,
        },
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(result_path, result)
    return result


def finalize(config: Mapping[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    result_path = output_dir / "data_order_result.json"
    records_path = output_dir / "data_order_records.jsonl"
    resource_path = output_dir / "run.resource_receipt.json"
    if not all(path.exists() for path in (result_path, records_path, resource_path)):
        raise RuntimeError("data-order run is incomplete")
    if (output_dir / "result_receipt.json").exists() or FINAL_RECEIPT.exists():
        raise RuntimeError("data-order result already sealed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    if result["config_sha256"] != config_hash or result["status"] != "complete":
        raise RuntimeError("data-order result does not match registered config")
    if not verify_file_sha256(records_path, result["records_sha256"]):
        raise RuntimeError("data-order records failed re-verification")
    if resource["status"] != "completed" or not resource["cleanup_passed"]:
        raise RuntimeError("data-order resource receipt did not pass")
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
        "fresh_persistence_count": result["summary"]["fresh_persistence_count"],
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
    config, config_hash, trainer_config = load_registered_config(args.config.resolve())
    _set_vram_fraction(config, args.vram_fraction)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        if args.phase == "validate":
            payload = validate(config, config_hash, output)
        elif args.phase == "run":
            payload = run(config, config_hash, trainer_config, output)
        else:
            payload = finalize(config, config_hash, output)
        print(json.dumps(payload, sort_keys=True))
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
