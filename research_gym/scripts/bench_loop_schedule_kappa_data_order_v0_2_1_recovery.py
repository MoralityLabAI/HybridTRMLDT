"""Recover result assembly from the complete v0.2.1 data-order records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from research_gym.analysis.lsa_kappa_transient_mechanism import decompose_estimate
from research_gym.integrity import canonical_file_sha256, verify_file_sha256
from research_gym.scripts.bench_loop_schedule_algebra_v0_1 import (
    ROOT,
    _cleanup,
    _git_head,
    _set_vram_fraction,
    _write_json,
)
from research_gym.scripts.bench_loop_schedule_kappa_data_order_v0_2_1 import (
    _read_records,
    _records_by_order,
    baseline_replay_gate,
    classify_order_trace,
)


DEFAULT_CONFIG = ROOT / "configs" / "loop_schedule_kappa_data_order_v0_2_1_recovery1.json"
REGISTRATION = (
    ROOT / "configs" / "loop_schedule_kappa_data_order_v0_2_1_recovery1_registration.json"
)
DEFAULT_OUTPUT = ROOT / "experiments" / "loop_schedule_kappa_data_order_v0_2_1_recovery1"
FINAL_RECEIPT = (
    ROOT / "data" / "benchmarks" / "lsa_kappa_data_order_v0_2_1_recovery1_receipt.json"
)


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if path != ROOT.resolve() and ROOT.resolve() not in path.parents:
        raise RuntimeError(f"registered path escapes repository: {value}")
    return path


def load_registered_config(path: Path) -> tuple[dict[str, Any], str]:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    config_hash = canonical_file_sha256(path)
    if config_hash != registration["config_sha256"]:
        raise RuntimeError("recovery config does not match frozen registration")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["protocol_id"] != registration["protocol_id"]:
        raise RuntimeError("recovery protocol id does not match registration")
    for parent in config["parents"].values():
        source = _root_path(parent["path"])
        if not verify_file_sha256(source, parent["sha256"]):
            raise RuntimeError(f"parent hash mismatch: {source.name}")
    return config, config_hash


def admitted_records(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = _read_records(_root_path(config["parents"]["failed_records"]["path"]))
    admission = config["admission"]
    if len(records) != int(admission["record_count"]):
        raise RuntimeError("failed attempt record count changed")
    if len({str(row["record_id"]) for row in records}) != len(records):
        raise RuntimeError("failed attempt contains duplicate record IDs")
    expected_orders = {int(value) for value in admission["data_order_seeds"]}
    expected_exposures = {int(value) for value in admission["measurement_exposures"]}
    grouped = _records_by_order(records)
    if set(grouped) != expected_orders:
        raise RuntimeError("failed attempt order-seed cells changed")
    for order_seed, rows in grouped.items():
        if {int(row["exposure"]) for row in rows} != expected_exposures:
            raise RuntimeError(f"order seed {order_seed} has an incomplete exposure grid")
        for row in rows:
            channels = row["seed_channels"]
            expected = {
                "model_seed": 103,
                "task_seed": 103,
                "data_order_seed": order_seed,
                "measurement_data_seed": 103,
                "probe_seed": 103,
            }
            if channels != expected:
                raise RuntimeError(f"order seed {order_seed} changed a frozen seed channel")
            if row.get("regime") != "tied" or int(row.get("rounds", 0)) != 64:
                raise RuntimeError("recovery admitted a non-tied or non-R64 record")
    return records


def validate(config: Mapping[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    resource = json.loads(
        _root_path(config["parents"]["failed_resource"]["path"]).read_text(encoding="utf-8")
    )
    if (
        resource["status"] != "failed"
        or resource["abort_reason"] != "process_exit_1"
        or not resource["cleanup_passed"]
    ):
        raise RuntimeError("failed-attempt resource status changed")
    records = admitted_records(config)
    outcomes = (
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
        "admitted_record_count": len(records),
        "new_training_cells": 0,
        "outcomes_present": False,
    }


def run(config: Mapping[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    result_path = output_dir / "recovery_result.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite recovery result")
    records = admitted_records(config)
    grouped = _records_by_order(records)
    parent_records = _read_records(_root_path(config["parents"]["surface_records"]["path"]))
    construction_seed = int(config["frozen_analysis"]["construction_seed"])
    replay = baseline_replay_gate(
        grouped[construction_seed],
        parent_records,
        tolerance=float(config["frozen_analysis"]["absolute_kappa_tolerance"]),
    )
    threshold = float(config["frozen_analysis"]["practical_log_change"])
    traces = {
        str(seed): classify_order_trace(seed_records, threshold)
        for seed, seed_records in sorted(grouped.items())
    }
    fresh = [int(seed) for seed in config["frozen_analysis"]["fresh_data_order_seeds"]]
    persistence_count = sum(
        traces[str(seed)]["classification"] == "e2048_trough_and_recovery"
        for seed in fresh
    )
    if not replay["passed"]:
        scientific_status = "instrument_drift"
    elif persistence_count >= int(config["frozen_analysis"]["minimum_persistent_fresh_orders"]):
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
        "source_records_path": config["parents"]["failed_records"]["path"],
        "source_records_sha256": config["parents"]["failed_records"]["sha256"],
        "record_count": len(records),
        "summary": {
            "scientific_status": scientific_status,
            "baseline_replay": replay,
            "per_order_trace": traces,
            "fresh_persistence_count": persistence_count,
            "fresh_order_count": len(fresh),
            "e2048_cancellation": cancellation,
            "recovery_changed_scientific_rules": False,
            "new_training_cells": 0,
        },
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(result_path, result)
    return result


def finalize(config: Mapping[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    result_path = output_dir / "recovery_result.json"
    resource_path = output_dir / "run.resource_receipt.json"
    source_records = _root_path(config["parents"]["failed_records"]["path"])
    if not all(path.exists() for path in (result_path, resource_path, source_records)):
        raise RuntimeError("recovery is incomplete")
    if (output_dir / "result_receipt.json").exists() or FINAL_RECEIPT.exists():
        raise RuntimeError("recovery result already sealed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    if result["config_sha256"] != config_hash or result["status"] != "complete":
        raise RuntimeError("recovery result does not match registered config")
    if not verify_file_sha256(source_records, result["source_records_sha256"]):
        raise RuntimeError("source records failed re-verification")
    if resource["status"] != "completed" or not resource["cleanup_passed"]:
        raise RuntimeError("recovery resource receipt did not pass")
    receipt = {
        "protocol_id": config["protocol_id"],
        "status": "sealed",
        "config_path": DEFAULT_CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": config_hash,
        "result_path": result_path.relative_to(ROOT).as_posix(),
        "result_sha256": canonical_file_sha256(result_path),
        "records_path": source_records.relative_to(ROOT).as_posix(),
        "records_sha256": canonical_file_sha256(source_records),
        "record_count": result["record_count"],
        "resource_path": resource_path.relative_to(ROOT).as_posix(),
        "resource_sha256": canonical_file_sha256(resource_path),
        "cleanup_passed": True,
        "scientific_status": result["summary"]["scientific_status"],
        "fresh_persistence_count": result["summary"]["fresh_persistence_count"],
        "new_training_cells": 0,
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
    config, config_hash = load_registered_config(args.config.resolve())
    _set_vram_fraction(config, args.vram_fraction)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        if args.phase == "validate":
            payload = validate(config, config_hash, output)
        elif args.phase == "run":
            payload = run(config, config_hash, output)
        else:
            payload = finalize(config, config_hash, output)
        print(json.dumps(payload, sort_keys=True))
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
