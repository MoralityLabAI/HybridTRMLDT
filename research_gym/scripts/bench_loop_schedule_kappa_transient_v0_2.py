"""Run the registered R=128 transient-timing discriminator."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

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


DEFAULT_CONFIG = ROOT / "configs" / "loop_schedule_kappa_transient_v0_2.json"
REGISTRATION = ROOT / "configs" / "loop_schedule_kappa_transient_v0_2_registration.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "loop_schedule_kappa_transient_v0_2"
FINAL_RECEIPT = ROOT / "data" / "benchmarks" / "lsa_kappa_transient_v0_2_receipt.json"


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


def _geometric_mean(values: Iterable[float]) -> float:
    samples = tuple(float(value) for value in values)
    if not samples or any(value <= 0.0 or not math.isfinite(value) for value in samples):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in samples) / len(samples))


def _grouped_kappa(records: Iterable[Mapping[str, Any]]) -> dict[int, float]:
    values: dict[int, list[float]] = defaultdict(list)
    for row in records:
        if row.get("regime") == "tied" and "kappa" in row:
            values[int(row["exposure"])].append(float(row["kappa"]))
    return {exposure: _geometric_mean(samples) for exposure, samples in values.items()}


def classify_timing(kappa: Mapping[int, float], threshold: float) -> dict[str, Any]:
    required = {1024, 2048, 4096, 8192}
    if set(kappa) != required:
        raise ValueError(f"timing classification requires {sorted(required)}")
    exposure_drop = math.log(kappa[2048] / kappa[1024])
    middle_change = math.log(kappa[4096] / kappa[2048])
    step_rebound = math.log(kappa[8192] / kappa[4096])
    exposure_pinned = exposure_drop <= -threshold and middle_change >= threshold
    step_pinned = middle_change <= -threshold and step_rebound >= threshold
    if exposure_pinned:
        timing = "exposure_pinned"
    elif step_pinned:
        timing = "step_pinned"
    else:
        timing = "timing_unresolved"
    return {
        "classification": timing,
        "log_changes": {
            "E1024_to_E2048": exposure_drop,
            "E2048_to_E4096": middle_change,
            "E4096_to_E8192": step_rebound,
        },
        "threshold": threshold,
    }


def _interval_maximums(
    records: Iterable[Mapping[str, Any]], rounds: int, exposure: int
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        if int(row.get("rounds", -1)) != rounds or int(row.get("exposure", -1)) != exposure:
            continue
        maximum = row.get("interval_gradient", {}).get("maximum")
        if maximum is not None:
            grouped[str(row["regime"])].append(float(maximum))
    if set(grouped) != {"tied", "untied"}:
        raise ValueError(f"missing matched gradient records for R={rounds}, E={exposure}")
    return {regime: _geometric_mean(values) for regime, values in grouped.items()}


def _stress_ratio(records: Iterable[Mapping[str, Any]], rounds: int, exposure: int) -> float:
    values = _interval_maximums(records, rounds, exposure)
    return values["tied"] / values["untied"]


def _power_exponent(ratios: Mapping[int, float]) -> float:
    xs = [math.log(float(rounds)) for rounds in sorted(ratios)]
    ys = [math.log(float(ratios[rounds])) for rounds in sorted(ratios)]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sum(
        (x - x_mean) ** 2 for x in xs
    )


def _retrospective_values(records: list[dict[str, Any]]) -> dict[int, dict[str, float | int]]:
    surface: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in records:
        if row.get("regime") == "tied" and "kappa" in row:
            surface[(int(row["rounds"]), int(row["exposure"]))].append(float(row["kappa"]))
    specification = {
        16: (0, 512, 1024),
        32: (512, 1024, 2048),
        64: (1024, 2048, 4096),
    }
    return {
        rounds: {
            "previous_exposure": points[0],
            "previous_kappa": _geometric_mean(surface[(rounds, points[0])]),
            "step_four_exposure": points[1],
            "step_four_kappa": _geometric_mean(surface[(rounds, points[1])]),
            "next_exposure": points[2],
            "next_kappa": _geometric_mean(surface[(rounds, points[2])]),
        }
        for rounds, points in specification.items()
    }


def load_registered_config(path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    config_hash = canonical_file_sha256(path)
    if config_hash != registration["config_sha256"]:
        raise RuntimeError("v0.2 config does not match the frozen registration")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["protocol_id"] != registration["protocol_id"]:
        raise RuntimeError("v0.2 protocol id does not match registration")
    for parent in config["parents"].values():
        source = _root_path(parent["path"])
        if not verify_file_sha256(source, parent["sha256"]):
            raise RuntimeError(f"parent hash mismatch: {source.name}")
    trainer_path = _root_path(config["parents"]["trainer_config"]["path"])
    trainer_config = json.loads(trainer_path.read_text(encoding="utf-8"))
    return config, config_hash, trainer_config


def validate(config: Mapping[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    frozen = config["frozen_training"]
    if frozen["rounds"] != [128] or frozen["seeds"] != [101, 103, 107]:
        raise RuntimeError("registered R128 cell or seed ensemble changed")
    quantum = int(frozen["batch_size"]) * 128
    exposures = [int(value) for value in frozen["surface_exposures"]]
    if exposures != [1024, 2048, 4096, 8192] or any(value % quantum for value in exposures):
        raise RuntimeError("registered R128 exposure grid changed or is unreachable")
    predictions = config["registered_predictions"]
    if predictions["exposure_pinned"]["optimizer_step"] != 2:
        raise RuntimeError("exposure-pinned prediction changed")
    if predictions["step_pinned"]["optimizer_step"] != 4:
        raise RuntimeError("step-pinned prediction changed")
    if predictions["recovery"]["optimizer_step"] != 8:
        raise RuntimeError("recovery checkpoint changed")
    parent_records = _read_records(_root_path(config["parents"]["surface_records"]["path"]))
    observed = _retrospective_values(parent_records)
    for expected in config["retrospective_step_four"]["cells"]:
        actual = observed[int(expected["rounds"])]
        for key, value in actual.items():
            if isinstance(value, float) and not math.isclose(
                value, float(expected[key]), rel_tol=0.0, abs_tol=1e-12
            ):
                raise RuntimeError(f"retrospective step-four value changed: R={expected['rounds']} {key}")
            if isinstance(value, int) and value != int(expected[key]):
                raise RuntimeError(f"retrospective step-four exposure changed: R={expected['rounds']} {key}")
    outcomes = (
        output_dir / "transient_records.jsonl",
        output_dir / "transient_result.json",
        output_dir / "result_receipt.json",
        FINAL_RECEIPT,
    )
    existing = [str(path) for path in outcomes if path.exists()]
    if existing:
        raise RuntimeError(f"registered outcome already exists: {existing}")
    return {
        "status": "valid",
        "config_sha256": config_hash,
        "new_training_cells": 6,
        "retrospective_cells": 3,
        "outcomes_present": False,
    }


def run(
    config: dict[str, Any], config_hash: str, trainer_config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    result_path = output_dir / "transient_result.json"
    records_path = output_dir / "transient_records.jsonl"
    event_path = output_dir / "transient_events.jsonl"
    if any(path.exists() for path in (result_path, records_path, event_path)):
        raise RuntimeError("refusing to overwrite R128 outcomes")
    frozen = config["frozen_training"]
    records: list[dict[str, Any]] = []
    training_cells: list[dict[str, Any]] = []
    for regime in frozen["regimes"]:
        for seed in frozen["seeds"]:
            cell_records, training = _train_surface_cell(
                config,
                trainer_config,
                rounds=128,
                regime=str(regime),
                seed=int(seed),
                output_dir=output_dir,
                event_path=event_path,
                kappa_exposures=frozen["surface_exposures"] if regime == "tied" else [],
                gradient_exposures=frozen["gradient_exposures"],
                checkpoint_exposures=frozen["checkpoint_exposures"],
                checkpoint_pacing_seconds=float(frozen["checkpoint_pacing_seconds"]),
            )
            records.extend(cell_records)
            training_cells.append(training)
    _write_records(records_path, records)
    failures = [
        row
        for row in training_cells
        if row["nonfinite"] or int(row["state_visit_exposures"]) != int(frozen["progress_target"])
    ]
    if failures:
        summary: dict[str, Any] = {
            "scientific_status": "nonfinite_boundary",
            "timing": None,
            "recovery": "nonfinite_boundary",
            "failed_cells": failures,
            "training_cells": training_cells,
        }
    else:
        kappa = _grouped_kappa(records)
        timing = classify_timing(
            kappa, float(config["registered_predictions"]["practical_log_change"])
        )
        if timing["classification"] in {"exposure_pinned", "step_pinned"}:
            recovery = "survived_recovered_excursion"
        else:
            drops = timing["log_changes"]
            has_drop = min(drops["E1024_to_E2048"], drops["E2048_to_E4096"]) <= -float(
                config["registered_predictions"]["practical_log_change"]
            )
            recovery = "survived_unrecovered_excursion" if has_drop else "no_registered_excursion"
        parent_records = _read_records(_root_path(config["parents"]["surface_records"]["path"]))
        ratios = {
            32: _stress_ratio(parent_records, 32, 1024),
            64: _stress_ratio(parent_records, 64, 2048),
            128: _stress_ratio(records, 128, 4096),
        }
        summary = {
            "scientific_status": "complete",
            "tied_geometric_kappa": {str(key): value for key, value in sorted(kappa.items())},
            "timing": timing,
            "recovery": recovery,
            "step_four_stress_ratios": {str(key): value for key, value in sorted(ratios.items())},
            "three_depth_stress_exponent": _power_exponent(ratios),
            "training_cells": training_cells,
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
        "summary": summary,
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(result_path, result)
    return result


def finalize(config: Mapping[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    result_path = output_dir / "transient_result.json"
    records_path = output_dir / "transient_records.jsonl"
    resource_path = output_dir / "run.resource_receipt.json"
    if not all(path.exists() for path in (result_path, records_path, resource_path)):
        raise RuntimeError("R128 run is incomplete")
    if (output_dir / "result_receipt.json").exists() or FINAL_RECEIPT.exists():
        raise RuntimeError("R128 result already sealed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    if result["config_sha256"] != config_hash or result["status"] != "complete":
        raise RuntimeError("R128 result does not match the registered config")
    if not verify_file_sha256(records_path, result["records_sha256"]):
        raise RuntimeError("R128 records failed re-verification")
    if resource["status"] != "completed" or not resource["cleanup_passed"]:
        raise RuntimeError("R128 resource receipt did not pass")
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
