"""Derive the v0.2 interference and data-stream audit from sealed records."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable

from research_gym.analysis.lsa_kappa_transient_mechanism import (
    decompose_estimate,
    training_location,
)
from research_gym.integrity import canonical_file_sha256


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SURFACE_RECORDS = (
    ROOT / "experiments" / "loop_schedule_kappa_surface_v1_recovery2" / "combined_records.jsonl"
)
DEFAULT_R128_RECORDS = (
    ROOT
    / "experiments"
    / "loop_schedule_kappa_transient_v0_2_recovery1"
    / "combined_records.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "loop_schedule_kappa_transient_v0_2_recovery1"
    / "mechanism_audit.json"
)


def _load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    return records


def _geometric_mean(values: Iterable[float]) -> float:
    samples = tuple(float(value) for value in values)
    if not samples or any(value <= 0.0 for value in samples):
        raise ValueError("geometric mean requires positive samples")
    return math.exp(sum(math.log(value) for value in samples) / len(samples))


def build_audit(
    records: list[dict[str, Any]],
    *,
    source_paths: Iterable[Path],
    batch_size: int = 8,
) -> dict[str, Any]:
    r128 = sorted(
        (
            row
            for row in records
            if row.get("regime") == "tied"
            and int(row.get("rounds", 0)) == 128
            and "estimate" in row
        ),
        key=lambda row: (int(row["seed"]), int(row["exposure"])),
    )
    if len(r128) != 12:
        raise ValueError("mechanism audit requires 12 tied R128 estimates")

    per_seed: list[dict[str, Any]] = []
    by_exposure: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in r128:
        decomposition = decompose_estimate(row["estimate"])
        entry = {
            "seed": int(row["seed"]),
            "exposure": int(row["exposure"]),
            "kappa": float(row["kappa"]),
            **decomposition,
        }
        per_seed.append(entry)
        by_exposure[entry["exposure"]].append(entry)

    aggregate = []
    for exposure, entries in sorted(by_exposure.items()):
        aggregate.append(
            {
                "exposure": exposure,
                **training_location(exposure, batch_size=batch_size, rounds=128),
                "geometric_kappa": _geometric_mean(entry["kappa"] for entry in entries),
                "geometric_sensitivity_interference_ratio": _geometric_mean(
                    entry["sensitivity_interference_ratio"] for entry in entries
                ),
                "geometric_gradient_interference_ratio": _geometric_mean(
                    entry["gradient_interference_ratio"] for entry in entries
                ),
                "destructive_sensitivity_seed_count": sum(
                    entry["sensitivity_interference_class"] == "net_destructive"
                    for entry in entries
                ),
                "destructive_gradient_seed_count": sum(
                    entry["gradient_interference_class"] == "net_destructive"
                    for entry in entries
                ),
            }
        )

    untied = [row for row in records if row.get("regime") == "untied"]
    untied_trace = []
    for rounds in (16, 32, 64, 128):
        round_rows = [row for row in untied if int(row.get("rounds", 0)) == rounds]
        for exposure in sorted({int(row["exposure"]) for row in round_rows}):
            entries = [row for row in round_rows if int(row["exposure"]) == exposure]
            untied_trace.append(
                {
                    "rounds": rounds,
                    "exposure": exposure,
                    **training_location(exposure, batch_size=batch_size, rounds=rounds),
                    "geometric_interval_max_gradient": _geometric_mean(
                        float(row["interval_gradient"]["maximum"]) for row in entries
                    ),
                    "geometric_latest_training_loss": _geometric_mean(
                        float(row["latest_training_loss"]) for row in entries
                    ),
                    "seed_count": len(entries),
                }
            )

    e2048_locations = {
        str(rounds): training_location(2048, batch_size=batch_size, rounds=rounds)
        for rounds in (16, 32, 64, 128)
    }
    return {
        "schema_version": "lsa-kappa-transient-mechanism-audit-v1",
        "source_records": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": canonical_file_sha256(path)}
            for path in source_paths
        ],
        "kappa_identity": {
            "interference_ratio": "||sum_r v_r||^2 / sum_r ||v_r||^2",
            "cross_term_identity": "ratio - 1 = 2 sum_{r<s}<v_r,v_s> / sum_r ||v_r||^2",
            "interpretation": "ratio below one proves a negative aggregate pairwise cross-term; it does not imply every pair is anti-aligned",
        },
        "r128_per_seed": per_seed,
        "r128_geometric_by_exposure": aggregate,
        "e2048_training_locations": e2048_locations,
        "same_exposure_uses_same_training_stream": len(
            {entry["training_stream"] for entry in e2048_locations.values()}
        )
        == 1,
        "untied_trace": untied_trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-records", type=Path, default=DEFAULT_SURFACE_RECORDS)
    parser.add_argument("--r128-records", type=Path, default=DEFAULT_R128_RECORDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source_paths = (args.surface_records.resolve(), args.r128_records.resolve())
    audit = build_audit(_load_records(source_paths), source_paths=source_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
