"""Materialize the post-hoc flat-null analysis for the mini Transformer control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from research_gym.analysis.lsa_v0_1 import bootstrap_gamma
from research_gym.scripts.bench_loop_schedule_algebra_v0_1 import ROOT, _write_json


DEFAULT_RECORDS = ROOT / "experiments" / "loop_schedule_algebra_v0_1" / "external_records.jsonl"
DEFAULT_OUTPUT = (
    ROOT / "experiments" / "loop_schedule_algebra_v0_1" / "external_flatness_analysis.json"
)


def analyze(records_path: Path) -> dict:
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    interval = bootstrap_gamma(
        rows,
        rounds=[2, 4, 8, 16],
        regime="untied",
        exposure=4096,
        seeds=[101, 103, 107],
        samples=20_000,
        bootstrap_seed=0,
        interval_mass=0.95,
    )
    return {
        "analysis_id": "lsa_v0_1_external_untied_flatness_posthoc_v1",
        "status": "post_hoc_gate_diagnostic_not_preregistered_confirmation",
        "source_records_path": str(records_path.relative_to(ROOT)),
        "source_records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "rounds": [2, 4, 8, 16],
        "regime": "untied",
        "exposure": 4096,
        "seeds": [101, 103, 107],
        "gamma_interval": interval,
        "flatness_check": {
            "interval_contains_zero": interval["lower"] <= 0.0 <= interval["upper"],
            "absolute_point_gamma_at_most_0_1": abs(interval["median"]) <= 0.1,
        },
        "interpretation": "The registered composite gate still fails; this analysis only diagnoses why power-law R-squared is unsuitable for an intended flat null.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze(args.records.resolve())
    _write_json(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
