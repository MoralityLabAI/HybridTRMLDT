from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from research_gym.analysis.controller_mesh_stalk_bridge import (
    load_registered_stalk,
    markdown_report,
    run_bridge_study,
)
from research_gym.scripts.bench_gaming_vs_improvement import _write_json


DEFAULT_CONFIG = Path("configs/controller_mesh_measured_stalk_bridge_v1.json")
DEFAULT_OUTPUT = Path("data/benchmarks/controller_mesh_measured_stalk_bridge_v1.json")
DEFAULT_REPORT = Path("reports/controller_mesh_measured_stalk_bridge_v1.md")
DEFAULT_EXPERIMENT = Path("experiments/controller_mesh_measured_stalk_bridge_v1")


def experiment_notes(result: Mapping[str, object]) -> str:
    bridge = result["bridge_result"]
    return "\n".join(
        [
            "# Measured-Stalk Controller Bridge Notes",
            "",
            f"- protocol: `{result['study_id']}`",
            f"- config SHA-256: `{result['config_sha256']}`",
            f"- source receipt SHA-256: `{result['measured_stalk']['source_receipt_sha256']}`",
            f"- prediction receipt SHA-256: `{result['stage_receipts']['triple_prediction_sha256']}`",
            f"- analysis receipt SHA-256: `{result['analysis_receipt_sha256']}`",
            f"- combined / controller / categorical rho: `{bridge['combined_rho']:+.4f} / {bridge['controller_rho']:+.4f} / {bridge['categorical_rho']:+.4f}`",
            f"- registered bridge gate passed: `{bridge['gate_passed']}`",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a measured Qwen stalk in the controller mesh."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()

    registration = json.loads(args.config.read_text(encoding="utf-8"))
    stalk = load_registered_stalk(registration, root=args.root)
    result = run_bridge_study(registration, stalk)
    _write_json(result, args.out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown_report(result), encoding="utf-8")
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_json(result, args.experiment_dir / "results.json")
    (args.experiment_dir / "training_notes.md").write_text(
        experiment_notes(result), encoding="utf-8"
    )
    print(markdown_report(result))


if __name__ == "__main__":
    main()
