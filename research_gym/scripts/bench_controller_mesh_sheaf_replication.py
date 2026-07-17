from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from research_gym.analysis.controller_mesh_forward_replication import (
    markdown_report,
    run_replication,
)
from research_gym.scripts.bench_gaming_vs_improvement import _write_json


DEFAULT_CONFIG = Path("configs/controller_mesh_sheaf_forward_replication_v2.json")
DEFAULT_OUTPUT = Path(
    "data/benchmarks/controller_mesh_sheaf_forward_replication_v2.json"
)
DEFAULT_REPORT = Path("reports/controller_mesh_sheaf_forward_replication_v2.md")
DEFAULT_EXPERIMENT = Path("experiments/controller_mesh_sheaf_forward_replication_v2")


def experiment_notes(result: Mapping[str, object]) -> str:
    replication = result["replication_result"]
    paired = replication["paired_bootstrap"]
    return "\n".join(
        [
            "# Whole-Family Spectral Replication Notes",
            "",
            f"- protocol: `{result['study_id']}`",
            f"- config SHA-256: `{result['config_sha256']}`",
            f"- task corpus SHA-256: `{result['task_corpus']['sha256']}`",
            f"- dual prediction SHA-256: `{result['stage_receipts']['dual_prediction_sha256']}`",
            f"- analysis receipt SHA-256: `{result['analysis_receipt_sha256']}`",
            f"- spectral / categorical rho: `{paired['spectral_rho']:+.4f} / {paired['categorical_rho']:+.4f}`",
            f"- paired rho delta: `{paired['rho_delta']:+.4f}`",
            f"- registered incremental gate passed: `{replication['gate_passed']}`",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replicate spectral controller prediction with whole-family holdouts."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()

    registration = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_replication(registration)
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
