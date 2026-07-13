from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.benchmarks.training_review_bench import (
    run_training_review_benchmark,
    summary_markdown,
)


def training_notes(payload: dict[str, object]) -> str:
    return "\n".join(
        [
            "# HRM Training Review Experiment Notes",
            "",
            "## Scope",
            "",
            "This is a deterministic review-contract experiment over synthetic sealed receipts. It ports scalar",
            "lineage and holonomy authorization math from the local RSITopology program into Conductor-HRM flow.",
            "It does not launch training, inspect live weights, or authorize model promotion outside the saved cases.",
            "",
            "## Frozen policy",
            "",
            f"- signed operator-error budget: `{payload['policy']['signed_error_budget']}`",
            f"- simultaneous false-authorization delta: `{payload['policy']['delta']}`",
            f"- minimum conservative edge retention: `{payload['policy']['minimum_worst_direction_retention']}`",
            f"- maximum realized KL for bundle allocation: `{payload['policy']['maximum_realized_kl']}`",
            f"- minimum grouped replicates: `{payload['policy']['minimum_replicates']}`",
            f"- hard RAM cap: `{payload['policy']['resource_preset']['ram_cap_mb']} MB`",
            f"- hard CPU cap: `{payload['policy']['resource_preset']['cpu_cap_pct']}%`",
            f"- hard I/O cap: `{payload['policy']['resource_preset']['io_cap_mb_s']} MB/s`",
            "",
            "## Review order",
            "",
            "1. Seal target-blind spectral geometry and bind model, dataset, and operator hashes.",
            "2. Audit checkpoint/context lineage, loops, orientation, uncertainty, and matched noise nulls.",
            "3. Reveal grouped held-out utility and matched-rank Haar controls only after the geometry seal.",
            "4. Join held-out damage and capped-run resource receipts.",
            "5. Route to authorize, local sectioning, additional audit, or rejection.",
            "",
            "## Interpretation",
            "",
            "- `holonomy_clean` is required for global signed coordinate control.",
            "- `lineage_certified` is sufficient for invariant bundle-energy allocation under its KL budget.",
            "- Ordinary model promotion is not blocked solely by non-identifiable signed internal coordinates.",
            "- Positive geometry cannot compensate for failed utility, damage, or resource gates.",
            "- The information coefficient is descriptive and is not used as the primary gate.",
            "- Aborts are retained as structured evidence but cannot be promoted as completed training runs.",
            "",
            "## Training status",
            "",
            "No model training or weight mutation was run in this experiment.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Review synthetic model-training candidates with RSITopology gates.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/benchmarks/hrm_training_review_results.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/hrm_training_review.md"),
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("experiments/hrm_training_review"),
    )
    args = parser.parse_args()

    payload = run_training_review_benchmark()
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    report = summary_markdown(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    args.out.write_text(serialized, encoding="utf-8")
    args.report.write_text(report, encoding="utf-8")
    (args.experiment_dir / "results.json").write_text(serialized, encoding="utf-8")
    (args.experiment_dir / "training_notes.md").write_text(training_notes(payload), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
