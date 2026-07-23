"""Materialize the fresh confirmatory panel for the RLM uncertainty pre-gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

from research_gym.integrity import canonical_file_sha256
from research_gym.scripts import materialize_rlm_evidence_acquisition_v0 as base


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_SEED = 494117
TASK_SUFFIX = "__pregate_v0_1"
CORPUS_ID = "rlm_uncertainty_pregate_v0_1"
DEFAULT_TASKS = ROOT / "data/benchmarks/rlm_uncertainty_pregate_v0_1_tasks.json"
DEFAULT_PROPOSALS = ROOT / "data/benchmarks/rlm_uncertainty_pregate_v0_1_proposals.jsonl"
DEFAULT_TRUTH = ROOT / "data/benchmarks/rlm_uncertainty_pregate_v0_1_truth.jsonl"
DEFAULT_RECEIPT = ROOT / "data/benchmarks/rlm_uncertainty_pregate_v0_1_materialization_receipt.json"


def materialize(
    tasks_path: Path,
    proposals_path: Path,
    truth_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    for path in (tasks_path, proposals_path, truth_path, receipt_path):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite uncertainty-pregate artifact: {path}")
    original_seed, original_suffix = base.GENERATOR_SEED, base.TASK_SUFFIX
    try:
        base.GENERATOR_SEED = GENERATOR_SEED
        base.TASK_SUFFIX = TASK_SUFFIX
        payload, truth_rows = base._renamed_payload()
    finally:
        base.GENERATOR_SEED = original_seed
        base.TASK_SUFFIX = original_suffix
    payload["suite_id"] = CORPUS_ID
    proposals = base._proposal_rows(payload, truth_rows)
    truth_rows = sorted(truth_rows, key=lambda row: row["task_id"])
    base._write_json(tasks_path, payload)
    base._write_jsonl(proposals_path, proposals)
    base._write_jsonl(truth_path, truth_rows)
    receipt = {
        "status": "materialized",
        "corpus_id": CORPUS_ID,
        "generator_seed": GENERATOR_SEED,
        "task_suffix": TASK_SUFFIX,
        "task_count": len(payload["tasks"]),
        "proposal_count": len(proposals),
        "truth_count": len(truth_rows),
        "checkpoint_path": base.CHECKPOINT.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": canonical_file_sha256(base.CHECKPOINT),
        "tasks_path": tasks_path.relative_to(ROOT).as_posix(),
        "tasks_sha256": canonical_file_sha256(tasks_path),
        "proposals_path": proposals_path.relative_to(ROOT).as_posix(),
        "proposals_sha256": canonical_file_sha256(proposals_path),
        "truth_path": truth_path.relative_to(ROOT).as_posix(),
        "truth_sha256": canonical_file_sha256(truth_path),
        "source_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "source_sha256": canonical_file_sha256(__file__),
        "provider_outcomes_present": False,
        "evaluation_outcomes_accessed": False,
        "git_head": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "materialized_utc": datetime.now(timezone.utc).isoformat(),
    }
    base._write_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("materialize",), default="materialize")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--vram-fraction", type=float, default=0.35)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    value = materialize(
        args.tasks.resolve(),
        args.proposals.resolve(),
        args.truth.resolve(),
        args.receipt.resolve(),
    )
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
