from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.architecture_discovery.planner import read_proposals
from research_gym.architecture_discovery.tasks import read_task_bundle
from research_gym.architecture_discovery.training import (
    TrainingCellConfig,
    result_receipt,
    run_training_cell,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one resumable LSPG architecture cell.")
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--proposal-id", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--token-visit-budget", type=int, required=True)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--microbatch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--evaluation-limit-per-family", type=int, default=256)
    parser.add_argument("--allow-locked-evaluation", action="store_true")
    parser.add_argument("--vram-fraction", type=float, default=0.60)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    proposals = {value.proposal_id: value for value in read_proposals(args.proposal_dir)}
    if args.proposal_id not in proposals:
        raise ValueError(f"unknown proposal: {args.proposal_id}")
    bundle = read_task_bundle(args.dataset_dir)
    config = TrainingCellConfig(
        stage=args.stage,
        scale_rung=args.scale,
        seed=args.seed,
        token_visit_budget=args.token_visit_budget,
        effective_batch_size=args.effective_batch_size,
        microbatch_size=args.microbatch_size,
        learning_rate=args.learning_rate,
        evaluation_limit_per_family=(
            None if args.evaluation_limit_per_family < 0 else args.evaluation_limit_per_family
        ),
        allow_locked_evaluation=args.allow_locked_evaluation,
        vram_fraction=args.vram_fraction,
    )
    result = run_training_cell(proposals[args.proposal_id], bundle, config, output_dir=args.out)
    result_path = args.out / result.cell_id / "result.json"
    receipt = result_receipt(result_path)
    receipt_path = args.out / result.cell_id / "result_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
