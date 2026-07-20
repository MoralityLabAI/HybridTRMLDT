"""Materialize a sealed LSPG architecture campaign stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from research_gym.architecture_discovery.campaign import (
    build_stage_cells,
    file_sha256,
    write_stage_manifest,
)
from research_gym.architecture_discovery.planner import read_proposals


ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument(
        "--proposal-dir",
        type=Path,
        default=ROOT / "experiments/loop_schedule_architecture_discovery_v1/proposals",
    )
    parser.add_argument(
        "--profile-selection",
        type=Path,
        default=ROOT / "experiments/loop_schedule_architecture_discovery_v1/calibration/profile_selection.json",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "configs/lsa/architecture_promotion_policy_v1.json",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    proposals = {value.proposal_id: value for value in read_proposals(args.proposal_dir)}
    candidate_ids = list(args.candidate_id)
    if args.stage == "A1" and not candidate_ids:
        candidate_ids = sorted(
            value.proposal_id
            for value in proposals.values()
            if value.batch == "batch_1" and value.role == "candidate"
        )
    if args.stage == "A1_reserve" and not candidate_ids:
        candidate_ids = sorted(
            value.proposal_id
            for value in proposals.values()
            if value.batch == "batch_2_reserve" and value.role == "candidate"
        )
    selection = _load(args.profile_selection)
    if selection["status"] != "selected" or selection["task_outcomes_observed"]:
        raise ValueError("campaign profile is not a valid resource-only selection")
    cells = build_stage_cells(
        stage_id=args.stage,
        candidate_ids=candidate_ids,
        proposals=proposals,
        policy=_load(args.policy),
        selected_profile=str(selection["selected_profile"]),
    )
    output = args.out or (
        ROOT
        / "experiments/loop_schedule_architecture_discovery_v1/campaign/manifests"
        / f"{args.stage}.json"
    )
    manifest = write_stage_manifest(
        cells,
        output,
        stage_id=args.stage,
        selected_profile=str(selection["selected_profile"]),
        profile_selection_hash=str(selection["selection_hash"]),
        proposal_manifest_sha256=file_sha256(args.proposal_dir / "proposal_manifest.json"),
        code_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
