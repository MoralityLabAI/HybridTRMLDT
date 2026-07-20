"""Create a hashed transition receipt from one completed LSPG stage."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path

from lsa.canonical import digest
from research_gym.architecture_discovery.analysis import (
    a1_shortlist,
    final_inference,
    provisional_selection,
)
from research_gym.architecture_discovery.campaign import read_stage_manifest
from research_gym.architecture_discovery.planner import read_proposals


ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _results(manifest, runs: Path):
    rows = []
    for cell in manifest["cells"]:
        result_path = runs / cell["cell_id"] / "result.json"
        receipt_path = runs / cell["cell_id"] / "result_receipt.json"
        result = _load(result_path)
        receipt = _load(receipt_path)
        if hashlib.sha256(result_path.read_bytes()).hexdigest() != receipt["result_sha256"]:
            raise ValueError(f"result receipt mismatch: {cell['cell_id']}")
        rows.append(result)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--prior-b-transition", type=Path)
    parser.add_argument(
        "--proposal-dir",
        type=Path,
        default=ROOT / "experiments/loop_schedule_architecture_discovery_v1/proposals",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_stage_manifest(args.stage_manifest)
    proposals = {value.proposal_id: value for value in read_proposals(args.proposal_dir)}
    rows = _results(manifest, args.runs)
    stage = str(manifest["stage_id"])
    candidate_ids = sorted(
        {cell["proposal_id"] for cell in manifest["cells"] if cell["role"] == "candidate"}
    )
    payload = {
        "schema_version": 1,
        "completed_stage": stage,
        "stage_manifest_hash": manifest["manifest_hash"],
        "task_outcomes_observed": True,
    }
    if stage in {"A1", "A1_reserve"}:
        selected, decisions = a1_shortlist(candidate_ids, proposals, rows)
        payload.update(
            {
                "action": "prepare_A2" if stage == "A1" else "prepare_A2_reserve",
                "selected_candidates": list(selected),
                "decisions": list(decisions),
            }
        )
    elif stage in {"A2", "A2_reserve", "B", "B_reserve", "C"}:
        maximum = 4 if stage != "C" else 2
        selected, decisions = provisional_selection(
            candidate_ids,
            proposals,
            rows,
            maximum=maximum,
            structurally_distinct=stage == "C",
        )
        decision_rows = [asdict(value) for value in decisions]
        if stage == "B" and len(selected) < 2:
            reserve = sorted(
                value.proposal_id
                for value in proposals.values()
                if value.batch == "batch_2_reserve" and value.role == "candidate"
            )
            payload.update(
                {
                    "action": "open_single_reserve_batch",
                    "selected_candidates": list(selected),
                    "next_candidates": reserve,
                    "decisions": decision_rows,
                }
            )
        elif stage == "B_reserve":
            if args.prior_b_transition is None:
                raise ValueError("B_reserve requires the first B transition receipt")
            prior = _load(args.prior_b_transition)
            combined = {
                row["proposal_id"]: row
                for row in [*prior["decisions"], *decision_rows]
                if row["passed"]
            }
            selected = tuple(
                row["proposal_id"]
                for row in sorted(
                    combined.values(),
                    key=lambda value: (-float(value["mean_macro_delta"]), value["proposal_id"]),
                )[:4]
            )
            payload.update(
                {
                    "action": "prepare_C",
                    "selected_candidates": list(selected),
                    "decisions": decision_rows,
                    "reserve_batches_consumed": 1,
                }
            )
        else:
            next_stage = {"A2": "B", "A2_reserve": "B_reserve", "B": "C", "C": "D"}[stage]
            payload.update(
                {
                    "action": f"prepare_{next_stage}",
                    "selected_candidates": list(selected),
                    "decisions": decision_rows,
                }
            )
    elif stage == "D":
        payload.update(
            {
                "action": "finalize",
                "final_inference": final_inference(
                    candidate_ids, proposals, rows, root=ROOT
                ),
            }
        )
    else:
        raise ValueError(f"unsupported completed stage: {stage}")
    payload["transition_hash"] = digest(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
