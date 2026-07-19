"""Generate and seal deterministic LSPG-v0 proposal batches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from research_gym.prognostics.planner import LoopSchedulePlanner
from research_gym.prognostics.schemas import ResourceProfile


ROOT = Path(__file__).resolve().parents[2]


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value))


def _write_jsonl(path: Path, values: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for value in values:
            handle.write(_canonical(value))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--scale-ladder", type=Path, required=True)
    parser.add_argument("--mutation-space", type=Path, required=True)
    parser.add_argument("--resource-profile", type=Path, required=True)
    parser.add_argument("--promotion-policy", type=Path, default=ROOT / "configs/lsa/promotion_policy_v0.json")
    parser.add_argument("--max-proposals", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    resource_data = _load(args.resource_profile)
    profile = ResourceProfile(**resource_data["profile"])
    planner = LoopSchedulePlanner(
        base_receipt=args.base_receipt.resolve(),
        scale_ladder=_load(args.scale_ladder),
        mutation_space=_load(args.mutation_space),
        resource_profile=profile,
        promotion_policy=_load(args.promotion_policy),
        code_commit=code_commit,
    )
    proposals = planner.propose(args.max_proposals)
    if len(proposals) != args.max_proposals:
        raise RuntimeError(f"requested {args.max_proposals} proposals, generated {len(proposals)}")
    output = args.out.resolve()
    rows = [proposal.to_dict() for proposal in proposals]
    files = {
        "proposal_manifest.json": {
            "protocol_id": "loop_schedule_prognostic_gym_v0",
            "status": "sealed_before_execution",
            "proposal_count": len(rows),
            "code_commit": code_commit,
            "proposal_hashes": [row["proposal_hash"] for row in rows],
            "execution_scope": "S0_local_screening_only",
        },
        "theory_predictions.json": {row["proposal_id"]: row["theory"] for row in rows},
        "empirical_predictions.json": {row["proposal_id"]: row["empirical"] for row in rows},
        "resource_forecasts.json": {row["proposal_id"]: row["resources"] for row in rows},
        "promotion_policy.json": _load(args.promotion_policy),
    }
    for name, value in files.items():
        _write_json(output / name, value)
    _write_jsonl(output / "proposal_table.jsonl", rows)
    receipt = {
        "protocol_id": "loop_schedule_prognostic_gym_v0",
        "status": "sealed_before_execution",
        "inputs": {
            "base_receipt_sha256": _sha(args.base_receipt),
            "scale_ladder_sha256": _sha(args.scale_ladder),
            "mutation_space_sha256": _sha(args.mutation_space),
            "resource_profile_sha256": _sha(args.resource_profile),
            "promotion_policy_sha256": _sha(args.promotion_policy),
        },
        "outputs": {
            name: _sha(output / name)
            for name in (*files, "proposal_table.jsonl")
        },
        "proposal_count": len(rows),
        "code_commit": code_commit,
        "outcomes_observed": False,
    }
    _write_json(output / "proposal_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
