"""Execute the sealed LSPG-v0 Stage-A proposal batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from research_gym.neural.training import screen_proposal


ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    proposal_dir = args.proposal_dir.resolve()
    receipt = json.loads((proposal_dir / "proposal_receipt.json").read_text())
    if receipt["status"] != "sealed_before_execution" or receipt["outcomes_observed"]:
        raise RuntimeError("proposal batch was not sealed before outcomes")
    table = proposal_dir / "proposal_table.jsonl"
    if _sha(table) != receipt["outputs"]["proposal_table.jsonl"]:
        raise RuntimeError("proposal table hash mismatch")
    proposals = [json.loads(line) for line in table.read_text().splitlines()]
    if any(proposal["scale_rung"] != "S0" for proposal in proposals):
        raise RuntimeError("Stage A may execute only S0 proposals")
    policy = json.loads((proposal_dir / "promotion_policy.json").read_text())
    stage = policy["stage_a"]
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / "screening_records.partial.jsonl"
    if partial_path.exists():
        raise RuntimeError("partial screening output already exists; use a fresh output directory")
    records = []
    for proposal in proposals:
        result = screen_proposal(
            proposal,
            output_dir=output,
            exposure_budget=int(stage["exposure_budget"]),
            checkpoints_pct=tuple(policy["checkpoints_pct"]),
            seed=int(stage["seed"]),
        )
        records.append(result.to_dict())
        with partial_path.open("ab") as handle:
            handle.write(_canonical(result.to_dict()))
    records_path = output / "screening_records.jsonl"
    with records_path.open("wb") as handle:
        for record in sorted(records, key=lambda row: row["proposal_id"]):
            handle.write(_canonical(record))
    summary = {
        "protocol_id": "loop_schedule_prognostic_gym_v0",
        "stage": "A",
        "scale": "S0",
        "proposal_receipt_sha256": _sha(proposal_dir / "proposal_receipt.json"),
        "records_sha256": _sha(records_path),
        "record_count": len(records),
        "completed": sum(row["status"] == "completed" for row in records),
        "stopped": sum(row["status"] != "completed" for row in records),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "claim_scope": "trainability-boundary and schedule-cost measurement only",
        "scale_promotion_evaluated": False,
    }
    (output / "screening_result.json").write_bytes(_canonical(summary))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
