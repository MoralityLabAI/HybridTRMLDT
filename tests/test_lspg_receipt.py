from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_lspg_receipt_rehashes_sealed_and_normalized_artifacts() -> None:
    receipt = json.loads((ROOT / "data/benchmarks/lspg_v0_receipt.json").read_text())

    assert _sha("experiments/loop_schedule_prognostic_gym_v0/proposals/proposal_table.jsonl") == receipt["proposals"]["table_sha256"]
    assert _sha("experiments/loop_schedule_prognostic_gym_v0/proposals/proposal_receipt.json") == receipt["proposals"]["receipt_sha256"]
    assert _sha("experiments/loop_schedule_prognostic_gym_v0/runs/stage_a/screening_records.partial.jsonl") == receipt["stage_a"]["raw_partial_sha256"]
    assert _sha("experiments/loop_schedule_prognostic_gym_v0/receipts/stage_a_records.normalized.jsonl") == receipt["stage_a"]["normalized_records_sha256"]

    normalized = [
        json.loads(line)
        for line in (ROOT / "experiments/loop_schedule_prognostic_gym_v0/receipts/stage_a_records.normalized.jsonl").read_text().splitlines()
    ]
    assert len(normalized) == receipt["stage_a"]["receipts"] == 3
    assert normalized[0]["initial_loss"] is None
    assert normalized[0]["final_over_initial_loss"] is None
    assert not receipt["promotion"]["allowed"]
    assert not receipt["promotion"]["larger_scale_execution"]
