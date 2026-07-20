from __future__ import annotations

import json

from research_gym.prognostics.reporting import (
    canonical_bytes,
    normalize_nonfinite,
    summarize_stage_a,
)


def test_nonfinite_partial_values_normalize_to_strict_json_null() -> None:
    normalized = normalize_nonfinite({"loss": float("nan"), "ratio": float("inf")})

    assert normalized == {"loss": None, "ratio": None}
    assert json.loads(canonical_bytes(normalized)) == normalized


def test_incomplete_timeout_blocks_promotion() -> None:
    proposals = [{"proposal_id": "p1", "resources": []}, {"proposal_id": "p2", "resources": []}]
    records = [
        {
            "proposal_id": "p1",
            "status": "stopped",
            "stop_reason": "gradient_norm_above_100",
            "mean_step_seconds": None,
        }
    ]

    summary = summarize_stage_a(
        proposals,
        records,
        {"status": "aborted", "abort_reason": "timeout"},
    )

    assert summary["status"] == "incomplete"
    assert summary["promotion"]["action"] == "do_not_promote"
    assert not summary["promotion"]["larger_scale_execution_allowed"]
    assert "incomplete_stage_a" in summary["promotion"]["reasons"]
    assert "stage_a_timeout" in summary["promotion"]["reasons"]
