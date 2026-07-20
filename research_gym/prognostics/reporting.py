"""Finalize complete or aborted Stage-A runs without blending partial evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def normalize_nonfinite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: normalize_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_nonfinite(item) for item in value]
    return value


def load_partial_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        normalize_nonfinite(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_stage_a(
    proposals: list[Mapping[str, Any]],
    records: list[Mapping[str, Any]],
    resource_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    by_id = {record["proposal_id"]: record for record in records}
    completed = [record for record in records if record["status"] == "completed"]
    stopped = [record for record in records if record["status"] == "stopped"]
    pending = [proposal["proposal_id"] for proposal in proposals if proposal["proposal_id"] not in by_id]
    forecast_ratios = []
    for proposal in proposals:
        record = by_id.get(proposal["proposal_id"])
        if not record or record.get("mean_step_seconds") is None:
            continue
        s0 = next(row for row in proposal["resources"] if row["scale_rung"] == "S0")
        forecast_ratios.append(
            {
                "proposal_id": proposal["proposal_id"],
                "predicted_step_seconds": s0["estimated_step_seconds"],
                "measured_step_seconds": record["mean_step_seconds"],
                "measured_over_predicted": record["mean_step_seconds"]
                / max(s0["estimated_step_seconds"], 1e-12),
            }
        )
    reasons = ["unresolved_grid_edge_censoring"]
    if pending:
        reasons.append("incomplete_stage_a")
    if resource_receipt.get("status") != "completed":
        reasons.append(f"stage_a_{resource_receipt.get('abort_reason') or resource_receipt.get('status')}")
    if forecast_ratios and max(row["measured_over_predicted"] for row in forecast_ratios) > 2:
        reasons.append("resource_forecast_miscalibrated")
    return {
        "protocol_id": "loop_schedule_prognostic_gym_v0",
        "stage": "A",
        "status": "complete" if not pending and resource_receipt.get("status") == "completed" else "incomplete",
        "proposal_count": len(proposals),
        "receipt_count": len(records),
        "completed_count": len(completed),
        "stopped_count": len(stopped),
        "pending_proposal_ids": pending,
        "gradient_stop_proposal_ids": [
            row["proposal_id"]
            for row in stopped
            if row.get("stop_reason") == "gradient_norm_above_100"
        ],
        "forecast_calibration": forecast_ratios,
        "promotion": {
            "action": "do_not_promote",
            "reasons": sorted(set(reasons)),
            "larger_scale_execution_allowed": False,
        },
        "claim_scope": "trainability-boundary and schedule-cost measurement only",
    }
