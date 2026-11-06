"""Pre-execution index-bank routing plans, never inferred task performance."""
from __future__ import annotations

import hashlib
import random
from typing import Any

VALID_POLICIES = ("trm", "ldt_verified", "hybrid", "hybrid_vpd", "random_matched")


def route(record: dict[str, Any], policy: str) -> dict:
    if policy not in VALID_POLICIES:
        raise ValueError(f"Unknown routing policy: {policy}")
    task_id = str(record["task_id"])
    candidates = list(record["candidate_indices"])
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("candidate_indices must be unique and nonempty")
    if any(type(c) is not int or not 0 <= c <= 15 for c in candidates):
        raise ValueError("Expected the 16 verified expert-index stacks (0 through 15)")
    k = record["bank_size"]
    if type(k) is not int or k < 1 or k > len(candidates):
        raise ValueError("Bank size must be within available index count")
    scores = record["trm_scores"]
    if set(scores) != {str(c) for c in candidates}:
        raise ValueError("Each candidate needs a TRM score")
    if any(type(scores[str(c)]) not in (int, float) for c in candidates):
        raise ValueError("TRM scores must be numeric")
    from math import isfinite
    if any(not isfinite(scores[str(c)]) for c in candidates):
        raise ValueError("TRM scores must be finite")

    independently_verified = (
        record.get("ldt_verification") == "independent"
        and bool(record.get("verifier_receipt"))
    )
    approved = record.get("ldt_allowed", [])
    if not isinstance(approved, list) or any(x not in candidates for x in approved):
        raise ValueError("Invalid LDT candidate set")
    eligible = candidates
    mode = "trm_unrestricted"
    if policy in ("ldt_verified", "hybrid", "hybrid_vpd"):
        if independently_verified:
            eligible = [c for c in candidates if c in approved]
            mode = "independent_hard_gate"
        elif policy == "ldt_verified":
            eligible = []
            mode = "abstain_missing_verifier"
        else:
            mode = "unverified_ldt_ignored"
    if policy == "hybrid_vpd":
        vpd = record.get("vpd_scores")
        if record.get("activation_source") != "routed_kimi":
            raise ValueError("VPD-guided routing requires routed Kimi activations")
        if not record.get("activation_receipt"):
            raise ValueError("VPD-guided routing requires an activation capture receipt")
        if not isinstance(vpd, dict) or set(vpd) != {str(c) for c in candidates}:
            raise ValueError("VPD scores required for every candidate")
        if any(type(vpd[str(c)]) not in (int, float)
               or not isfinite(vpd[str(c)]) for c in candidates):
            raise ValueError("VPD scores must be finite")
        alpha = record.get("vpd_weight", 0.25)
        if type(alpha) not in (int, float) or not 0 <= alpha <= 1:
            raise ValueError("VPD coefficient must be calibrated in [0,1]")
        # Calibration-only tuning is a caller obligation, recorded in the trace.
        if not record.get("calibration_receipt"):
            raise ValueError("VPD coefficient requires calibration provenance")
        scores = {str(c): (1 - alpha) * scores[str(c)] + alpha * vpd[str(c)]
                  for c in candidates}
    if policy == "random_matched":
        seed = int(record.get("seed", 11))
        keyed_seed = int.from_bytes(
            hashlib.sha256(f"{seed}/{task_id}".encode()).digest()[:8], "big"
        )
        eligible = random.Random(keyed_seed).sample(eligible, len(eligible))
        selected = eligible[:k]
        mode = "random_matched_budget"
    elif policy == "ldt_verified":
        selected = sorted(eligible)[:k]
    else:
        selected = sorted(eligible, key=lambda c: (-scores[str(c)], c))[:k]
    return {
        "task_id": task_id,
        "policy": policy,
        "selected_indices": selected,
        "requested_bank_size": k,
        "executed_bank_size": len(selected),
        "abstained": not bool(selected),
        "verification_mode": mode,
        "status": "routing_plan_only",
        "top_k_per_token": record.get("top_k_per_token"),
        "inference_measurement": None,
    }
