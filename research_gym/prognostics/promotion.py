"""Sealed screening, promotion, and stopping rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PromotionDecision:
    action: str
    reasons: tuple[str, ...]


def edge_censoring_present(prior: Mapping[str, Any] | Any) -> bool:
    observations = (
        prior.boundary_observations
        if hasattr(prior, "boundary_observations")
        else prior.get("boundary_observations", ())
    )
    return any(
        row["observation"]["kind"] in {"left_censored", "right_censored"}
        for row in observations
    )


def promotion_decision(
    *,
    integrity_passed: bool,
    stable_sign_or_ordering: bool,
    posterior_contracted: bool,
    edge_censored: bool,
    redundant: bool,
) -> PromotionDecision:
    reasons = []
    if not integrity_passed:
        reasons.append("integrity_failure")
    if not stable_sign_or_ordering:
        reasons.append("unstable_sign_or_ordering")
    if not posterior_contracted:
        reasons.append("insufficient_posterior_contraction")
    if edge_censored:
        reasons.append("unresolved_grid_edge_censoring")
    if redundant:
        reasons.append("redundant_candidate")
    return PromotionDecision(
        "promote" if not reasons else "stop_or_extend_grid",
        tuple(reasons),
    )
