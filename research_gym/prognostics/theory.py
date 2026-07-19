"""Deterministic algebraic predictions that are never empirically overwritten."""

from __future__ import annotations

from typing import Any, Mapping

from .schemas import TheoryPrediction


def theory_prediction(
    candidate: Mapping[str, Any],
    *,
    unique_parameters: int,
    module_parameters: int,
    gamma_assumption: float,
) -> TheoryPrediction:
    visits = int(candidate["expanded_visits"])
    gradient_visits = int(candidate["gradient_visible_visits"])
    alpha = float(candidate.get("alpha", 1.0))
    beta = float(candidate.get("beta", 1.0))
    kappa = float(candidate.get("kappa", 1.0))
    residual_sublayers = int(candidate.get("residual_sublayers", 2))
    stability = (
        residual_sublayers
        * gradient_visits
        * kappa
        * (beta / alpha) ** 2
    )
    return TheoryPrediction(
        unique_parameters=unique_parameters,
        applied_parameters_per_token=module_parameters * visits,
        gradient_applied_parameters_per_token=module_parameters * gradient_visits,
        expanded_visits=visits,
        gradient_visible_visits=gradient_visits,
        retained_state_edges=int(candidate["retained_state_edges"]),
        gamma_assumption=gamma_assumption,
        p_star=(1.0 + gamma_assumption) / 4.0,
        stability_functional=stability,
    )
