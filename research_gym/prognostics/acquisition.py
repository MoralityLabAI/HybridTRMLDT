"""Resource-constrained information and falsification scoring."""

from __future__ import annotations

from .schemas import DecisionPrediction, ResourceForecast


def acquisition_decision(
    *,
    information_gain: float,
    falsification_value: float,
    diversity: float,
    estimated_compute: float,
    decision_kind: str,
    resource: ResourceForecast,
    known_invalid: bool = False,
) -> DecisionPrediction:
    exclusions = list(resource.exclusion_reasons)
    if known_invalid:
        exclusions.append("known_invalid_control")
    rankable = resource.locally_executable and not known_invalid
    score = 0.0
    if rankable:
        score = (
            information_gain
            * (1.0 + falsification_value)
            * (1.0 + diversity)
            / max(estimated_compute, 1e-12)
        )
    return DecisionPrediction(
        information_gain=information_gain,
        falsification_value=falsification_value,
        diversity=diversity,
        estimated_compute=estimated_compute,
        acquisition_score=score,
        decision_kind=decision_kind,
        rankable=rankable,
        exclusion_reasons=tuple(sorted(set(exclusions))),
    )
