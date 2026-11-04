"""Dual-LDT-04: finite proof-carrying preference-repair certificate.

Authority is supplied by an independent execution checker. The maximization
checks only a fixed finite candidate grammar, not all possible language actions.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Mapping, FrozenSet

@dataclass(frozen=True)
class PreferenceCertificate:
    chosen: str
    reference: str
    witness: str
    target_score: float
    chosen_score: float
    chosen_cost: int
    measured_regret: float
    witness_regret: float
    certified_upper_regret: float
    adjusted_optimum: float
    viable_count: int


def certify_preference_bound(
    reference: str,
    chosen: str,
    feasible: FrozenSet[str],
    score: Mapping[str,float],
    costs: Mapping[str,int],
    *,
    rho: float = 0.10,
    tolerance: float = 1e-10,
) -> PreferenceCertificate:
    """Prove a bounded raw preference loss from the argmax adjusted score.

    If v is any certified alternative then
    u(chosen)-rho*c(chosen) >= u(v)-rho*c(v),
    hence max(0,u(reference)-u(chosen))
       <= max(0,u(reference)-u(v)) + rho*c(v)
    when costs are nonnegative. Minimize this upper bound over all v.
    """
    if not feasible or chosen not in feasible or reference not in score:
        raise ValueError("nonempty feasible set and scored reference/chosen needed")
    if not isfinite(rho) or rho < 0 or not isfinite(tolerance) or tolerance < 0:
        raise ValueError("invalid pricing or tolerance")
    if not feasible <= set(score) or not feasible <= set(costs):
        raise ValueError("all feasible plans need scores and costs")
    if any(not isfinite(score[a]) or type(costs[a]) is not int or costs[a] < 0
           for a in feasible) or not isfinite(score[reference]):
        raise ValueError("nonfinite score or invalid resource price")
    adjusted={a:score[a]-rho*costs[a] for a in feasible}
    optimum=max(adjusted.values())
    if adjusted[chosen] + tolerance < optimum:
        raise ValueError("selected plan is not a maximum of certified preference")
    target=score[reference]
    witness=min(sorted(feasible), key=lambda a:
       (max(0.,target-score[a])+rho*costs[a],a))
    delta=max(0.,target-score[witness])
    bound=delta+rho*costs[witness]
    measured=max(0.,target-score[chosen])
    if measured > bound+tolerance:
        raise AssertionError("bounded-regret mathematical invariant failed")
    return PreferenceCertificate(chosen,reference,witness,target,score[chosen],
        costs[chosen],measured,delta,bound,optimum,len(feasible))


def verify_superset_preference(
    baseline: FrozenSet[str], enlarged: FrozenSet[str],
    baseline_choice: str, enlarged_choice: str,
    scores: Mapping[str,float], costs: Mapping[str,int], *,
    rho:float=0.10, tolerance:float=1e-10,
) -> float:
    """A superset can never reduce the maximum of the same adjusted score."""
    if not baseline or not baseline <= enlarged:
        raise ValueError("expansion must contain a nonempty baseline")
    if baseline_choice not in baseline or enlarged_choice not in enlarged:
        raise ValueError("selections outside their certified sets")
    if any(a not in scores or a not in costs for a in enlarged):
        raise ValueError("missing preference inputs")
    v=lambda a:scores[a]-rho*costs[a]
    if v(baseline_choice)+tolerance < max(v(a) for a in baseline):
        raise ValueError("baseline choice is not argmax")
    if v(enlarged_choice)+tolerance < max(v(a) for a in enlarged):
        raise ValueError("enlarged choice is not argmax")
    gain=v(enlarged_choice)-v(baseline_choice)
    if gain < -tolerance:
        raise AssertionError("superset optimum fell")
    return gain
