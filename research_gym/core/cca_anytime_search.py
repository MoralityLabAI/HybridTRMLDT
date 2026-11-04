"""CCA-09: proof-checked anytime search with neural proposals as ordering only.

Every upper bound and feasibility/utility verdict requires an independent
host-owned checker. Optimality claims depend on the soundness of these checkers.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, Mapping, Sequence

@dataclass(frozen=True)
class UpperCertificate:
    action: str
    upper: float
    binding: str
    evidence: Any

@dataclass(frozen=True)
class ActionVerdict:
    action: str
    feasible: bool
    utility: float | None
    evidence: Any

@dataclass(frozen=True)
class AnytimeStep:
    evaluated: str
    feasible: bool
    incumbent: str | None
    lower: float | None
    upper: float | None
    gap: float | None
    checked: int

@dataclass(frozen=True)
class AnytimeResult:
    action: str | None
    status: str
    steps: tuple[AnytimeStep, ...]
    lower: float | None
    upper: float | None
    gap: float | None
    checked: tuple[str, ...]
    skipped_duplicates: int

def certified_anytime_search(
    context: Mapping[str, Any],
    actions: Sequence[str],
    proposal_order: Sequence[str],
    upper_bounds: Mapping[str, UpperCertificate],
    *,
    check_bound: Callable[[Mapping[str, Any], str, UpperCertificate], bool],
    evaluate: Callable[[Mapping[str, Any], str], ActionVerdict],
    check_verdict: Callable[[Mapping[str, Any], str, ActionVerdict], bool],
    max_evaluations: int,
    epsilon: float = 0.0,
) -> AnytimeResult:
    """Best-so-far certified action and valid upper bound on regret.

    Negative verdicts must also be independently checked: eliminating a
    feasible action on a self-reported negative would falsify optimality.
    """
    if not actions or len(set(actions)) != len(actions):
        raise ValueError("actions must be nonempty and unique")
    domain = tuple(actions)
    aset = set(domain)
    if set(upper_bounds) != aset:
        raise ValueError("bounds required for every action and no others")
    if type(max_evaluations) is not int or max_evaluations < 0:
        raise ValueError("nonnegative integer evaluation budget required")
    if not isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite nonnegative")
    for action in domain:
        proof = upper_bounds[action]
        if proof.action != action or not isfinite(proof.upper):
            raise ValueError("malformed upper-bound certificate")
        if check_bound(context, action, proof) is not True:
            raise ValueError(f"unverified upper bound: {action}")

    ordered: list[str] = []
    seen_suggestions: set[str] = set()
    duplicates = 0
    for action in proposal_order:
        if action not in aset:
            raise ValueError("neural proposer nominated action outside domain")
        if action in seen_suggestions:
            duplicates += 1
        else:
            ordered.append(action)
            seen_suggestions.add(action)
    ordered.extend(a for a in domain if a not in seen_suggestions)

    evaluated: set[str] = set()
    incumbent: str | None = None
    lower: float | None = None
    upper: float | None = max(proof.upper for proof in upper_bounds.values())
    priority = {a: i for i, a in enumerate(domain)}
    records: list[AnytimeStep] = []
    status = "budget_exhausted"

    for action in ordered[:min(max_evaluations, len(ordered))]:
        verdict = evaluate(context, action)
        if verdict.action != action or check_verdict(context, action, verdict) is not True:
            raise ValueError(f"unverified action verdict: {action}")
        if verdict.feasible:
            if verdict.utility is None or not isfinite(verdict.utility):
                raise ValueError("certified feasible action lacks finite utility")
            if verdict.utility > upper_bounds[action].upper:
                raise ValueError("observed utility exceeds trusted upper bound")
            if lower is None or verdict.utility > lower or (
                verdict.utility == lower and priority[action] < priority[incumbent]
            ):
                incumbent, lower = action, verdict.utility
        elif verdict.utility is not None:
            raise ValueError("infeasible verdict must have no utility")
        evaluated.add(action)
        unseen_upper = [proof.upper for a, proof in upper_bounds.items() if a not in evaluated]
        if lower is None:
            upper = max(unseen_upper) if unseen_upper else None
            gap = None
        else:
            upper = max([lower, *unseen_upper])
            gap = max(0.0, upper - lower)
        records.append(AnytimeStep(action, verdict.feasible, incumbent,
                                   lower, upper, gap, len(evaluated)))
        if incumbent is not None and gap <= epsilon:
            status = "epsilon_optimal"
            break
    else:
        if len(evaluated) == len(domain):
            status = "proven_empty" if incumbent is None else "epsilon_optimal"
    return AnytimeResult(incumbent, status, tuple(records), lower, upper,
                         None if lower is None or upper is None else max(0.0, upper - lower),
                         tuple(step.evaluated for step in records), duplicates)
