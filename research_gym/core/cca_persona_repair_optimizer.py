"""CCA-08: globally optimal finite repair over a host-certified action set."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from math import isfinite
from typing import Callable, Hashable, Mapping
from .cca_compositional import FiniteRetraction, scalar_persona_metrics

Action = Hashable

@dataclass(frozen=True)
class RepairScore:
    repair: FiniteRetraction
    intervention_cost: float
    fisher_loss: float
    task_regret: float

    def weighted_score(self, *, fisher_weight: float, regret_weight: float) -> float:
        if (not isfinite(fisher_weight) or fisher_weight < 0 or
                not isfinite(regret_weight) or regret_weight < 0):
            raise ValueError("weights must be finite and nonnegative")
        return (self.intervention_cost + fisher_weight * self.fisher_loss
                + regret_weight * self.task_regret)


def evaluate_retractions(
    universe: frozenset[Action],
    certified: frozenset[Action],
    baseline: Mapping[Action, float],
    persona_utility: Mapping[Action, float],
    task_utility: Mapping[Action, float],
    distance: Callable[[Action, Action], float],
    *,
    strength: float = 0.0,
    max_retractions: int = 10000,
) -> tuple[RepairScore, ...]:
    """Enumerate all safe, identity-preserving repairs within a hard size cap.

    A neural proposer cannot enlarge the host-supplied certified action set.
    """
    if not universe or not certified or not certified <= universe:
        raise ValueError("nonempty certified subset of a nonempty universe required")
    if set(baseline) != universe or set(persona_utility) != universe:
        raise ValueError("persona scores must cover exact universe")
    if set(task_utility) != universe or any(not isfinite(v) for v in task_utility.values()):
        raise ValueError("finite task utilities must cover universe")
    if type(max_retractions) is not int or max_retractions < 1:
        raise ValueError("positive integer enumeration budget required")
    ordered_safe = sorted(certified, key=repr)
    outside = sorted(universe - certified, key=repr)
    total = len(ordered_safe) ** len(outside)
    if total > max_retractions:
        raise ValueError(f"enumeration cap exceeded: {total} repairs")
    best_task = max(task_utility[a] for a in certified)
    out = []
    for images in product(ordered_safe, repeat=len(outside)):
        mapping = {a: a for a in certified}
        mapping.update(zip(outside, images))
        repair = FiniteRetraction(universe, certified, mapping)
        moments = scalar_persona_metrics(baseline, persona_utility, repair, strength)
        p = moments.distribution
        costs = {a: distance(a, repair(a)) for a in universe}
        if any(not isfinite(c) or c < 0 for c in costs.values()):
            raise ValueError("distance must be finite and nonnegative")
        cost = sum(p[a] * costs[a] for a in universe)
        regret = sum(p[a] * (best_task - task_utility[repair(a)]) for a in universe)
        out.append(RepairScore(repair, cost, moments.fisher_lost, regret))
    return tuple(out)


def pareto_repair_frontier(
    scores: tuple[RepairScore, ...], *, tolerance: float = 1e-12
) -> tuple[RepairScore, ...]:
    """Nondominated repairs in intervention cost, Fisher loss and task regret."""
    if not scores or tolerance < 0:
        raise ValueError("nonempty scores and nonnegative tolerance required")
    def vec(s):
        return (s.intervention_cost, s.fisher_loss, s.task_regret)
    result = []
    for s in scores:
        v = vec(s)
        if not any(all(x <= y + tolerance for x, y in zip(vec(t), v)) and
                   any(x < y - tolerance for x, y in zip(vec(t), v))
                   for t in scores):
            result.append(s)
    return tuple(result)


def optimal_repair(
    scores: tuple[RepairScore, ...], *, fisher_weight: float = 0.0,
    regret_weight: float = 0.0,
) -> RepairScore:
    """Global weighted minimum over the fully enumerated feasible repair family."""
    if not scores:
        raise ValueError("no certified repairs")
    return min(scores, key=lambda x: x.weighted_score(
        fisher_weight=fisher_weight, regret_weight=regret_weight))
