"""CCA-06: finite repair composition and persona information under repair.

Theorems assume that admissible sets and repair maps describe the deployed
system. Probabilistic summaries evaluate exact finite sums in floating point.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from types import MappingProxyType
from typing import Hashable, Mapping, Sequence

Action = Hashable


@dataclass(frozen=True)
class FiniteRetraction:
    """A total deterministic repair into a nonempty subset, fixing that subset."""

    universe: frozenset[Action]
    permitted: frozenset[Action]
    outputs: Mapping[Action, Action]

    def __post_init__(self) -> None:
        if not self.universe or not self.permitted:
            raise ValueError("universe and permitted set must be nonempty")
        if not self.permitted <= self.universe:
            raise ValueError("permitted actions must belong to universe")
        if set(self.outputs) != self.universe:
            raise ValueError("repair must map every action in universe")
        if not set(self.outputs.values()) <= self.permitted:
            raise ValueError("repair maps outside permitted set")
        if any(self.outputs[a] != a for a in self.permitted):
            raise ValueError("repair must fix every permitted action")
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))

    def __call__(self, action: Action) -> Action:
        return self.outputs[action]

    def commutes_with(self, other: FiniteRetraction) -> bool:
        if self.universe != other.universe:
            raise ValueError("different action universes")
        return all(self(other(a)) == other(self(a)) for a in self.universe)

    def compose(self, other: FiniteRetraction) -> FiniteRetraction:
        """Verified retraction onto intersection, when the operators commute."""
        if not self.commutes_with(other):
            raise ValueError("repairs do not commute: joint retraction uncertified")
        return FiniteRetraction(
            self.universe,
            self.permitted & other.permitted,
            {a: self(other(a)) for a in self.universe},
        )


def maximal_sound_actions(
    possible_states: frozenset[Hashable],
    allowed_by_state: Mapping[Hashable, frozenset[Action]],
) -> frozenset[Action]:
    """Greatest action set safe for every represented concrete state.

    Empty or unknown interpretation sets fail closed. Coverage of the true
    state by possible_states is an independent semantic assumption.
    """
    if not possible_states:
        return frozenset()
    if not possible_states <= set(allowed_by_state):
        return frozenset()
    it = iter(possible_states)
    result = allowed_by_state[next(it)]
    for state in it:
        result = result & allowed_by_state[state]
    return frozenset(result)


@dataclass(frozen=True)
class ScalarPersonaMetrics:
    distribution: Mapping[Action, float]
    repaired_distribution: Mapping[Action, float]
    mean_pre: float
    mean_post: float
    post_derivative: float
    fisher_pre: float
    fisher_post: float
    fisher_lost: float


def _tilted_distribution(
    baseline: Mapping[Action, float],
    values: Mapping[Action, float],
    strength: float,
) -> dict[Action, float]:
    if not baseline or set(baseline) != set(values):
        raise ValueError("baseline and values need identical nonempty support")
    if not isfinite(strength):
        raise ValueError("nonfinite strength")
    if any(not isfinite(p) or p <= 0 for p in baseline.values()):
        raise ValueError("baseline must be finite and strictly positive")
    if any(not isfinite(v) for v in values.values()):
        raise ValueError("nonfinite utility")
    logits = {a: log(p) + strength * values[a] for a, p in baseline.items()}
    if any(not isfinite(x) for x in logits.values()):
        raise ValueError("nonfinite tilted logit")
    peak = max(logits.values())
    weights = {a: exp(x - peak) for a, x in logits.items()}
    denom = sum(weights.values())
    return {a: w / denom for a, w in weights.items()}


def scalar_persona_metrics(
    baseline: Mapping[Action, float],
    utility: Mapping[Action, float],
    repair: FiniteRetraction,
    strength: float,
) -> ScalarPersonaMetrics:
    """Compute covariance dose derivative and Fisher loss for a scalar persona."""
    if set(baseline) != repair.universe:
        raise ValueError("baseline support must equal repair universe")
    distribution = _tilted_distribution(baseline, utility, strength)
    outputs = {y: sum(p for a, p in distribution.items() if repair(a) == y)
               for y in repair.permitted}
    pre_mean = sum(p * utility[a] for a, p in distribution.items())
    post_mean = sum(p * utility[repair(a)] for a, p in distribution.items())
    derivative = sum(p * (utility[a] - pre_mean) *
                     (utility[repair(a)] - post_mean)
                     for a, p in distribution.items())
    if any(w <= 0 for w in outputs.values()):
        raise ValueError("repaired action underflow")
    fisher_pre = sum(p * (utility[a] - pre_mean) ** 2
                     for a, p in distribution.items())
    conditional = {
        y: sum(p * utility[a] for a, p in distribution.items()
               if repair(a) == y) / outputs[y]
        for y in repair.permitted
    }
    fisher_post = sum(outputs[y] * (conditional[y] - pre_mean) ** 2
                      for y in repair.permitted)
    fisher_lost = sum(p * (utility[a] - conditional[repair(a)]) ** 2
                      for a, p in distribution.items())
    return ScalarPersonaMetrics(distribution, outputs, pre_mean, post_mean,
                                derivative, fisher_pre, fisher_post, fisher_lost)


@dataclass(frozen=True)
class VectorPersonaFisher:
    before: tuple[tuple[float, ...], ...]
    after: tuple[tuple[float, ...], ...]
    lost: tuple[tuple[float, ...], ...]


def vector_persona_fisher(
    baseline: Mapping[Action, float],
    features: Mapping[Action, Sequence[float]],
    theta: Sequence[float],
    repair: FiniteRetraction,
) -> VectorPersonaFisher:
    """Finite-data processing: I(before)-I(after)=E[Cov(f|repair(A))]."""
    if not theta or any(not isfinite(x) for x in theta):
        raise ValueError("finite nonempty theta required")
    n = len(theta)
    if set(baseline) != repair.universe or set(features) != repair.universe:
        raise ValueError("features, baseline and repair must share domain")
    if any(len(x) != n or any(not isfinite(v) for v in x)
           for x in features.values()):
        raise ValueError("feature dimensions or values invalid")
    log_tilt = {a: sum(t * f for t, f in zip(theta, row))
                for a, row in features.items()}
    p = _tilted_distribution(baseline, log_tilt, 1.0)
    mean = [sum(p[a] * features[a][i] for a in p) for i in range(n)]
    py = {y: sum(p[a] for a in p if repair(a) == y) for y in repair.permitted}
    if any(w <= 0 for w in py.values()):
        raise ValueError("repaired action underflow")
    conditional = {
        y: [sum(p[a] * features[a][i] for a in p if repair(a) == y) / py[y]
            for i in range(n)]
        for y in repair.permitted
    }
    before = [[sum(p[a] * (features[a][i] - mean[i]) *
                   (features[a][j] - mean[j]) for a in p)
               for j in range(n)] for i in range(n)]
    after = [[sum(py[y] * (conditional[y][i] - mean[i]) *
                  (conditional[y][j] - mean[j]) for y in py)
              for j in range(n)] for i in range(n)]
    lost = [[sum(p[a] * (features[a][i] - conditional[repair(a)][i]) *
                 (features[a][j] - conditional[repair(a)][j]) for a in p)
             for j in range(n)] for i in range(n)]
    freeze = lambda m: tuple(tuple(row) for row in m)
    return VectorPersonaFisher(freeze(before), freeze(after), freeze(lost))
