"""CCA-01: small, dependency-free reference kernel for constitutional control.

The verifier and the semantic reader MUST be supplied by the host, independently
of the neural proposer. Exact guarantees are conditional on their correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from math import exp, isfinite, log
from typing import Callable, FrozenSet, Iterable, Mapping


class ConstitutionOp(str, Enum):
    BEAST_ONLY = "beast_only"
    JINN_ONLY = "jinn_only"
    MEET = "meet"
    PARALLAX = "parallax"


@dataclass(frozen=True)
class Constitution:
    universe: FrozenSet[str]
    hard_safe: FrozenSet[str]
    jinn: FrozenSet[str]
    beast: FrozenSet[str]

    def __post_init__(self) -> None:
        if not self.universe:
            raise ValueError("empty action universe")
        for name in ("hard_safe", "jinn", "beast"):
            if not getattr(self, name).issubset(self.universe):
                raise ValueError(f"{name} contains actions outside universe")

    def allowed(self, operation: ConstitutionOp | str) -> FrozenSet[str]:
        op = ConstitutionOp(operation)
        candidates = {
            ConstitutionOp.BEAST_ONLY: self.beast,
            ConstitutionOp.JINN_ONLY: self.jinn,
            ConstitutionOp.MEET: self.jinn & self.beast,
            ConstitutionOp.PARALLAX: self.jinn | self.beast,
        }[op]
        return frozenset(candidates & self.hard_safe)


@dataclass(frozen=True)
class LaneHead:
    """Three-lane rational linear head; reject has lowest tie priority.

    The score order matches AdditiveCandidateHead: train, hold, reject.
    Fractions certify the supplied coefficients, not GPU floating-point effects.
    """

    train: tuple[Fraction, ...]
    hold: tuple[Fraction, ...]
    reject: tuple[Fraction, ...]
    biases: tuple[Fraction, Fraction, Fraction]

    def __post_init__(self) -> None:
        if len(self.train) != len(self.hold) or len(self.train) != len(self.reject):
            raise ValueError("lane dimension mismatch")
        if not self.train:
            raise ValueError("empty feature domain")
        if len(self.biases) != 3:
            raise ValueError("three biases required")
        if not all(isinstance(v, (int, Fraction)) for lane in
                   (self.train, self.hold, self.reject, self.biases) for v in lane):
            raise TypeError("exact certification requires integer or rational coefficients")

    def winner(self, bits: tuple[int, ...]) -> str:
        if len(bits) != len(self.train) or any(v not in (0, 1) for v in bits):
            raise ValueError("expected binary tag vector")
        scores = [b + sum(w * x for w, x in zip(ws, bits))
                  for b, ws in zip(self.biases, (self.train, self.hold, self.reject))]
        return ("train", "hold", "reject")[max(range(3), key=lambda k: scores[k])]

    def max_competing_margin(self, lane: str, critical_index: int) -> Fraction:
        """Exact maximum of score(lane) - score(reject) given x_j = 1."""
        if lane not in ("train", "hold"):
            raise ValueError("lane must be train or hold")
        if not 0 <= critical_index < len(self.train):
            raise IndexError("critical tag index out of range")
        index = 0 if lane == "train" else 1
        competitor = (self.train, self.hold)[index]
        delta = tuple(a - b for a, b in zip(competitor, self.reject))
        return (self.biases[index] - self.biases[2] + delta[critical_index]
                + sum(max(Fraction(0), d) for i, d in enumerate(delta)
                      if i != critical_index))

    def certifies_critical_rejection(self, critical_indices: Iterable[int]) -> bool:
        indices = tuple(critical_indices)
        if not indices:
            raise ValueError("critical indices required")
        return all(self.max_competing_margin(lane, j) < 0
                   for j in indices for lane in ("train", "hold"))


def robust_tag_certificate(
    description: str,
    reader: Callable[[str], Iterable[FrozenSet[str]]],
    forbidden_tags: FrozenSet[str],
) -> bool:
    """Authorize only if every independently supplied possible tag set is safe.

    Empty evidence fails closed. Coverage of the true tags by the reader is a
    separate statistical or semantic assumption, not proved here.
    """
    interpretations = tuple(reader(description))
    return bool(interpretations) and all(not (tags & forbidden_tags)
                                         for tags in interpretations)


def persona_distribution(
    baseline: Mapping[str, float],
    utility: Mapping[str, float],
    allowed: FrozenSet[str],
    strength: float,
) -> dict[str, float]:
    """KL-regularized preference distribution on certified actions."""
    if not allowed:
        raise ValueError("empty certified action set: abstain upstream")
    if not isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and nonnegative")
    actions = sorted(allowed)
    for a in actions:
        if a not in baseline or a not in utility:
            raise ValueError(f"missing score for {a}")
        if not (0 < baseline[a] <= 1 and isfinite(baseline[a]) and isfinite(utility[a])):
            raise ValueError(f"invalid score for {a}")
    logits = [log(baseline[a]) + strength * utility[a] for a in actions]
    if not all(isfinite(v) for v in logits):
        raise ValueError("non-finite logit")
    peak = max(logits)
    weights = [exp(v - peak) for v in logits]
    total = sum(weights)
    return {a: weight / total for a, weight in zip(actions, weights)}


@dataclass(frozen=True)
class Repair:
    action: str | None
    abstained: bool
    cost: float | None


def minimal_repair(
    proposed: str,
    certified: FrozenSet[str],
    distance: Callable[[str, str], float],
) -> Repair:
    """Deterministic nearest safe action; abstain if no action is certified."""
    if not certified:
        return Repair(action=None, abstained=True, cost=None)
    if proposed in certified:
        return Repair(action=proposed, abstained=False, cost=0.0)
    choices = [(distance(proposed, candidate), candidate) for candidate in certified]
    if not all(isfinite(cost) and cost >= 0 for cost, _ in choices):
        raise ValueError("distance must be finite and nonnegative")
    cost, action = min(choices)
    return Repair(action=action, abstained=False, cost=cost)


def viability_kernel(
    safe_states: FrozenSet[str],
    actions: Mapping[str, FrozenSet[str]],
    successors: Mapping[tuple[str, str], FrozenSet[str]],
) -> FrozenSet[str]:
    """Greatest finite-state region permitting an indefinitely safe action.

    A transition's nonempty successor set must be wholly within the region.
    Missing/empty successor sets are treated as unsupported, never safe.
    This assumes all physically possible successors are represented.
    """
    region = safe_states
    while True:
        updated = frozenset(s for s in region if any(
            successors.get((s, a), frozenset()) and
            successors[(s, a)].issubset(region)
            for a in actions.get(s, frozenset())
        ))
        if updated == region:
            return region
        region = updated


def shield_actions(
    state: str,
    region: FrozenSet[str],
    actions: Mapping[str, FrozenSet[str]],
    successors: Mapping[tuple[str, str], FrozenSet[str]],
) -> FrozenSet[str]:
    """Allow exactly the actions whose possible successors stay in the region."""
    if state not in region:
        return frozenset()
    return frozenset(a for a in actions.get(state, frozenset())
                     if successors.get((state, a), frozenset()) and
                     successors[(state, a)].issubset(region))
