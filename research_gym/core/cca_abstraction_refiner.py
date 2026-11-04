"""CCA-07: finite counterexample-guided refinement of action abstractions.

The host supplies a complete state space, sound permissions, and a bank of
predicates with environment-owned truth values. The model cannot author facts.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Hashable, Mapping

from .cca_compositional import maximal_sound_actions

State = Hashable
Action = Hashable


@dataclass(frozen=True)
class PredicateBank:
    states: frozenset[State]
    values: Mapping[str, Mapping[State, bool]]

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("empty host-owned state space")
        frozen = {}
        for name, observations in self.values.items():
            if not name or set(observations) != self.states:
                raise ValueError("predicate needs a name and a total truth table")
            if any(type(x) is not bool for x in observations.values()):
                raise ValueError("predicate values must be booleans")
            frozen[name] = MappingProxyType(dict(observations))
        object.__setattr__(self, "values", MappingProxyType(frozen))

    def class_of(self, target: State, predicates: tuple[str, ...]) -> frozenset[State]:
        if target not in self.states:
            raise ValueError("unknown target")
        if len(predicates) != len(set(predicates)):
            raise ValueError("duplicate predicates")
        if not set(predicates) <= set(self.values):
            raise ValueError("unknown predicate")
        return frozenset(x for x in self.states if all(
            self.values[name][x] == self.values[name][target] for name in predicates
        ))


@dataclass(frozen=True)
class RefinementStep:
    predicate: str
    witness: State
    before: frozenset[State]
    after: frozenset[State]
    removed_counterexamples: frozenset[State]


@dataclass(frozen=True)
class RefinementReceipt:
    target: State
    proposed_action: Action
    selected_predicates: tuple[str, ...]
    possible_states: frozenset[State]
    robust_allowed: frozenset[Action]
    steps: tuple[RefinementStep, ...]
    status: str

    @property
    def authorized(self) -> bool:
        return self.status == "authorized"


def refine_to_authorize(
    bank: PredicateBank,
    allowed_by_state: Mapping[State, frozenset[Action]],
    target: State,
    desired: Action,
    *,
    initial_predicates: tuple[str, ...] = (),
) -> RefinementReceipt:
    """Refine until the safe desired action is authorized or no split exists.

    Each chosen predicate removes at least one counterexample, preserving the
    known true state. Choice is deterministic by number excluded, then name.
    The predicate bank is host-owned; caller-supplied labels have no authority.
    """
    if set(allowed_by_state) != bank.states:
        raise ValueError("incomplete or extraneous environment permission table")
    if any(not isinstance(acts, frozenset) for acts in allowed_by_state.values()):
        raise TypeError("permission tables must use frozensets")
    current = bank.class_of(target, initial_predicates)
    chosen = list(initial_predicates)
    steps: list[RefinementStep] = []
    if desired not in allowed_by_state[target]:
        return RefinementReceipt(target, desired, tuple(chosen), current,
                                 maximal_sound_actions(current, allowed_by_state),
                                 (), "denied_at_true_state")
    while True:
        authorized = maximal_sound_actions(current, allowed_by_state)
        if desired in authorized:
            return RefinementReceipt(target, desired, tuple(chosen), current,
                                     authorized, tuple(steps), "authorized")
        negative = frozenset(s for s in current if desired not in allowed_by_state[s])
        options = []
        for name in sorted(set(bank.values) - set(chosen)):
            kept = frozenset(s for s in current if
                             bank.values[name][s] == bank.values[name][target])
            removed_bad = negative - kept
            if removed_bad:
                options.append((-len(removed_bad), -len(current - kept), name,
                                kept, removed_bad))
        if not options:
            return RefinementReceipt(target, desired, tuple(chosen), current,
                                     authorized, tuple(steps), "unresolved_predicates")
        _, _, name, kept, removed_bad = min(options)
        witness = min(removed_bad, key=repr)
        steps.append(RefinementStep(name, witness, current, kept, removed_bad))
        chosen.append(name)
        current = kept
