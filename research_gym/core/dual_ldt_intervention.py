"""Dual-LDT-02: preserve future host intervention under transition uncertainty.

This kernel checks a finite game supplied by an independent environment owner.
It establishes bounded reachability of states where direct HALT is available,
not an unconditional property of real-world AI systems or their overseers.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .dual_ldt import BeastSnapshot


@dataclass(frozen=True)
class InterventionGame:
    states: frozenset[str]
    halt_ready: frozenset[str]
    host_actions: Mapping[str, frozenset[str]]
    agent_actions: Mapping[str, frozenset[str]]
    successors: Mapping[tuple[str, str], frozenset[str]]

    def __post_init__(self) -> None:
        if not self.states or not self.halt_ready <= self.states:
            raise ValueError("finite states and halt-ready subset required")
        for label, actions in (("host", self.host_actions), ("agent", self.agent_actions)):
            if not set(actions) <= self.states:
                raise ValueError(f"{label} actions contain an unknown state")
            if any(not isinstance(v, frozenset) for v in actions.values()):
                raise TypeError(f"{label} action options must be frozen sets")
        for (state, action), dest in self.successors.items():
            if state not in self.states or not action:
                raise ValueError("transition names an unknown state/action")
            if not isinstance(dest, frozenset) or not dest <= self.states:
                raise ValueError("transition successors must belong to state space")
        object.__setattr__(self, "states", frozenset(self.states))
        object.__setattr__(self, "halt_ready", frozenset(self.halt_ready))
        object.__setattr__(self, "host_actions", MappingProxyType(dict(self.host_actions)))
        object.__setattr__(self, "agent_actions", MappingProxyType(dict(self.agent_actions)))
        object.__setattr__(self, "successors", MappingProxyType(dict(self.successors)))


def halt_winning_regions(game: InterventionGame, horizon: int) -> tuple[frozenset[str], ...]:
    """States from which the host can force direct-HALT availability in <= h steps.

    The outer existential chooses a host action at each observed state.
    The inner universal requires all represented disturbance successors to be
    winning. Missing and empty transition sets never prove progress.
    """
    if type(horizon) is not int or horizon < 0:
        raise ValueError("nonnegative integer horizon required")
    regions = [game.halt_ready]
    for _ in range(horizon):
        earlier = regions[-1]
        new = frozenset(
            s for s in game.states if s in earlier or any(
                game.successors.get((s, action), frozenset())
                and game.successors[(s, action)].issubset(earlier)
                for action in game.host_actions.get(s, frozenset())
            )
        )
        regions.append(new)
    return tuple(regions)


def intervention_preserving_actions(
    game: InterventionGame,
    snapshot: BeastSnapshot,
    state: str,
    horizon: int,
) -> frozenset[str]:
    """Apply Beast authorization, then require every successor to be corrigible.

    The returned actions preserve a host recovery path of <= horizon steps
    after the proposed action; the host must still exercise that capability.
    """
    if state not in game.states:
        raise ValueError("unknown current state")
    regions = halt_winning_regions(game, horizon)
    if state not in regions[-1] or snapshot.halted:
        return frozenset()
    choices = game.agent_actions.get(state, frozenset()) & snapshot.authorized()
    return frozenset(a for a in choices if
                     game.successors.get((state, a), frozenset()) and
                     game.successors[(state, a)].issubset(regions[-1]))
