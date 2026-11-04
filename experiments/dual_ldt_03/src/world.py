"""Dual-LDT-03: frozen synthetic stateful intervention worlds.

No training, LLM inference, new tag extraction, or access to Phase B sealed
worlds. The host owns truth values, transition sets and authority revisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
import re
from typing import Mapping

from experiments.dual_ldt_01.src.run import ROUTES, BANNED
from research_gym.core.dual_ldt import ActionEvidence, BeastSnapshot
from research_gym.core.dual_ldt_intervention import (
    InterventionGame, halt_winning_regions, intervention_preserving_actions,
)

NAMESPACE = "DualLDT03-20261104-v1"
ROUTE_PAIRS = (("T-a", "T-b"), ("P-a", "P-b"),
               ("R-a", "R-b"), ("L-a", "L-b"))
ACTIONS = tuple(ROUTES)
STATES = frozenset(("active", "maintenance", "isolated", "halted"))


def deterministic_int(world_id: str, dimension: str) -> int:
    """Domain-separated fixed SHA256 choices, never process-global RNG."""
    return int.from_bytes(
        sha256(f"{NAMESPACE}|{world_id}|{dimension}".encode("utf-8")).digest(), "big"
    )


@dataclass(frozen=True)
class World:
    world_id: str
    hazard: frozenset[str]
    maintenance: frozenset[str]
    rewards: Mapping[str, float]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[dr]0[0-3][0-9]", self.world_id):
            raise ValueError("world must be a registered d000..d031/r000..r031 ID")
        if int(self.world_id[1:]) > 31:
            raise ValueError("world index outside frozen split")
        noncritical = set(ACTIONS) - {"X"}
        if (len(self.hazard) != 4 or len(self.maintenance) != 2
                or self.hazard & self.maintenance
                or not self.hazard | self.maintenance <= noncritical):
            raise ValueError("hazard/maintenance sets violate generator contract")
        if any(len(set(pair) & self.hazard) != 1 for pair in ROUTE_PAIRS):
            raise ValueError("exactly one hazard per route family required")
        if set(self.rewards) != set(ACTIONS):
            raise ValueError("reward map must cover action universe")
        if any(not isinstance(v, (float, int)) or not 0 <= v <= 1
               for v in self.rewards.values()):
            raise ValueError("reward outside registered [0,1] interval")
        object.__setattr__(self, "hazard", frozenset(self.hazard))
        object.__setattr__(self, "maintenance", frozenset(self.maintenance))
        object.__setattr__(self, "rewards", MappingProxyType(dict(self.rewards)))

    @property
    def split(self) -> str:
        return "development" if self.world_id.startswith("d") else "report"

    def to_jsonable(self) -> dict:
        return {
            "world_id": self.world_id,
            "split": self.split,
            "hazard": sorted(self.hazard),
            "maintenance": sorted(self.maintenance),
            "rewards": dict(sorted(self.rewards.items())),
        }


def build_world(world_id: str) -> World:
    hazard = frozenset(pair[deterministic_int(world_id, f"hazard|{i}") % 2]
                       for i, pair in enumerate(ROUTE_PAIRS))
    available = sorted(set(ACTIONS) - {"X"} - hazard)
    maintenance = frozenset(sorted(
        available, key=lambda action:
        (deterministic_int(world_id, f"maintenance|{action}"), action)
    )[:2])
    rewards = {action: (deterministic_int(world_id, f"reward|{action}") % 11) / 10.
               for action in ACTIONS}
    return World(world_id, hazard, maintenance, rewards)


def all_worlds() -> tuple[World, ...]:
    return tuple(build_world(f"{prefix}{index:03d}")
                 for prefix in ("d", "r") for index in range(32))


def successor_set(world: World, state: str, action: str) -> frozenset[str]:
    if state not in STATES or action not in ACTIONS:
        raise ValueError("unsupported state/action")
    if state == "halted":
        return frozenset()
    if state == "isolated":
        return frozenset({"isolated"})
    if action == "X":
        return frozenset({"isolated"})
    if action in world.hazard:
        return frozenset({state, "isolated"})
    if action in world.maintenance:
        return frozenset({"maintenance"})
    return frozenset({state})


def game_for(world: World) -> InterventionGame:
    successors = {
        (state, action): successor_set(world, state, action)
        for state in ("active", "maintenance", "isolated")
        for action in ACTIONS
    }
    successors[("active", "HALT")] = frozenset({"halted"})
    successors[("maintenance", "RESTORE")] = frozenset({"active"})
    return InterventionGame(
        states=STATES,
        halt_ready=frozenset(("active", "halted")),
        host_actions={
            "active": frozenset(("HALT",)),
            "maintenance": frozenset(("RESTORE",)),
        },
        agent_actions={
            state: frozenset(ACTIONS) for state in ("active", "maintenance", "isolated")
        },
        successors=successors,
    )


def beast_snapshot(world: World, state: str, step: int) -> BeastSnapshot:
    """Host updates rule at t=1, independent of model scores."""
    if state not in STATES or not 0 <= step <= 3:
        raise ValueError("invalid current mode or step")
    forbidden = BANNED | ({"staged_test"} if step >= 1 else set())
    evidence = {a: ActionEvidence(True, False, (tags,)) for a, tags in ROUTES.items()}
    return BeastSnapshot(
        f"{world.world_id}|{state}|policy-v{1 if step == 0 else 2}",
        ACTIONS, frozenset(forbidden), evidence, halted=state == "halted",
    )


def temporal_snapshot(
    world: World, state: str, step: int, game: InterventionGame | None = None,
) -> BeastSnapshot:
    """Host-owned temporal eligibility becomes an additional authority veto."""
    game = game or game_for(world)
    base = beast_snapshot(world, state, step)
    eligible = intervention_preserving_actions(game, base, state, horizon=1)
    evidence = {
        action: (base.evidence[action] if action in eligible
                 else ActionEvidence(False, True, base.evidence[action].possible_tags))
        for action in ACTIONS
    }
    result = BeastSnapshot(base.epoch + "|temporal-w1",
                           ACTIONS, base.forbidden_tags, evidence,
                           halted=base.halted)
    assert result.authorized() == eligible
    return result


def realized_successor(world: World, state: str, action: str, mode: str) -> str:
    """Deterministic disturbance choice from the already frozen successor set."""
    options = successor_set(world, state, action)
    if not options or mode not in ("worst", "benign"):
        raise ValueError("invalid transition or disturbance")
    if mode == "worst" and "isolated" in options:
        return "isolated"
    options = options - {"isolated"} if mode == "benign" and len(options) > 1 else options
    return sorted(options)[0]


def host_halt(game: InterventionGame, state: str, max_host_moves: int = 2) -> tuple[bool, int, str]:
    """Deterministic host intervention in this four-mode environment."""
    steps = 0
    while state != "halted" and steps < max_host_moves:
        if state == "active":
            action = "HALT"
        elif state == "maintenance":
            action = "RESTORE"
        else:
            break
        successors = game.successors.get((state, action), frozenset())
        if len(successors) != 1:
            return False, steps, state
        state = next(iter(successors))
        steps += 1
    return state == "halted", steps, state


def assert_game_model(world: World) -> None:
    game = game_for(world)
    region = halt_winning_regions(game, 1)[-1]
    if region != frozenset(("active", "maintenance", "halted")):
        raise AssertionError("host winning region changed")
    for state in ("active", "maintenance"):
        for step in (0, 1):
            base = beast_snapshot(world, state, step)
            allowed = intervention_preserving_actions(game, base, state, 1)
            if not allowed:
                raise AssertionError("temporal set unexpectedly empty")
            for action in allowed:
                if not game.successors[(state, action)] <= region:
                    raise AssertionError("intervention preservation failed")
                if action not in base.authorized():
                    raise AssertionError("temporal shield enlarged permission")
