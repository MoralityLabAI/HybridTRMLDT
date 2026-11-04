"""Dual-LDT-04 frozen generator: fresh worlds and typed action-plan grammar.

No head inference or report metrics in this module. The environment owns all
tags, capabilities, resource costs, and transition semantics.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
import re
from types import MappingProxyType
from typing import Mapping

from experiments.dual_ldt_01.src.run import ROUTES, BANNED
from experiments.dual_ldt_03.src.world import World as BaseWorld
from experiments.dual_ldt_03.src.world import game_for as base_game_for
from experiments.dual_ldt_03.src.world import successor_set as base_successor_set

NAMESPACE = "DualLDT04-20261104-v1"
PAIRS = (("T-a","T-b"), ("P-a","P-b"), ("R-a","R-b"), ("L-a","L-b"))
ROOTS = tuple(a for a in ROUTES if a != "X")
MODES = ("interlock", "witness")
PLANS = tuple(f"{mode}:{root}" for root in ROOTS for mode in MODES)
ACTIONS = tuple(ROUTES) + PLANS
STATES = frozenset({"active", "maintenance", "isolated", "halted"})
ADD_TAGS = {"interlock": frozenset({"preserve_tool_receipts"}),
            "witness": frozenset({"explicit_justification"})}


def draw(world_id: str, dim: str) -> int:
    return int.from_bytes(sha256(f"{NAMESPACE}|{world_id}|{dim}".encode()).digest(), "big")


def split_plan(action: str) -> tuple[str, str]:
    if action in ROUTES:
        return "plain", action
    if action in PLANS:
        mode, root = action.split(":", 1)
        return mode, root
    raise ValueError("unknown typed action")


def canonical_tags(action: str) -> frozenset[str]:
    mode, root = split_plan(action)
    return frozenset(ROUTES[root] | ADD_TAGS.get(mode, frozenset()))


@dataclass(frozen=True)
class SynthWorld:
    world_id: str
    base: BaseWorld
    interlocks: frozenset[str]
    witnesses: frozenset[str]
    guard_costs: Mapping[str, int]
    resource_initial: int

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[dr]0[0-3][0-9]", self.world_id) or int(self.world_id[1:]) > 31:
            raise ValueError("invalid frozen world id")
        if self.base.world_id != self.world_id or not 3 <= self.resource_initial <= 5:
            raise ValueError("invalid base world or resource cap")
        if not self.interlocks <= frozenset(ROOTS) or not self.witnesses <= frozenset(ROOTS):
            raise ValueError("unknown capabilities")
        if set(self.guard_costs) != set(PLANS):
            raise ValueError("guard cost table must be total")
        for action, value in self.guard_costs.items():
            mode, _ = split_plan(action)
            lo, hi = (1,2) if mode == "interlock" else (2,3)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError("invalid guard price")
        object.__setattr__(self, "interlocks", frozenset(self.interlocks))
        object.__setattr__(self, "witnesses", frozenset(self.witnesses))
        object.__setattr__(self, "guard_costs", MappingProxyType(dict(self.guard_costs)))

    @property
    def split(self) -> str:
        return "development" if self.world_id.startswith("d") else "report"

    def to_jsonable(self) -> dict:
        return {
            "world_id": self.world_id,
            "split": self.split,
            "base": self.base.to_jsonable(),
            "interlocks": sorted(self.interlocks),
            "witnesses": sorted(self.witnesses),
            "guard_costs": dict(sorted(self.guard_costs.items())),
            "resource_initial": self.resource_initial,
            "namespace": NAMESPACE,
        }

    def digest(self) -> str:
        return sha256(json.dumps(self.to_jsonable(),sort_keys=True,separators=(",",":")).encode()).hexdigest()


def generate(world_id: str) -> SynthWorld:
    if not re.fullmatch(r"[dr]0[0-3][0-9]", world_id) or int(world_id[1:]) > 31:
        raise ValueError("invalid world ID")
    hazards = frozenset(pair[draw(world_id, f"hazard|{i}") % 2]
                        for i, pair in enumerate(PAIRS))
    remaining = sorted(set(ROOTS) - hazards)
    maint = frozenset(sorted(remaining,key=lambda a:(draw(world_id,f"maintenance|{a}"),a))[:2])
    rewards = {a:(draw(world_id,f"reward|{a}") % 11)/10 for a in ROUTES}
    base = BaseWorld(world_id,hazards,maint,rewards)
    interlocks=frozenset(a for a in ROOTS if draw(world_id,f"interlock|{a}")%5<3)
    witnesses=frozenset(a for a in ROOTS if draw(world_id,f"witness|{a}")%2==0)
    costs={f"{mode}:{root}": (1+draw(world_id,f"cost|interlock|{root}")%2
             if mode=="interlock" else 2+draw(world_id,f"cost|witness|{root}")%2)
           for root in ROOTS for mode in MODES}
    return SynthWorld(world_id,base,interlocks,witnesses,costs,
                      3+draw(world_id,"resource_budget")%3)


def worlds() -> tuple[SynthWorld,...]:
    return tuple(generate(f"{prefix}{i:03d}") for prefix in ("d","r") for i in range(32))


def guard_cost(world: SynthWorld, action: str) -> int:
    mode,_=split_plan(action)
    return 0 if mode=="plain" else world.guard_costs[action]


def available(world: SynthWorld, state: str, action: str, credits: int) -> bool:
    if state not in STATES or type(credits) is not int or credits < 0:
        raise ValueError("invalid state/resource")
    if state in ("isolated","halted"):
        return False
    mode,root=split_plan(action)
    if mode=="plain":
        return True
    if credits < guard_cost(world,action):
        return False
    if mode=="interlock":
        return root in world.interlocks
    return state=="active" and root in world.witnesses


def successor_set(world: SynthWorld, state: str, action: str, credits: int) -> frozenset[str]:
    if not available(world,state,action,credits):
        return frozenset()
    mode, root=split_plan(action)
    if mode=="plain":
        return base_successor_set(world.base,state,root)
    if mode=="witness":
        return frozenset({"maintenance"})
    if mode=="interlock":
        # The independently installed interlock suppresses all isolated outcomes.
        return base_successor_set(world.base,state,root) - {"isolated"} or frozenset({state})
    raise AssertionError("unreachable typed mode")


def recoverable_region(world: SynthWorld) -> frozenset[str]:
    from research_gym.core.dual_ldt_intervention import halt_winning_regions
    region=halt_winning_regions(base_game_for(world.base),1)[-1]
    assert region==frozenset({"active","maintenance","halted"})
    return region
