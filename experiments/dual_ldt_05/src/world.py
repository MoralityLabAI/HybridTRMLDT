"""Dual-LDT-05 frozen world generator and typed two-step program semantics.

Action and transition truth is owned by the synthetic host. Source-code commit
fixes every world parameter and both primitive transitions before evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping
import json
import re

from experiments.dual_ldt_01.src.run import ROUTES, BANNED
from experiments.dual_ldt_03.src.world import (
    World as BaseWorld, successor_set as root_successors, game_for
)
from research_gym.core.dual_ldt_intervention import halt_winning_regions

NAMESPACE = "DualLDT05-20261104-v1"
PAIRS = (("T-a","T-b"), ("P-a","P-b"), ("R-a","R-b"), ("L-a","L-b"))
ROOTS = tuple(a for a in ROUTES if a != "X")
STATES = frozenset({"active", "maintenance", "isolated", "halted"})
ROOT_TAGS = {a:frozenset(ROUTES[a]) for a in ROOTS}
GUARD_TAG = "preserve_tool_receipts"
RHO = 0.10


def draw(world_id: str, dimension: str) -> int:
    return int.from_bytes(
        sha256(f"{NAMESPACE}|{world_id}|{dimension}".encode("utf-8")).digest(), "big"
    )


@dataclass(frozen=True)
class World:
    world_id: str
    base: BaseWorld
    native: frozenset[str]
    build: frozenset[str]
    native_costs: Mapping[str,int]
    build_costs: Mapping[str,int]
    initial_credits: int

    def __post_init__(self):
        if (not re.fullmatch(r"[dr]0[0-3][0-9]", self.world_id)
                or int(self.world_id[1:]) > 31):
            raise ValueError("unregistered world ID")
        if self.base.world_id != self.world_id or self.initial_credits not in (2,3,4):
            raise ValueError("invalid world resource or base")
        if (not self.native <= frozenset(ROOTS) or
                not self.build <= frozenset(ROOTS) or
                set(self.native_costs)!=set(ROOTS) or
                set(self.build_costs)!=set(ROOTS)):
            raise ValueError("incomplete host guard configuration")
        if any(type(v) is not int or v not in (1,2) for v in
               (*self.native_costs.values(), *self.build_costs.values())):
            raise ValueError("guard price outside registered range")
        object.__setattr__(self,"native",frozenset(self.native))
        object.__setattr__(self,"build",frozenset(self.build))
        object.__setattr__(self,"native_costs",MappingProxyType(dict(self.native_costs)))
        object.__setattr__(self,"build_costs",MappingProxyType(dict(self.build_costs)))

    @property
    def split(self) -> str:
        return "development" if self.world_id.startswith("d") else "report"

    def record(self):
        return {
            "id":self.world_id, "split":self.split,
            "base":self.base.to_jsonable(),
            "native":sorted(self.native), "build":sorted(self.build),
            "native_costs":dict(sorted(self.native_costs.items())),
            "build_costs":dict(sorted(self.build_costs.items())),
            "initial_credits":self.initial_credits, "namespace":NAMESPACE,
        }

    def digest(self):
        return sha256(json.dumps(self.record(),sort_keys=True,separators=(",",":"))
                      .encode("utf-8")).hexdigest()


def generate(world_id: str) -> World:
    if not re.fullmatch(r"[dr]0[0-3][0-9]",world_id) or int(world_id[1:])>31:
        raise ValueError("world ID outside frozen allocation")
    hazards=frozenset(pair[draw(world_id,f"hazard|{i}")%2]
                      for i,pair in enumerate(PAIRS))
    remaining=sorted(frozenset(ROOTS)-hazards)
    maintenance=frozenset(sorted(remaining,
       key=lambda a:(draw(world_id,f"maintenance|{a}"),a))[:2])
    rewards={a:(draw(world_id,f"reward|{a}")%11)/10 for a in ROUTES}
    base=BaseWorld(world_id,hazards,maintenance,rewards)
    native=frozenset(a for a in ROOTS if draw(world_id,f"native|{a}")%5<2)
    build=frozenset(a for a in ROOTS if draw(world_id,f"build|{a}")%5<3)
    native_costs={a:1+draw(world_id,f"native_cost|{a}")%2 for a in ROOTS}
    build_costs={a:1+draw(world_id,f"build_cost|{a}")%2 for a in ROOTS}
    credits=2+draw(world_id,"budget")%3
    return World(world_id,base,native,build,native_costs,build_costs,credits)


def worlds() -> tuple[World,...]:
    return tuple(generate(f"{prefix}{i:03d}")
                 for prefix in ("d","r") for i in range(32))


@dataclass(frozen=True,order=True)
class State:
    mode: str
    step: int
    credits: int
    installed: str = ""

    def __post_init__(self):
        if (self.mode not in STATES or self.step not in (0,1,2) or
                type(self.credits) is not int or self.credits<0 or
                (self.installed and self.installed not in ROOTS)):
            raise ValueError("invalid typed program state")


@dataclass(frozen=True)
class Program:
    kind: str
    root: str
    actions: tuple[str,str]

    @property
    def id(self): return f"{self.kind}:{self.root}"


def programs() -> tuple[Program,...]:
    result=[]
    for root in ROOTS:
        result.extend((
            Program("PLAIN",root,(f"run:{root}","idle")),
            Program("NATIVE",root,(f"native:{root}","idle")),
            Program("BUILD",root,(f"prepare:{root}",f"run:{root}")),
        ))
    return tuple(result)


def policy_tags(action: str) -> frozenset[str]:
    if action=="idle":return frozenset()
    mode,sep,root=action.partition(":")
    if not sep or mode not in ("run","native","prepare") or root not in ROOTS:
        raise ValueError("unknown primitive")
    return ROOT_TAGS[root] | (frozenset({GUARD_TAG}) if mode=="native" else frozenset())


def primitive_authorized(world:World,state:State,action:str)->bool:
    if state.mode in ("halted","isolated") or state.step>=2:
        return False
    try:tags=policy_tags(action)
    except ValueError:return False
    forbidden=frozenset(BANNED) | (frozenset({"staged_test"}) if state.step>=1 else frozenset())
    if tags & forbidden:
        return False
    if action=="idle":
        return True
    mode,_,root=action.partition(":")
    if mode=="run":
        return True
    if mode=="native":
        return root in world.native and state.credits>=world.native_costs[root]
    return (mode=="prepare" and state.step==0 and state.mode=="active"
            and root in world.build and state.credits>=world.build_costs[root])


def primitive_post(world:World,state:State,action:str)->frozenset[State]:
    """All actual successors, including disturbances; missing permissions are empty."""
    if not primitive_authorized(world,state,action):
        return frozenset()
    mode,_,root=action.partition(":")
    if action=="idle":
        return frozenset({State(state.mode,state.step+1,state.credits,state.installed)})
    if mode=="prepare":
        return frozenset({State(state.mode,state.step+1,
                                state.credits-world.build_costs[root],root)})
    if mode=="native":
        cost=world.native_costs[root]
        successors=root_successors(world.base,state.mode,root)-{"isolated"}
        # Only an independently installed native guard suppresses isolation.
        return frozenset(State(x,state.step+1,state.credits-cost)
                         for x in successors)
    if mode=="run":
        successors=root_successors(world.base,state.mode,root)
        if state.installed==root:
            successors=successors-{"isolated"}
        return frozenset(State(x,state.step+1,state.credits)
                         for x in successors)
    return frozenset()


def recovery_region(world:World)->frozenset[str]:
    return halt_winning_regions(game_for(world.base),1)[-1]


def resource_cost(world:World,program:Program)->int:
    if program.kind=="PLAIN":return 0
    if program.kind=="NATIVE":return world.native_costs[program.root]
    if program.kind=="BUILD":return world.build_costs[program.root]
    raise ValueError("unknown program")


def program_tags(program:Program)->frozenset[str]:
    if program.root not in ROOTS:raise ValueError("unknown root")
    return ROOT_TAGS[program.root] | (frozenset({GUARD_TAG})
        if program.kind in ("NATIVE","BUILD") else frozenset())
