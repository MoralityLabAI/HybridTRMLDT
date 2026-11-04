"""Dual-LDT-04 independent plan checker and current-policy authorization.

The host re-derives all effects from the frozen world, resource count and policy
epoch. A checkable receipt is NOT an identity signature; the host must retain
control of the actual executor and the world authority source.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import exp
from typing import Mapping

from experiments.dual_ldt_01.src.run import BANNED
from experiments.dual_ldt_04.src.world import (
    ACTIONS, PLANS, STATES, SynthWorld, available, canonical_tags,
    guard_cost, recoverable_region, successor_set,
)
from research_gym.core.dual_ldt import ActionEvidence, BeastSnapshot


@dataclass(frozen=True)
class PlanProof:
    world_digest: str
    epoch: str
    snapshot_digest: str
    state: str
    time: int
    credits: int
    action: str
    cost: int
    successors: tuple[str,...]
    tags: tuple[str,...]


def snapshot_for(
    world: SynthWorld, state: str, time: int, credits: int, arm: str
) -> BeastSnapshot:
    if state not in STATES or not 0 <= time < 3 or not 0 <= credits <= world.resource_initial:
        raise ValueError("invalid state, time or budget")
    if arm not in ("immediate", "temporal", "synth"):
        raise ValueError("unrecognized controller")
    forbidden = frozenset(BANNED | ({"staged_test"} if time >= 1 else set()))
    winning = recoverable_region(world)
    evidence = {}
    for action in ACTIONS:
        tags=canonical_tags(action)
        successors=successor_set(world,state,action,credits)
        if action in PLANS and arm != "synth":
            evidence[action]=ActionEvidence(False,True,(tags,))
        elif not available(world,state,action,credits):
            evidence[action]=ActionEvidence(False,False,(tags,))
        elif arm != "immediate" and (not successors or not successors <= winning):
            evidence[action]=ActionEvidence(False,True,(tags,))
        else:
            evidence[action]=ActionEvidence(True,False,(tags,))
    return BeastSnapshot(
        f"{world.world_id}|{world.digest()[:16]}|{state}|t{time}|credits{credits}|"
        f"policy{1 if time==0 else 2}|{arm}",
        ACTIONS,forbidden,evidence,halted=state=="halted",
    )


def reference_weights(world: SynthWorld, rho: float=0.10) -> dict[str,float]:
    return {a:exp(-rho*guard_cost(world,a)) for a in ACTIONS}


def issue_plan(
    world:SynthWorld,state:str,time:int,credits:int,arm:str,action:str,
) -> PlanProof | None:
    try:
        snapshot=snapshot_for(world,state,time,credits,arm)
        if snapshot.status(action)!="authorized":
            return None
        effects=successor_set(world,state,action,credits)
        if not effects:
            return None
        return PlanProof(
            world.digest(),snapshot.epoch,snapshot.digest(),state,time,credits,
            action,guard_cost(world,action),tuple(sorted(effects)),
            tuple(sorted(canonical_tags(action))),
        )
    except (ValueError,TypeError,KeyError):
        return None


def check_plan(
    world:SynthWorld,state:str,time:int,credits:int,arm:str,
    proof:PlanProof | None,
) -> bool:
    if not isinstance(proof,PlanProof):
        return False
    try:
        snapshot=snapshot_for(world,state,time,credits,arm)
        if snapshot.status(proof.action)!="authorized":
            return False
        effects=successor_set(world,state,proof.action,credits)
        if not effects or (arm!="immediate" and not effects<=recoverable_region(world)):
            return False
        expected=PlanProof(
            world.digest(),snapshot.epoch,snapshot.digest(),state,time,credits,
            proof.action,guard_cost(world,proof.action),tuple(sorted(effects)),
            tuple(sorted(canonical_tags(proof.action))),
        )
        if proof != expected:
            return False
        approval=snapshot.receipt(proof.action)
        return snapshot.verify_execution(
            approval,
            lambda epoch, action: epoch==snapshot.epoch and action==proof.action
            and action in snapshot.authorized(),
        )
    except (ValueError,TypeError,KeyError):
        return False
