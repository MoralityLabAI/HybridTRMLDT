"""Typed LDT defender bridge for native HybridTRMLDT and future neural heads."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from research_gym.core.dual_ldt import BeastSnapshot, JinnHead, sort_with_jinn
from .world import MoralControlWorld
from .eval import snapshot

@dataclass
class TypedLDTDefender:
    """Native Beast+Jinn preference sorter; host authorization stays external."""
    head:JinnHead
    def choose(self,world:MoralControlWorld)->str|None:
        bound=snapshot(world)
        decision=sort_with_jinn(bound,self.head)
        if decision.action is not None and decision.action not in bound.authorized():
            raise AssertionError("native typed LDT crossed host authority")
        return decision.action


def execute_checked(world:MoralControlWorld,requested_action:str,
                    *,defender:TypedLDTDefender|None=None)->dict:
    """A red proposal enters as untrusted data; only host receipts execute."""
    s=snapshot(world)
    if requested_action not in s.actions:raise ValueError("unknown red action")
    admitted=s.receipt(requested_action)
    authorized=s.verify_execution(admitted,lambda epoch,a:
        epoch==world.authority().epoch and world.allow(a))
    fallback=None if authorized else (
        defender.choose(world) if defender is not None else world.best_host_action())
    selected=requested_action if authorized else fallback
    if selected is None:
        return {"requested":requested_action,"executed":None,"authorized":False,"blocked":True}
    signed=s.receipt(selected)
    if not s.verify_execution(signed,lambda epoch,a:
        epoch==world.authority().epoch and world.allow(a)):
        raise AssertionError("host declined LDT-selected action")
    world.step(selected)
    return {"requested":requested_action,"executed":selected,
            "authorized":authorized,"blocked":not authorized,
            "epoch":signed.epoch,"digest":signed.snapshot_digest}
