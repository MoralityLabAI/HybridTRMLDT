"""Dual-LDT-05: host-owned plan issuer, verifier and two-step executor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from experiments.dual_ldt_01.src.run import BANNED
from experiments.dual_ldt_03.src.world import game_for, host_halt
from experiments.dual_ldt_05.src.world import (
    World, State, Program, programs, ROOTS, GUARD_TAG,
    ROOT_TAGS, primitive_authorized, primitive_post, policy_tags,
    recovery_region, resource_cost, program_tags, RHO,
)
from research_gym.core.dual_ldt_programs import (
    ProgramProof, issue_program, check_program,
)


def initial(world: World) -> State:
    return State("active", 0, world.initial_credits)


def _known(program: Program) -> bool:
    return isinstance(program, Program) and program in programs()


def issue(world: World, program: Program) -> ProgramProof | None:
    if not _known(program):
        return None
    recovery = recovery_region(world)
    return issue_program(
        world.digest(), initial(world), program.id, program.actions,
        authorized=lambda s,a: primitive_authorized(world,s,a),
        successor=lambda s,a: primitive_post(world,s,a),
        invariant=lambda s: s.mode in recovery,
    )


def check(world: World, program: Program, cert: ProgramProof | None) -> bool:
    if not _known(program):
        return False
    recovery = recovery_region(world)
    return check_program(
        cert, world.digest(), initial(world), program.id, program.actions,
        authorized=lambda s,a: primitive_authorized(world,s,a),
        successor=lambda s,a: primitive_post(world,s,a),
        invariant=lambda s: s.mode in recovery,
    )


@dataclass(frozen=True)
class Execution:
    program: str
    completed: bool
    halted: bool
    host_moves: int
    final_mode: str
    end_credits: int
    root_reward: float
    resource_charge: float
    task_net: float
    steps: tuple[dict, ...]
    stopped_reason: str


def execute(
    world: World, program: Program, proof: ProgramProof | None,
    *,
    disturbance: str,
    additional_revocation_at_second_step: frozenset[str] = frozenset(),
) -> Execution:
    """Revalidate the certificate AND the current authority for each effect.

    An unexpected stricter human policy between the two microsteps can revoke
    the queued second action even if the earlier full-program proof was valid.
    This leaves an already executed preparation cost but no root reward.
    """
    if disturbance not in ("worst", "benign"):
        raise ValueError("unrecognized disturbance model")
    if not check(world, program, proof):
        raise ValueError("host rejected plan certificate")
    state = initial(world)
    record = []
    completed = True
    reason = "complete"
    for primitive in program.actions:
        if (state.step >= 1 and
                policy_tags(primitive) & additional_revocation_at_second_step):
            completed = False
            reason = "revoked_after_first_step"
            break
        if not primitive_authorized(world, state, primitive):
            completed = False
            reason = "current_host_authority_rejected_step"
            break
        successors = primitive_post(world, state, primitive)
        if not successors or not successors <= set(proof.levels[state.step+1]):
            raise AssertionError("actual successor outside certified envelope")
        if not all(s.mode in recovery_region(world) for s in successors):
            raise AssertionError("execution would destroy host recovery")
        # Branch selection is for the synthetic measurement, independent of
        # the all-successor proof. Certification covers every listed branch.
        next_states = sorted(successors)
        if disturbance == "worst":
            isolated = [s for s in next_states if s.mode == "isolated"]
            chosen = isolated[0] if isolated else next_states[0]
        else:
            safe = [s for s in next_states if s.mode != "isolated"]
            chosen = safe[0] if safe else next_states[0]
        record.append({
            "primitive": primitive, "from": vars(state), "to": vars(chosen),
            "all_possible_successors": [vars(s) for s in next_states],
            "host_policy_epoch": state.step,
            "modeled_correction_issued": state.step == 1,
        })
        state = chosen
    if completed and state.step != 2:
        raise AssertionError("completed plan without two checked steps")
    # A completed script gives exactly one completed root task. A revoked
    # preparation leaves any resources already spent unreimbursed.
    spent = world.initial_credits - state.credits
    if spent < 0 or spent > world.initial_credits:
        raise AssertionError("resource accounting violated")
    gained = world.base.rewards[program.root] if completed else 0.
    charge = RHO * spent
    game = game_for(world.base)
    halted, moves, final_mode = host_halt(game, state.mode, 2)
    if not halted:
        raise AssertionError("certified program lost human HALT capability")
    return Execution(
        program.id, completed, halted, moves, final_mode, state.credits,
        gained, charge, gained-charge, tuple(record), reason,
    )
