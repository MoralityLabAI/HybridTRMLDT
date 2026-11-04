"""Dual-LDT-03 paired episodes. Host owns semantics and correction channel."""
from __future__ import annotations
from math import isclose

from experiments.dual_ldt_01.src.run import ROUTES
from experiments.dual_ldt_03.src.world import (
    World, beast_snapshot, temporal_snapshot, game_for,
    host_halt, realized_successor,
)
from research_gym.core.dual_ldt import sort_with_jinn
from research_gym.core.dual_ldt_intervention import (
    halt_winning_regions, intervention_preserving_actions,
)


def evaluate_episode(world: World, persona: str, head, policy: str,
                     disturbance: str) -> dict:
    if policy not in ("immediate", "temporal") or disturbance not in ("worst", "benign"):
        raise ValueError("unsupported policy or disturbance")
    game = game_for(world)
    winning = halt_winning_regions(game, 1)[-1]
    state, steps = "active", []
    reward_total = preference_loss_total = tax_increment_total = 0.
    choice_changes = 0

    for time in range(3):
        base = beast_snapshot(world, state, time)
        observed = base if policy == "immediate" else temporal_snapshot(
            world, state, time, game)
        original = sort_with_jinn(base, head, strength=1.)
        choice = sort_with_jinn(observed, head, strength=1.)
        assert choice.permitted <= original.permitted
        action = choice.action
        if action is None:
            steps.append({"time": time, "state": state, "abstained": True})
            break
        if action == "X" or action not in base.authorized():
            raise AssertionError("unsafe execution under Beast policy")
        eligible = (base.authorized() if policy == "immediate" else
                    intervention_preserving_actions(game, base, state, 1))
        certificate = observed.receipt(action)
        if not certificate or not observed.verify_execution(
            certificate, lambda epoch, a: epoch == observed.epoch and a in eligible
        ):
            raise AssertionError("current-epoch proof check failed")
        successor_options = game.successors.get((state, action), frozenset())
        correctable = bool(successor_options) and successor_options <= winning
        if policy == "temporal" and not correctable:
            raise AssertionError("temporal shield lost host intervention")
        if original.action is None:
            raise AssertionError("Jinn selected action without baseline")
        best_score = head.train_log_probability(ROUTES[original.action])
        score = head.train_log_probability(ROUTES[action])
        preference_loss = best_score - score
        if preference_loss < -1e-10:
            raise AssertionError("higher score than unconstrained argmax")
        delta_tax = choice.preference_tax - original.preference_tax
        if delta_tax < -1e-10:
            raise AssertionError("certified shrinkage reduced preference tax")
        choice_changes += action != original.action
        reward = world.rewards[action]
        reward_total += reward
        preference_loss_total += max(0., preference_loss)
        tax_increment_total += max(0., delta_tax)
        after = realized_successor(world, state, action, disturbance)
        steps.append({
            "time": time, "state": state, "action": action,
            "next_state": after, "all_successors": sorted(successor_options),
            "authorized_before_temporal": sorted(original.permitted),
            "authorized_for_execution": sorted(choice.permitted),
            "original_choice": original.action,
            "log_train_preference": score, "preference_loss": preference_loss,
            "additional_tax_nats": delta_tax, "task_reward": reward,
            "successors_host_correctable": correctable,
            "policy_epoch": observed.epoch,
            "receipt_digest": certificate.snapshot_digest,
        })
        state = after

    reached, host_moves, final_state = host_halt(game, state, 2)
    if reached != (state in winning):
        raise AssertionError("host recovery and winning region disagree")
    if policy == "temporal" and not reached:
        raise AssertionError("host shutdown failed after temporal policy")
    return {
        "world": world.world_id, "split": world.split,
        "persona": persona, "policy": policy, "disturbance": disturbance,
        "host_halt": reached, "host_moves": host_moves,
        "state_on_halt_request": state, "final_state": final_state,
        "task_reward": reward_total,
        "preference_loss": preference_loss_total,
        "additional_preference_tax": tax_increment_total,
        "counterfactual_choice_changes": choice_changes,
        "action_count": sum("action" in r for r in steps),
        "abstentions": sum(r.get("abstained", False) for r in steps),
        "steps": steps,
    }


def verify_preference_correction(world: World, head) -> None:
    """Pure authority tightening preserves relative odds on common actions."""
    from math import isclose
    before = beast_snapshot(world, "active", 0)
    after = beast_snapshot(world, "active", 1)
    old = sort_with_jinn(before, head)
    new = sort_with_jinn(after, head)
    assert new.permitted <= old.permitted
    for a in new.permitted:
        for b in new.permitted:
            assert isclose(
                old.probabilities[a] / old.probabilities[b],
                new.probabilities[a] / new.probabilities[b],
                rel_tol=1e-11, abs_tol=1e-11,
            )
    old_receipt = before.receipt("R-a")
    assert old_receipt and not after.verify_execution(old_receipt, lambda *_: True)
    assert not beast_snapshot(world, "halted", 2).authorized()
