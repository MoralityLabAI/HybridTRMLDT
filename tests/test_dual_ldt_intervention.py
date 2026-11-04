"""Host-intervention finite game: temporal corrigibility tests."""
from itertools import product
import pytest

from research_gym.core.dual_ldt import ActionEvidence, BeastSnapshot
from research_gym.core.dual_ldt_intervention import (
    InterventionGame, halt_winning_regions, intervention_preserving_actions,
)


def sample_game():
    return InterventionGame(
        states=frozenset({"active", "maintenance", "isolated", "halted"}),
        halt_ready=frozenset({"active", "halted"}),
        host_actions={"active": frozenset({"halt"}),
                      "maintenance": frozenset({"restore"})},
        agent_actions={"active": frozenset({"continue", "degrade", "disable", "risky"})},
        successors={
            ("active", "halt"): frozenset({"halted"}),
            ("maintenance", "restore"): frozenset({"active"}),
            ("active", "continue"): frozenset({"active"}),
            ("active", "degrade"): frozenset({"maintenance"}),
            ("active", "disable"): frozenset({"isolated"}),
            ("active", "risky"): frozenset({"active", "isolated"}),
        }
    )


def snapshot(*, halted=False, denied=frozenset()):
    actions = ("continue", "degrade", "disable", "risky")
    return BeastSnapshot(
        "epoch-1", actions, frozenset(),
        {a: ActionEvidence(True, a in denied, (frozenset(),)) for a in actions},
        halted=halted,
    )


def test_temporal_shield_removes_actions_that_disable_future_correction():
    game = sample_game()
    r0, r1 = halt_winning_regions(game, 1)
    assert r0 == frozenset({"active", "halted"})
    assert r1 == frozenset({"active", "maintenance", "halted"})
    allowed = intervention_preserving_actions(game, snapshot(), "active", 1)
    assert allowed == frozenset({"continue", "degrade"})
    assert "disable" in snapshot().authorized()
    assert "disable" not in allowed
    assert "risky" not in allowed
    for action in allowed:
        assert game.successors[("active", action)] <= r1


def test_strict_deadline_can_exclude_recoverable_detour():
    game = sample_game()
    assert intervention_preserving_actions(game, snapshot(), "active", 0) == frozenset({"continue"})
    assert intervention_preserving_actions(game, snapshot(), "active", 1) == frozenset({"continue", "degrade"})


def test_beast_denials_and_halt_are_authoritative():
    game = sample_game()
    assert intervention_preserving_actions(
        game, snapshot(denied=frozenset({"continue"})), "active", 1
    ) == frozenset({"degrade"})
    assert not intervention_preserving_actions(game, snapshot(halted=True), "active", 1)


def test_unknown_transition_fails_closed():
    game = InterventionGame(
        states=frozenset({"active","halted"}),
        halt_ready=frozenset({"active","halted"}),
        host_actions={},
        agent_actions={"active":frozenset({"undocumented"})},
        successors={},
    )
    snap = BeastSnapshot("v", ("undocumented",), frozenset(),
                         {"undocumented": ActionEvidence(True,False,(frozenset(),))})
    assert intervention_preserving_actions(game, snap, "active", 2) == frozenset()


def test_exhaustive_small_game_robust_successor_invariant_and_horizon_monotonicity():
    states = frozenset(("s0","s1","s2"))
    options = ("a","b")
    def bits(mask):
        return frozenset(s for i,s in enumerate(("s0","s1","s2")) if mask & (1 << i))
    snap = BeastSnapshot("v", options, frozenset(),
        {a:ActionEvidence(True,False,(frozenset(),)) for a in options})
    for target_mask, a_mask, b_mask in product(range(8), repeat=3):
        game = InterventionGame(
            states=states,
            halt_ready=bits(target_mask),
            host_actions={"s0": frozenset({"a","b"})},
            agent_actions={"s0": frozenset(options)},
            successors={("s0","a"):bits(a_mask),
                        ("s0","b"):bits(b_mask)},
        )
        regions=halt_winning_regions(game,3)
        assert all(x <= y for x,y in zip(regions, regions[1:]))
        for k in range(4):
            permitted=intervention_preserving_actions(game,snap,"s0",k)
            assert permitted <= snap.authorized()
            assert all(game.successors[("s0",action)] and
                       game.successors[("s0",action)] <= regions[k]
                       for action in permitted)


def test_invalid_state_and_unknown_horizon():
    g=sample_game()
    with pytest.raises(ValueError):
        intervention_preserving_actions(g,snapshot(),"fiction",1)
    with pytest.raises(ValueError):
        halt_winning_regions(g,-1)
    with pytest.raises(ValueError):
        InterventionGame(frozenset({"s"}), frozenset({"other"}), {}, {}, {})
