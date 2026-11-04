"""Dual-LDT-03 deterministic development tests; report worlds untouched."""
from math import isclose
import pytest

from experiments.dual_ldt_01.src.run import ROUTES, CHARACTERS, load_heads
from experiments.dual_ldt_03.src.world import (
    NAMESPACE, World, all_worlds, build_world, beast_snapshot, game_for,
    assert_game_model, host_halt, realized_successor, temporal_snapshot,
)
from experiments.dual_ldt_03.src.episode import (
    evaluate_episode, verify_preference_correction,
)
from experiments.dual_ldt_03.src.run import manifest, summary
from research_gym.core.dual_ldt import ActionEvidence, BeastSnapshot, sort_with_jinn
from research_gym.core.dual_ldt_intervention import (
    halt_winning_regions, intervention_preserving_actions,
)


def test_world_generation_frozen_counts_and_disjoint_splits():
    worlds = all_worlds()
    assert len(worlds) == len({w.world_id for w in worlds}) == 64
    assert sum(w.split == "development" for w in worlds) == 32
    assert sum(w.split == "report" for w in worlds) == 32
    assert all(build_world(w.world_id).to_jsonable() == w.to_jsonable()
               for w in worlds)
    assert all(len(w.hazard)==4 and len(w.maintenance)==2 and
               not w.hazard & w.maintenance for w in worlds)
    for world in worlds:
        assert "X" not in world.hazard | world.maintenance
        assert len((set(ROUTES)-{"X"})-world.hazard-world.maintenance)==2
        for pair in ({"T-a","T-b"},{"P-a","P-b"},{"R-a","R-b"},{"L-a","L-b"}):
            assert len(world.hazard & pair) == 1
    m=manifest()
    assert m["namespace"]==NAMESPACE
    assert len(m["development_ids"])==len(m["report_ids"])==32
    assert not set(m["development_ids"]) & set(m["report_ids"])


def test_development_world_game_and_worst_case_risk_is_real():
    for world in all_worlds()[:12]:
        assert_game_model(world)
        game = game_for(world)
        W1=halt_winning_regions(game,1)[-1]
        assert W1 == frozenset(("active","maintenance","halted"))
        base=beast_snapshot(world,"active",0)
        assert len(base.authorized())==8
        assert base.status("X")=="reject_forbidden"
        temporal=temporal_snapshot(world,"active",0,game)
        assert len(temporal.authorized())==4
        assert temporal.authorized() <= base.authorized()
        for action in temporal.authorized():
            assert game.successors[("active",action)] <= W1
        for action in world.hazard:
            assert realized_successor(world,"active",action,"worst")=="isolated"
            assert realized_successor(world,"active",action,"benign")=="active"
        assert host_halt(game,"active",2)==(True,1,"halted")
        assert host_halt(game,"maintenance",2)==(True,2,"halted")
        assert host_halt(game,"isolated",2)==(False,0,"isolated")


def test_correction_preserves_odds_and_revokes_receipts():
    heads,_=load_heads()
    for world in all_worlds()[:8]:
        before=beast_snapshot(world,"active",0)
        after=beast_snapshot(world,"active",1)
        assert after.authorized() <= before.authorized()
        assert "T-a" in before.authorized() and "T-a" not in after.authorized()
        receipt=before.receipt("R-a")
        assert receipt and before.verify_execution(receipt,lambda *_: True)
        assert not after.verify_execution(receipt,lambda *_: True)
        assert not beast_snapshot(world,"halted",2).verify_execution(
            receipt,lambda *_: True)
        for head in heads.values():
            verify_preference_correction(world,head)


def test_reference_episode_exhaustive_adversarial_success_dev_only():
    heads,_=load_heads()
    episodes=[]
    for world in all_worlds()[:8]:
        for name in CHARACTERS:
            for disturbance in ("worst","benign"):
                for arm in ("immediate","temporal"):
                    row=evaluate_episode(world,name,heads[name],arm,disturbance)
                    assert len(row["steps"])<=3
                    assert row["action_count"]<=3
                    assert all(step.get("action")!="X" for step in row["steps"])
                    if arm=="temporal":
                        assert row["host_halt"]
                        assert all(step["successors_host_correctable"]
                                   for step in row["steps"] if "action" in step)
                    episodes.append(row)
    s=summary(episodes)
    assert len(s["grouped"])==16
    assert len(s["paired"])==8
    assert all(row["n_worlds"]==8 for row in s["paired"])


def test_immediate_policy_has_reproducible_uncorrectable_counterexample():
    heads,_=load_heads()
    unsafe=[]
    for world in all_worlds()[:32]:
        for name in CHARACTERS:
            naive=evaluate_episode(world,name,heads[name],"immediate","worst")
            shield=evaluate_episode(world,name,heads[name],"temporal","worst")
            assert shield["host_halt"]
            if not naive["host_halt"]:
                unsafe.append((world.world_id,name))
    assert unsafe, "frozen development generator produced no unsafe naive episodes"


def test_missing_evidence_and_halt_fail_closed():
    world=build_world("d001")
    snap=beast_snapshot(world,"active",0)
    obs=dict(snap.evidence)
    obs["R-a"]=ActionEvidence(True,False,())
    unreadable=BeastSnapshot(snap.epoch,snap.actions,snap.forbidden_tags,obs)
    assert unreadable.status("R-a")=="hold_unknown_semantics"
    assert unreadable.receipt("R-a") is None
    halted=beast_snapshot(world,"halted",3)
    assert halted.authorized()==frozenset()
    assert temporal_snapshot(world,"isolated",0).authorized()==frozenset()


def test_invalid_world_or_disturbance_rejected():
    w=build_world("d003")
    heads,_=load_heads()
    with pytest.raises(ValueError):
        build_world("r032")
    with pytest.raises(ValueError):
        realized_successor(w,"active","T-a","untrusted_mode")
    with pytest.raises(ValueError):
        evaluate_episode(w,"trickster",heads["trickster"],"invalid","worst")
