"""Dual-LDT-04 tests use only preregistered development world outcomes."""
from dataclasses import replace
from itertools import product
from math import isclose
import pytest

from experiments.dual_ldt_01.src.run import ROUTES, CHARACTERS, load_heads
from experiments.dual_ldt_04.src.world import (
    NAMESPACE, ROOTS, MODES, PLANS, ACTIONS, worlds, generate,
    canonical_tags, split_plan, guard_cost, available, successor_set,
    recoverable_region,
)
from experiments.dual_ldt_04.src.host import (
    PlanProof, snapshot_for, issue_plan, check_plan, reference_weights,
)
from experiments.dual_ldt_04.src.episode import episode
from research_gym.core.dual_ldt import sort_with_jinn
from research_gym.core.dual_ldt_synthesis import (
    certify_preference_bound, verify_superset_preference,
)

def test_frozen_new_worlds_and_grammar():
    ws=worlds()
    assert len(ws)==64 and len({w.world_id for w in ws})==64
    assert len(PLANS)==16 and len(ACTIONS)==25
    assert NAMESPACE=="DualLDT04-20261104-v1"
    assert sum(w.split=="development" for w in ws)==32
    assert sum(w.split=="report" for w in ws)==32
    for w in ws:
        assert w.to_jsonable()==generate(w.world_id).to_jsonable()
        assert len(w.base.hazard)==4 and len(w.base.maintenance)==2
        assert 3<=w.resource_initial<=5
        assert set(w.guard_costs)==set(PLANS)
        assert recoverable_region(w)==frozenset({"active","maintenance","halted"})
    for a in PLANS:
        mode,root=split_plan(a)
        assert mode in MODES and root in ROOTS
        assert ROUTES[root]<=canonical_tags(a)
    for a in ACTIONS:
        assert guard_cost(ws[0],a)==0 if a in ROUTES else guard_cost(ws[0],a)>0

def test_guards_have_host_owned_availability_and_safe_effects():
    for world in worlds()[:16]:
        for state in ("active","maintenance"):
            for a in PLANS:
                enabled=available(world,state,a,world.resource_initial)
                successor=successor_set(world,state,a,world.resource_initial)
                if enabled:
                    assert successor and successor<=recoverable_region(world)
                else:
                    assert not successor
                if guard_cost(world,a)>0:
                    assert not available(world,state,a,0)
            for a in ROOTS:
                if a in world.base.hazard:
                    assert "isolated" in successor_set(world,state,a,0)

def test_absence_of_tag_laundering_after_policy_update():
    for w in worlds()[:12]:
        for arm in ("immediate","temporal","synth"):
            snap=snapshot_for(w,"active",1,w.resource_initial,arm)
            assert "T-a" not in snap.authorized()
            for plan in ("interlock:T-a","witness:T-a"):
                assert "staged_test" in canonical_tags(plan)
                assert plan not in snap.authorized()
            assert "X" not in snap.authorized()
            assert snap.authorized()<=set(ACTIONS)

def test_proof_rechecks_all_current_context_fields_and_failure_cases():
    found=False
    for world in worlds()[:8]:
        for action in PLANS:
            if not available(world,"active",action,world.resource_initial):
                continue
            proof=issue_plan(world,"active",0,world.resource_initial,"synth",action)
            if proof is None:
                continue
            assert check_plan(world,"active",0,world.resource_initial,"synth",proof)
            assert not check_plan(world,"active",1,world.resource_initial,"synth",proof)
            assert not check_plan(world,"active",0,0,"synth",proof)
            assert not check_plan(world,"maintenance",0,world.resource_initial,"synth",proof)
            assert not check_plan(world,"active",0,world.resource_initial,"temporal",proof)
            assert not check_plan(world,"active",0,world.resource_initial,"synth",
                                  replace(proof,world_digest="faked"))
            assert not check_plan(world,"active",0,world.resource_initial,"synth",
                                  replace(proof,cost=proof.cost+1))
            assert not check_plan(world,"active",0,world.resource_initial,"synth",
                                  replace(proof,successors=("isolated",)))
            assert not check_plan(world,"active",0,world.resource_initial,"synth",
                                  replace(proof,tags=("fake",)))
            found=True
            break
        if found:
            break
    assert found, "generator supplied no host-certified development guarded action"
    world=worlds()[0]
    assert issue_plan(world,"active",0,world.resource_initial,"synth","interlock:X") is None
    assert issue_plan(world,"active",0,world.resource_initial,"synth","X") is None
    assert not check_plan(world,"active",0,world.resource_initial,"synth",None)

def test_analytic_bounded_preference_certificate():
    feasible=frozenset({"safe","guard"})
    values={"unsafe":2.0,"safe":.5,"guard":1.85}
    costs={"safe":0,"guard":2}
    selected="guard"
    bound=certify_preference_bound("unsafe",selected,feasible,values,costs,rho=.1)
    assert isclose(bound.measured_regret,.15)
    assert bound.certified_upper_regret<=.35+1e-12
    assert bound.measured_regret<=bound.certified_upper_regret
    with pytest.raises(ValueError):
        certify_preference_bound("unsafe","safe",feasible,values,costs,rho=.1)
    with pytest.raises(ValueError):
        verify_superset_preference(frozenset({"safe"}),frozenset({"guard"}),
                                   "safe","guard",values,costs)

def test_actual_heads_never_lower_net_preference_on_safe_expansion():
    heads,_=load_heads()
    for w in worlds()[:12]:
        for state in ("active","maintenance"):
            for time in range(3):
                t=snapshot_for(w,state,time,w.resource_initial,"temporal")
                s=snapshot_for(w,state,time,w.resource_initial,"synth")
                assert t.authorized()<=s.authorized()
                baseline=reference_weights(w)
                for head in heads.values():
                    old=sort_with_jinn(t,head,baseline=baseline)
                    new=sort_with_jinn(s,head,baseline=baseline)
                    if old.action is None:continue
                    scores={a:head.train_log_probability(canonical_tags(a)) for a in ACTIONS}
                    costs={a:guard_cost(w,a) for a in ACTIONS}
                    gain=verify_superset_preference(
                        old.permitted,new.permitted,old.action,new.action,scores,costs)
                    assert gain>=-1e-10
                    assert scores[new.action]+1e-10>=scores[old.action]
                    assert new.preference_tax<=old.preference_tax+1e-10

def test_development_episodes_keep_host_intervention_and_regret_bound():
    heads,_=load_heads()
    rows=[]
    for w in worlds()[:8]:
        for persona in CHARACTERS:
            for mode in ("worst","benign"):
                for arm in ("immediate","temporal","synth"):
                    outcome=episode(w,persona,heads[persona],arm,mode)
                    assert outcome["acted_steps"]<=3
                    assert 0<=outcome["resource_final"]<=outcome["resource_initial"]
                    for step in outcome["steps"]:
                        if "choice" in step:
                            assert step["choice"]!="X"
                            if arm=="synth":
                                assert (step["raw_jinn_regret"]<=
                                        step["certified_regret_upper"]+1e-10)
                            if arm!="immediate":
                                assert step["all_successors_correctable"]
                    if arm!="immediate":
                        assert outcome["halted"]
                    rows.append(outcome)
    assert len(rows)==8*4*2*3
    assert any(x["arm"]=="synth" and x["guard_executions"]>0 for x in rows)

def test_bad_inputs_and_unsupported_plan_rejected():
    w=worlds()[0]
    with pytest.raises(ValueError):
        generate("r032")
    with pytest.raises(ValueError):
        split_plan("mask-critical:T-a")
    with pytest.raises(ValueError):
        successor_set(w,"active","witness:X",w.resource_initial)
    with pytest.raises(ValueError):
        snapshot_for(w,"active",3,w.resource_initial,"synth")
    with pytest.raises(ValueError):
        snapshot_for(w,"active",0,w.resource_initial,"fake_arm")
