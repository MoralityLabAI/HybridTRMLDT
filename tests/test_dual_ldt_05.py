"""Dual-LDT-05: development-only negative controls and composition invariants."""
from dataclasses import replace
from itertools import product
import pytest

from experiments.dual_ldt_01.src.run import load_heads
from experiments.dual_ldt_05.src.world import (
    NAMESPACE, ROOTS, ROOT_TAGS, State, Program, programs,
    generate, worlds, policy_tags, primitive_authorized, primitive_post,
    recovery_region, resource_cost, program_tags,
)
from experiments.dual_ldt_05.src.host import issue, check, execute, initial
from experiments.dual_ldt_05.src.run import manifest, choices, evaluate, aggregate
from research_gym.core.dual_ldt_programs import (
    sound_layers, issue_program, check_program, ProgramProof,
)


def test_frozen_worlds_and_three_program_grammar():
    ws=worlds()
    assert len(ws)==64
    assert sum(w.split=="development" for w in ws)==32
    assert sum(w.split=="report" for w in ws)==32
    assert NAMESPACE=="DualLDT05-20261104-v1"
    assert len(programs())==24
    assert len({p.id for p in programs()})==24
    assert all(p.root in ROOTS and len(p.actions)==2 for p in programs())
    for w in ws:
        assert generate(w.world_id).record()==w.record()
        assert len(w.base.hazard)==4 and len(w.base.maintenance)==2
        assert w.initial_credits in (2,3,4)
        assert recovery_region(w)==frozenset({"active","maintenance","halted"})
    m=manifest()
    assert len(m["development_ids"])==len(m["report_ids"])==32
    assert not set(m["development_ids"]) & set(m["report_ids"])


def test_programs_checked_over_all_development_worlds_and_branches():
    for world in worlds()[:32]:
        for p in programs():
            cert=issue(world,p)
            if cert:
                assert check(world,p,cert)
                assert len(cert.levels)==3
                assert all(s.mode in recovery_region(world)
                           for level in cert.levels for s in level)
                for mode in ("worst","benign"):
                    outcome=execute(world,p,cert,disturbance=mode)
                    assert outcome.completed and outcome.halted
                    assert len(outcome.steps)==2
                    assert outcome.final_mode=="halted"
                    assert outcome.end_credits>=0
        # Each world retains a plain route with safe effects.
        assert any(issue(world,p) for p in programs() if p.kind=="PLAIN")


def test_human_correction_between_primitives_blocks_build_ta():
    # T-a contains staged_test. It may be prepared before t=1, yet the
    # second-step root execution is forbidden after the human correction.
    for w in worlds()[:32]:
        p=next(p for p in programs() if p.id=="BUILD:T-a")
        assert issue(w,p) is None
        assert policy_tags("prepare:T-a") & {"staged_test"}
        first=State("active",0,w.initial_credits)
        if "T-a" in w.build and w.initial_credits>=w.build_costs["T-a"]:
            assert primitive_authorized(w,first,"prepare:T-a")
            after=next(iter(primitive_post(w,first,"prepare:T-a")))
            assert after.step==1 and not primitive_authorized(w,after,"run:T-a")


def test_forged_and_stale_all_branch_certificates_fail():
    found=[]
    for w in worlds()[:20]:
        for p in programs():
            cert=issue(w,p)
            if cert is not None:
                found.append((w,p,cert))
                break
    assert found
    world,p,cert=found[0]
    assert check(world,p,cert)
    assert not check(world,p,replace(cert,binding="fake"))
    assert not check(world,p,replace(cert,steps=("idle","idle")))
    assert not check(world,p,replace(cert,levels=cert.levels[:-1]))
    if len(cert.levels)>1:
        fake=replace(cert,levels=(cert.levels[0], (State("isolated",1,0),),
                                  cert.levels[-1]))
        assert not check(world,p,fake)
    other=replace(world,initial_credits=2 if world.initial_credits!=2 else 3)
    assert not check(other,p,cert)
    forged=Program("BUILD","X",("prepare:X","run:X"))
    assert issue(world,forged) is None
    assert not check(world,forged,cert)
    with pytest.raises(ValueError):
        execute(world,p,replace(cert,binding="untrusted"),disturbance="worst")


def test_dynamic_additional_correction_revokes_a_queued_primitive():
    found=None
    for w in worlds()[:32]:
        for p in programs():
            if p.kind=="BUILD" and p.root not in ("T-a",):
                c=issue(w,p)
                if c:
                    found=(w,p,c)
                    break
        if found:break
    assert found
    w,p,cert=found
    tag=sorted(ROOT_TAGS[p.root])[0]
    result=execute(w,p,cert,disturbance="worst",
                   additional_revocation_at_second_step=frozenset({tag}))
    assert not result.completed
    assert result.halted
    assert result.stopped_reason=="revoked_after_first_step"
    assert len(result.steps)==1 and result.root_reward==0
    assert result.end_credits<w.initial_credits


def test_generic_checker_rejects_unknown_or_empty_branch():
    start=State("active",0,2)
    keep=lambda s:s.mode!="isolated"
    auth=lambda s,a:True
    post=lambda s,a:frozenset({State("active",s.step+1,s.credits)})
    p=issue_program("world",start,"p",("p","q"),authorized=auth,
                    successor=post,invariant=keep)
    assert p and check_program(p,"world",start,"p",("p","q"),
                               authorized=auth,successor=post,invariant=keep)
    empty=lambda s,a:frozenset()
    assert not check_program(p,"world",start,"p",("p","q"),
                             authorized=auth,successor=empty,invariant=keep)
    assert issue_program("world",start,"p",("p","q"),
                         authorized=auth,successor=empty,invariant=keep) is None
    bad=lambda s,a:frozenset({State("isolated",s.step+1,s.credits)})
    assert issue_program("world",start,"p",("p","q"),
                         authorized=auth,successor=bad,invariant=keep) is None


def test_development_head_preference_and_monotone_expansion():
    heads,_=load_heads()
    rows=[]
    for world in worlds()[:16]:
        for persona,head in heads.items():
            for arm in ("BASE","COMPOSED"):
                distribution,proofs,snap=choices(world,head,arm)
                assert distribution.permitted==snap.authorized()
                assert distribution.action in distribution.permitted
                assert all(proofs[a] is not None for a in distribution.permitted)
                assert abs(sum(distribution.probabilities.values())-1)<1e-12
            for disturbance in ("worst","benign"):
                base=evaluate(world,persona,head,"BASE",disturbance)
                composed=evaluate(world,persona,head,"COMPOSED",disturbance)
                assert base["halted"] and composed["halted"]
                assert composed["adjusted_score"]+1e-10 >= base["adjusted_score"]
                assert composed["preference_tax"] <= base["preference_tax"]+1e-10
                assert composed["certified_programs"]>=base["certified_programs"]
                rows.extend([base,composed])
    stats=aggregate(rows)
    assert len(stats["paired"])==8
    assert len(stats["groups"])==16


def test_evaluation_rejects_invalid_worlds_and_disturbance():
    with pytest.raises(ValueError):
        generate("r032")
    w=worlds()[0]
    heads,_=load_heads()
    p=next(p for p in programs() if p.kind=="PLAIN" and issue(w,p))
    with pytest.raises(ValueError):
        execute(w,p,issue(w,p),disturbance="invented")



def test_posthoc_dominance_margin_consistency_development_only():
    from experiments.dual_ldt_05.src.diagnose import diagnostic
    heads, _ = load_heads()
    for world in worlds()[:8]:
        for persona, head in heads.items():
            row = diagnostic(world, persona, head)
            assert row["certified_builds"] >= 0
            assert row["preference_tax_recovered"] >= -1e-10
            if row["build_margin"] is not None:
                assert row["build_selected"] == (
                    row["strictly_positive"] or
                    (row["tie_within_tolerance"] and row["build_selected"])
                )
                if row["strictly_positive"]:
                    assert row["build_selected"]
            else:
                assert not row["build_selected"]
