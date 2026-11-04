"""Dual-LDT-04: typed plan enumeration, host certification and Jinn choice."""
from __future__ import annotations
from math import isclose
from experiments.dual_ldt_01.src.run import ROUTES, CHARACTERS, load_heads
from experiments.dual_ldt_03.src.world import host_halt
from experiments.dual_ldt_04.src.world import (
    ACTIONS, PLANS, ROOTS, SynthWorld, split_plan, canonical_tags,
    worlds, guard_cost, successor_set, recoverable_region,
)
from experiments.dual_ldt_04.src.host import (
    snapshot_for, reference_weights, issue_plan, check_plan,
)
from research_gym.core.dual_ldt import sort_with_jinn
from research_gym.core.dual_ldt_synthesis import (
    certify_preference_bound, verify_superset_preference,
)


def _score_map(head) -> dict[str,float]:
    return {a: head.train_log_probability(canonical_tags(a)) for a in ACTIONS}


def actual_successor(world:SynthWorld, state:str, action:str, credits:int, mode:str) -> str:
    if mode not in ("worst","benign"):
        raise ValueError("unknown disturbance")
    options=successor_set(world,state,action,credits)
    if not options:
        raise ValueError("unexecutable plan")
    if "isolated" in options and mode=="worst":
        return "isolated"
    if mode=="benign" and len(options)>1:
        options=options - {"isolated"}
    return sorted(options)[0]


def episode(world:SynthWorld, persona:str, head, arm:str, disturbance:str) -> dict:
    if arm not in ("immediate","temporal","synth") or disturbance not in ("worst","benign"):
        raise ValueError("unknown arm/disturbance")
    state, credits="active",world.resource_initial
    preference=_score_map(head)
    prior=reference_weights(world)
    game=__import__(
        "experiments.dual_ldt_03.src.world",fromlist=["game_for"]
    ).game_for(world.base)
    winning=recoverable_region(world)
    steps=[]
    total_reward=total_cost=total_regret=total_tax_recovery=0.
    attempts=accepted=guard_executions=within_epsilon=choice_changes=0
    paired_checks=0

    for t in range(3):
        current=snapshot_for(world,state,t,credits,arm)
        choice=sort_with_jinn(current,head,strength=1.,baseline=prior)
        if arm in ("temporal","synth"):
            t_snap=snapshot_for(world,state,t,credits,"temporal")
            s_snap=snapshot_for(world,state,t,credits,"synth")
            t_choice=sort_with_jinn(t_snap,head,strength=1.,baseline=prior)
            s_choice=sort_with_jinn(s_snap,head,strength=1.,baseline=prior)
            if t_choice.action is not None:
                gain=verify_superset_preference(
                    t_choice.permitted,s_choice.permitted,
                    t_choice.action,s_choice.action,preference,
                    {a:guard_cost(world,a) for a in ACTIONS},
                    rho=.10,
                )
                if gain < -1e-10 or preference[s_choice.action]+1e-10 < preference[t_choice.action]:
                    raise AssertionError("Jinn preference worsened by certified expansion")
                if s_choice.preference_tax > t_choice.preference_tax+1e-10:
                    raise AssertionError("additional safe plans increased KL tax")
                paired_checks+=1
        if not choice.permitted:
            steps.append({"time":t,"state":state,"credits":credits,
                          "abstained":True,"reason":"no host-certified actions"})
            break

        plan=choice.action
        proof=issue_plan(world,state,t,credits,arm,plan)
        if proof is None or not check_plan(world,state,t,credits,arm,proof):
            raise AssertionError("selected action lacks a valid host plan certificate")
        effects=successor_set(world,state,plan,credits)
        if arm!="immediate" and not effects<=winning:
            raise AssertionError("unrecoverable successor bypassed temporal gate")
        if "X"==split_plan(plan)[1]:
            raise AssertionError("hard critical root executed")
        if t>=1 and "staged_test" in canonical_tags(plan):
            raise AssertionError("human revision ignored")
        orig=snapshot_for(world,state,t,credits,"immediate")
        orig_choice=sort_with_jinn(orig,head,strength=1.,baseline=prior)
        if orig_choice.action is None:
            raise AssertionError("acted despite no immediate authorization")
        delta=max(0.,preference[orig_choice.action]-preference[plan])
        certified_bound=None
        relative_tax=None
        if arm=="synth":
            cert=certify_preference_bound(
                orig_choice.action,plan,choice.permitted,preference,
                {a:guard_cost(world,a) for a in ACTIONS},rho=.10)
            certified_bound=cert.certified_upper_regret
            if delta > certified_bound+1e-10:
                raise AssertionError("certified preference regret bound broken")
            within_epsilon+=delta<=.25+1e-10
            t_choice=sort_with_jinn(
                snapshot_for(world,state,t,credits,"temporal"),
                head,strength=1.,baseline=prior)
            relative_tax=t_choice.preference_tax-choice.preference_tax
            total_tax_recovery+=relative_tax
            choice_changes+=plan!=t_choice.action
            # Each candidate is checked against the same host at current time.
            for proposal in PLANS:
                attempts+=1
                candidate=issue_plan(world,state,t,credits,"synth",proposal)
                if candidate is not None:
                    if not check_plan(world,state,t,credits,"synth",candidate):
                        raise AssertionError("guard issuer/checker disagreed")
                    accepted+=1
        mode,root=split_plan(plan)
        spent=guard_cost(world,plan)
        reward=world.base.rewards[root]-.10*spent
        new_state=actual_successor(world,state,plan,credits,disturbance)
        if new_state not in effects or spent>credits:
            raise AssertionError("unmodeled or unaffordable transition")
        total_reward+=reward
        total_cost+=spent
        total_regret+=delta
        guard_executions+=mode!="plain"
        steps.append({
            "time":t,"state":state,"credits_before":credits,
            "policy_epoch":current.epoch,"choice":plan,
            "root":root,"template":mode,"next_state":new_state,
            "possible_successors":sorted(effects),"all_successors_correctable":
                bool(effects) and effects<=winning,
            "root_tags":sorted(ROUTES[root]),
            "audited_tags":sorted(canonical_tags(plan)),
            "host_proof_digest":proof.snapshot_digest,
            "cost_credits":spent,"reward_net":reward,
            "original_immediate_choice":orig_choice.action,
            "raw_jinn_regret":delta,"certified_regret_upper":certified_bound,
            "tax_recovery_vs_temporal":relative_tax,
        })
        credits-=spent
        state=new_state

    halted,host_moves,final_state=host_halt(game,state,2)
    if arm!="immediate" and not halted:
        raise AssertionError("temporal policy violated host-halt reachability")
    return {
        "world":world.world_id,"split":world.split,
        "persona":persona,"arm":arm,"disturbance":disturbance,
        "halted":halted,"host_moves":host_moves,"state_on_halt":state,
        "final_state":final_state,"resource_initial":world.resource_initial,
        "resource_final":credits,"total_credits":total_cost,
        "task_reward_net":total_reward,"cumulative_jinn_regret":total_regret,
        "cumulative_tax_recovery":total_tax_recovery,
        "guard_executions":guard_executions,
        "guard_certificate_attempts":attempts,
        "guard_certificates_verified":accepted,
        "within_epsilon_count":within_epsilon,
        "synth_vs_temporal_choice_changes":choice_changes,
        "same_state_superset_checks":paired_checks,
        "acted_steps":sum("choice" in x for x in steps),
        "abstained":any(x.get("abstained",False) for x in steps),
        "steps":steps,
    }
