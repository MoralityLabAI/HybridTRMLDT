"""Dual-LDT-05 post-hoc dominance diagnosis on the frozen world family.

This code must not alter registration, model weights, program grammar, cost
parameters or evaluation outcomes. It calculates a deterministic mathematical
diagnostic: can any newly certified BUILD beat the baseline adjusted score?
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from math import isfinite
from pathlib import Path
from statistics import mean
import sys

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))

from experiments.dual_ldt_01.src.run import CHARACTERS, load_heads  # noqa
from experiments.dual_ldt_05.src.world import (  # noqa
    worlds, programs, program_tags, resource_cost, RHO,
)
from experiments.dual_ldt_05.src.run import choices, save  # noqa

PROGRAMS=programs()


def diagnostic(world,persona,head):
    base,_,_=choices(world,head,"BASE")
    enlarged,_,_=choices(world,head,"COMPOSED")
    scores={
        p.id:head.train_log_probability(program_tags(p))-RHO*resource_cost(world,p)
        for p in PROGRAMS
    }
    baseline_opt=max(scores[a] for a in base.permitted)
    new=[p for p in PROGRAMS if p.kind=="BUILD" and p.id in enlarged.permitted]
    newly_certified=frozenset(p.id for p in new)
    if enlarged.permitted != base.permitted | newly_certified:
        raise AssertionError("unexpected expanded set")
    new_opt=max((scores[p.id] for p in new),default=None)
    margin=(new_opt-baseline_opt) if new_opt is not None else None
    winner=enlarged.action
    if winner is None:
        raise AssertionError("no certified program")
    if winner in newly_certified and margin is not None and margin < -1e-10:
        raise AssertionError("new action chosen despite strict negative margin")
    if winner not in newly_certified and margin is not None and margin > 1e-10:
        raise AssertionError("best new program strictly dominates but was not chosen")
    return {
        "world":world.world_id,"split":world.split,"persona":persona,
        "baseline_choice":base.action,
        "enlarged_choice":winner,
        "certified_builds":len(new),
        "best_build":max((p.id for p in new),key=lambda a:scores[a],default=None),
        "baseline_adjusted_optimum":baseline_opt,
        "best_build_adjusted":new_opt,
        "build_margin":margin,
        "strictly_positive":margin is not None and margin > 1e-10,
        "tie_within_tolerance":margin is not None and abs(margin)<=1e-10,
        "build_selected":winner in newly_certified,
        "preference_tax_recovered":base.preference_tax-enlarged.preference_tax,
    }


def run(split="all"):
    heads,hashes=load_heads()
    selected=[w for w in worlds() if split=="all" or w.split==split]
    rows=[diagnostic(w,persona,heads[persona]) for w in selected
          for persona in CHARACTERS]
    groups=defaultdict(list)
    for row in rows:
        groups[(row["split"],row["persona"])].append(row)
    summaries=[]
    for (sp,persona),rs in sorted(groups.items()):
        margins=[r["build_margin"] for r in rs if r["build_margin"] is not None]
        summaries.append({
            "split":sp,"persona":persona,"worlds":len(rs),
            "worlds_with_certified_build":len(margins),
            "mean_certified_builds":mean(r["certified_builds"] for r in rs),
            "strict_positive_margins":sum(r["strictly_positive"] for r in rs),
            "tied_margins":sum(r["tie_within_tolerance"] for r in rs),
            "build_selected":sum(r["build_selected"] for r in rs),
            "max_build_margin":max(margins,default=None),
            "min_build_margin":min(margins,default=None),
            "mean_build_margin":mean(margins) if margins else None,
            "mean_tax_recovered":mean(r["preference_tax_recovered"] for r in rs),
        })
    return {"status":"post_hoc_analytical_dominance_diagnostic",
            "model_source_hashes":hashes,"rows":rows,"groups":summaries}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,
                    default=Path("experiments/dual_ldt_05/outputs/diagnostic.json"))
    ap.add_argument("--split",choices=["development","report","all"],default="all")
    args=ap.parse_args()
    data=run(args.split)
    save(args.out,data)
    for row in data["groups"]:
        print("DUAL05 GAP",row["split"],row["persona"],
              "max",row["max_build_margin"],
              "mean",row["mean_build_margin"],
              "positive",row["strict_positive_margins"],
              "selected",row["build_selected"])
    print("WROTE",args.out)

if __name__=="__main__":
    main()
