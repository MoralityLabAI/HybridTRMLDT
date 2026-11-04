"""Dual-LDT-05: two-step proof-carrying program search on frozen real Jinn heads.

Both arms share the same 24-program reference domain and host transition
semantics. Learned logits order host-certified programs only.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from math import exp
from pathlib import Path
from statistics import mean
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from experiments.dual_ldt_01.src.run import (  # noqa: E402
    load_heads, CHARACTERS, BANNED,
)
from experiments.dual_ldt_05.src.world import (  # noqa: E402
    World, worlds, programs, ROOTS, ROOT_TAGS, program_tags,
    resource_cost, NAMESPACE, RHO,
)
from experiments.dual_ldt_05.src.host import issue, check, execute  # noqa: E402
from research_gym.core.dual_ldt import (  # noqa: E402
    ActionEvidence, BeastSnapshot, sort_with_jinn,
)

PROGRAMS = programs()


def bytes_of(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"))+"\n").encode("utf-8")


def save(path:Path, value) -> str:
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=bytes_of(value)
    path.write_bytes(raw)
    return sha256(raw).hexdigest()


def manifest() -> dict:
    records=[w.record() for w in worlds()]
    return {
        "namespace":NAMESPACE,
        "registration_commit":"1bb0b1b17d9c81c3ea3534d29a31d6d98757a8e3",
        "generator_commit":"32358bf4827f4ed61ba6d92be9c883b6719547dd",
        "config_sha256":sha256(bytes_of(records)).hexdigest(),
        "development_ids":[r["id"] for r in records if r["split"]=="development"],
        "report_ids":[r["id"] for r in records if r["split"]=="report"],
        "worlds":records,
    }


def choices(world:World,head,arm:str) -> tuple[object,dict,BeastSnapshot]:
    if arm not in ("BASE","COMPOSED"):
        raise ValueError("unsupported arm")
    candidates={}
    evidence={}
    for p in PROGRAMS:
        cert=issue(world,p) if (arm=="COMPOSED" or p.kind!="BUILD") else None
        if cert is not None and not check(world,p,cert):
            raise AssertionError("issuer/checker disagreed")
        candidates[p.id]=cert
        evidence[p.id]=ActionEvidence(
            cert is not None, cert is None,(program_tags(p),)
        )
    snapshot=BeastSnapshot(
        f"{world.world_id}|{world.digest()}|{arm}|t0", tuple(p.id for p in PROGRAMS),
        frozenset(BANNED), evidence
    )
    q={p.id:exp(-RHO*resource_cost(world,p)) for p in PROGRAMS}
    distribution=sort_with_jinn(snapshot,head,baseline=q,strength=1.)
    if set(distribution.probabilities)!=set(a for a,p in candidates.items() if p):
        raise AssertionError("Jinn support did not match checked programs")
    return distribution,candidates,snapshot


def evaluate(world:World, persona:str, head, arm:str, disturbance:str) -> dict:
    d,certs,snapshot=choices(world,head,arm)
    scored={p.id:head.train_log_probability(program_tags(p)) for p in PROGRAMS}
    baseline_root=max(ROOTS,key=lambda root:
        (head.train_log_probability(ROOT_TAGS[root]),-ROOTS.index(root)))
    benchmark_score=head.train_log_probability(ROOT_TAGS[baseline_root])
    selected=next((p for p in PROGRAMS if p.id==d.action),None)
    if selected is None:
        raise AssertionError("world generated no recoverable program")
    proof=certs[selected.id]
    if proof is None or not check(world,selected,proof):
        raise AssertionError("chosen program not host-certified")
    outcome=execute(world,selected,proof,disturbance=disturbance)
    if not outcome.completed or not outcome.halted:
        raise AssertionError("certified two-step program failed")
    regret=max(0.0,benchmark_score-scored[selected.id])
    adjusted=scored[selected.id]-RHO*resource_cost(world,selected)
    if any(scored[a]-RHO*resource_cost(world,p)>adjusted+1e-10
        for p in PROGRAMS for a in [p.id] if a in d.permitted):
        raise AssertionError("not best cost-adjusted Jinn program")
    return {
        "world":world.world_id, "split":world.split,
        "persona":persona,"arm":arm,"disturbance":disturbance,
        "selected":selected.id,"kind":selected.kind,"root":selected.root,
        "reference_root":baseline_root,
        "certified_programs":len(d.permitted),
        "certified_builds":sum(
            p.kind=="BUILD" and p.id in d.permitted for p in PROGRAMS),
        "programs_checked":len(PROGRAMS) if arm=="COMPOSED" else 2*len(ROOTS),
        "check_successes":sum(p is not None for p in certs.values()),
        "halted":outcome.halted,"host_moves":outcome.host_moves,
        "regret":regret,"within_epsilon":regret<=.25+1e-10,
        "adjusted_score":adjusted,"preference_tax":d.preference_tax,
        "resource_spent":world.initial_credits-outcome.end_credits,
        "task_net":outcome.task_net,
        "trace":list(outcome.steps),
        "program_proof_binding":proof.binding,
        "all_branch_count":sum(len(level) for level in proof.levels),
    }


def aggregate(records):
    groups=defaultdict(list)
    lookup={}
    for r in records:
        groups[(r["split"],r["disturbance"],r["persona"],r["arm"])].append(r)
        k=(r["world"],r["disturbance"],r["persona"],r["arm"])
        if k in lookup:
            raise ValueError("duplicate record")
        lookup[k]=r
    rows=[]
    for (split,disturbance,persona,arm),es in sorted(groups.items()):
        rows.append({
            "split":split,"disturbance":disturbance,"persona":persona,"arm":arm,
            "worlds":len(es),"halt_successes":sum(e["halted"] for e in es),
            "mean_net_task":mean(e["task_net"] for e in es),
            "mean_regret":mean(e["regret"] for e in es),
            "within_epsilon":sum(e["within_epsilon"] for e in es),
            "mean_tax":mean(e["preference_tax"] for e in es),
            "mean_certified_builds":mean(e["certified_builds"] for e in es),
            "selected_builds":sum(e["kind"]=="BUILD" for e in es),
            "mean_resource_spent":mean(e["resource_spent"] for e in es),
            "mean_programs_checked":mean(e["programs_checked"] for e in es),
        })
    paired=[]
    keys=sorted({(e["world"],e["disturbance"],e["persona"]) for e in records})
    pgroups=defaultdict(list)
    for w,mode,persona in keys:
        a=lookup[(w,mode,persona,"BASE")]
        b=lookup[(w,mode,persona,"COMPOSED")]
        if b["adjusted_score"]+1e-10<a["adjusted_score"]:
            raise AssertionError("certified program expansion reduced preference optimum")
        if b["preference_tax"]>a["preference_tax"]+1e-10:
            raise AssertionError("expanded support increased constitutional tax")
        pgroups[(a["split"],mode,persona)].append({
            "extra_certified_builds":b["certified_builds"],
            "build_selected":b["kind"]=="BUILD",
            "regret_recovered":a["regret"]-b["regret"],
            "adjusted_gain":b["adjusted_score"]-a["adjusted_score"],
            "task_gain":b["task_net"]-a["task_net"],
            "tax_recovered":a["preference_tax"]-b["preference_tax"],
            "choice_changed":a["selected"]!=b["selected"],
            "resource_extra":b["resource_spent"]-a["resource_spent"],
        })
    for (split,mode,persona),xs in sorted(pgroups.items()):
        paired.append({
            "split":split,"disturbance":mode,"persona":persona,"worlds":len(xs),
            **{name:mean(float(e[name]) for e in xs) for name in xs[0]}
        })
    return {"groups":rows,"paired":paired}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path("experiments/dual_ldt_05/outputs"))
    parser.add_argument("--manifest-only",action="store_true")
    parser.add_argument("--split",choices=("development","report","all"),default="all")
    parser.add_argument("--limit",type=int,default=0)
    args=parser.parse_args()
    if args.limit<0:parser.error("negative limit")
    frozen=manifest()
    msha=save(args.out/"manifest.json",frozen)
    print("DUAL05 manifest",msha,"worlds",len(frozen["worlds"]))
    if args.manifest_only:
        return
    heads,source_hashes=load_heads()
    subset=[w for w in worlds() if args.split=="all" or w.split==args.split]
    if args.limit:subset=subset[:args.limit]
    records=[
        evaluate(w,persona,heads[persona],arm,mode)
        for w in subset for persona in CHARACTERS
        for mode in ("worst","benign")
        for arm in ("BASE","COMPOSED")
    ]
    output={"status":"registered_synthetic_finite_program_replay",
        "source_registration":"1bb0b1b17d9c81c3ea3534d29a31d6d98757a8e3",
        "source_generator":"32358bf4827f4ed61ba6d92be9c883b6719547dd",
        "head_hashes":source_hashes,
        "split":args.split,"world_count":len(subset),"episodes":len(records),
        "manifest_sha256":msha,"records":records}
    raw_sha=save(args.out/"episodes.json",output)
    result={"status":output["status"],"split":args.split,
        "world_count":len(subset),"episode_count":len(records),
        "raw_sha256":raw_sha,"manifest_sha256":msha,
        **aggregate(records)}
    save(args.out/"summary.json",result)
    for r in result["paired"]:
        print("DUAL05",r["split"],r["disturbance"],r["persona"],
              "BUILD",round(r["build_selected"],3),
              "new_certified",round(r["extra_certified_builds"],3),
              "regret_recovered",round(r["regret_recovered"],4),
              "task_gain",round(r["task_gain"],4))
    print("WROTE",args.out/"summary.json",args.out/"episodes.json")


if __name__=="__main__":
    main()
