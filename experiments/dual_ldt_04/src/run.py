"""Dual-LDT-04 paired CPU replay. No neural training or semantic monitor calls."""
from __future__ import annotations
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
import sys

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))

from experiments.dual_ldt_01.src.run import CHARACTERS, load_heads  # noqa
from experiments.dual_ldt_04.src.world import NAMESPACE, worlds  # noqa
from experiments.dual_ldt_04.src.episode import episode  # noqa

ARMS=("immediate","temporal","synth")
DISTURBANCES=("worst","benign")


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"))+"\n").encode("utf-8")


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    data=canonical(value)
    path.write_bytes(data)
    return sha256(data).hexdigest()


def manifest():
    configs=[w.to_jsonable() for w in worlds()]
    return {
       "namespace":NAMESPACE,
       "status":"frozen_synthetic_config",
       "registered_source_commit":"70663f49b0a1a179587951f2f80ae2405df78a15",
       "registration_commit":"390f8e6dfc0874810398e77bed54e132f1a4c3c4",
       "worlds":configs,
       "worlds_sha256":sha256(canonical(configs)).hexdigest(),
       "development_ids":[c["world_id"] for c in configs if c["split"]=="development"],
       "report_ids":[c["world_id"] for c in configs if c["split"]=="report"],
    }


def summarize(rows):
    groups=defaultdict(list)
    keys={}
    for r in rows:
        group=(r["split"],r["disturbance"],r["persona"],r["arm"])
        groups[group].append(r)
        key=(r["world"],r["persona"],r["disturbance"],r["arm"])
        if key in keys:
            raise AssertionError("duplicate evaluation cell")
        keys[key]=r
    grouped=[]
    for (split,disturbance,persona,arm),episodes in sorted(groups.items()):
        steps=[s for r in episodes for s in r["steps"] if "choice" in s]
        grouped.append({
            "split":split,"disturbance":disturbance,"persona":persona,"arm":arm,
            "worlds":len(episodes),
            "host_halt_successes":sum(r["halted"] for r in episodes),
            "halt_rate":mean(r["halted"] for r in episodes),
            "mean_net_task_reward":mean(r["task_reward_net"] for r in episodes),
            "mean_credits_consumed":mean(r["total_credits"] for r in episodes),
            "mean_raw_regret_per_decision":mean(
                s["raw_jinn_regret"] for s in steps) if steps else None,
            "mean_guard_executions":mean(r["guard_executions"] for r in episodes),
            "guard_certificates_verified":sum(
                r["guard_certificates_verified"] for r in episodes),
            "guard_certificate_attempts":sum(
                r["guard_certificate_attempts"] for r in episodes),
            "synth_preference_epsilon_passes":sum(r["within_epsilon_count"] for r in episodes),
            "synth_preference_epsilon_denominator":len(steps) if arm=="synth" else 0,
            "mean_kl_tax_recovery":mean(r["cumulative_tax_recovery"] for r in episodes),
            "mean_synth_vs_temporal_choice_changes":mean(
                r["synth_vs_temporal_choice_changes"] for r in episodes),
            "mean_steps":mean(r["acted_steps"] for r in episodes),
            "abstention_episodes":sum(r["abstained"] for r in episodes),
        })
    paired=[]
    cases=sorted({(r["world"],r["persona"],r["disturbance"]) for r in rows})
    bucket=defaultdict(list)
    for world,persona,disturbance in cases:
        immediate=keys[(world,persona,disturbance,"immediate")]
        temporal=keys[(world,persona,disturbance,"temporal")]
        synth=keys[(world,persona,disturbance,"synth")]
        bucket[(immediate["split"],disturbance,persona)].append({
            "halt_synth_minus_immediate":int(synth["halted"])-int(immediate["halted"]),
            "halt_synth_minus_temporal":int(synth["halted"])-int(temporal["halted"]),
            "reward_synth_minus_temporal":
                synth["task_reward_net"]-temporal["task_reward_net"],
            "reward_synth_minus_immediate":
                synth["task_reward_net"]-immediate["task_reward_net"],
            "jinn_regret_synth_minus_temporal":
                synth["cumulative_jinn_regret"]-temporal["cumulative_jinn_regret"],
            "first_choice_diff":
                synth["steps"][0].get("choice")!=temporal["steps"][0].get("choice"),
            "guards_executed":synth["guard_executions"],
        })
    for (split,disturbance,persona),sample in sorted(bucket.items()):
        paired.append({
            "split":split,"disturbance":disturbance,"persona":persona,
            "worlds":len(sample),
            **{k:mean(x[k] for x in sample) for k in sample[0]}
        })
    return {"grouped":grouped,"paired":paired}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("experiments/dual_ldt_04/outputs"))
    p.add_argument("--split",choices=("development","report","all"),default="all")
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--manifest-only",action="store_true")
    args=p.parse_args()
    if args.limit<0:
        p.error("negative limits forbidden")
    frozen=manifest()
    frozen_sha=write(args.out/"manifest.json",frozen)
    print("DUAL-LDT-04 manifest",frozen_sha,"worlds",len(frozen["worlds"]))
    if args.manifest_only:return
    heads,head_hashes=load_heads()
    selected=[w for w in worlds() if args.split=="all" or w.split==args.split]
    if args.limit:selected=selected[:args.limit]
    entries=[
        episode(w,persona,heads[persona],arm,mode)
        for w in selected for persona in CHARACTERS
        for mode in DISTURBANCES for arm in ARMS
    ]
    if any(e["arm"]!="immediate" and not e["halted"] for e in entries):
        raise AssertionError("host intervention failure under certified temporal control")
    if any(x["raw_jinn_regret"]>x["certified_regret_upper"]+1e-9
           for e in entries if e["arm"]=="synth" for x in e["steps"] if "choice" in x):
        raise AssertionError("preference regret bound violated")
    summary=summarize(entries)
    meta={
       "status":"prospectively_generated_synthetic_worlds_with_fixed_reused_heads",
       "split":args.split,"worlds":len(selected),"episodes":len(entries),
       "world_manifest_sha256":frozen_sha,"world_config_sha256":frozen["worlds_sha256"],
       "original_paper_head_hashes":head_hashes,
       "source_registration_commit":"390f8e6dfc0874810398e77bed54e132f1a4c3c4",
       "source_generator_commit":"70663f49b0a1a179587951f2f80ae2405df78a15",
       "rho":0.1,"epsilon":0.25,
       "results":entries,
    }
    episodes_sha=write(args.out/"episodes.json",meta)
    report={"status":meta["status"],"split":args.split,"world_count":len(selected),
            "episode_count":len(entries),"manifest_sha256":frozen_sha,
            "episodes_sha256":episodes_sha,
            "world_config_sha256":frozen["worlds_sha256"],**summary}
    write(args.out/"summary.json",report)
    for row in report["paired"]:
        print("DUAL-LDT-04",row["split"],row["disturbance"],row["persona"],
              "HALT[S-I]",round(row["halt_synth_minus_immediate"],4),
              "reward[S-T]",round(row["reward_synth_minus_temporal"],4),
              "regret[S-T]",round(row["jinn_regret_synth_minus_temporal"],4),
              "guards",round(row["guards_executed"],4))
    print("WROTE",args.out/"summary.json",args.out/"episodes.json")


if __name__=="__main__": main()
