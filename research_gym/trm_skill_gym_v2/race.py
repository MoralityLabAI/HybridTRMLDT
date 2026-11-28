"""Fresh training race of archived finalist architectures on paired strata.

Every candidate gets identical roots, train steps, seeds and per-job wall cap.
Incomplete jobs remain explicit. Selection freezes before held-out evaluation.
No API. Cooperatively bounded; no hard process kill.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from research_gym.trm_skill_gym import campaign as C, dsl, modules as M, tasks as T, train as TR
from .runner import SOURCE, CAL, TEST, GRAPH, hard_graph, pools, evaluate
from .core import paired_rank


def roster(archive, skill, max_archive):
    configs=C.arm_configs()
    out={k:configs[k] for k in ("A_existing_skill","B_feedforward","C_gru","D_trm_direct")}
    seen={dsl.config_id(cfg) for k,cfg in out.items() if k!="A_existing_skill"}
    added=0
    for row in sorted(archive,key=lambda r:(r["arm"],r["round"],r["proposal_index"])):
        if row["skill"]!=skill or (row.get("rung1") or {}).get("dev_exact") is None:
            continue
        cfg=dsl.validate(row["config"])
        key=dsl.config_id(cfg)
        if key in seen or not dsl.PARAM_MIN<=M.SkillModule(cfg).n_params()<=dsl.PARAM_MAX:
            continue
        if max_archive is not None and added>=max_archive:
            break
        out["archive_"+key]=cfg
        seen.add(key)
        added+=1
    return out


def training_pool(skill,count):
    strata=list(T.FAMILIES[skill]["dev"])
    if skill in GRAPH:
        strata += [2,4,6,8]
    inst=[]
    for i in range(count):
        s=strata[i%len(strata)]
        root=1_800_000+i
        inst.append(hard_graph(skill,s,root) if isinstance(s,int) else T.make_instance(skill,s,root))
    return T.stack(inst)


def run_job(path,skill,name,cfg,seed,steps,train,cal,deadline,job_seconds):
    if path.exists():
        old=C.read_json(path)
        if old["status"]=="ok":
            return old
    if time.time()>=deadline:
        return {"status":"deadline"}
    try:
        if name=="A_existing_skill":
            model=None
            info={"steps":0,"train_seconds":0}
            cfg=C.base_trm()
        else:
            model,info=TR.train_module(skill,cfg,steps,seed,train,
                deadline=min(deadline,time.time()+job_seconds))
            if info["steps"]!=steps:
                out={"status":"incomplete","train":info}
                C.write_json(path,out)
                return out
            model.eval()
        ev={k:evaluate(skill,model,cfg,b)[1] for k,b in cal.items()}
        if model is not None:
            ck=path.with_suffix(".pt")
            tmp=ck.with_suffix(".tmp")
            torch.save({"config":cfg,"state_dict":model.state_dict()},tmp)
            tmp.replace(ck)
        out={"status":"ok","train":info,"cal":ev}
    except (TR.RamCapExceeded,RuntimeError) as exc:
        out={"status":"error","error":type(exc).__name__}
    C.write_json(path,out)
    return out


def aggregate(jobs):
    return {k:{"exact":float(np.mean([j["cal"][k]["exact"] for j in jobs])),
               "flops":max(j["cal"][k]["flops"] for j in jobs),
               "calls":max(j["cal"][k]["calls"] for j in jobs)}
            for k in jobs[0]["cal"]}


def paired_interval(delta,seed=19):
    # Seed-averaged paired outcomes; bootstrap roots, not individual seed/root cells.
    delta=np.asarray(delta)
    d=delta.mean(0)
    rng=np.random.default_rng(seed)
    boot=d[rng.integers(0,len(d),(2000,len(d)))].mean(1)
    return {"delta":float(d.mean()),"root_bootstrap_95":np.quantile(boot,[.025,.975]).tolist(),
            "seed_deltas":delta.mean(1).tolist()}


def load(path,name):
    if name=="A_existing_skill":
        return None,C.base_trm()
    ck=torch.load(path.with_suffix(".pt"),map_location="cpu",weights_only=True)
    model=M.SkillModule(ck["config"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(),ck["config"]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=SOURCE)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--skills",nargs="+",choices=T.SKILLS,default=list(T.SKILLS))
    p.add_argument("--seeds",type=int,nargs="+",default=[10,11,12,13])
    p.add_argument("--steps",type=int,default=800)
    p.add_argument("--count",type=int,default=128)
    p.add_argument("--train-count",type=int,default=4096)
    p.add_argument("--max-archive",type=int)
    p.add_argument("--hours",type=float,default=8)
    p.add_argument("--job-seconds",type=float,default=900)
    a=p.parse_args()
    if (not 0<a.hours<=8 or min(a.steps,a.count,a.train_count,a.job_seconds)<=0
        or len(set(a.seeds))!=len(a.seeds) or len(set(a.skills))!=len(a.skills)
        or (a.max_archive is not None and a.max_archive<0)):
        p.error("bad budget/roster")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    a.out.mkdir(parents=True,exist_ok=True)
    archive_bytes=(a.source/"search/archive.jsonl").read_bytes()
    archive=[json.loads(l) for l in archive_bytes.splitlines() if l.strip()]
    configs={s:roster(archive,s,a.max_archive) for s in a.skills}
    profile={"skills":a.skills,"seeds":a.seeds,"steps":a.steps,"count":a.count,
             "train_count":a.train_count,"hours":a.hours,"job_seconds":a.job_seconds,
             "archive_sha256":hashlib.sha256(archive_bytes).hexdigest(),"configs":configs,
             "cal_root":CAL,"test_root":TEST,"train_root":1800000}
    manifest=a.out/"manifest.json"
    if manifest.exists():
        old=C.read_json(manifest)
        if old["profile"]!=profile:
            p.error("manifest mismatch; use a new output directory")
        deadline=old["deadline"]
    else:
        deadline=time.time()+3600*a.hours
        C.write_json(manifest,{"profile":profile,"deadline":deadline})
    def jobpath(skill,name,seed):
        return a.out/"jobs"/skill/f"{name}__s{seed}.json"
    frozen={}
    freeze=a.out/"frozen.json"
    if freeze.exists():
        frozen=C.read_json(freeze)
    else:
        for skill in a.skills:
            if time.time()>=deadline:
                break
            train=training_pool(skill,a.train_count)
            cal=pools(skill,CAL,a.count)
            metrics={}
            incomplete=[]
            for name,cfg in configs[skill].items():
                jobs=[]
                for seed in a.seeds:
                    j=run_job(jobpath(skill,name,seed),skill,name,cfg,seed,a.steps,train,cal,deadline,a.job_seconds)
                    jobs.append(j)
                if all(j["status"]=="ok" for j in jobs):
                    metrics[name]=aggregate(jobs)
                else:
                    incomplete.append(name)
            required=("A_existing_skill","B_feedforward","C_gru","D_trm_direct")
            if not all(k in metrics for k in required):
                C.write_json(a.out/"partial.json",{"skill":skill,"incomplete":incomplete,"frozen_so_far":frozen})
                return
            cap=max(m["flops"] for m in metrics["D_trm_direct"].values())
            eligible=[k for k,m in metrics.items() if all(v["flops"]<=cap and v["calls"]<=2 for v in m.values())]
            selected=max(eligible,key=lambda k:paired_rank(metrics[k],cal))
            baseline=max([k for k in eligible if k in required],key=lambda k:paired_rank(metrics[k],cal))
            frozen[skill]={"selected":selected,"baseline":baseline,"cap_model_flops":cap,
                           "incomplete":incomplete,"eligible":eligible,"calibration":metrics}
            print(skill,selected,baseline,"incomplete",len(incomplete),flush=True)
        if len(frozen)!=len(a.skills):
            C.write_json(a.out/"partial.json",{"frozen_so_far":frozen,"status":"deadline"})
            return
        C.write_json(freeze,frozen)
    test_rows={}
    for skill,row in frozen.items():
        if time.time()>=deadline:
            break
        test=pools(skill,TEST,a.count,True)
        paired={k:[] for k in test}
        metrics={}
        for seed in a.seeds:
            acc={}
            for name in set((row["selected"],row["baseline"])):
                model,cfg=load(jobpath(skill,name,seed),name)
                acc[name]={}
                metrics[f"{name}/s{seed}"]={}
                for k,b in test.items():
                    acc[name][k],metrics[f"{name}/s{seed}"][k]=evaluate(skill,model,cfg,b)
            for k in test:
                paired[k].append(acc[row["selected"]][k]-acc[row["baseline"]][k])
        test_rows[skill]={"metrics":metrics,"paired":{k:paired_interval(v) for k,v in paired.items()}}
        C.write_json(a.out/"heldout.json",{"complete":len(test_rows)==len(a.skills),"results":test_rows})


if __name__=="__main__":
    main()
