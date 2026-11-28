"""FG-06: inference-only elastic draft clocks and read-only prefix-bank transport."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import time

import torch

from . import gym as G

SCHEMA = "hybrid.gsf.floodgates.fg06.v1"
MODES = ("first", "deep")
ARMS = ("single", "plain2", "plain4", "rlt4", "damp4", "cycle4", "reverse4", "cycle_rlt4")
PANELS = ("cal", "dev", "shift", "trace_train")
OFFSET = 20000
RANK = 4
DEFAULT = dict(G.DEFAULT, seeds=[137, 163])
CONTRASTS = tuple((a,"single") for a in ARMS if a!="single") + (
    ("rlt4","plain4"),("rlt4","damp4"),("cycle4","plain4"),
    ("cycle4","reverse4"),("cycle_rlt4","cycle4"),("cycle_rlt4","rlt4"))


def source_manifest():
    here=Path(__file__).resolve().parent
    return {n:hashlib.sha256((here/n).read_bytes()).hexdigest()
            for n in ("gym.py","fg06_elastic.py","FG06_PROTOCOL.md")}


def partitions(cfg):
    n={"base_train":cfg["base_n"],"adapter_train":cfg["adapter_n"]}
    sets={s:[G.world(s,OFFSET+i) for i in range(n.get(s,cfg["eval_n"]))] for s in G.SPLITS}
    seen=set()
    for ws in sets.values():
        keys={tuple(w["prefix"]) for w in ws}
        if seen&keys or len(keys)!=len(ws):
            raise ValueError("prefix lineage overlap")
        seen.update(keys)
    return sets


def hashes(sets):
    return {s:G.digest([w["root"] for w in ws]) for s,ws in sets.items()}


def schedule(mode,arm):
    if mode not in MODES or arm not in ARMS:
        raise ValueError("invalid elastic policy")
    b=0 if mode=="first" else G.LOOPS-1
    passes=1 if arm=="single" else 2 if arm=="plain2" else 4
    if arm in ("cycle4","cycle_rlt4"):
        depths=[(b+i)%G.LOOPS for i in range(passes)]
    elif arm=="reverse4":
        depths=[(b-i)%G.LOOPS for i in range(passes)]
    else:
        depths=[b]*passes
    alpha=[1.]*passes
    if arm in ("rlt4","cycle_rlt4"):
        alpha=[1.,1/3,1/2,1.]
    elif arm=="damp4":
        alpha=[1.,5/12,5/12,1.]
    return tuple(zip(depths,alpha,strict=True))


@torch.no_grad()
def draft(model,adapter,anchor,past,k,mode,arm):
    if not 1<=k<=G.BLOCK or len(past)!=G.LOOPS:
        raise ValueError("invalid draft state")
    offset=past[0][0].shape[1]
    if any(a.shape!=b.shape or a.shape[1]!=offset for a,b in past):
        raise ValueError("inconsistent cache lengths")
    positions=torch.arange(offset+1,offset+k+1,device=anchor.device)
    masks=adapter.mask[None,None,:]+model.position(positions)[None,:,:]
    h=torch.cat((model.embed(anchor,offset),masks.expand(anchor.shape[0],-1,-1)),dim=1)
    rows=[]
    for i,(depth,alpha) in enumerate(schedule(mode,arm)):
        kv=past[depth]
        proposal,_=model.block(h,kv,adapter if i==0 else None)
        h=proposal if alpha==1. else h+alpha*(proposal-h)
        z=model.head(h[:,1:])
        rows.append({"pass":i+1,"cache_depth":depth,"alpha":alpha,
                     "adapter_active":i==0,"draft_ids":z.argmax(-1).tolist(),
                     "block_token_visits":h.shape[0]*h.shape[1],
                     "logical_prefix_kv_bytes":sum(x.numel()*x.element_size() for x in kv)})
    return z,rows


@torch.no_grad()
def generate(model,adapter,mode,arm,prefix):
    tokens=list(prefix)
    _,past=model(torch.tensor([tokens[:-1]]))
    rounds=[]
    work=Counter()
    start=time.perf_counter()
    while len(tokens)<G.PREFIX+G.GENERATE:
        remaining=G.PREFIX+G.GENERATE-len(tokens)
        k=min(G.BLOCK,remaining)
        proposal,passes=draft(model,adapter,torch.tensor([[tokens[-1]]]),past,k,mode,arm)
        guesses=proposal[0].argmax(-1).tolist()
        tokens,past,receipt=G.verify_step(model,tokens,past,guesses,remaining)
        # The base verifier helper charges a one-pass draft. Replace it
        # with the actual, fully accounted executed draft path.
        receipt["draft_block_token_visits"]=sum(r["block_token_visits"] for r in passes)
        receipt["draft_passes"]=len(passes)
        receipt["logical_draft_prefix_kv_bytes"]=sum(r["logical_prefix_kv_bytes"] for r in passes)
        receipt["serialized_block_calls"]=G.LOOPS+len(passes)
        receipt["draft_trace"]=passes
        rounds.append(receipt)
        for key in ("verify_calls","verify_block_token_visits","draft_block_token_visits",
                    "draft_passes","accepted_prefix","logical_draft_prefix_kv_bytes",
                    "serialized_block_calls","discarded_cache_positions"):
            work[key]+=receipt[key]
        work["draft_calls"]+=1
        work["draft_tokens"]+=len(guesses)
        work["emitted_tokens"]+=len(receipt["emitted"])
        work["peak_provisional_kv_bytes"]=max(work["peak_provisional_kv_bytes"],receipt["peak_provisional_kv_bytes"])
    elapsed=time.perf_counter()-start
    if work["emitted_tokens"]!=G.GENERATE or work["draft_passes"]!=len(schedule(mode,arm))*len(rounds):
        raise AssertionError("work conservation")
    return tokens,rounds,dict(work),elapsed


def row_for(model,adapter,mode,arm,w,ref):
    tokens,rounds,work,seconds=generate(model,adapter,mode,arm,w["prefix"])
    if tokens!=ref:
        raise AssertionError(f"exact greedy mismatch {w['root']} {mode}/{arm}")
    truth=G.truth(w)[G.PREFIX:]
    output=tokens[G.PREFIX:]
    return {"public":w,"draft_steps":rounds,"work":work,"verified_output":output,
            "reference_output":ref[G.PREFIX:],"truth_output":truth,"greedy_parity":True,
            "truth_correct_tokens":sum(a==b for a,b in zip(output,truth,strict=True))},seconds


def aggregate(rows):
    total=Counter()
    eligible=[0]*G.BLOCK
    accepted=[0]*G.BLOCK
    for r in rows:
        for k,v in r["work"].items():
            total[k]=max(total[k],v) if k.startswith("peak_") else total[k]+v
        total["truth_correct_tokens"]+=r["truth_correct_tokens"]
        total["truth_exact_sequences"]+=r["truth_output"]==r["verified_output"]
        for step in r["draft_steps"]:
            for j in range(len(step["draft"])):
                eligible[j]+=1
                accepted[j]+=step["accepted_prefix"]>=j+1
    n=len(rows)
    serial_calls=n*G.GENERATE*G.LOOPS
    return {"n":n,**dict(total),"serial_verify_calls":n*G.GENERATE,
            "serial_block_calls":serial_calls,"serial_block_token_visits":serial_calls,
            "total_block_token_visits":total["verify_block_token_visits"]+total["draft_block_token_visits"],
            "prefill_block_token_visits":n*(G.PREFIX-1)*G.LOOPS,
            "emitted_per_verification":total["emitted_tokens"]/total["verify_calls"],
            "serialized_call_reduction":serial_calls/total["serialized_block_calls"],
            "truth_accuracy":total["truth_correct_tokens"]/(n*G.GENERATE),
            "acceptance_survival":{"eligible":eligible,"accepted":accepted}}


def evaluate(model,adapter,mode,arm,worlds,refs):
    rows=[]
    elapsed=0.
    for w in worlds:
        r,t=row_for(model,adapter,mode,arm,w,refs[w["root"]])
        rows.append(r)
        elapsed+=t
    return {**aggregate(rows),"draft_verify_seconds":elapsed},rows


def paired(left,right):
    if [r["public"]["root"] for r in left]!=[r["public"]["root"] for r in right]:
        raise ValueError("unpaired roots")
    c=Counter()
    for x,y in zip(left,right,strict=True):
        a,b=x["work"],y["work"]
        d=b["serialized_block_calls"]-a["serialized_block_calls"]
        c["n"]+=1
        c["serialized_calls_saved"]+=d
        c["target_calls_saved"]+=b["verify_calls"]-a["verify_calls"]
        c["block_token_visits_saved"]+=b["verify_block_token_visits"]+b["draft_block_token_visits"]-a["verify_block_token_visits"]-a["draft_block_token_visits"]
        c["wins"]+=d>0
        c["ties"]+=d==0
        c["losses"]+=d<0
    return dict(c)


def record(seed,mode,arm,target_hash,adapter_hash,row):
    r={"schema":SCHEMA+".rollout","split":"trace_train","root":row["public"]["root"],
       "seed":seed,"mode":mode,"arm":arm,"rank":RANK,
       "target_weights_hash":target_hash,"adapter_weights_hash":adapter_hash,
       **row,"training_ready":False}
    r["receipt_hash"]=G.digest(r)
    return r


def run(out,cfg=None):
    cfg=dict(DEFAULT if cfg is None else cfg)
    G.validate(cfg)
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    sets=partitions(cfg)
    report={"schema":SCHEMA,"status":"public_development_only","config":cfg,
            "source_manifest":source_manifest(),"split_hashes":hashes(sets),
            "targets":[],"adapters":[],"cells":[],"paired":[]}
    path=out/"fg06_rollouts.jsonl"
    with path.open("w",encoding="utf-8") as trace:
        for seed in cfg["seeds"]:
            model,t=G.fit_target("final_only",seed,sets["base_train"],cfg)
            t.update(seed=seed,**G.save(out,f"{seed}-target.pt",model))
            report["targets"].append(t)
            refs={s:{w["root"]:G.serial(model,w["prefix"]) for w in sets[s]} for s in PANELS}
            for mode in MODES:
                adapter,info=G.fit_adapter(model,seed,mode,RANK,sets["adapter_train"],cfg)
                adapter.requires_grad_(False)
                info.update(seed=seed,mode=mode,rank=RANK,
                            **G.save(out,f"{seed}-{mode}-adapter.pt",adapter))
                report["adapters"].append(info)
                results={}
                for arm in ARMS:
                    cell={"seed":seed,"mode":mode,"arm":arm,
                          "target_weights_hash":t["weights_hash"],
                          "adapter_weights_hash":info["weights_hash"],
                          "schedule":[list(x) for x in schedule(mode,arm)],"panels":{}}
                    results[arm]={}
                    for split in PANELS:
                        stats,rows=evaluate(model,adapter,mode,arm,sets[split],refs[split])
                        cell["panels"][split]=stats
                        results[arm][split]=rows
                        if split=="trace_train":
                            for r in rows:
                                obj=record(seed,mode,arm,t["weights_hash"],info["weights_hash"],r)
                                trace.write(json.dumps(obj,sort_keys=True,allow_nan=False)+"\n")
                    report["cells"].append(cell)
                for a,b in CONTRASTS:
                    for split in PANELS:
                        report["paired"].append({"seed":seed,"mode":mode,"arm":a,"control":b,
                              "split":split,**paired(results[a][split],results[b][split])})
                if G.weight_hash(adapter)!=info["weights_hash"] or G.weight_hash(model)!=t["weights_hash"]:
                    raise AssertionError("inference mutated frozen weights")
    report["trace_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/"fg06_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def audit(out):
    torch.set_num_threads(1)
    report=json.loads((out/"fg06_report.json").read_text(encoding="utf-8"))
    cfg=report["config"]
    G.validate(cfg)
    sets=partitions(cfg)
    if report["schema"]!=SCHEMA or report["source_manifest"]!=source_manifest() or report["split_hashes"]!=hashes(sets):
        raise ValueError("source/split identity")
    tr={r["seed"]:r for r in report["targets"]}
    ar={(r["seed"],r["mode"]):r for r in report["adapters"]}
    cells={(r["seed"],r["mode"],r["arm"]):r for r in report["cells"]}
    if (len(tr)!=len(report["targets"]) or set(tr)!=set(cfg["seeds"]) or
        len(ar)!=len(report["adapters"]) or set(ar)!=set(itertools.product(cfg["seeds"],MODES)) or
        len(cells)!=len(report["cells"]) or set(cells)!=set(itertools.product(cfg["seeds"],MODES,ARMS))):
        raise ValueError("cell coverage")
    targets={s:G.load(G.Target(),out,c) for s,c in tr.items()}
    adapters={}
    for key,c in ar.items():
        if c["target_hash"]!=tr[key[0]]["weights_hash"] or c["rank"]!=RANK:
            raise ValueError("adapter target/rank identity")
        adapters[key]=G.load(G.DraftAdapter(RANK),out,c)
    for key,c in cells.items():
        if (c["target_weights_hash"]!=tr[key[0]]["weights_hash"] or
            c["adapter_weights_hash"]!=ar[key[:2]]["weights_hash"] or
            c["schedule"]!=[list(x) for x in schedule(*key[1:])]):
            raise ValueError("schedule or weight binding")
    refs={seed:{w["root"]:G.serial(model,w["prefix"]) for w in sets["trace_train"]}
          for seed,model in targets.items()}
    worlds={w["root"]:w for w in sets["trace_train"]}
    path=out/"fg06_rollouts.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report["trace_sha256"]:
        raise ValueError("trace inventory")
    seen=set()
    rows={k:[] for k in cells}
    for line in path.read_text(encoding="utf-8").splitlines():
        r=json.loads(line)
        bound=r.pop("receipt_hash")
        key=(r["seed"],r["mode"],r["arm"])
        ident=(key,r["root"])
        if (G.digest(r)!=bound or r["schema"]!=SCHEMA+".rollout" or
            r["split"]!="trace_train" or r["training_ready"] is not False or
            key not in cells or r["root"] not in worlds or ident in seen):
            raise ValueError("trace custody")
        w=worlds[r["root"]]
        replay,_=row_for(targets[key[0]],adapters[key[:2]],key[1],key[2],w,refs[key[0]][r["root"]])
        if record(*key,tr[key[0]]["weights_hash"],ar[key[:2]]["weights_hash"],replay)["receipt_hash"]!=bound:
            raise ValueError("elastic decision/work replay")
        rows[key].append(replay)
        seen.add(ident)
    if len(seen)!=len(cells)*len(worlds):
        raise ValueError("incomplete traces")
    for key,rs in rows.items():
        expected={k:v for k,v in cells[key]["panels"]["trace_train"].items() if k!="draft_verify_seconds"}
        if aggregate(rs)!=expected:
            raise ValueError("trace aggregate mismatch")
    for s,m in itertools.product(cfg["seeds"],MODES):
        for a,b in CONTRASTS:
            expected={"seed":s,"mode":m,"arm":a,"control":b,"split":"trace_train",
                      **paired(rows[s,m,a],rows[s,m,b])}
            if expected not in report["paired"]:
                raise ValueError("paired contrast replay")
    return {"status":"audited_development_only","targets":len(tr),"adapters":len(ar),
            "cells":len(cells),"rollouts":len(seen),"greedy_mismatches":0}


@torch.no_grad()
def selftest():
    torch.set_num_threads(1)
    torch.manual_seed(137)
    model=G.Target().eval().requires_grad_(False)
    adapter=G.DraftAdapter(RANK).eval().requires_grad_(False)
    w=G.world("dev",OFFSET)
    anchor=torch.tensor([[w["prefix"][-1]]])
    _,past=model(torch.tensor([w["prefix"][:-1]]))
    saved=tuple((a.clone(),b.clone()) for a,b in past)
    ref=G.serial(model,w["prefix"])
    th,ah=G.weight_hash(model),G.weight_hash(adapter)
    for mode in MODES:
        x=model.draft(anchor,past,adapter,G.BLOCK,mode)
        y,steps=draft(model,adapter,anchor,past,G.BLOCK,mode,"single")
        assert torch.equal(x,y) and len(steps)==1
        assert sorted(x[0] for x in schedule(mode,"cycle4"))==sorted(x[0] for x in schedule(mode,"reverse4"))
        assert abs(sum(x[1] for x in schedule(mode,"rlt4"))-sum(x[1] for x in schedule(mode,"damp4")))<1e-12
        for arm in ARMS:
            z,rows=draft(model,adapter,anchor,past,G.BLOCK,mode,arm)
            assert rows[0]["draft_ids"]==x.argmax(-1).tolist()
            assert sum(r["adapter_active"] for r in rows)==1
            r,_=row_for(model,adapter,mode,arm,w,ref)
            assert r["greedy_parity"]
    for x,y in zip(past,saved,strict=True):
        assert all(torch.equal(a,b) for a,b in zip(x,y,strict=True))
    assert G.weight_hash(model)==th and G.weight_hash(adapter)==ah
    print("FG-06 common first pass, cache schedule, immutable weights and greedy parity passed")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path)
    p.add_argument("--audit",type=Path)
    p.add_argument("--selftest",action="store_true")
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    torch.set_num_threads(1)
    if a.selftest:
        selftest()
    if a.out:
        cfg=dict(DEFAULT,base_steps=4,adapter_steps=4,batch=4,base_n=16,adapter_n=8,eval_n=4,seeds=[137]) if a.smoke else None
        r=run(a.out,cfg)
        print(json.dumps({"status":r["status"],"targets":len(r["targets"]),"adapters":len(r["adapters"]),"cells":len(r["cells"])}))
    if a.audit:
        print(json.dumps(audit(a.audit),sort_keys=True))
    if not (a.selftest or a.out or a.audit):
        p.error("choose --selftest, --out or --audit")


if __name__=="__main__":
    main()
