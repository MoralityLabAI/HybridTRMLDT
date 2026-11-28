"""SKILL-02D: public certificate dispatch with actual skipped neural computation.

This is an inference-only composition experiment on graph skills. Model receives
no target, aux, verifier verdict or certificate feature during inference. A
bounded public certificate constrains final decisions; fully determined cases
bypass the network. Grade after all decisions and meter symbolic work separately.
"""
import argparse
from collections import deque
from pathlib import Path
import json
import time

import numpy as np
import torch

from research_gym.trm_skill_gym import modules as M, tasks as T, campaign as C
from .core import graph_certificate
from .runner import SOURCE, GRAPH, pools, checkpoint

ROOT=2_200_000


def graph_inputs(b,i,skill):
    active=torch.where(b["mask"][i]>0)[0].tolist()
    r=b["rel"][i].numpy()
    parents=[[j for j,v in enumerate(active) if r[u,v]>0] for u in active]
    q=None
    if skill=="retrieval":
        query=[j for j,v in enumerate(active) if b["cand"][i,v,0]==1]
        if len(query)!=1:
            raise ValueError("expected one public query")
        q=query[0]
    return active,parents,q


def exact_graph(skill,parents,q):
    """Linear-time graph baseline; no labels. Edge work excludes graph extraction."""
    n=len(parents)
    if skill=="retrieval":
        seen={q}
        todo=[q]
        work=0
        while todo:
            u=todo.pop()
            for v in parents[u]:
                work+=1
                if v not in seen:
                    seen.add(v)
                    todo.append(v)
        return [1 if i in seen and i!=q else -1 for i in range(n)],work
    if skill!="deduction":
        raise ValueError(skill)
    children=[[] for _ in parents]
    deg=list(map(len,parents))
    work=0
    for u,ps in enumerate(parents):
        for v in ps:
            children[v].append(u)
            work+=1
    queue=deque(i for i in range(n) if deg[i]==0)
    st=[0]*n
    blocked=[False]*n
    while queue:
        u=queue.popleft()
        st[u]=-1 if blocked[u] else 1
        for v in children[u]:
            work+=1
            blocked[v]|=st[u]==1
            deg[v]-=1
            if deg[v]==0:
                queue.append(v)
    if any(s==0 for s in st):
        raise ValueError("deduction exact solver requires a DAG")
    return st,work


@torch.no_grad()
def neural(model,cfg,skill,public):
    """Physical model execution with no verifier calls and no target-bearing fields."""
    cand=public["cand"].clone()
    cand[...,T.FS:]=0
    st=model.init_state(public)
    pred=None
    for _ in range(cfg["visits"]):
        logits,conf,value,st=model.visit(cand,public,st,cfg["visits"])
        pred=T.decide(skill,logits,public["mask"])
    return pred


@torch.no_grad()
def run_arm(model,cfg,skill,b,mode,rounds=0):
    t0=time.process_time()
    n=len(b["mask"])
    # Full certificate path solves all slots without any calls to the network.
    cert=torch.zeros_like(b["mask"])
    work=0
    if mode!="neural":
        for i in range(n):
            active,parents,q=graph_inputs(b,i,skill)
            c,w=(exact_graph(skill,parents,q) if mode=="exact" else
                 graph_certificate(skill,parents,q,rounds))
            work+=w
            cert[i,active]=torch.tensor(c,dtype=cert.dtype)
    unresolved=((cert==0)&(b["mask"]>0)).any(1)
    pred=(cert>0).float()
    flops=0
    if unresolved.any():
        if mode=="exact":
            raise AssertionError("exact solver left unknowns")
        pub={k:b[k][unresolved] for k in ("cand","rel","glob","mask")}
        M.flops_reset()
        proposed=neural(model,cfg,skill,pub)
        flops=M.flops_read()
        c=cert[unresolved]
        pred[unresolved]=torch.where(c==0,proposed,(c>0).float())
    elapsed=time.process_time()-t0
    # Scoring starts only after the inference timer stops.
    acc=T.exact(skill,b,pred).numpy()
    known=(cert!=0)&(b["mask"]>0)
    wrong=int((known&((cert>0)!=(b["label"]>0))).sum())
    return acc,{"exact":float(acc.mean()),"n":n,"nn_instances":int(unresolved.sum()),
                "bypass_fraction":float((~unresolved).float().mean()),
                "model_flops_per_instance":flops/n,"symbolic_edge_exams_per_instance":work/n,
                "certificate_coverage":float(known.sum()/b["mask"].sum()),
                "certificate_violations":wrong,"inference_cpu_s":elapsed,
                "online_verifier_calls":0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=SOURCE)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--count",type=int,default=128)
    p.add_argument("--seeds",type=int,nargs="+",default=[0,1])
    a=p.parse_args()
    if not 1<=a.count<=1024:
        p.error("count must be 1..1024")
    torch.set_num_threads(1)
    results={}
    for skill in GRAPH:
        test=pools(skill,ROOT,a.count,True)
        for seed in a.seeds:
            model,cfg=checkpoint(a.source,skill,"D_trm_direct",seed)
            rows={}
            for k,b in test.items():
                rows[k]={}
                for name,mode,r in [("neural","neural",0),("hybrid_1","hybrid",1),
                     ("hybrid_2","hybrid",2),("hybrid_4","hybrid",4),
                     ("hybrid_8","hybrid",8),("hybrid_16","hybrid",16),("exact","exact",0)]:
                    _,row=run_arm(model,cfg,skill,b,mode,r)
                    rows[k][name]=row
            results[f"{skill}/s{seed}"]=rows
    C.write_json(a.out/"hybrid.json",results)
    print(json.dumps({k:{arm:{"macro":float(np.mean([v[arm]["exact"] for v in rows.values()])),
                                  "bypass":float(np.mean([v[arm]["bypass_fraction"] for v in rows.values()])),
                                  "flops":float(np.mean([v[arm]["model_flops_per_instance"] for v in rows.values()])),
                                  "violations":sum(v[arm]["certificate_violations"] for v in rows.values())}
                         for arm in next(iter(rows.values()))} for k,rows in results.items()},indent=2))


if __name__=="__main__":
    main()
