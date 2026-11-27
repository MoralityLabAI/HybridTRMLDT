"""GSF-05: trainable graph-local loop schedules, paired work, adaptive exits.

LSA-inspired finite development fixture, not a reproduction of full HRM/MoLT.
The input is a pair of permutations, a start node and an explicit traversal
program. The exact solver is evaluator-owned. All neural operators have the same
schema and parameter counts; order, register writes and visit frequency vary.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

import torch
from torch import nn

SCHEMA = "hybrid.gsf.loop_clocks.v1"
N = 8
TICKS = 8
SPLITS = ("train", "cal", "dev", "shift", "trace_train")
WORDS = {"flat_ab": "ABABABAB", "coupled_ab": "ABABABAB",
         "coupled_grouped": "AABBAABB", "hier_fast": "AAABAAAB",
         "hier_reverse": "BAAABAAA"}
ARMS = tuple(WORDS)
THRESHOLDS = (.5, .65, .8, .9, .95, .99, 1.01)
FAMILIES = ("chain", "compose")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def world(split, seed, index):
    if split not in SPLITS:
        raise ValueError(split)
    root = digest([SCHEMA, split, seed, index])
    rng = random.Random(int(root,16))
    maps = [rng.sample(range(N),N) for _ in range(2)]
    family = FAMILIES[index%2]
    depth = rng.randint(5,8) if split=="shift" else rng.randint(1,4)
    if family=="chain" or depth==1:
        a,b = (depth,0) if rng.randrange(2)==0 else (0,depth)
    else:
        a=rng.randint(1,depth-1)
        b=depth-a
    reverse=bool(rng.randrange(2))
    return {"schema":SCHEMA,"root":root,"split":split,"seed":seed,"index":index,
            "family":family,"maps":maps,"start":rng.randrange(N),
            "a":a,"b":b,"reverse":reverse}


def program(w):
    return (("B",)*w["b"]+("A",)*w["a"]) if w["reverse"] else (("A",)*w["a"]+("B",)*w["b"])


def exact(w):
    node=w["start"]
    path=[node]
    for op in program(w):
        node=w["maps"][op=="B"][node]
        path.append(node)
    return path


def independent_target(w):
    # Permutation composition over the full universe, independent of the
    # single-node path traversal used to produce the reference receipt.
    state=list(range(N))
    for op in program(w):
        mapping=w["maps"][int(op=="B")]
        state=[mapping[j] for j in state]
    return state[w["start"]]


def inputs(worlds):
    # Graph transport via inverse permutation: gather the unique incoming node.
    inverse=[]
    x=[]
    for w in worlds:
        inv=[[m.index(i) for i in range(N)] for m in w["maps"]]
        inverse.append(inv)
        x.append([[float(i==w["start"]),w["a"]/TICKS,w["b"]/TICKS,
                   float(w["reverse"])] for i in range(N)])
    return torch.tensor(x,dtype=torch.float32),torch.tensor(inverse,dtype=torch.long)


class ClockModel(nn.Module):
    def __init__(self,width,arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm=arm
        self.width=width
        self.encoder=nn.Linear(4,width)
        self.blocks=nn.ModuleDict({op:nn.Linear(5*width,width) for op in "AB"})
        self.head=nn.Linear(2*width,1)

    def initialize(self,x):
        z=torch.tanh(self.encoder(x))
        return z,z,z

    def tick(self,registers,edges,op):
        z,f,s=registers
        flat=self.arm=="flat_ab"
        h,other=(f,s) if op=="A" or flat else (s,f)
        msg=[torch.gather(h,1,edges[:,j,:,None].expand(-1,-1,self.width)) for j in range(2)]
        update=torch.tanh(self.blocks[op](torch.cat((z,h,other,*msg),dim=-1)))
        h=h+update
        if flat:
            f=s=h
        elif op=="A":
            f=h
        else:
            s=h
        return (z,f,s),self.head(torch.cat((f,s),dim=-1)).squeeze(-1)

    def forward(self,x,edges,word=None):
        word=word or WORDS[self.arm]
        state=self.initialize(x)
        scores=[]
        for op in word:
            state,z=self.tick(state,edges,op)
            scores.append(z)
        return torch.stack(scores,dim=1)


def weight_hash(model):
    return digest({k:v.detach().cpu().tolist() for k,v in model.state_dict().items()})


def fit(arm,width,seed,worlds,steps,batch):
    torch.manual_seed(seed)
    model=ClockModel(width,arm)
    before=weight_hash(model)
    x,edges=inputs(worlds)
    y=torch.tensor([independent_target(w) for w in worlds])
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    rng=torch.Generator().manual_seed(seed+103)
    for _ in range(steps):
        idx=torch.randint(len(x),(batch,),generator=rng)
        logits=model(x[idx],edges[idx])
        final=nn.functional.cross_entropy(logits[:,-1],y[idx])
        all_sites=nn.functional.cross_entropy(logits.flatten(0,1),y[idx].repeat_interleave(TICKS))
        loss=final+.2*all_sites
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    after=weight_hash(model)
    if before==after:
        raise RuntimeError("unchanged weights")
    params=sum(p.numel() for p in model.parameters())
    return model,{"initial_hash":before,"weights_hash":after,"params":params,
                  "train_ticks":steps*batch*TICKS,"train_node_visits":steps*batch*TICKS*N,
                  "updates":steps,"word":WORDS[arm]}


@torch.no_grad()
def propose(model,w,threshold=None,word=None):
    x,edges=inputs([w])
    state=model.initialize(x)
    visits=[]
    word=word or WORDS[model.arm]
    for i,op in enumerate(word):
        state,z=model.tick(state,edges,op)
        p=z.softmax(-1)[0]
        action=int(p.argmax())
        confidence=float(p[action])
        visits.append({"tick":i+1,"operator":op,"action":action,"confidence":confidence})
        if threshold is not None and confidence>=threshold:
            break
    return visits


def grade(w,visits):
    path=exact(w)
    target=independent_target(w)
    if path[-1]!=target:
        raise AssertionError("reference mismatch")
    if not visits or not 0<=visits[-1]["action"]<N:
        raise ValueError("invalid proposal")
    return {"target":target,"correct":visits[-1]["action"]==target,
            "exact_path":path,"exact_hops":len(path)-1,"ticks":len(visits),
            "node_visits":len(visits)*N,"message_gathers":2*len(visits)*N}


def evaluate(model,worlds,threshold=None,word=None,trace=None,seed=None,policy=None,identity=None):
    counts=Counter()
    by_family={f:Counter() for f in FAMILIES}
    start=time.process_time()
    for w in worlds:
        visits=propose(model,w,threshold,word)
        g=grade(w,visits)
        for c in (counts,by_family[w["family"]]):
            c["n"]+=1
            c["correct"]+=g["correct"]
            c["ticks"]+=g["ticks"]
            c["node_visits"]+=g["node_visits"]
            c["message_gathers"]+=g["message_gathers"]
            c["exact_hops"]+=g["exact_hops"]
        if trace is not None:
            if w["split"]!="trace_train":
                raise ValueError("trace custody")
            row={"schema":SCHEMA+".rollout","split":"trace_train","root":w["root"],
                 "seed":seed,"arm":model.arm,"policy":policy,"threshold":threshold,
                 "word":word or WORDS[model.arm],"identity":identity,"public":w,
                 "visits":visits,"receipt":g,"training_ready":False,
                 "target_kind":"verified_success" if g["correct"] else "schedule_repair"}
            row["receipt_hash"]=digest(row)
            trace.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
    def stats(c):
        return {**c,"accuracy":c["correct"]/c["n"],"mean_ticks":c["ticks"]/c["n"]}
    return {**stats(counts),"families":{f:stats(c) for f,c in by_family.items()},
            "cpu_seconds_including_grading":time.process_time()-start}


def calibrate(model,worlds):
    # Choose from the frozen CAL threshold grid with a one percentage point
    # raw accuracy guardrail relative to this checkpoint's fixed eight ticks.
    fixed=evaluate(model,worlds)
    trials=[]
    for threshold in THRESHOLDS:
        m=evaluate(model,worlds,threshold)
        trials.append({"threshold":threshold,"accuracy":m["accuracy"],"ticks":m["ticks"]})
    eligible=[t for t in trials if t["accuracy"]>=fixed["accuracy"]-.01]
    chosen=min(eligible,key=lambda t:(t["ticks"],-t["accuracy"],t["threshold"]))
    return chosen["threshold"],trials


def run(out,width=24,steps=300,batch=32,train_n=512,eval_n=64,seeds=(137,163)):
    if not (1<=width<=64 and 1<=steps<=1024 and 1<=batch<=128
            and 2<=train_n<=2048 and 2<=eval_n<=512 and
            1<=len(seeds)<=8 and len(set(seeds))==len(seeds)):
        raise ValueError("CPU budget outside contract")
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    cfg=dict(width=width,steps=steps,batch=batch,train_n=train_n,eval_n=eval_n,seeds=list(seeds))
    report={"schema":SCHEMA,"status":"public_development_only","config":cfg,"cells":[]}
    path=out/"clock_rollouts.jsonl"
    with path.open("w",encoding="utf-8") as trace:
        for seed in seeds:
            sets={s:[world(s,seed,i) for i in range(train_n if s=="train" else eval_n)] for s in SPLITS}
            seen=set()
            for split,ws in sets.items():
                # Group all queries on the same graph pair; reject cross-split
                # graph collisions rather than silently credit distinct roots.
                graphs={digest(w["maps"]) for w in ws}
                if seen&graphs:
                    raise ValueError("graph lineage overlap")
                seen.update(graphs)
            for arm in ARMS:
                model,info=fit(arm,width,seed,sets["train"],steps,batch)
                threshold,trials=calibrate(model,sets["cal"])
                artifact=out/f"{seed}-{arm}.pt"
                torch.save(model.state_dict(),artifact)
                row={"seed":seed,"arm":arm,**info,"threshold":threshold,"calibration":trials,
                     "artifact":artifact.name,"artifact_sha256":hashlib.sha256(artifact.read_bytes()).hexdigest()}
                # Same fitted weights, fixed architecture, schedule-order swap.
                # Coupled AB/grouped swap preserves A/B counts. Hier swaps also
                # preserve A6/B2. Flat has only the prefix budget diagnostic.
                swap={"coupled_ab":"AABBAABB","coupled_grouped":"ABABABAB",
                      "hier_fast":"BAAABAAA","hier_reverse":"AAABAAAB"}.get(arm)
                row["swap_word"]=swap
                for split in ("cal","dev","shift"):
                    ws=sets[split]
                    row[split]={"fixed":evaluate(model,ws),"adaptive":evaluate(model,ws,threshold),
                                "prefix4":evaluate(model,ws,word=WORDS[arm][:4])}
                    if swap:
                        row[split]["swap"]=evaluate(model,ws,word=swap)
                ws=sets["trace_train"]
                for policy,th in (("fixed",None),("adaptive",threshold)):
                    row["trace_"+policy]=evaluate(model,ws,th,trace=trace,seed=seed,policy=policy,
                                                  identity=info["weights_hash"])
                report["cells"].append(row)
    report["trace_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/"clock_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def audit(out):
    report=json.loads((out/"clock_report.json").read_text(encoding="utf-8"))
    cfg=report["config"]
    cells={(r["seed"],r["arm"]):r for r in report["cells"]}
    if len(cells)!=len(report["cells"]) or set(cells)!=set(itertools.product(cfg["seeds"],ARMS)):
        raise ValueError("cell coverage")
    models={}
    for key,c in cells.items():
        model=ClockModel(cfg["width"],c["arm"])
        path=out/c["artifact"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=c["artifact_sha256"]:
            raise ValueError("artifact mismatch")
        model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
        model.eval()
        if weight_hash(model)!=c["weights_hash"]:
            raise ValueError("weight identity")
        if c["word"]!=WORDS[c["arm"]]:
            raise ValueError("schedule identity")
        cal=[world("cal",c["seed"],i) for i in range(cfg["eval_n"])]
        threshold,trials=calibrate(model,cal)
        if threshold!=c["threshold"] or trials!=c["calibration"]:
            raise ValueError("CAL stopping policy mismatch")
        models[key]=model
    traces=out/"clock_rollouts.jsonl"
    if hashlib.sha256(traces.read_bytes()).hexdigest()!=report["trace_sha256"]:
        raise ValueError("trace inventory")
    seen=set()
    counts=Counter()
    for line in traces.read_text(encoding="utf-8").splitlines():
        r=json.loads(line)
        bound=r.pop("receipt_hash")
        p=r["public"]
        key=(r["seed"],r["arm"])
        i=p["index"]
        ident=(key,r["policy"],i)
        if (digest(r)!=bound or r["schema"]!=SCHEMA+".rollout" or r["split"]!="trace_train"
                or r["training_ready"] is not False or key not in cells
                or r["policy"] not in ("fixed","adaptive") or not 0<=i<cfg["eval_n"]
                or ident in seen):
            raise ValueError("rollout custody")
        w=world("trace_train",r["seed"],i)
        c=cells[key]
        th=None if r["policy"]=="fixed" else c["threshold"]
        if (w!=p or r["root"]!=w["root"] or r["identity"]!=c["weights_hash"]
                or r["word"]!=c["word"] or r["threshold"]!=th):
            raise ValueError("packet/weight/schedule mismatch")
        replay=propose(models[key],w,th)
        g=grade(w,replay)
        if replay!=r["visits"] or g!=r["receipt"]:
            raise ValueError("policy/target replay mismatch")
        kind="verified_success" if g["correct"] else "schedule_repair"
        if kind!=r["target_kind"]:
            raise ValueError("target kind mismatch")
        seen.add(ident)
        counts[kind]+=1
    if len(seen)!=len(cells)*2*cfg["eval_n"]:
        raise ValueError("missing rollout")
    return {"status":"audited_development_only","fits":len(cells),"rollouts":len(seen),
            "targets":dict(counts)}


def selftest():
    for split in SPLITS:
        for i in range(20):
            w=world(split,137,i)
            assert exact(w)[-1]==independent_target(w)
            assert len(program(w))==(w["a"]+w["b"])
    ws=[world("train",137,i) for i in range(16)]
    hashes=set()
    params=set()
    for arm in ARMS:
        m,r=fit(arm,8,137,ws,2,4)
        hashes.add(r["initial_hash"])
        params.add(r["params"])
        assert len(propose(m,ws[0]))==TICKS
        assert len(propose(m,ws[0],0.))==1
        assert len(propose(m,ws[0],1.01))==TICKS
    assert len(hashes)==len(params)==1
    print("GSF-05 matched clocks and exact solver selftest passed")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path)
    p.add_argument("--audit",type=Path)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--selftest",action="store_true")
    a=p.parse_args()
    torch.set_num_threads(1)
    if a.selftest:
        selftest()
    if a.out:
        kwargs=dict(width=8,steps=4,batch=8,train_n=32,eval_n=8,seeds=(137,)) if a.smoke else {}
        r=run(a.out,**kwargs)
        print(json.dumps({"cells":len(r["cells"]),"status":r["status"]}))
    if a.audit:
        print(json.dumps(audit(a.audit),sort_keys=True))
    if not (a.selftest or a.out or a.audit):
        p.error("choose --selftest, --out or --audit")


if __name__=="__main__":
    main()
