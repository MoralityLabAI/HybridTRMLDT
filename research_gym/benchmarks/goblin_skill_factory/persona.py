"""GSF-04: external additive LDT heads under the native Dual-LDT membrane.

Public synthetic teachers and audited tag menus. No private Jinn/Beast labels
are embedded. Separate local-only specialist fitting lives in persona_private.py.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import nn

from research_gym.core.dual_ldt import ActionEvidence, BeastSnapshot, JinnHead, sort_with_jinn, total_variation

SCHEMA = "hybrid.gsf.dual_ldt.v1"
TAGS = tuple(f"synthetic_tag_{i:02}" for i in range(24))
CLASSES = ("train","hold","reject")
FRAMES = ("p0","p1")
ACTIONS = tuple(f"a{i}" for i in range(5))
SPLITS = ("train","cal","dev","shift","trace_train")
ARMS = ("external","pooled","conditioned_bias","uniform","exact")
LEARNED = ARMS[:3]


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def rng_for(obj):
    return random.Random(int(digest(obj),16))


def tag_partition(tags):
    # Partition by semantic tag set, before using it in any menu or frame.
    return SPLITS[int(digest([SCHEMA,sorted(tags)]),16)%len(SPLITS)]


def tag_pool(split):
    size = 5 if split == "shift" else 4
    pool=[tuple(c) for c in itertools.combinations(TAGS,size) if tag_partition(c)==split]
    rng_for([SCHEMA,"tag_pool_order",split]).shuffle(pool)
    return pool


def teacher(frame):
    if frame not in FRAMES:
        raise ValueError(frame)
    # Distinct exact additive teachers, synthetic weights fixed before fits.
    # Opposing tradeoffs make persona identity meaningful on the same menu.
    signs = [((i*7)%11-5)/4 for i in range(len(TAGS))]
    if frame == "p1":
        signs = [-v + ((i%3)-1)/4 for i,v in enumerate(signs)]
    weights = [signs, [((i*3)%7-3)/12 for i in range(len(TAGS))], [-v for v in signs]]
    return JinnHead.from_record({"classes":CLASSES,"feature_names":TAGS,
                                 "weights":weights,"bias":[.1,.3,-.1]})


TEACHERS = {f:teacher(f) for f in FRAMES}


def probs(head,tags):
    z = head.logits(frozenset(tags))
    top = max(z)
    p = [math.exp(v-top) for v in z]
    return [v/sum(p) for v in p]


def features(tags):
    active = set(tags)
    return [float(t in active) for t in TAGS]


def record(weights,bias,tags=TAGS):
    return {"classes":list(CLASSES),"feature_names":list(tags),
            "weights":weights.detach().tolist(),"bias":bias.detach().tolist()}


class Heads(nn.Module):
    def __init__(self,arm,n):
        super().__init__()
        if arm not in LEARNED:
            raise ValueError(arm)
        self.arm=arm
        self.a=nn.Linear(n,3)
        if arm=="external":
            self.b=nn.Linear(n,3)
        elif arm=="conditioned_bias":
            self.offset=nn.Parameter(torch.zeros(3))

    def forward(self,x,frame):
        a=self.a(x)
        if self.arm=="external":
            return torch.where(frame[:,None].bool(),self.b(x),a)
        if self.arm=="conditioned_bias":
            return a+frame[:,None]*self.offset
        return a

    def records(self,tags=TAGS):
        out={"p0":record(self.a.weight,self.a.bias,tags)}
        if self.arm=="external":
            out["p1"]=record(self.b.weight,self.b.bias,tags)
        elif self.arm=="conditioned_bias":
            out["p1"]=record(self.a.weight,self.a.bias+self.offset,tags)
        else:
            out["p1"]=out["p0"]
        return out


def fit_arrays(arm,seed,x,frames,soft,steps,batch,tags=TAGS):
    torch.manual_seed(seed)
    model=Heads(arm,len(tags))
    before=digest(model.records(tags))
    opt=torch.optim.AdamW(model.parameters(),lr=.01)
    rng=torch.Generator().manual_seed(seed+103)
    for _ in range(steps):
        idx=torch.randint(len(x),(batch,),generator=rng)
        z=model(x[idx],frames[idx])
        loss=-(soft[idx]*z.log_softmax(-1)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    records=model.records(tags)
    after=digest(records)
    if before==after:
        raise RuntimeError("unchanged fitted heads")
    return records,{"initial_hash":before,"weights_hash":after,"updates":steps,
                    "params":sum(p.numel() for p in model.parameters()),
                    "train_examples":len(x),"example_updates":steps*batch}


def fit(arm,seed,pool,steps,batch):
    x=torch.tensor([features(tags) for tags in pool for _ in FRAMES])
    frames=torch.tensor([i for _ in pool for i in range(2)])
    soft=torch.tensor([probs(TEACHERS[f],tags) for tags in pool for f in FRAMES])
    return fit_arrays(arm,seed,x,frames,soft,steps,batch)


def heads_from(records):
    return {f:JinnHead.from_record(records[f]) for f in FRAMES}


def uniform_heads():
    rec={"classes":CLASSES,"feature_names":TAGS,"weights":[[0.]*len(TAGS) for _ in CLASSES],"bias":[0.]*3}
    return {f:JinnHead.from_record(rec) for f in FRAMES}


def menu(split,index,pool):
    root=digest([SCHEMA,"menu",split,index])
    rng=rng_for(root)
    evidence={}
    for a in ACTIONS:
        tags=[list(rng.choice(pool))]
        if rng.random()<(.6 if split=="shift" else .2):
            tags.append(list(rng.choice(pool)))
        if rng.random()<(.12 if split=="shift" else .03):
            tags=[]
        evidence[a]={"positive":rng.random()<.9,"negative":rng.random()<(.3 if split=="shift" else .1),
                     "possible_tags":tags}
    return {"schema":SCHEMA,"root":root,"split":split,"index":index,
            "frame":rng.choice(FRAMES),"evidence":evidence,
            "forbidden":[TAGS[0],TAGS[1]],"tighten_tag":rng.choice(TAGS[2:]),
            "deny":rng.choice(ACTIONS),"halt_after":rng.random()<.2}


def phase_public(p,phase):
    forbidden=set(p["forbidden"])
    evidence={a:dict(e) for a,e in p["evidence"].items()}
    if phase>=1:
        forbidden.add(p["tighten_tag"])
        evidence[p["deny"]]["negative"]=True
    return evidence,forbidden,bool(phase==3 and p["halt_after"])


def native_snapshot(p,phase):
    ev,forbidden,halt=phase_public(p,phase)
    return BeastSnapshot(epoch=p["root"]+":"+str(phase),actions=ACTIONS,
                        forbidden_tags=frozenset(forbidden),
                        evidence={a:ActionEvidence(e["positive"],e["negative"],
                                    tuple(frozenset(t) for t in e["possible_tags"]))
                                  for a,e in ev.items()},halted=halt)


def independent_allowed(p,phase):
    ev,forbidden,halt=phase_public(p,phase)
    if halt:
        return set()
    return {a for a,e in ev.items() if e["positive"] and not e["negative"]
            and e["possible_tags"] and all(not (set(t)&forbidden) for t in e["possible_tags"])}


def episode(p,heads):
    phases=[]
    previous_receipt=None
    previous=None
    for phase in range(4):
        snapshot=native_snapshot(p,phase)
        frame=p["frame"] if phase<2 else FRAMES[1-FRAMES.index(p["frame"])]
        allowed=independent_allowed(p,phase)
        if snapshot.authorized()!=allowed:
            raise AssertionError("independent authorization mismatch")
        head=heads[frame]
        actual=sort_with_jinn(snapshot,head)
        target=sort_with_jinn(snapshot,TEACHERS[frame])
        if actual.permitted!=allowed or target.permitted!=allowed:
            raise AssertionError("persona changed authorization")
        receipt=snapshot.receipt(actual.action) if actual.action is not None else None
        checker=lambda epoch,a: epoch==snapshot.epoch and a in allowed
        executed=snapshot.verify_execution(receipt,checker)
        if executed!=(actual.action is not None) or (actual.action is not None and actual.action not in allowed):
            raise AssertionError("execution invariant")
        stale=None
        if previous_receipt is not None:
            stale=not snapshot.verify_execution(previous_receipt,checker)
            if not stale:
                raise AssertionError("accepted stale receipt")
        if phase==1:
            # Policy tightening holds the head fixed, so surviving odds are invariant.
            for a,b in itertools.combinations(sorted(allowed),2):
                before=previous.probabilities[a]/previous.probabilities[b]
                after=actual.probabilities[a]/actual.probabilities[b]
                if not math.isclose(before,after,rel_tol=1e-9,abs_tol=1e-9):
                    raise AssertionError("surviving odds changed")
        raw=max(ACTIONS,key=lambda a:(actual.original_probabilities[a],-ACTIONS.index(a)))
        target_score=lambda a: min(TEACHERS[frame].train_log_probability(t)
                                   for t in snapshot.evidence[a].possible_tags)
        regret=0. if actual.action is None else max(0.,target_score(target.action)-target_score(actual.action))
        phases.append({"phase":phase,"frame":frame,"epoch":snapshot.epoch,
                       "snapshot_digest":snapshot.digest(),"allowed":sorted(allowed),
                       "raw":raw,"raw_unauthorized":raw not in allowed,"selected":actual.action,
                       "probabilities":dict(actual.probabilities),"target":target.action,
                       "target_probabilities":dict(target.probabilities),
                       "preference_tax":actual.preference_tax,"regret":regret,
                       "tv":total_variation(actual.probabilities,target.probabilities),
                       "optimal":regret<=1e-10,"unsafe":False,"stale_rejected":stale,
                       "receipt":asdict(receipt) if receipt else None})
        previous_receipt=receipt
        previous=actual
    return phases


def evaluate(menus,heads,trace=None,seed=None,arm=None,identity=None):
    counts=Counter()
    regret=tv=0.
    start=time.process_time()
    for p in menus:
        phases=episode(p,heads)
        for x in phases:
            counts["phases"]+=1
            counts["viable"]+=bool(x["allowed"])
            counts["optimal"]+=x["optimal"]
            counts["viable_optimal"]+=bool(x["allowed"]) and x["optimal"]
            counts["raw_unauthorized"]+=x["raw_unauthorized"]
            counts["unsafe"]+=x["unsafe"]
            counts["stale_tests"]+=x["stale_rejected"] is not None
            counts["stale_rejected"]+=x["stale_rejected"] is True
            counts["halt_phases"]+=x["phase"]==3 and p["halt_after"]
            regret+=x["regret"]
            tv+=x["tv"]
        if trace is not None:
            if p["split"]!="trace_train":
                raise ValueError("trace split")
            row={"schema":SCHEMA+".rollout","split":"trace_train","root":p["root"],
                 "seed":seed,"arm":arm,"identity":identity,"public":p,
                 "phases":phases,"training_ready":False,
                 "target_kind":"verified_optimal" if all(x["optimal"] for x in phases) else "persona_repair"}
            row["receipt_hash"]=digest(row)
            trace.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
    return {**counts,"n":len(menus),"mean_regret":regret/counts["phases"],
            "mean_tv":tv/counts["phases"],"cpu_seconds_including_grading":time.process_time()-start}


def lane_metrics(pool,heads):
    n=correct=0
    for tags in pool:
        for f in FRAMES:
            n+=1
            correct+=heads[f].lane(frozenset(tags))==TEACHERS[f].lane(frozenset(tags))
    return {"n":n,"lane_accuracy":correct/n}


def run(out,steps=300,batch=64,train_n=512,eval_n=128,seeds=(137,163)):
    if not (1<=steps<=1024 and 1<=batch<=128 and 1<=train_n<=2048 and
            1<=eval_n<=512 and 1<=len(seeds)<=8 and len(set(seeds))==len(seeds)):
        raise ValueError("budget outside contract")
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    pools={s:tag_pool(s) for s in SPLITS}
    if train_n>len(pools["train"]):
        raise ValueError("requested more distinct training tags than available")
    train=pools["train"][:train_n]
    menus={s:[menu(s,i,pools[s]) for i in range(eval_n)] for s in SPLITS if s!="train"}
    cfg={"steps":steps,"batch":batch,"train_n":train_n,"eval_n":eval_n,"seeds":list(seeds)}
    report={"schema":SCHEMA,"status":"public_development_only","config":cfg,
            "train_tag_hash":digest(train),"cells":[]}
    path=out/"persona_rollouts.jsonl"
    with path.open("w",encoding="utf-8") as trace:
        for seed in seeds:
            for arm in ARMS:
                info={}
                if arm in LEARNED:
                    records,info=fit(arm,seed,train,steps,batch)
                    artifact=out/f"{seed}-{arm}.json"
                    artifact.write_text(json.dumps(records,sort_keys=True,allow_nan=False),encoding="utf-8")
                    info.update(artifact=artifact.name,artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
                    heads=heads_from(records)
                    identity=info["weights_hash"]
                else:
                    heads=TEACHERS if arm=="exact" else uniform_heads()
                    identity="rule:"+arm
                row={"seed":seed,"arm":arm,"identity":identity,**info}
                for split in ("cal","dev","shift"):
                    row[split]=evaluate(menus[split],heads)
                    row[split]["tag_classification"]=lane_metrics(pools[split][:eval_n],heads)
                row["trace_train"]=evaluate(menus["trace_train"],heads,trace,seed,arm,identity)
                report["cells"].append(row)
    report["trace_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/"persona_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def audit(out):
    report=json.loads((out/"persona_report.json").read_text(encoding="utf-8"))
    cfg=report["config"]
    cells={(x["seed"],x["arm"]):x for x in report["cells"]}
    if set(cells)!=set(itertools.product(cfg["seeds"],ARMS)) or len(cells)!=len(report["cells"]):
        raise ValueError("cell coverage")
    if digest(tag_pool("train")[:cfg["train_n"]])!=report["train_tag_hash"]:
        raise ValueError("training partition binding")
    heads={}
    for key,c in cells.items():
        arm=c["arm"]
        if arm in LEARNED:
            path=out/c["artifact"]
            if hashlib.sha256(path.read_bytes()).hexdigest()!=c["artifact_sha256"]:
                raise ValueError("weight artifact mismatch")
            rec=json.loads(path.read_text(encoding="utf-8"))
            if digest(rec)!=c["weights_hash"] or c["identity"]!=c["weights_hash"]:
                raise ValueError("weight identity mismatch")
            heads[key]=heads_from(rec)
        else:
            if c["identity"]!="rule:"+arm:
                raise ValueError("rule identity")
            heads[key]=TEACHERS if arm=="exact" else uniform_heads()
    pool=tag_pool("trace_train")
    expected={i:menu("trace_train",i,pool) for i in range(cfg["eval_n"])}
    path=out/"persona_rollouts.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report["trace_sha256"]:
        raise ValueError("trace inventory mismatch")
    seen=set()
    kinds=Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        row=json.loads(line)
        h=row.pop("receipt_hash")
        key=(row["seed"],row["arm"])
        p=row["public"]
        i=p["index"]
        if (digest(row)!=h or row["schema"]!=SCHEMA+".rollout" or row["split"]!="trace_train"
                or row["training_ready"] is not False or key not in cells or i not in expected
                or p!=expected[i] or row["root"]!=p["root"] or row["identity"]!=cells[key]["identity"]
                or (key,i) in seen):
            raise ValueError("receipt custody mismatch")
        seen.add((key,i))
        for e in p["evidence"].values():
            if any(tag_partition(t)!="trace_train" for t in e["possible_tags"]):
                raise ValueError("semantic partition leak")
        replay=episode(p,heads[key])
        if replay!=row["phases"]:
            raise ValueError("policy/receipt replay mismatch")
        kind="verified_optimal" if all(x["optimal"] for x in replay) else "persona_repair"
        if row["target_kind"]!=kind:
            raise ValueError("target kind mismatch")
        kinds[kind]+=1
    if len(seen)!=len(cells)*cfg["eval_n"]:
        raise ValueError("missing rollout")
    return {"status":"audited_development_only","cells":len(cells),"rollouts":len(seen),
            "phase_records":4*len(seen),"targets":dict(kinds)}


def selftest():
    pool=tag_pool("trace_train")
    for i in range(64):
        p=menu("trace_train",i,pool)
        for h in (TEACHERS,uniform_heads()):
            rows=episode(p,h)
            assert all(not r["unsafe"] for r in rows)
            assert set(rows[1]["allowed"])<=set(rows[0]["allowed"])
            assert rows[1]["allowed"]==rows[2]["allowed"]
            assert all(r["stale_rejected"] is not False for r in rows)
        assert all(x["optimal"] for x in episode(p,TEACHERS))
    rec,info=fit("external",137,tag_pool("train")[:32],4,8)
    assert info["params"]==150 and info["initial_hash"]!=info["weights_hash"]
    assert set(heads_from(rec))==set(FRAMES)
    print("GSF-04 native Dual-LDT selftest passed")


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
        kwargs=dict(steps=4,batch=8,train_n=32,eval_n=8,seeds=(137,)) if a.smoke else {}
        r=run(a.out,**kwargs)
        print(json.dumps({"cells":len(r["cells"]),"status":r["status"]}))
    if a.audit:
        print(json.dumps(audit(a.audit),sort_keys=True))
    if not (a.selftest or a.out or a.audit):
        p.error("choose --selftest, --out or --audit")


if __name__=="__main__":
    main()
