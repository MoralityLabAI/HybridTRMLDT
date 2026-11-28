"""GSF-07: learned public-frontier geometry over the native MCP-02 runtime.

A CPU graph-message router, not activation geometry or native neural KV cache.
Native Composer, Env02, candidate refinement, and exact judge are unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

import torch
from torch import nn

from MCP.mcp02 import agent as A
from MCP.mcp02.textutil import BM25, tokenize
from MCP.mcp02.world import make_world

SCHEMA = "hybrid.gsf.mcp_geometry.v1"
LEARNED = ("direct", "local", "geometry", "shuffled")
CONTROLS = ("deterministic_rules", "lexical_bm25", "sham_random", "oracle_upper_bound")
ARMS = LEARNED + CONTROLS
LAYOUTS = (("search", ""), ("layered_ldt", "PSCB"))
SPLITS = ("train", "dev", "shift", "trace_train")
NATIVE = {"train": ("train", 81701), "dev": ("dev", 81702),
          "shift": ("test", 81703), "trace_train": ("train", 81704)}
WIDTH_IN = 17
PASSES = 4
GLUE = set("resource branch search need name entity server collection fields scope status unit unknown page remaining count sketch for of the and results".split())
DEFAULT = dict(width=24, steps=160, batch=4, train_n=32, eval_n=16,
               max_calls=32, prefilter_k=16, handoff_bytes=2048, seeds=[137,163])


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def clone(x):
    return json.loads(json.dumps(x,allow_nan=False))


def validate(cfg):
    lim={"width":(1,64),"steps":(1,512),"batch":(1,16),"train_n":(8,128),
         "eval_n":(8,64),"max_calls":(4,64),"prefilter_k":(4,32),"handoff_bytes":(256,8192)}
    if set(cfg)!=set(lim)|{"seeds"} or any(type(cfg[k]) is not int or not lo<=cfg[k]<=hi
                                        for k,(lo,hi) in lim.items()):
        raise ValueError("invalid bounded configuration")
    seeds=cfg["seeds"]
    if not isinstance(seeds,list) or not 1<=len(seeds)<=4 or any(type(s) is not int for s in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError("invalid seeds")


def world(split,index):
    native,seed=NATIVE[split]
    return make_world(native,seed,index,128 if split=="shift" else 64)


def source_manifest():
    root=Path(__file__).resolve().parents[3]
    paths=("MCP/mcp02/agent.py","MCP/mcp02/env.py","MCP/mcp02/world.py",
           "MCP/mcp02/textutil.py","research_gym/core/hybrid.py",
           "research_gym/benchmarks/goblin_skill_factory/mcp_geometry.py")
    return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}


def root_inventory(cfg):
    inv={s:[world(s,i)["group_id"] for i in range(cfg["train_n"] if s=="train" else cfg["eval_n"])] for s in SPLITS}
    all_roots=[x for v in inv.values() for x in v]
    if len(all_roots)!=len(set(all_roots)):
        raise ValueError("overlapping world groups")
    return {s:digest(v) for s,v in inv.items()}


def features(obs):
    """Only public need records and candidate metadata/text enter the model."""
    cs=obs["candidates"]
    need=set(tokenize(" ".join(obs["needs"])))-GLUE
    texts=[set(tokenize(c["text"]))-GLUE for c in cs]
    bm=BM25([c["text"] for c in cs])
    lexical=bm.scores(" ".join(obs["needs"])+" "+obs["task"]["request"])
    rules=A.DeterministicPolicy().score(obs)
    n=len(cs)
    xs=[]
    for i,c in enumerate(cs):
        reg=c["meta"].get("registry",{})
        intersection=len(need&texts[i])
        xs.append(list(c["struct"])+[
            intersection/max(1,len(need)),
            intersection/max(1,len(texts[i])),
            math.tanh(lexical[i]/10),
            math.tanh(rules[i]/50),
            float(reg.get("status")=="active"),
            float(reg.get("scope") is not None),
            min(obs["budget_left"],64)/64,
            min(len(obs["evidence"]),8)/8])
    # Target-blind candidate graph: lexical affinity plus shared public origin.
    # Self-loops keep isolated vertices defined. Symmetric affinity before
    # row normalization; shuffling is an exact permutation conjugation.
    matrix=[]
    for i in range(n):
        row=[]
        for j in range(n):
            overlap=len(texts[i]&texts[j])/max(1,len(texts[i]|texts[j]))
            same_origin=bool(cs[i].get("origin")) and cs[i].get("origin")==cs[j].get("origin")
            row.append(1. if i==j else overlap+.25*same_origin)
        total=sum(row)
        matrix.append([v/total for v in row])
    g=torch.tensor(matrix,dtype=torch.float32)
    perm=list(range(n))
    random.Random(digest([obs["task"]["id"],obs["step"],[c["id"] for c in cs]])).shuffle(perm)
    p=torch.tensor(perm,dtype=torch.long)
    return torch.tensor(xs,dtype=torch.float32),g,g[p][:,p],bm.operations,n*n


class Router(nn.Module):
    def __init__(self,width,arm):
        super().__init__()
        if arm not in LEARNED:
            raise ValueError(arm)
        self.width,self.arm=width,arm
        self.encoder=nn.Linear(WIDTH_IN,width)
        self.cell=nn.Linear(3*width,width)
        self.head=nn.Linear(width,1)

    @property
    def passes(self):
        return 1 if self.arm=="direct" else PASSES

    def forward(self,x,g,shuffled):
        z=torch.tanh(self.encoder(x))
        h=z
        matrix=g if self.arm=="geometry" else shuffled if self.arm=="shuffled" else torch.eye(len(x),dtype=x.dtype)
        for _ in range(self.passes):
            h=h+torch.tanh(self.cell(torch.cat((z,h,matrix@h),dim=-1)))
        return self.head(h).squeeze(-1)


def weight_hash(model):
    return digest({k:v.detach().cpu().tolist() for k,v in model.state_dict().items()})


class NeuralPolicy(A.Policy):
    family="gsf07_graph_router"
    def __init__(self,model):
        self.model=model.eval()
        self.name="gsf07_"+model.arm
        self._cost={}

    @torch.no_grad()
    def score(self,obs):
        x,g,s,lex,pairs=features(obs)
        scores=self.model(x,g,s)
        n,w,p=len(x),self.model.width,self.model.passes
        # Exact dense multiply/add convention (2 per MAC), excludes
        # activations, graph construction, normalization and scalar logic.
        self._cost={"flops":2*(n*WIDTH_IN*w+p*(n*3*w*w+n*n*w)+n*w),
                    "router_tokens":0,"lexical_ops":lex}
        return scores.tolist()

    def propose_exclusions(self,obs,rows):
        self._cost={}
        return A.symbolic_exclusions(obs["task"],rows)

    def last_cost(self):
        return self._cost

    def describe(self):
        return {"name":self.name,"family":self.family,
                "parameters":sum(p.numel() for p in self.model.parameters()),
                "recurrent_depth":self.model.passes,"proposer":"symbolic",
                "uses_private_oracle":False,"geometry":self.model.arm,
                "flops_scope":"dense_MAC_only"}


def episode(case,layout,policy,cfg,record=None):
    arm,word=layout
    result=A.run_episode(case,arm=arm,layout=word,policy=policy,
                         max_tool_calls=cfg["max_calls"],prefilter_k=cfg["prefilter_k"],
                         handoff_budget_bytes=cfg["handoff_bytes"],record=record,debug=True)
    if result["failure"]=="transport_or_environment_error":
        raise RuntimeError(result["error_detail"])
    return result


def train_data(cfg):
    examples=[]
    counts=Counter()
    def collect(event):
        if event["kind"]!="decide":
            return
        counts["observations"]+=1
        labels=event["labels"]
        if not any(labels):
            counts["zero_positive"]+=1
            return
        x,g,s,_,_=features(event["obs"])
        examples.append((x,g,s,torch.tensor(labels,dtype=torch.bool)))
    for i in range(cfg["train_n"]):
        case=world("train",i)
        for layout in LAYOUTS:
            for name in ("oracle_upper_bound","deterministic_rules"):
                r=episode(case,layout,A.make_control(name),cfg,collect)
                counts["episodes"]+=1
                counts["accepted"]+=r["accepted"]
    if not examples:
        raise ValueError("no supervised frontier observations")
    return examples,dict(counts)


def fit(arm,seed,examples,cfg):
    torch.manual_seed(seed)
    model=Router(cfg["width"],arm)
    before=weight_hash(model)
    rng=random.Random(seed+103)
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    cells=0
    graph_pairs=0
    batch_hashes=[]
    for _ in range(cfg["steps"]):
        indices=[rng.randrange(len(examples)) for _ in range(cfg["batch"])]
        batch_hashes.append(digest(indices))
        loss=0.
        for i in indices:
            x,g,s,y=examples[i]
            pred=model(x,g,s)
            loss=loss+torch.logsumexp(pred,0)-torch.logsumexp(pred[y],0)
            cells+=len(x)*model.passes
            graph_pairs+=len(x)**2*model.passes
        opt.zero_grad()
        (loss/cfg["batch"]).backward()
        opt.step()
    model.eval()
    after=weight_hash(model)
    if before==after:
        raise ValueError("unchanged weights")
    return model,{"initial_hash":before,"weights_hash":after,
                  "params":sum(p.numel() for p in model.parameters()),
                  "updates":cfg["steps"],"observation_exposures":cfg["steps"]*cfg["batch"],
                  "candidate_cell_visits":cells,"message_pairs":graph_pairs,
                  "batch_order_sha256":digest(batch_hashes)}


def policy_for(arm,model=None):
    return NeuralPolicy(model) if arm in LEARNED else A.make_control(arm)


def stable(result):
    r=clone(result)
    r.pop("elapsed_seconds",None)
    r["compute"].pop("router_seconds",None)
    return r


def score_episode(case,layout,arm,model,cfg,trace=False):
    observations=[]
    def capture(event):
        if event["kind"]=="decide":
            # Private labels live in a separate post-action supervision field.
            observations.append({"public":clone(event["obs"]),"oracle_usefulness":list(event["labels"])})
    r=episode(case,layout,policy_for(arm,model),cfg,capture if trace else None)
    if trace:
        decisions=[d for d in r["decisions"] if d["kind"]=="decide"]
        if len(decisions)!=len(observations):
            raise ValueError("decision/observation count mismatch")
        for d,o in zip(decisions,observations,strict=True):
            if d["obs_sha"]!=A.observation_digest(o["public"])[:16] or d["n"]!=len(o["oracle_usefulness"]):
                raise ValueError("frontier binding mismatch")
            o["choice_index"]=d["choice_index"]
    return stable(r),observations


def metrics(rows,arm):
    n=len(rows)
    c=Counter()
    failures=Counter()
    by_family={}
    depth=1 if arm=="direct" else PASSES if arm in LEARNED else 0
    for r in rows:
        c["accepted"]+=r["accepted"]
        c["answer_correct"]+=r["grade"]["answer_correct"]
        c["evidence_valid"]+=r["grade"]["evidence_valid"]
        c["budget_exceeded"]+=r["handoff_budget_exceeded"]
        c["failed"]+=r["failure"] is not None
        c["tool_calls"]+=r["tool_calls"]
        c["handoff_bytes"]+=r["handoff_utf8_bytes"]
        c["observation_bytes"]+=r["compute"]["observation_bytes"]
        c["result_bytes"]+=r["compute"]["result_bytes"]
        c["candidates_scored"]+=r["compute"]["candidates_scored"]
        c["dense_MAC_flops"]+=r["compute"]["flops"]
        c["epoch_rolls"]+=r["rolled"]
        c["revalidations"]+=r["composer"]["revalidations"]
        c["refine_calls"]+=r["compute"]["refine_calls"]
        if depth:
            pairs=sum(d["n"]**2 for d in r["decisions"] if d["kind"]=="decide")
            c["graph_preparation_pairs"]+=pairs
            c["candidate_message_pairs"]+=pairs*depth
        if r["failure"]:
            failures[r["failure"]]+=1
        failures.update(r["grade"]["reasons"])
        f=by_family.setdefault(r["family"],Counter())
        f["n"]+=1
        f["accepted"]+=r["accepted"]
        f["tool_calls"]+=r["tool_calls"]
        f["rolled"]+=r["rolled"]
        f["revalidations"]+=r["composer"]["revalidations"]
    return {"n":n,**{k:int(v) for k,v in c.items()},
            "acceptance":c["accepted"]/n,
            "mean_evidence_recall":sum(r["grade"]["evidence_recall"] for r in rows)/n,
            "tool_calls_per_accepted":c["tool_calls"]/c["accepted"] if c["accepted"] else None,
            "candidate_cell_visits":c["candidates_scored"]*depth,
            "families":{k:dict(v) for k,v in by_family.items()},"failure_reasons":dict(failures)}


def trace_record(split,index,case,seed,arm,layout,identity,result,observations):
    row={"schema":SCHEMA+".rollout","split":split,"root":case["group_id"],
         "native_case_sha256":result["case_sha256"],"index":index,"seed":seed,
         "arm":arm,"layout":list(layout),"identity":identity,
         "public_decisions":[{"public":o["public"],"choice_index":o["choice_index"]} for o in observations],
         "evaluator_supervision":[o["oracle_usefulness"] for o in observations],
         "result":result,"target_kind":"verified_packet" if result["accepted"] else "retrieval_repair",
         "uses_private_oracle":arm=="oracle_upper_bound","training_ready":False}
    row["receipt_hash"]=digest(row)
    return row


def run(out,cfg=None):
    cfg=clone(DEFAULT if cfg is None else cfg)
    validate(cfg)
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    examples,counts=train_data(cfg)
    report={"schema":SCHEMA,"status":"public_development_only","config":cfg,
            "training":counts,"training_positive_observations":len(examples),
            "source_manifest":source_manifest(),"root_hashes":root_inventory(cfg),"cells":[]}
    trace_path=out/"mcp_rollouts.jsonl.gz"
    with gzip.open(trace_path,"wt",encoding="utf-8") as trace:
        for seed in cfg["seeds"]:
            for arm in ARMS:
                model=None
                info={}
                if arm in LEARNED:
                    model,info=fit(arm,seed,examples,cfg)
                    artifact=out/f"{seed}-{arm}.pt"
                    torch.save(model.state_dict(),artifact)
                    info.update(artifact=artifact.name,artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
                identity=info.get("weights_hash","rule:"+arm)
                row={"seed":seed,"arm":arm,"identity":identity,**info}
                for split in ("dev","shift","trace_train"):
                    row[split]={}
                    for layout in LAYOUTS:
                        rows=[]
                        for i in range(cfg["eval_n"]):
                            case=world(split,i)
                            r,obs=score_episode(case,layout,arm,model,cfg,split=="trace_train")
                            rows.append(r)
                            if split=="trace_train":
                                rec=trace_record(split,i,case,seed,arm,layout,identity,r,obs)
                                trace.write(json.dumps(rec,sort_keys=True,allow_nan=False)+"\n")
                        row[split][layout[0]]=metrics(rows,arm)
                low_cfg=dict(cfg,max_calls=max(4,cfg["max_calls"]//2))
                row["low_budget"]={}
                for split in ("dev","shift"):
                    row["low_budget"][split]={}
                    for layout in LAYOUTS:
                        rows=[score_episode(world(split,i),layout,arm,model,low_cfg)[0]
                              for i in range(cfg["eval_n"])]
                        row["low_budget"][split][layout[0]]=metrics(rows,arm)
                report["cells"].append(row)
    report["trace_sha256"]=hashlib.sha256(trace_path.read_bytes()).hexdigest()
    (out/"mcp_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def audit(out):
    torch.set_num_threads(1)
    report=json.loads((out/"mcp_report.json").read_text(encoding="utf-8"))
    cfg=report["config"]
    validate(cfg)
    if report["source_manifest"]!=source_manifest() or report["root_hashes"]!=root_inventory(cfg):
        raise ValueError("source or root manifest mismatch")
    cells={(c["seed"],c["arm"]):c for c in report["cells"]}
    if len(cells)!=len(report["cells"]) or set(cells)!=set(itertools.product(cfg["seeds"],ARMS)):
        raise ValueError("cell coverage mismatch")
    models={}
    for key,c in cells.items():
        if c["arm"] in LEARNED:
            artifact=out/c["artifact"]
            if hashlib.sha256(artifact.read_bytes()).hexdigest()!=c["artifact_sha256"]:
                raise ValueError("artifact digest mismatch")
            m=Router(cfg["width"],c["arm"])
            m.load_state_dict(torch.load(artifact,map_location="cpu",weights_only=True))
            if weight_hash(m)!=c["identity"] or c["weights_hash"]!=c["identity"]:
                raise ValueError("weight identity mismatch")
            models[key]=m.eval()
        elif c["identity"]!="rule:"+c["arm"]:
            raise ValueError("rule identity mismatch")
    path=out/"mcp_rollouts.jsonl.gz"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report["trace_sha256"]:
        raise ValueError("trace inventory mismatch")
    seen=set()
    counts=Counter()
    replay_rows={}
    with gzip.open(path,"rt",encoding="utf-8") as trace:
        for line in trace:
            r=json.loads(line)
            bound=r.pop("receipt_hash")
            key=(r["seed"],r["arm"])
            layout=tuple(r["layout"])
            i=r["index"]
            ident=(key,layout,i)
            if (digest(r)!=bound or r["schema"]!=SCHEMA+".rollout" or r["split"]!="trace_train"
                    or r["training_ready"] is not False or key not in cells or layout not in LAYOUTS
                    or type(i) is not int or not 0<=i<cfg["eval_n"] or ident in seen):
                raise ValueError("trace custody")
            case=world("trace_train",i)
            if r["root"]!=case["group_id"] or r["identity"]!=cells[key]["identity"]:
                raise ValueError("root/model binding")
            result,obs=score_episode(case,layout,r["arm"],models.get(key),cfg,True)
            replay=trace_record("trace_train",i,case,r["seed"],r["arm"],layout,r["identity"],result,obs)
            if replay["receipt_hash"]!=bound:
                raise ValueError("native policy and exact judge replay mismatch")
            replay_rows.setdefault((key,layout),[]).append(result)
            seen.add(ident)
            counts[r["target_kind"]]+=1
    if len(seen)!=len(cells)*len(LAYOUTS)*cfg["eval_n"]:
        raise ValueError("missing traces")
    for (key,layout),rows in replay_rows.items():
        if metrics(rows,key[1])!=cells[key]["trace_train"][layout[0]]:
            raise ValueError("trace summary mismatch")
    return {"status":"audited_development_only","cells":len(cells),
            "rollouts":len(seen),"targets":dict(counts)}


def selftest():
    torch.set_num_threads(1)
    cfg=dict(DEFAULT,train_n=8,eval_n=8,steps=2,batch=2,seeds=[137],max_calls=12)
    examples,counts=train_data(cfg)
    x,g,s,y=examples[0]
    assert x.shape[1]==WIDTH_IN and torch.allclose(g.sum(1),torch.ones(len(g)))
    assert torch.allclose(s.sum(1),torch.ones(len(s)))
    hashes=set()
    for arm in LEARNED:
        m,info=fit(arm,137,examples,cfg)
        hashes.add(info["initial_hash"])
        assert m(x,g,s).shape==y.shape
        r,obs=score_episode(world("trace_train",4),LAYOUTS[0],arm,m,cfg,True)
        assert r["tool_calls"]<=cfg["max_calls"] and obs
    assert len(hashes)==1
    print("GSF-07 native MCP geometry selftest passed")


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
        cfg=dict(DEFAULT,train_n=8,eval_n=8,steps=4,batch=2,seeds=[137],max_calls=16) if a.smoke else None
        r=run(a.out,cfg)
        print(json.dumps({"status":r["status"],"cells":len(r["cells"]),"training":r["training"]}))
    if a.audit:
        print(json.dumps(audit(a.audit),sort_keys=True))
    if not (a.selftest or a.out or a.audit):
        p.error("choose --selftest, --out or --audit")


if __name__=="__main__":
    main()
