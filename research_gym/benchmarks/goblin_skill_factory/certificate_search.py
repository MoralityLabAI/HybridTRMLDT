"""GSF-08: native certified anytime search with bounded verdict memory.

Host-owned exact verdicts and upper bounds; advisory neural ordering only.
Same-context verified records may be reused within a trajectory. Changed
contexts never inherit authority. Public development, CPU only.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

import torch

from experiments.cca05_certified_trm_repair.src import semantics as M
from research_gym.core.cca_anytime_search import ActionVerdict, UpperCertificate, certified_anytime_search
from research_gym.core.cca_optimistic_bound import action_count_vectors, optimistic_moral_upper
from research_gym.core.certified_repair import ActionCertificate, bind, execute_certified
from . import certified_skill as S

SCHEMA="hybrid.gsf.certificate_search.v1"
LEARNED=("direct","refresh","rlt","damp")
ORDERS=LEARNED+("lexical",)
CACHES=("cold","lru","forecast")
QUERIES=6
BUDGET=3
CAPACITY=6
DEFAULT=dict(width=24,steps=100,batch=32,train_n=256,eval_n=64,seeds=[137,163])
SPLITS=("dev","shift","trace_train")


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),
                                     allow_nan=False).encode()).hexdigest()


def clone(x):
    return json.loads(json.dumps(x,allow_nan=False))


def validate(cfg):
    limits={"width":(1,64),"steps":(1,512),"batch":(1,128),"train_n":(8,512),"eval_n":(4,80)}
    if set(cfg)!=set(limits)|{"seeds"} or any(type(cfg[k]) is not int or not lo<=cfg[k]<=hi
                                            for k,(lo,hi) in limits.items()):
        raise ValueError("invalid budget")
    seeds=cfg["seeds"]
    if not isinstance(seeds,list) or not 1<=len(seeds)<=4 or any(type(s) is not int for s in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError("invalid seeds")


def contexts(state,split):
    horizons=(1,2) if split=="shift" else (3,6)
    return [M.context(sc,state,k) for sc in M.SCENARIOS for k in horizons]


def transition_weights(current,split):
    # Public Markov forecast, with different persistence on shift.
    return [((3 if split=="shift" else 8) if i==current else 1) for i in range(4)]


def trajectory(state,split):
    root=S.state_root(state)
    rng=random.Random(int(digest([SCHEMA,root,split,"query_path"]),16))
    first=rng.randrange(4)
    path=[first]
    for _ in range(QUERIES-1):
        path.append(rng.choices(range(4),weights=transition_weights(path[-1],split),k=1)[0])
    return {"schema":SCHEMA,"root":root,"split":split,"state":list(M.st(state)),
            "contexts":contexts(state,split),"query_ids":path}


def packet(ctx):
    return {"state":ctx["state"],"scenario":ctx["scenario"],"horizon":ctx["horizon"]}


def order_for(ctx,arm,model):
    if arm=="lexical":
        return list(M.ACTIONS),[],0
    scores=S.learned_scores(model,packet(ctx),S.FULL)
    order=sorted(M.ACTIONS,key=lambda a:(-scores[-1][M.ACTIONS.index(a)],M.ACTIONS.index(a)))
    return order,scores,len(scores)


def bound_value(ctx,action):
    if ctx["scenario"]!="moral_optimization":
        return 1.,"global_1",0
    state=M.StoryState(*ctx["state"])
    nxt=M.successor(state,action)
    k=max(0,ctx["horizon"]-1)
    return optimistic_moral_upper(*M.st(nxt),k),"cca10_count_envelope",len(action_count_vectors(k))


def verdict_json(v):
    return {"action":v.action,"feasible":v.feasible,"utility":v.utility,
            "evidence":v.evidence.to_jsonable() if isinstance(v.evidence,ActionCertificate) else v.evidence}


@dataclass(frozen=True)
class Entry:
    binding:str
    action:str
    verdict:ActionVerdict
    seal:str


def entry_seal(binding,action,verdict):
    return digest([SCHEMA,binding,action,verdict_json(verdict)])


def valid_entry(entry,ctx,action,verdict):
    return (entry is not None and entry.binding==bind(ctx) and entry.action==action
            and entry.seal==entry_seal(entry.binding,entry.action,entry.verdict)
            and verdict==entry.verdict)


class Host:
    """Trusted in-process store, populated only after exact checking."""
    def __init__(self,mode):
        if mode not in CACHES:
            raise ValueError(mode)
        self.mode=mode
        self.cache=OrderedDict()
        self.work=Counter()
        self.events=[]
        self.forecast={}
        self.pending={}

    def set_forecast(self,ctxs,current,split):
        self.forecast={bind(c):w for c,w in zip(ctxs,transition_weights(current,split),strict=True)}

    def bounds(self,ctx):
        out={}
        for a in M.ACTIONS:
            v,rule,n=bound_value(ctx,a)
            self.work["bound_issues"]+=1
            self.work["bound_issue_envelope_terms"]+=n
            out[a]=UpperCertificate(a,v,bind(ctx),{"rule":rule,"env":M.ENV_DIGEST})
        return out

    def check_bound(self,ctx,a,b):
        self.work["bound_checks"]+=1
        v,rule,n=bound_value(ctx,a)
        self.work["bound_check_envelope_terms"]+=n
        return (ctx["env"]==M.ENV_DIGEST and a in M.ACTIONS and b.action==a
                and b.binding==bind(ctx) and b.upper==v
                and b.evidence=={"rule":rule,"env":M.ENV_DIGEST})

    def evaluate(self,ctx,a):
        key=(bind(ctx),a)
        self.work["verdict_requests"]+=1
        self.work["cache_lookups"]+=self.mode!="cold"
        entry=self.cache.get(key)
        if entry is not None:
            self.cache.move_to_end(key)
            self.work["cache_hits"]+=1
            self.pending[key]="hit"
            self.events.append({"event":"cache_hit","binding":key[0],"action":a,"seal":entry.seal})
            return entry.verdict
        self.work["cold_issues"]+=1
        sc,s,k=ctx["scenario"],M.StoryState(*ctx["state"]),ctx["horizon"]
        cert=M.issue(sc,s,k,a)
        if cert is None:
            v=ActionVerdict(a,False,None,"host-negative-solver")
        else:
            self.work["issuer_utility_calls"]+=1
            v=ActionVerdict(a,True,M.utility(sc,s,k,a),cert)
        self.pending[key]="cold"
        self.events.append({"event":"cold_issue","binding":key[0],"action":a,"verdict":verdict_json(v)})
        return v

    def insert(self,key,verdict):
        if self.mode=="cold":
            return
        self.cache[key]=Entry(key[0],key[1],verdict,entry_seal(key[0],key[1],verdict))
        self.cache.move_to_end(key)
        self.work["cache_inserts"]+=1
        if len(self.cache)>CAPACITY:
            if self.mode=="lru":
                victim=next(iter(self.cache))
            else:
                # Prefer entries whose exact context is forecast to recur.
                # Equal probability uses oldest entry, independent of outcomes.
                victim=min(self.cache,key=lambda k:self.forecast.get(k[0],0))
            removed=self.cache.pop(victim)
            self.work["evictions"]+=1
            self.events.append({"event":"evict","binding":victim[0],"action":victim[1],"seal":removed.seal})
        self.work["peak_entries"]=max(self.work["peak_entries"],len(self.cache))

    def check_verdict(self,ctx,a,v):
        key=(bind(ctx),a)
        origin=self.pending.pop(key,None)
        self.work["verdict_checks"]+=1
        if ctx["env"]!=M.ENV_DIGEST or a not in M.ACTIONS or v.action!=a:
            return False
        if origin=="hit":
            self.work["cache_binding_checks"]+=1
            if not valid_entry(self.cache.get(key),ctx,a,v):
                return False
        elif origin!="cold":
            return False
        sc,s,k=ctx["scenario"],M.StoryState(*ctx["state"]),ctx["horizon"]
        if not v.feasible:
            if v.utility is not None or v.evidence!="host-negative-solver":
                return False
            if origin=="cold":
                self.work["negative_exact_checks"]+=1
                if M.admissible(sc,s,k,a):
                    return False
        else:
            if not isinstance(v.evidence,ActionCertificate) or v.utility is None or not math.isfinite(v.utility):
                return False
            # Even an admitted exact positive receipt replays its witness on hit.
            self.work["witness_checks"]+=1
            self.work["witness_steps"]+=len(v.evidence.witness)+1
            if execute_certified(ctx,a,v.evidence,M.check_certificate)!=a:
                return False
            if origin=="cold":
                self.work["exact_utility_checks"]+=1
                if v.utility!=M.utility(sc,s,k,a):
                    return False
        if origin=="cold":
            self.insert(key,v)
        return True


def episode(public,arm,cache,model=None):
    host=Host(cache)
    steps=[]
    for q,idx in enumerate(public["query_ids"]):
        ctx=public["contexts"][idx]
        host.set_forecast(public["contexts"],idx,public["split"])
        order,scores,visits=order_for(ctx,arm,model)
        host.work["neural_cell_visits"]+=visits
        begin=len(host.events)
        bounds=host.bounds(ctx)
        result=certified_anytime_search(ctx,M.ACTIONS,order,bounds,check_bound=host.check_bound,
                     evaluate=host.evaluate,check_verdict=host.check_verdict,
                     max_evaluations=BUDGET,epsilon=0.)
        if host.pending:
            raise AssertionError("unresolved evaluator result")
        sc,s,k=ctx["scenario"],M.StoryState(*ctx["state"]),ctx["horizon"]
        # Final oracle calculation is separate from charged search operations.
        safe=M.certified_set(sc,s,k)
        best=M.oracle_action(sc,s,k,safe)
        optimum=M.utility(sc,s,k,best) if best is not None else 0.
        selected=result.action
        value=result.lower if selected is not None else 0.
        if selected is not None and (selected not in safe or value!=M.utility(sc,s,k,selected)):
            raise AssertionError("uncertified action")
        if result.gap is not None and optimum-value>result.gap+1e-12:
            raise AssertionError("regret bound violated")
        if result.status=="epsilon_optimal" and optimum-value>1e-12:
            raise AssertionError("false optimality claim")
        if result.status=="proven_empty" and safe:
            raise AssertionError("false empty claim")
        for old,new in zip(result.steps,result.steps[1:]):
            if old.lower is not None and (new.lower is None or new.lower<old.lower):
                raise AssertionError("lower bound fell")
            if old.upper is not None and new.upper is not None and new.upper>old.upper+1e-12:
                raise AssertionError("upper bound increased")
        steps.append({"query":q,"context_id":idx,"context":ctx,
                      "public_observation":{"context":ctx,"forecast_weights":transition_weights(idx,public["split"])},
                      "order":order,"scores":scores,"bounds":{a:asdict(b) for a,b in bounds.items()},
                      "result":asdict(result),"memory_events":host.events[begin:],
                      "grade":{"viable":bool(safe),"oracle":best,"oracle_value":optimum,
                               "selected":selected,"value":value,"regret":optimum-value,
                               "missed_viable":bool(safe) and selected is None,"unsafe":False}})
    w=dict(host.work)
    if w.get("verdict_requests",0)!=w.get("cold_issues",0)+w.get("cache_hits",0):
        raise AssertionError("work conservation")
    if w.get("peak_entries",0)>CAPACITY:
        raise AssertionError("cache capacity")
    return {"queries":steps,"work":w}


def decision_signature(r):
    return digest([{"result":q["result"],"grade":q["grade"]} for q in r["queries"]])


def metrics(rows):
    c=Counter()
    work=Counter()
    regret=0.
    for row in rows:
        work.update(row["work"])
        for q in row["queries"]:
            g=q["grade"]
            c["queries"]+=1
            c["viable"]+=g["viable"]
            c["missed_viable"]+=g["missed_viable"]
            c["unsafe"]+=g["unsafe"]
            c["optimal_certificates"]+=q["result"]["status"]=="epsilon_optimal"
            c["proven_empty"]+=q["result"]["status"]=="proven_empty"
            regret+=g["regret"]
    return {"trajectories":len(rows),**dict(c),"mean_regret":regret/c["queries"],"work":dict(work),
            "peak_entries":max((r["work"].get("peak_entries",0) for r in rows),default=0)}


def source_manifest():
    root=Path(__file__).resolve().parents[3]
    paths=("experiments/cca05_certified_trm_repair/src/semantics.py",
           "research_gym/core/cca_anytime_search.py","research_gym/core/cca_optimistic_bound.py",
           "research_gym/core/certified_repair.py","research_gym/envs/coupled_storyworld.py",
           "research_gym/benchmarks/goblin_skill_factory/certified_skill.py",
           "research_gym/benchmarks/goblin_skill_factory/certificate_search.py")
    return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}


def trace_record(public,seed,arm,cache,identity,r):
    # query_ids and grades are evaluator-owned realized trajectory, not the
    # per-turn observation provided to the learned orderer.
    row={"schema":SCHEMA+".rollout","split":"trace_train","root":public["root"],
         "seed":seed,"arm":arm,"cache":cache,"identity":identity,"state":public["state"],
         "realized_query_ids":public["query_ids"],"queries":r["queries"],"work":r["work"],
         "target_kind":"verified_search" if all(q["grade"]["regret"]<=1e-12 for q in r["queries"])
                       else "certificate_search_repair","training_ready":False}
    row["receipt_hash"]=digest(row)
    return row


def run(out,cfg=None):
    cfg=clone(DEFAULT if cfg is None else cfg)
    validate(cfg)
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    parts=S.partitions(cfg["train_n"],cfg["eval_n"])
    arrays=S.training_arrays(S.cases(parts["train"],"train"))
    report={"schema":SCHEMA,"status":"public_development_only","config":cfg,
            "source_manifest":source_manifest(),"env":M.ENV_DIGEST,
            "split_hashes":{s:digest([S.state_root(x) for x in states]) for s,states in parts.items()},
            "cells":[],"fits":[]}
    path=out/"certificate_rollouts.jsonl.gz"
    with gzip.open(path,"wt",encoding="utf-8") as trace:
        for seed in cfg["seeds"]:
            for arm in ORDERS:
                model=None
                identity="rule:"+arm
                if arm in LEARNED:
                    model,info=S.fit(arm,cfg["width"],seed,arrays,cfg["steps"],cfg["batch"])
                    artifact=out/f"{seed}-{arm}.pt"
                    torch.save(model.state_dict(),artifact)
                    identity=info["weights_hash"]
                    report["fits"].append({"seed":seed,"arm":arm,**info,"artifact":artifact.name,
                                            "artifact_sha256":hashlib.sha256(artifact.read_bytes()).hexdigest()})
                cells={cache:{"seed":seed,"arm":arm,"cache":cache,"identity":identity} for cache in CACHES}
                for split in SPLITS:
                    rows={cache:[] for cache in CACHES}
                    for state in parts[split]:
                        public=trajectory(state,split)
                        signature=None
                        for cache in CACHES:
                            r=episode(public,arm,cache,model)
                            sig=decision_signature(r)
                            if signature is None:
                                signature=sig
                            elif signature!=sig:
                                raise AssertionError("cache changed certified search decisions")
                            rows[cache].append(r)
                            if split=="trace_train":
                                record=trace_record(public,seed,arm,cache,identity,r)
                                trace.write(json.dumps(record,sort_keys=True,allow_nan=False)+"\n")
                    for cache in CACHES:
                        cells[cache][split]=metrics(rows[cache])
                report["cells"].extend(cells.values())
    report["trace_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/"certificate_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def audit(out):
    torch.set_num_threads(1)
    report=json.loads((out/"certificate_report.json").read_text(encoding="utf-8"))
    cfg=report["config"]
    validate(cfg)
    parts=S.partitions(cfg["train_n"],cfg["eval_n"])
    if report["source_manifest"]!=source_manifest() or report["env"]!=M.ENV_DIGEST or report["split_hashes"]!={
        s:digest([S.state_root(x) for x in states]) for s,states in parts.items()}:
        raise ValueError("source or partition mismatch")
    fits={(c["seed"],c["arm"]):c for c in report["fits"]}
    cells={(c["seed"],c["arm"],c["cache"]):c for c in report["cells"]}
    if len(fits)!=len(report["fits"]) or set(fits)!=set(itertools.product(cfg["seeds"],LEARNED)):
        raise ValueError("fit coverage")
    if len(cells)!=len(report["cells"]) or set(cells)!=set(itertools.product(cfg["seeds"],ORDERS,CACHES)):
        raise ValueError("cell coverage")
    models={}
    for key,c in fits.items():
        artifact=out/c["artifact"]
        if hashlib.sha256(artifact.read_bytes()).hexdigest()!=c["artifact_sha256"]:
            raise ValueError("artifact digest")
        m=S.RepairTRM(cfg["width"],c["arm"])
        m.load_state_dict(torch.load(artifact,map_location="cpu",weights_only=True))
        if S.weight_hash(m)!=c["weights_hash"]:
            raise ValueError("weight identity")
        models[key]=m.eval()
    expected={S.state_root(s):s for s in parts["trace_train"]}
    path=out/"certificate_rollouts.jsonl.gz"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report["trace_sha256"]:
        raise ValueError("rollout digest")
    seen=set()
    counts=Counter()
    rows={}
    paired={}
    with gzip.open(path,"rt",encoding="utf-8") as trace:
        for line in trace:
            record=json.loads(line)
            bound=record.pop("receipt_hash")
            seed,arm,cache,root=(record[k] for k in ("seed","arm","cache","root"))
            key=(seed,arm,cache)
            ident=(key,root)
            if (digest(record)!=bound or record["schema"]!=SCHEMA+".rollout" or record["split"]!="trace_train"
                    or record["training_ready"] is not False or key not in cells or root not in expected or ident in seen):
                raise ValueError("trace custody")
            identity=fits[(seed,arm)]["weights_hash"] if arm in LEARNED else "rule:"+arm
            if identity!=cells[key]["identity"] or identity!=record["identity"]:
                raise ValueError("checkpoint binding")
            public=trajectory(expected[root],"trace_train")
            replay=episode(public,arm,cache,models.get((seed,arm)))
            if trace_record(public,seed,arm,cache,identity,replay)["receipt_hash"]!=bound:
                raise ValueError("native certificate and cache replay mismatch")
            sig=decision_signature(replay)
            pair=(seed,arm,root)
            if pair in paired and paired[pair]!=sig:
                raise ValueError("cache changed search result")
            paired[pair]=sig
            seen.add(ident)
            rows.setdefault(key,[]).append(replay)
            counts[record["target_kind"]]+=1
    if len(seen)!=len(cells)*len(expected):
        raise ValueError("missing traces")
    if any(metrics(rs)!=cells[key]["trace_train"] for key,rs in rows.items()):
        raise ValueError("trace summary mismatch")
    return {"status":"audited_development_only","fits":len(fits),"cells":len(cells),
            "rollouts":len(seen),"queries":len(seen)*QUERIES,"targets":dict(counts)}


def selftest():
    torch.set_num_threads(1)
    # Exact same state partition as GSF-03, to preserve cross-gym custody.
    parts=S.partitions(32,8)
    assert len({S.state_root(s) for ss in parts.values() for s in ss})==sum(map(len,parts.values()))
    checked=0
    for state in parts["dev"]:
        public=trajectory(state,"dev")
        results=[episode(public,"lexical",cache) for cache in CACHES]
        assert len({decision_signature(r) for r in results})==1
        assert all(r["work"]["cold_issues"]<=results[0]["work"]["cold_issues"] for r in results)
        for ctx in public["contexts"]:
            a=M.ACTIONS[0]
            host=Host("lru")
            v=host.evaluate(ctx,a)
            assert host.check_verdict(ctx,a,v)
            assert host.check_verdict(ctx,a,host.evaluate(ctx,a))
            entry=host.cache[(bind(ctx),a)]
            stale=dict(ctx,horizon=ctx["horizon"]+1)
            assert not valid_entry(entry,stale,a,v)
            assert not valid_entry(entry,ctx,M.ACTIONS[1],v)
            forged=Entry(entry.binding,entry.action,entry.verdict,"bad")
            assert not valid_entry(forged,ctx,a,v)
            bad=UpperCertificate(a,0.,bind(ctx),{"rule":"forged","env":M.ENV_DIGEST})
            assert not host.check_bound(ctx,a,bad)
            checked+=1
    arrays=S.training_arrays(S.cases(parts["train"],"train"))
    for arm in LEARNED:
        m,info=S.fit(arm,8,137,arrays,2,8)
        assert info["initial_hash"]!=info["weights_hash"]
        r=episode(trajectory(parts["trace_train"][0],"trace_train"),arm,"forecast",m)
        assert len(r["queries"])==QUERIES
    print("GSF-08 native anytime search and cache binding selftest passed",checked)


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
        cfg=dict(DEFAULT,width=8,steps=4,batch=8,train_n=32,eval_n=8,seeds=[137]) if a.smoke else None
        r=run(a.out,cfg)
        print(json.dumps({"status":r["status"],"fits":len(r["fits"]),"cells":len(r["cells"])}))
    if a.audit:
        print(json.dumps(audit(a.audit),sort_keys=True))
    if not (a.selftest or a.out or a.audit):
        p.error("choose --selftest, --out or --audit")


if __name__=="__main__":
    main()
