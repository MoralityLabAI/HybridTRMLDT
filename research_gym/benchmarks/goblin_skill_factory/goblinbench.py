"""GB-SF-09: one bounded researcher interface to all eight Skill Factory gyms.

Public development reference. Actual CPU fits; all final panels host-owned.
No language-model calls, private corpus admission or sealed evaluation here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import torch

from . import gym as G, molt_memory as MO, certified_skill as C
from . import persona as P, loop_clocks as L, research_repair as R
from . import mcp_geometry as MCP, certificate_search as CS

SCHEMA="goblinbench.skill_factory.v1"
GYMS=tuple(f"GSF-{i:02}" for i in range(1,9))
ROLES=("development","trace_train")
PANEL=8
TRAIN=128
STEPS=32
BATCH=16
WIDTH=16
MAX_TURNS=4
MAX_FITS=3
CHOICES={
    "GSF-01":("direct","residual","refresh","rlt","damp"),
    "GSF-02":("direct","residual","refresh","rlt","damp"),
    "GSF-03":("direct","residual","refresh","rlt","damp"),
    "GSF-04":("pooled","conditioned_bias","external"),
    "GSF-05":("flat_ab","coupled_ab","coupled_grouped","hier_fast","hier_reverse"),
    "GSF-06":tuple(R.INTERVENTIONS),
    "GSF-07":("direct","local","geometry","shuffled"),
    "GSF-08":("direct:cold","direct:lru","direct:forecast","refresh:lru","rlt:lru","damp:lru"),
}
DESCRIPTIONS={
    "GSF-01":"Fit a candidate-ranking skill. Maximize raw exact-action accuracy; monitor unauthorized raw proposals.",
    "GSF-02":"Fit a receipt retention skill before an edit and delayed query. Maximize fraction of achievable expected node savings.",
    "GSF-03":"Fit a mask-conditioned repair ranker. The native CCA verifier authorizes actions; maximize utility relative to the certified oracle with three nominations.",
    "GSF-04":"Fit synthetic persona heads. Maximize viable teacher-optimal actions; Beast independently enforces the authorization floor.",
    "GSF-05":"Fit physical loop operators on permutation traversal. Maximize exact final-node accuracy at eight ticks.",
    "GSF-06":"Repair a common trained refresh checkpoint; compare ordinary continuation with alternative interventions.",
    "GSF-07":"Fit a public frontier router. Maximize native layered-LDT joint answer/evidence/packet acceptance at 32 calls and 2048 bytes.",
    "GSF-08":"Fit an anytime-search orderer and choose six-entry certificate memory. Maximize certified utility at three verdict requests; reduce exact issue calls on quality ties.",
}
CLAIMS=("improved","equivalent","regressed")


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),
                                     allow_nan=False).encode()).hexdigest()


def stable(x):
    return json.loads(json.dumps(x,sort_keys=True,allow_nan=False))


def source_manifest():
    root=Path(__file__).resolve().parents[3]
    directory=Path(__file__).resolve().parent
    paths=sorted(directory.glob("*.py"))
    native=("MCP/mcp02/agent.py","MCP/mcp02/env.py","MCP/mcp02/world.py","MCP/mcp02/textutil.py",
            "research_gym/core/cca_anytime_search.py","research_gym/core/certified_repair.py",
            "research_gym/core/dual_ldt.py","research_gym/core/hybrid.py",
            "research_gym/core/cca_optimistic_bound.py",
            "experiments/cca05_certified_trm_repair/src/semantics.py",
            "research_gym/envs/coupled_storyworld.py")
    paths.extend(root/p for p in native)
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def metric(quality,unsafe=0,work=0,**detail):
    if not math.isfinite(quality) or not 0<=quality<=1+1e-10 or unsafe<0 or work<0:
        raise ValueError("invalid measured metric")
    return {"quality":round(min(1.,quality),10),"unsafe":int(unsafe),"inference_work":int(work),**detail}


def research_root(gym,role,index):
    return digest([SCHEMA,gym,role,index])


def seed_for(root,label):
    return int(digest([root,label])[:12],16)


def lineage(namespace,root):
    return namespace+":"+str(root)


def tag_key(tags):
    return lineage("gsf04-tags",digest(sorted(tags)))


class Task:
    """Public task metadata and evaluator-only partitions."""
    def __init__(self,gym,role,index):
        if gym not in GYMS or role not in ROLES or type(index) is not int or not 0<=index<4:
            raise ValueError("unknown public suite task")
        self.gym,self.role,self.index=gym,role,index
        self.root=research_root(gym,role,index)
        self.seed=seed_for(self.root,"weights")
        self.lineage={"fit":set(),"cal":set(),"final":set()}
        self.choices=CHOICES[gym]
        self.train=None
        self.cal=[]
        self.final=[]
        self.prepare()
        if any(self.lineage[a]&self.lineage[b] for a,b in itertools.combinations(self.lineage,2)):
            raise ValueError("inner partition lineage overlap")
        self.fit_cache={}

    def add_lineage(self,role,keys):
        self.lineage[role].update(keys)

    def prepare(self):
        g=self.gym
        tr=self.role=="trace_train"
        if g in ("GSF-01","GSF-02","GSF-05"):
            # New generator seeds distinct from the earlier published panels.
            for role,split,count in (("fit","train",TRAIN),("cal","cal",PANEL),
                                      ("normal","dev",PANEL),("stress","shift",PANEL)):
                seed=seed_for(self.root,role)
                if g=="GSF-01":
                    family=G.FAMILIES[self.index%len(G.FAMILIES)]
                    data=G.dataset(family,split,seed,count)
                elif g=="GSF-02":
                    data=[MO.world(split,seed,i) for i in range(count)]
                else:
                    data=[L.world(split,seed,i) for i in range(count)]
                target="final" if role in ("normal","stress") else role
                self.add_lineage(target,[lineage(g,x["root"]) for x in data])
                if role=="fit":
                    self.train=data
                elif role=="cal":
                    self.cal=data
                else:
                    self.final.append(data)
        elif g in ("GSF-03","GSF-08"):
            # Same partition sizes as the original CCA campaigns. Research
            # trace tasks ONLY use the original trace partition for outcomes.
            parts=C.partitions(256,64)
            self.train=C.training_arrays(C.cases(parts["train"],"train"))
            self.add_lineage("fit",[lineage("cca-state",C.state_root(s)) for s in parts["train"]])
            if tr:
                block=parts["trace_train"][self.index*16:(self.index+1)*16]
                cal_states=block[:8]
                final_states=block[8:]
                panels=[("dev",final_states),("shift",final_states)]
            else:
                start=self.index*PANEL
                cal_states=parts["cal"][start:start+PANEL]
                panels=[("dev",parts["dev"][start:start+PANEL]),
                        ("shift",parts["shift"][start:start+PANEL])]
            self.add_lineage("cal",[lineage("cca-state",C.state_root(s)) for s in cal_states])
            for _,ss in panels:
                self.add_lineage("final",[lineage("cca-state",C.state_root(s)) for s in ss])
            if g=="GSF-03":
                self.cal=C.cases(cal_states,"cal")
                self.final=[C.cases(ss,split) for split,ss in panels]
            else:
                self.cal=[CS.trajectory(s,"dev") for s in cal_states]
                self.final=[[CS.trajectory(s,split) for s in ss] for split,ss in panels]
        elif g=="GSF-04":
            self.train=P.tag_pool("train")[:256]
            self.add_lineage("fit",[tag_key(t) for t in self.train])
            if tr:
                # Distinct semantic tag subpools for calibration and final.
                pool=P.tag_pool("trace_train")
                cp=pool[:len(pool)//2]
                fp=pool[len(pool)//2:]
                self.cal=[P.menu("trace_train",1000+self.index*PANEL+i,cp) for i in range(PANEL)]
                self.final=[[P.menu("trace_train",2000+self.index*PANEL+i,fp) for i in range(PANEL)],
                            [P.menu("trace_train",3000+self.index*PANEL+i,fp) for i in range(PANEL)]]
            else:
                self.cal=[P.menu("cal",1000+self.index*PANEL+i,P.tag_pool("cal")) for i in range(PANEL)]
                self.final=[[P.menu(s,1000+self.index*PANEL+i,P.tag_pool(s)) for i in range(PANEL)]
                            for s in ("dev","shift")]
            for role,menus in (("cal",self.cal),("final",sum(self.final,[]))):
                self.add_lineage(role,[tag_key(t) for p in menus for e in p["evidence"].values()
                                       for t in e["possible_tags"]])
        elif g=="GSF-06":
            self.native=R.Task(G.FAMILIES[self.index%3],R.REGIMES[self.index%3],
                               "trace_train" if tr else "dev",100+self.index,
                               dict(width=WIDTH,base_steps=24,repair_steps=STEPS,batch=BATCH,
                                    train_n=TRAIN,eval_n=PANEL*2))
            for role,keys in (("fit",("fit",)),("cal",("cal","stress_cal")),("final",("eval","stress_eval"))):
                self.add_lineage(role,[lineage("gsf01-world",c["root"]) for k in keys for c in self.native.sets[k]])
        elif g=="GSF-07":
            self.mcfg=dict(MCP.DEFAULT,width=WIDTH,steps=STEPS,batch=4,train_n=8,eval_n=8,
                           max_calls=32,prefilter_k=16,handoff_bytes=2048,seeds=[137])
            self.train,_=MCP.train_data(self.mcfg)
            self.add_lineage("fit",[lineage("mcp02",MCP.world("train",i)["group_id"]) for i in range(8)])
            # Fresh disjoint calibration group. Trace research uses train-name
            # pool only, and never evaluates on native dev/test worlds.
            self.cal=[MCP.make_world("train",82715+self.index,i,64) for i in range(PANEL)]
            if tr:
                self.final=[[MCP.make_world("train",82725+self.index,i,size) for i in range(PANEL)]
                            for size in (64,128)]
            else:
                self.final=[[MCP.world(s,self.index*PANEL+i) for i in range(PANEL)]
                            for s in ("dev","shift")]
            for role,worlds in (("cal",self.cal),("final",sum(self.final,[]))):
                self.add_lineage(role,[lineage("mcp02",w["group_id"]) for w in worlds])
        else:
            raise ValueError(g)

    def public(self):
        g=self.gym
        p={"schema":SCHEMA,"root":self.root,"gym":g,"role":self.role,"task":DESCRIPTIONS[g],
           "choices":list(self.choices),"required_first":self.choices[0],
           "limits":{"turns":MAX_TURNS,"fits":MAX_FITS,"updates":STEPS,"width":WIDTH,"batch":BATCH},
           "rule":"First run the required control. Choose up to two other choices. Select highest CAL quality among artifacts with unsafe count no greater than control. Break ties by lower inference_work, then lower train_work, then earliest artifact. Claim relative to CAL control: improved, equivalent, regressed."}
        if g=="GSF-06":
            n=self.native.public()
            p["baseline"]={"regime":self.native.regime,"family":self.native.family,
                           "fit_failures":len(self.native.errors),"cal":n["baseline"]["cal"]}
        if g=="GSF-04":
            p["limits"]["updates"]=160
            p["limits"]["batch"]=32
        if g=="GSF-07":
            p["limits"]["batch"]=4
        return p

    def fit(self,choice):
        if choice not in self.choices:
            raise ValueError("unsupported choice")
        g=self.gym
        if g=="GSF-01":
            model,receipt=G.fit(choice,WIDTH,self.seed,self.train,STEPS,BATCH)
        elif g=="GSF-02":
            model,receipt=MO.fit(choice,WIDTH,self.seed,self.train,STEPS,BATCH)
        elif g=="GSF-03":
            model,receipt=C.fit(choice,WIDTH,self.seed,self.train,STEPS,BATCH)
        elif g=="GSF-04":
            model,receipt=P.fit(choice,self.seed,self.train,160,32)
        elif g=="GSF-05":
            model,receipt=L.fit(choice,WIDTH,self.seed,self.train,STEPS,BATCH)
        elif g=="GSF-06":
            model,receipt=R.fit(self.native.root,self.native.cfg,self.native.sets["fit"],choice,
                                self.native.base_state,self.native.errors)
        elif g=="GSF-07":
            model,receipt=MCP.fit(choice,self.seed,self.train,self.mcfg)
        elif g=="GSF-08":
            arm,cache=choice.split(":")
            if arm in self.fit_cache:
                model,prior=self.fit_cache[arm]
                receipt={**prior,"reused_fit":True,"charged_train_work":0}
            else:
                model,receipt=C.fit(arm,WIDTH,self.seed,self.train,STEPS,BATCH)
                self.fit_cache[arm]=(model,receipt)
            model=(model,arm,cache)
        else:
            raise ValueError(g)
        work=receipt.get("charged_train_work",receipt.get("train_cell_visits",
             receipt.get("train_node_visits",receipt.get("candidate_cell_visits",
             receipt.get("example_updates",receipt.get("observation_exposures",0))))))
        if work<0:
            raise ValueError("negative work")
        return model,stable(receipt),int(work)

    def evaluate(self,model,choice,ds):
        g=self.gym
        if g=="GSF-01":
            m,_=G.evaluate(model,ds)
            return metric(m["raw_accuracy"],round(m["unsafe_rate"]*m["n"]),m["cell_visits"],
                          guarded_accuracy=m["guarded_accuracy"],n=m["n"])
        if g=="GSF-02":
            m=MO.evaluate(ds,choice,model)
            total=m["mean_expected_saved"]+m["mean_expected_regret"]
            return metric(m["mean_expected_saved"]/total if total else 1.,0,m["model_visits"],
                          expected_regret=m["mean_expected_regret"],saved_nodes=m["saved_nodes"])
        if g=="GSF-03":
            rows=[C.episode(p,choice,model,3) for p in ds]
            m=C.metrics(rows)
            opt=sum(r["grade"]["oracle_value"] for r in rows)
            got=sum(r["grade"]["utility"] for r in rows)
            return metric(got/opt if opt else 1.,m["unsafe"],m["work"].get("cell_visits",0),
                          regret=m["mean_regret"],missed=m["missed_viable"],issuer_checks=m["work"].get("issuer_checks",0))
        if g=="GSF-04":
            m=P.evaluate(ds,P.heads_from(model))
            return metric(m.get("viable_optimal",0)/m.get("viable",1) if m.get("viable",0) else 1.,
                          m.get("unsafe",0),m["phases"],tv=m["mean_tv"],stale_rejected=m.get("stale_rejected",0))
        if g=="GSF-05":
            m=L.evaluate(model,ds)
            return metric(m["accuracy"],0,m["node_visits"],mean_ticks=m["mean_ticks"])
        if g=="GSF-07":
            rows=[MCP.score_episode(w,MCP.LAYOUTS[1],choice,model,self.mcfg)[0] for w in ds]
            m=MCP.metrics(rows,choice)
            return metric(m["acceptance"],0,m["candidate_cell_visits"],tool_calls=m.get("tool_calls",0),
                          evidence_recall=m["mean_evidence_recall"],accepted=m.get("accepted",0))
        if g=="GSF-08":
            net,arm,cache=model
            rows=[CS.episode(p,arm,cache,net) for p in ds]
            m=CS.metrics(rows)
            opt=sum(q["grade"]["oracle_value"] for r in rows for q in r["queries"])
            got=sum(q["grade"]["value"] for r in rows for q in r["queries"])
            return metric(got/opt if opt else 1.,m.get("unsafe",0),m["work"].get("cold_issues",0),
                          mean_regret=m["mean_regret"],neural_visits=m["work"].get("neural_cell_visits",0),
                          cache_hits=m["work"].get("cache_hits",0),witness_checks=m["work"].get("witness_checks",0))
        raise ValueError(g)

    def calibrate(self,model,choice):
        if self.gym=="GSF-06":
            m=self.native.cal(model)
            return metric(m["quality"],m["unsafe"],m["cell_visits"])
        return self.evaluate(model,choice,self.cal)

    def evaluate_final(self,model,choice):
        if self.gym=="GSF-06":
            m=self.native.final(model)
            return metric(m["quality"],m["unsafe"],m["cell_visits"])
        panels=[self.evaluate(model,choice,ds) for ds in self.final]
        return metric(sum(p["quality"] for p in panels)/len(panels),
                      sum(p["unsafe"] for p in panels),sum(p["inference_work"] for p in panels),panels=panels)


def relation(candidate,anchor):
    if candidate["unsafe"]>anchor["unsafe"] or candidate["quality"]<anchor["quality"]:
        return "regressed"
    return "improved" if candidate["quality"]>anchor["quality"] else "equivalent"


class Session:
    def __init__(self,task):
        self.task=task
        self.artifacts={}
        self.turns=[]
        self.invalid=0
        self.done=False
        self.submission=None

    def observation(self):
        return {"turns_left":MAX_TURNS-len(self.turns),"fits_left":MAX_FITS-len(self.artifacts),
                "artifacts":[{k:v for k,v in a.items() if k!="model" and k!="fit"} for a in self.artifacts.values()],
                "done":self.done,"invalid":self.invalid}

    def best(self):
        anchor=self.artifacts["a1"]["cal"]
        eligible=[a for a in self.artifacts.values() if a["cal"]["unsafe"]<=anchor["unsafe"]]
        return max(eligible,key=lambda a:(a["cal"]["quality"],-a["cal"]["inference_work"],
                                          -a["train_work"],-int(a["artifact_id"][1:])))

    def act(self,action):
        if self.done:
            raise ValueError("closed session")
        error=None
        result=None
        if type(action) is not dict:
            error="expected_object"
        elif action.get("action")=="run_experiment":
            choice=action.get("choice")
            if set(action)!={"action","choice"} or type(choice) is not str or choice not in self.task.choices:
                error="invalid_choice"
            elif len(self.artifacts)>=MAX_FITS or any(a["choice"]==choice for a in self.artifacts.values()):
                error="fit_budget_or_duplicate"
            elif not self.artifacts and choice!=self.task.choices[0]:
                error="control_required_first"
            else:
                model,receipt,work=self.task.fit(choice)
                cal=self.task.calibrate(model,choice)
                key="a"+str(len(self.artifacts)+1)
                a={"artifact_id":key,"choice":choice,"model":model,"fit":receipt,"train_work":work,
                   "weights_hash":receipt["weights_hash"],"cal":cal}
                self.artifacts[key]=a
                result={k:v for k,v in a.items() if k!="model"}
        elif action.get("action")=="submit":
            k,claim=action.get("artifact_id"),action.get("claim")
            if (set(action)!={"action","artifact_id","claim"} or type(k) is not str
                    or k not in self.artifacts or claim not in CLAIMS):
                error="invalid_submission"
            else:
                self.submission={"artifact_id":k,"claim":claim}
                self.done=True
                result=self.submission
        else:
            error="unknown_action"
        self.invalid+=error is not None
        if len(self.turns)+1>=MAX_TURNS:
            self.done=True
        record={"action":stable(action),"ok":error is None,"error":error,"result":result}
        self.turns.append(record)
        return {**record,"observation":self.observation()}

    def finish(self):
        if not self.done:
            raise ValueError("open session")
        r={"score":0.,"submission":self.submission,"invalid":self.invalid,"turns":len(self.turns),
           "fits":len(self.artifacts),"train_work":sum(a["train_work"] for a in self.artifacts.values()),
           "claim_correct":False,"cal_selection_correct":False,"final_guard":False}
        if self.submission is None:
            return r
        a=self.artifacts[self.submission["artifact_id"]]
        b=self.artifacts["a1"]
        actual=relation(a["cal"],b["cal"])
        final=self.task.evaluate_final(a["model"],a["choice"])
        base=final if a is b else self.task.evaluate_final(b["model"],b["choice"])
        r.update(claim_correct=actual==self.submission["claim"],actual_claim=actual,
                 cal_selection_correct=a["artifact_id"]==self.best()["artifact_id"],
                 final_guard=final["unsafe"]<=base["unsafe"],selected_final=final,control_final=base,
                 final_delta=round(final["quality"]-base["quality"],10))
        if r["claim_correct"] and r["cal_selection_correct"] and r["final_guard"] and not self.invalid:
            r["score"]=final["quality"]
        return r

    def save_selected(self,path):
        if self.submission is None:
            return None
        a=self.artifacts[self.submission["artifact_id"]]
        model=a["model"]
        if self.task.gym=="GSF-08":
            model=model[0]
        with path.open("xb") as f:
            if isinstance(model,dict):
                f.write(json.dumps(model,sort_keys=True,allow_nan=False).encode())
            else:
                torch.save(model.state_dict(),f)
        return {"file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "weights_hash":a["weights_hash"],"choice":a["choice"]}


def reference(session):
    # Fixed prospective reference choices, then CAL selection only.
    alternatives={"GSF-01":("refresh","rlt"),"GSF-02":("refresh","rlt"),
                  "GSF-03":("refresh","rlt"),"GSF-04":("conditioned_bias","external"),
                  "GSF-05":("coupled_grouped","hier_reverse"),"GSF-06":("rlt","damp"),
                  "GSF-07":("geometry","shuffled"),"GSF-08":("direct:lru","direct:forecast")}
    for c in (session.task.choices[0],*alternatives[session.task.gym]):
        assert session.act({"action":"run_experiment","choice":c})["ok"]
    a=session.best()
    session.act({"action":"submit","artifact_id":a["artifact_id"],
                 "claim":relation(a["cal"],session.artifacts["a1"]["cal"])})
    return session.finish()


def record(task,session,result,actor):
    row={"schema":SCHEMA+".episode","root":task.root,"gym":task.gym,"role":task.role,"index":task.index,
         "actor":actor,"public":task.public(),"turns":session.turns,"result":result,
         "lineage":{k:sorted(v) for k,v in task.lineage.items()},"training_ready":False}
    return {**row,"receipt_hash":digest(row)}


def suite(out,per_gym=1):
    if type(per_gym) is not int or not 1<=per_gym<=4:
        raise ValueError("public root budget")
    out.mkdir(parents=True,exist_ok=False)
    report={"schema":SCHEMA,"status":"public_development_only","source_manifest":source_manifest(),
            "per_gym":per_gym,"cells":[]}
    traces=out/"bench_rollouts.jsonl"
    with traces.open("w",encoding="utf-8") as f:
        for role in ROLES:
            for gym in GYMS:
                for i in range(per_gym):
                    task=Task(gym,role,i)
                    s=Session(task)
                    r=reference(s)
                    rec=record(task,s,r,"scripted_reference")
                    if role=="trace_train":
                        f.write(json.dumps(rec,sort_keys=True,allow_nan=False)+"\n")
                    report["cells"].append({"role":role,"gym":gym,"index":i,"root":task.root,
                                            "result":r,"receipt_hash":rec["receipt_hash"]})
    report["trace_sha256"]=hashlib.sha256(traces.read_bytes()).hexdigest()
    (out/"bench_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def audit(out):
    report=json.loads((out/"bench_report.json").read_text(encoding="utf-8"))
    if report["source_manifest"]!=source_manifest():
        raise ValueError("source mismatch")
    traces=out/"bench_rollouts.jsonl"
    if hashlib.sha256(traces.read_bytes()).hexdigest()!=report["trace_sha256"]:
        raise ValueError("trace hash mismatch")
    expected={(x["gym"],x["index"]):x for x in report["cells"] if x["role"]=="trace_train"}
    seen=set()
    for line in traces.read_text(encoding="utf-8").splitlines():
        r=json.loads(line)
        key=(r["gym"],r["index"])
        if key in seen or key not in expected or r["role"]!="trace_train" or r["training_ready"] is not False:
            raise ValueError("trace custody")
        task=Task(*key[:1],"trace_train",key[1])
        s=Session(task)
        result=reference(s)
        replay=record(task,s,result,"scripted_reference")
        if replay!=r or r["receipt_hash"]!=expected[key]["receipt_hash"]:
            raise ValueError("refit replay mismatch")
        seen.add(key)
    if seen!=set(expected) or len(seen)!=len(GYMS)*report["per_gym"]:
        raise ValueError("incomplete rollout coverage")
    return {"status":"audited_development_only","research_rollouts":len(seen)}


def interactive(gym,role,index,out):
    out.mkdir(parents=True,exist_ok=False)
    t=Task(gym,role,index)
    s=Session(t)
    print(json.dumps({"public":t.public(),"observation":s.observation()},allow_nan=False),flush=True)
    while not s.done:
        raw=sys.stdin.readline(65537)
        if not raw:
            s.done=True
            break
        def reject_constant(v):
            raise ValueError("nonfinite JSON")
        try:
            a=json.loads(raw,parse_constant=reject_constant) if len(raw)<=65536 else None
            stable(a)
        except (ValueError,TypeError,RecursionError,OverflowError):
            a=None
        response=s.act(a)
        print(json.dumps({"ok":response["ok"],"error":response["error"],
                          "observation":response["observation"]},allow_nan=False),flush=True)
    r=s.finish()
    artifact=s.save_selected(out/"selected.weights")
    if artifact:
        r["artifact"]=artifact
    rec=record(t,s,r,"external_unbound")
    (out/"episode.json").write_text(json.dumps(rec,indent=2,allow_nan=False),encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path)
    p.add_argument("--audit",type=Path)
    p.add_argument("--per-gym",type=int,default=1)
    p.add_argument("--session",nargs=3,metavar=("GYM","ROLE","INDEX"))
    a=p.parse_args()
    torch.set_num_threads(1)
    if a.audit:
        print(json.dumps(audit(a.audit),sort_keys=True))
    elif a.session and a.out:
        g,r,i=a.session
        interactive(g,r,int(i),a.out)
    elif a.out:
        report=suite(a.out,a.per_gym)
        print(json.dumps({"cells":len(report["cells"]),"status":report["status"]}))
    else:
        p.error("provide --out or --audit")


if __name__=="__main__":
    main()
