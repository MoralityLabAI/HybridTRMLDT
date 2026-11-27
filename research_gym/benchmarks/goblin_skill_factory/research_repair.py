"""GSF-06: bounded propose-test-repair research episodes on real skill weights.

Uses the GSF-01 skill contract and an independent scorer. Reference policies
are scripted researchers, not language-model rollouts. Public development only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import nn

from . import gym as G
from .audit import independent_grade

SCHEMA = "hybrid.gsf.research_repair.v1"
REGIMES = ("uniform", "abstain_skew", "veto_underexposed")
META_SPLITS = ("dev", "trace_train")
INTERVENTIONS = ("control", "rlt", "damp", "failure_replay", "final_only", "residual")
HYPOTHESES = {
    "control": "ordinary_continuation",
    "rlt": "scheduled_interpolation",
    "damp": "constant_interpolation",
    "failure_replay": "failure_directed_data",
    "final_only": "readout_supervision",
    "residual": "input_transport",
}
POLICIES = ("control_only", "fixed_clock", "fixed_data", "random_budget", "diagnostic")
CLAIMS = ("improved", "equivalent", "regressed")
MAX_TURNS = 4
MAX_EXPERIMENTS = 3
DEFAULT = dict(width=24, base_steps=24, repair_steps=32, batch=32, train_n=192, eval_n=64)


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),
                                     allow_nan=False).encode()).hexdigest()


def root_id(family,regime,split,index):
    return digest([SCHEMA,family,regime,split,index])


def seed_for(root,role):
    return int(digest([root,role])[:15],16)


def validate_config(cfg):
    limits = {"width":(1,64),"base_steps":(1,256),"repair_steps":(1,256),
              "batch":(1,128),"train_n":(8,1024),"eval_n":(4,256)}
    if set(cfg)!=set(limits) or any(type(cfg[k]) is not int or not lo<=cfg[k]<=hi
                                    for k,(lo,hi) in limits.items()):
        raise ValueError("invalid CPU experiment configuration")


def clone_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def fit(root,cfg,train,intervention,base_state=None,errors=(),regime="uniform"):
    """All continuations reset AdamW and share initialization, steps and visits."""
    is_base=base_state is None
    arm={"rlt":"rlt","damp":"damp","residual":"residual"}.get(intervention,"refresh")
    torch.manual_seed(seed_for(root,"model"))
    model=G.Skill(cfg["width"],arm)
    if not is_base:
        model.load_state_dict(base_state)
    before=G.weight_hash(model)
    x,y=G.tensors(train)
    rng=random.Random(seed_for(root,"base_batches" if is_base else "repair_batches"))
    # Baseline skew is a declared exposure intervention, never label corruption.
    weights=[1.0]*len(train)
    if is_base and regime=="abstain_skew":
        weights=[8.0 if int(v)==G.ABSTAIN else 1.0 for v in y]
    elif is_base and regime=="veto_underexposed":
        weights=[.1 if any(row[-1] for row in c["rows"]) else 1.0 for c in train]
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    steps=cfg["base_steps"] if is_base else cfg["repair_steps"]
    batch=cfg["batch"]
    exposure=Counter()
    schedule_hashes=[]
    for _ in range(steps):
        # Consume the same uniform draws for matched ordinary arms.
        indices=rng.choices(range(len(train)),weights=weights,k=batch)
        if not is_base and intervention=="failure_replay" and errors:
            # Separate deterministic stream so ordinary sampling remains matched.
            rr=random.Random(seed_for(root,"failure_replay:"+str(len(schedule_hashes))))
            indices[:batch//2]=rr.choices(errors,k=batch//2)
        schedule_hashes.append(digest(indices))
        for i in indices:
            exposure["examples"]+=1
            exposure["abstain_labels"]+=int(y[i])==G.ABSTAIN
            exposure["veto_cases"]+=any(row[-1] for row in train[i]["rows"])
        idx=torch.tensor(indices)
        logits=model(x[idx],G.PASSES)
        if intervention=="final_only" and not is_base:
            loss=nn.functional.cross_entropy(logits[:,-1],y[idx])
        else:
            loss=nn.functional.cross_entropy(logits.flatten(0,1),y[idx].repeat_interleave(G.PASSES))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    after=G.weight_hash(model)
    if before==after:
        raise RuntimeError("fit changed no weights")
    return model,{"initial_hash":before,"weights_hash":after,"parameters":sum(p.numel() for p in model.parameters()),
                  "optimizer_updates":steps,"example_exposures":steps*batch,
                  "train_cell_visits":steps*batch*G.PASSES,"batch_order_sha256":digest(schedule_hashes),
                  "exposure":dict(exposure),"runtime_arm":arm}


@torch.no_grad()
def predictions(model,cases):
    logits=model(G.features(cases),G.PASSES)
    return logits.argmax(-1).tolist()


def panel(model,cases):
    preds=predictions(model,cases)
    counts=Counter()
    pass_correct=[0]*G.PASSES
    for c,actions in zip(cases,preds,strict=True):
        receipts=[independent_grade(c,a) for a in actions]
        g=receipts[-1]
        viable=g["gold"]!=G.ABSTAIN
        counts["n"]+=1
        counts["viable" if viable else "no_viable"]+=1
        counts["correct"]+=g["raw_correct"]
        counts["viable_correct" if viable else "abstain_correct"]+=g["raw_correct"]
        counts["unsafe"]+=g["unsafe"]
        counts["guarded_correct"]+=g["guarded_correct"]
        counts["changed_after_first"]+=actions[0]!=actions[-1]
        for j,r in enumerate(receipts):
            pass_correct[j]+=r["raw_correct"]
    for k in ("viable","no_viable","correct","viable_correct","abstain_correct",
              "unsafe","guarded_correct","changed_after_first"):
        counts.setdefault(k,0)
    strata=[Fraction(counts["viable_correct"],counts["viable"])] if counts["viable"] else []
    if counts["no_viable"]:
        strata.append(Fraction(counts["abstain_correct"],counts["no_viable"]))
    q=sum(strata,Fraction(0))/len(strata)
    return {**counts,"balanced_quality":str(q),"quality":float(q),
            "accuracy":counts["correct"]/counts["n"],"pass_correct":pass_correct,
            "cell_visits":len(cases)*G.PASSES}


def pair_panel(model,normal,stress):
    a,b=panel(model,normal),panel(model,stress)
    q=(Fraction(a["balanced_quality"])+Fraction(b["balanced_quality"]))/2
    return {"normal":a,"stress":b,"quality_exact":str(q),"quality":float(q),
            "unsafe":a["unsafe"]+b["unsafe"],"cell_visits":a["cell_visits"]+b["cell_visits"]}


def relation(candidate,anchor):
    # A raw unauthorized-proposal regression fails the guardrail.
    cq,aq=Fraction(candidate["quality_exact"]),Fraction(anchor["quality_exact"])
    if candidate["unsafe"]>anchor["unsafe"] or cq<aq:
        return "regressed"
    return "improved" if cq>aq else "equivalent"


class Task:
    """Evaluator-owned data and common trained checkpoint for a research root."""
    def __init__(self,family,regime,split,index,cfg):
        validate_config(cfg)
        if family not in G.FAMILIES or regime not in REGIMES or split not in META_SPLITS or type(index) is not int or not 0<=index<10000:
            raise ValueError("invalid research root")
        self.family,self.regime,self.split,self.index=family,regime,split,index
        self.cfg=dict(cfg)
        self.root=root_id(family,regime,split,index)
        self.sets={}
        for role,g_split in (("fit","train"),("cal","cal"),("stress_cal","shift"),
                             ("eval","dev"),("stress_eval","shift")):
            self.sets[role]=G.dataset(family,g_split,seed_for(self.root,role),
                                      cfg["train_n"] if role=="fit" else cfg["eval_n"])
        seen=set()
        for cs in self.sets.values():
            worlds={digest(c["rows"]) for c in cs}
            if seen&worlds:
                raise ValueError("duplicate semantic world across roles")
            seen.update(worlds)
        self.base,self.base_receipt=fit(self.root,cfg,self.sets["fit"],"control",regime=regime)
        self.base_state=clone_state(self.base)
        actions=predictions(self.base,self.sets["fit"])
        self.errors=tuple(i for i,(c,aa) in enumerate(zip(self.sets["fit"],actions,strict=True))
                          if not independent_grade(c,aa[-1])["raw_correct"])
        self.base_cal=self.cal(self.base)

    def cal(self,model):
        return pair_panel(model,self.sets["cal"],self.sets["stress_cal"])

    def final(self,model):
        return pair_panel(model,self.sets["eval"],self.sets["stress_eval"])

    def public(self):
        return {"schema":SCHEMA,"root":self.root,"family":self.family,"baseline_regime":self.regime,
                "skill_contract":{"fields":list(G.FIELDS),"action_count":G.N,"abstain":G.ABSTAIN,
                                  "hard_gate":"host checks permitted candidates; invalid becomes abstain"},
                "baseline":{"artifact_id":"a0","fit":self.base_receipt,"cal":self.base_cal,
                            "fit_failure_count":len(self.errors),
                            "fit_failure_scan_cell_visits":len(self.sets["fit"])*G.PASSES},
                "budget":{"max_turns":MAX_TURNS,"max_experiments":MAX_EXPERIMENTS,
                          "updates_per_experiment":self.cfg["repair_steps"],
                          "batch":self.cfg["batch"],"visits":G.PASSES,
                          "max_train_cell_visits":MAX_EXPERIMENTS*self.cfg["repair_steps"]*self.cfg["batch"]*G.PASSES},
                "interventions":HYPOTHESES,
                "protocol":{"run_experiment":{"action":"run_experiment","intervention":"control",
                                              "hypothesis":"ordinary_continuation"},
                            "submit":{"action":"submit","artifact_id":"a0","claim":"improved|equivalent|regressed"}},
                "selection_rule":"Compare to matched control: unsafe count must not increase; maximize mean balanced raw quality. Prefer unchanged checkpoint on exact ties; then control. Claims refer only to CAL."}


class Session:
    """Bounded action interface. No eval outcomes are included in observations."""
    def __init__(self,task):
        self.task=task
        self.artifacts={"a0":{"model":task.base,"cal":task.base_cal,"intervention":"baseline","fit":task.base_receipt}}
        self.attempts=[]
        self.turns=0
        self.experiments=0
        self.train_visits=0
        self.cal_visits=0
        self.done=False
        self.submission=None
        self.invalid=0

    def control_id(self):
        return next((k for k,v in self.artifacts.items() if v["intervention"]=="control"),None)

    def public_artifact(self,k):
        a=self.artifacts[k]
        return {"artifact_id":k,"intervention":a["intervention"],"fit":a["fit"],"cal":a["cal"]}

    def observe(self):
        return {"turns_left":MAX_TURNS-self.turns,"experiments_left":MAX_EXPERIMENTS-self.experiments,
                "done":self.done,"artifacts":[self.public_artifact(k) for k in self.artifacts],
                "control_id":self.control_id(),"charged_train_cell_visits":self.train_visits,
                "charged_cal_cell_visits":self.cal_visits,"invalid_actions":self.invalid}

    def act(self,action):
        if self.done:
            raise ValueError("episode closed")
        self.turns+=1
        error=None
        result=None
        if not isinstance(action,dict):
            error="expected_object"
        elif action.get("action")=="run_experiment":
            intervention=action.get("intervention")
            if (set(action)!={"action","intervention","hypothesis"} or not isinstance(intervention,str)
                    or intervention not in INTERVENTIONS or action.get("hypothesis")!=HYPOTHESES.get(intervention)):
                error="invalid_experiment_contract"
            elif self.experiments>=MAX_EXPERIMENTS:
                error="experiment_budget"
            elif any(a["intervention"]==intervention for a in self.artifacts.values()):
                error="duplicate_experiment"
            elif self.control_id() is None and intervention!="control":
                error="matched_control_required_first"
            else:
                self.experiments+=1
                charge=self.task.cfg["repair_steps"]*self.task.cfg["batch"]*G.PASSES
                self.train_visits+=charge
                model,fit_receipt=fit(self.task.root,self.task.cfg,self.task.sets["fit"],intervention,
                                      self.task.base_state,self.task.errors)
                if fit_receipt["initial_hash"]!=self.task.base_receipt["weights_hash"] or fit_receipt["train_cell_visits"]!=charge:
                    raise AssertionError("matched checkpoint/work failure")
                cal=self.task.cal(model)
                self.cal_visits+=cal["cell_visits"]
                k="a"+str(self.experiments)
                self.artifacts[k]={"model":model,"cal":cal,"fit":fit_receipt,"intervention":intervention}
                result=self.public_artifact(k)
        elif action.get("action")=="submit":
            k,claim=action.get("artifact_id"),action.get("claim")
            if (set(action)!={"action","artifact_id","claim"} or not isinstance(k,str)
                    or k not in self.artifacts or claim not in CLAIMS or self.control_id() is None):
                error="invalid_submission"
            else:
                self.submission={"artifact_id":k,"claim":claim}
                self.done=True
                result={"submitted":k}
        else:
            error="unknown_action"
        if error:
            self.invalid+=1
        if self.turns>=MAX_TURNS:
            self.done=True
        response={"ok":error is None,"error":error,"result":result,"observation":self.observe()}
        self.attempts.append({"action":action,"response":response})
        return response

    def best_cal(self):
        cid=self.control_id()
        if cid is None:
            raise ValueError("matched control unavailable")
        anchor=self.artifacts[cid]["cal"]
        eligible=[k for k,a in self.artifacts.items() if a["cal"]["unsafe"]<=anchor["unsafe"]]
        def rank(k):
            return (Fraction(self.artifacts[k]["cal"]["quality_exact"]),k=="a0",k==cid,-int(k[1:]))
        return max(eligible,key=rank)

    def finish(self):
        if not self.done:
            raise ValueError("episode open")
        control=self.control_id()
        base=self.task.final(self.task.base)
        result={"submission":self.submission,"turns":self.turns,"experiments":self.experiments,
                "invalid_actions":self.invalid,"train_cell_visits":self.train_visits,
                "cal_cell_visits":self.cal_visits,"baseline_eval":base,
                "control_eval":None,"selected_eval":None,"claim_correct":False,
                "cal_selection_correct":False,"eval_guard":False,"score":0.0}
        if self.submission is None or control is None:
            return result
        k=self.submission["artifact_id"]
        a=self.artifacts[k]
        anchor=self.artifacts[control]
        claim=relation(a["cal"],anchor["cal"])
        selected_eval=self.task.final(a["model"])
        control_eval=self.task.final(anchor["model"])
        cal_best=self.best_cal()
        result.update(control_eval=control_eval,selected_eval=selected_eval,
                      actual_cal_claim=claim,claim_correct=self.submission["claim"]==claim,
                      cal_selection_correct=k==cal_best,cal_best_artifact_id=cal_best,
                      eval_guard=selected_eval["unsafe"]<=control_eval["unsafe"],
                      eval_quality_delta=float(Fraction(selected_eval["quality_exact"])-Fraction(control_eval["quality_exact"])),
                      eval_unsafe_delta=selected_eval["unsafe"]-control_eval["unsafe"])
        result["score"]=(selected_eval["quality"] if result["claim_correct"] and result["eval_guard"]
                         and self.invalid==0 else 0.)
        return result


def experiment(session,intervention):
    return session.act({"action":"run_experiment","intervention":intervention,"hypothesis":HYPOTHESES[intervention]})


def reference(session,policy):
    """Only consumes the declared observations, never final scores."""
    if policy not in POLICIES:
        raise ValueError(policy)
    initial=session.task.public()
    experiment(session,"control")
    if policy=="fixed_clock":
        arms=("rlt","damp")
    elif policy=="fixed_data":
        arms=("failure_replay","final_only")
    elif policy=="random_budget":
        rng=random.Random(seed_for(initial["root"],"policy_random"))
        arms=tuple(rng.sample(INTERVENTIONS[1:],2))
    elif policy=="diagnostic":
        cal=initial["baseline"]["cal"]
        early=sum(cal[s]["pass_correct"][0] for s in ("normal","stress"))
        late=sum(cal[s]["pass_correct"][-1] for s in ("normal","stress"))
        if late<early:
            arms=("rlt","damp")
        elif initial["baseline_regime"]!="uniform" or initial["baseline"]["fit_failure_count"]>initial["baseline"]["fit"]["exposure"]["examples"]/4:
            arms=("failure_replay","final_only")
        else:
            arms=("residual","final_only")
    else:
        arms=()
    for arm in arms:
        experiment(session,arm)
    k=session.best_cal()
    claim=relation(session.artifacts[k]["cal"],session.artifacts[session.control_id()]["cal"])
    session.act({"action":"submit","artifact_id":k,"claim":claim})
    return session.finish()


def record(task,session,policy,result):
    row={"schema":SCHEMA+".rollout","split":task.split,"root":task.root,
         "family":task.family,"regime":task.regime,"index":task.index,"policy":policy,
         "public":task.public(),"turns":session.attempts,"result":result,
         "training_ready":False}
    row["receipt_hash"]=digest(row)
    return row


def summary(rows):
    n=len(rows)
    return {"n":n,"mean_score":sum(r["score"] for r in rows)/n,
            "mean_eval_delta":sum(r.get("eval_quality_delta",0.) for r in rows)/n,
            "positive_delta":sum(r.get("eval_quality_delta",0.)>0 for r in rows),
            "negative_delta":sum(r.get("eval_quality_delta",0.)<0 for r in rows),
            "guard_failures":sum(not r["eval_guard"] for r in rows),
            "correct_claims":sum(r["claim_correct"] for r in rows),
            "correct_cal_selections":sum(r["cal_selection_correct"] for r in rows),
            "train_cell_visits":sum(r["train_cell_visits"] for r in rows),
            "cal_cell_visits":sum(r["cal_cell_visits"] for r in rows)}


def run(out,cfg=None,roots_per_stratum=2):
    cfg=dict(DEFAULT if cfg is None else cfg)
    validate_config(cfg)
    if type(roots_per_stratum) is not int or not 1<=roots_per_stratum<=8:
        raise ValueError("root budget")
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    report={"schema":SCHEMA,"status":"public_development_only","config":cfg,
            "roots_per_stratum":roots_per_stratum,"summaries":{},"baseline_roots":[]}
    trace_path=out/"research_rollouts.jsonl"
    grouped={(split,policy):[] for split in META_SPLITS for policy in POLICIES}
    with trace_path.open("w",encoding="utf-8") as trace:
        for split in META_SPLITS:
            for family in G.FAMILIES:
                for regime in REGIMES:
                    for i in range(roots_per_stratum):
                        task=Task(family,regime,split,i,cfg)
                        report["baseline_roots"].append({"root":task.root,"split":split,
                                                        "weights_hash":task.base_receipt["weights_hash"]})
                        for policy in POLICIES:
                            session=Session(task)
                            result=reference(session,policy)
                            grouped[split,policy].append(result)
                            if split=="trace_train":
                                trace.write(json.dumps(record(task,session,policy,result),
                                                       sort_keys=True,allow_nan=False)+"\n")
    for (split,policy),rows in grouped.items():
        report["summaries"].setdefault(split,{})[policy]=summary(rows)
    report["trace_sha256"]=hashlib.sha256(trace_path.read_bytes()).hexdigest()
    (out/"research_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def audit(out):
    """Re-fit every trace root and replay every reference decision/receipt."""
    torch.set_num_threads(1)
    report=json.loads((out/"research_report.json").read_text(encoding="utf-8"))
    cfg=report["config"]
    validate_config(cfg)
    n=report["roots_per_stratum"]
    if not 1<=n<=8:
        raise ValueError("root budget")
    path=out/"research_rollouts.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report["trace_sha256"]:
        raise ValueError("trace inventory mismatch")
    seen=set()
    tasks={}
    results={p:[] for p in POLICIES}
    for line in path.read_text(encoding="utf-8").splitlines():
        row=json.loads(line)
        bound=row.pop("receipt_hash")
        f,g,i,p=row["family"],row["regime"],row["index"],row["policy"]
        key=(f,g,i,p)
        if (digest(row)!=bound or row["schema"]!=SCHEMA+".rollout" or row["split"]!="trace_train"
                or row["training_ready"] is not False or f not in G.FAMILIES or g not in REGIMES
                or type(i) is not int or not 0<=i<n or p not in POLICIES or key in seen):
            raise ValueError("receipt custody")
        seen.add(key)
        if key[:3] not in tasks:
            tasks[key[:3]]=Task(f,g,"trace_train",i,cfg)
        task=tasks[key[:3]]
        session=Session(task)
        result=reference(session,p)
        replay=record(task,session,p,result)
        if replay["receipt_hash"]!=bound:
            raise ValueError("refit/action/receipt mismatch")
        results[p].append(result)
    if len(seen)!=len(G.FAMILIES)*len(REGIMES)*n*len(POLICIES):
        raise ValueError("missing rollouts")
    if any(summary(rows)!=report["summaries"]["trace_train"][p] for p,rows in results.items()):
        raise ValueError("aggregate replay mismatch")
    return {"status":"audited_development_only","research_roots":len(tasks),
            "rollouts":len(seen),"refitted_experiments":sum(r["experiments"] for rows in results.values() for r in rows)}


def selftest():
    cfg=dict(width=8,base_steps=2,repair_steps=2,batch=8,train_n=24,eval_n=8)
    task=Task("repair","abstain_skew","trace_train",0,cfg)
    s=Session(task)
    assert not experiment(s,"rlt")["ok"]
    assert experiment(s,"control")["ok"]
    assert not experiment(s,"control")["ok"]
    assert s.act({"action":"submit","artifact_id":"a1","claim":"equivalent"})["ok"]
    assert s.finish()["score"]==0. and s.finish()["invalid_actions"]==2
    s=Session(task)
    for arm in ("control","rlt","damp"):
        assert experiment(s,arm)["ok"]
    fits=[s.artifacts[k]["fit"] for k in ("a1","a2","a3")]
    assert len({r["initial_hash"] for r in fits})==1
    assert len({r["batch_order_sha256"] for r in fits})==1
    assert len({r["train_cell_visits"] for r in fits})==1
    assert s.act({"action":"submit","artifact_id":"a1","claim":"equivalent"})["ok"]
    assert s.finish()["claim_correct"]
    try:
        s.act({"action":"submit","artifact_id":"a1","claim":"equivalent"})
    except ValueError:
        pass
    else:
        raise AssertionError("closed session accepted action")
    print("GSF-06 bounded research episode selftest passed")


def interactive(family,regime,index,out):
    """Public JSONL session; final evaluator receipt written to separate file."""
    import sys
    torch.set_num_threads(1)
    task=Task(family,regime,"dev",index,DEFAULT)
    session=Session(task)
    print(json.dumps({"public":task.public(),"observation":session.observe()}),flush=True)
    while not session.done:
        line=sys.stdin.readline(65537)
        if not line:
            break
        try:
            action=json.loads(line) if len(line)<=65536 else None
        except (ValueError,RecursionError):
            action=None
        print(json.dumps(session.act(action),allow_nan=False),flush=True)
    if not session.done:
        session.done=True
    result=session.finish()
    with out.open("x",encoding="utf-8") as f:
        json.dump(record(task,session,"external",result),f,indent=2,allow_nan=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path)
    p.add_argument("--audit",type=Path)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--selftest",action="store_true")
    p.add_argument("--session",nargs=3,metavar=("FAMILY","REGIME","INDEX"))
    p.add_argument("--private-receipt",type=Path)
    a=p.parse_args()
    torch.set_num_threads(1)
    if a.selftest:
        selftest()
    if a.out:
        cfg=dict(width=8,base_steps=2,repair_steps=2,batch=8,train_n=24,eval_n=8) if a.smoke else None
        r=run(a.out,cfg,1 if a.smoke else 2)
        print(json.dumps({"status":r["status"],"roots":len(r["baseline_roots"])}))
    if a.audit:
        print(json.dumps(audit(a.audit),sort_keys=True))
    if a.session:
        if not a.private_receipt:
            p.error("--session requires --private-receipt")
        family,regime,index=a.session
        interactive(family,regime,int(index),a.private_receipt)
    if not (a.selftest or a.out or a.audit or a.session):
        p.error("choose --selftest, --out, --audit or --session")


if __name__=="__main__":
    main()
