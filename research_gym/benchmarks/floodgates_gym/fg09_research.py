"""FG-09: GSF-10 researcher-facing optimization under GB-SF-09's session grammar."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import torch

from . import gym as G
from . import fg07_controller as F
from research_gym.benchmarks.goblin_skill_factory import goblinbench as B

SCHEMA=B.SCHEMA
GYM="GSF-10"
ROOT_SCHEMA="hybrid.gsf.floodgates.fg09.v1"
ROLES=B.ROLES
CHOICES=("fixed_first4","serial","fixed_first1","fixed_first2","fixed_deep1","fixed_deep2",
         "fixed_deep4","fixed_first_elastic2","fixed_deep_elastic2","fixed_first_cycle_rlt4",
         "fixed_deep_cycle_rlt4","heuristic","imitation","cost_regression")
REFERENCE=("fixed_first4","heuristic","cost_regression")
FIXED={
    "fixed_first1":("first","single",1),"fixed_first2":("first","single",2),
    "fixed_first4":("first","single",4),"fixed_deep1":("deep","single",1),
    "fixed_deep2":("deep","single",2),"fixed_deep4":("deep","single",4),
    "fixed_first_elastic2":("first","plain2",4),"fixed_deep_elastic2":("deep","plain2",4),
    "fixed_first_cycle_rlt4":("first","cycle_rlt4",4),"fixed_deep_cycle_rlt4":("deep","cycle_rlt4",4)}
DEFAULT=dict(F.DEFAULT,controller_n=32,eval_n=8,seeds=[137])
PANELS=("cal","normal","stress")


def validate(cfg):
    F.validate(cfg)
    if len(cfg["seeds"])!=1 or cfg["base_n"]>256 or cfg["adapter_n"]>96 or cfg["controller_n"]>32 or cfg["eval_n"]>8:
        raise ValueError("FG-09 task resource cap")


def source_manifest():
    here=Path(__file__).resolve().parent
    files=[here/n for n in ("gym.py","fg06_elastic.py","fg07_controller.py","fg09_research.py","FG09_RESEARCH_PROTOCOL.md")]
    files.append(Path(B.__file__).resolve())
    root=here.parents[2]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def task_root(role,index):
    return G.digest([ROOT_SCHEMA,role,index])


def root_offset(role,index):
    return 50000+(ROLES.index(role)*4+index)*1024


def partitions(role,index,cfg):
    offset=root_offset(role,index)
    spec={"base_train":("base_train",0,cfg["base_n"]),
          "adapter_train":("adapter_train",0,cfg["adapter_n"]),
          "controller_train":("adapter_train",256,cfg["controller_n"]),
          "cal":("cal",0,cfg["eval_n"]),"normal":("dev",0,cfg["eval_n"]),
          "stress":("shift",0,cfg["eval_n"])}
    sets={name:[G.world(split,offset+delta+i) for i in range(n)]
          for name,(split,delta,n) in spec.items()}
    seen=set()
    for ws in sets.values():
        keys={tuple(w["prefix"]) for w in ws}
        if len(keys)!=len(ws) or seen&keys:
            raise ValueError("inner prefix overlap")
        seen.update(keys)
    return sets


def lineage_keys(worlds):
    return {key for w in worlds for key in
            (B.lineage("floodgates-root",w["root"]),B.lineage("floodgates-prefix",G.digest(w["prefix"])))}


def reference_ledger(worlds):
    n=len(worlds)
    return {"reference_roots":n,
            "reference_token_block_visits":n*((G.PREFIX-1)*G.LOOPS+G.GENERATE*G.LOOPS+
                sum((G.PREFIX+t)*G.LOOPS for t in range(G.GENERATE)))}


def stable_model_descriptor(choice,controller,target_hash,adapter_hashes):
    state=None if controller is None else {k:v.detach().cpu().tolist() for k,v in controller.state_dict().items()}
    return {"schema":ROOT_SCHEMA+".policy","choice":choice,"target_hash":target_hash,
            "adapter_hashes":adapter_hashes,"controller_state":state}


def load_controller(descriptor):
    state=descriptor["controller_state"]
    if state is None:
        return None
    m=F.Controller()
    m.load_state_dict({k:torch.tensor(v,dtype=m.state_dict()[k].dtype) for k,v in state.items()})
    return m.eval().requires_grad_(False)


@torch.no_grad()
def evaluate_policy(model,adapters,descriptor,worlds,refs):
    choice=descriptor["choice"]
    ctrl=load_controller(descriptor)
    work=Counter()
    actions=Counter()
    true_correct=0
    for w in worlds:
        tokens=list(w["prefix"])
        _,past=model(torch.tensor([tokens[:-1]]))
        work["prefill_token_block_visits"]+=(G.PREFIX-1)*G.LOOPS
        ref=refs[w["root"]]
        while len(tokens)<G.PREFIX+G.GENERATE:
            obs=F.public_observation(tokens)
            if choice in FIXED:
                a=F.ACTIONS.index(FIXED[choice])
            else:
                a=F.choose(obs,choice,ctrl)
            tokens,past,r=F.execute(model,adapters,tokens,past,a)
            if tokens!=ref[:len(tokens)]:
                raise AssertionError("research policy greedy mismatch")
            for k,v in r["work"].items():
                work[k]=max(work[k],v) if k.startswith("peak_") else work[k]+v
            actions[str(a)]+=1
            if ctrl is not None:
                work["controller_calls"]+=1
                work["controller_dense_MACs"]+=F.FEATURES*F.HIDDEN+F.HIDDEN*len(F.ACTIONS)
        if tokens!=ref:
            raise AssertionError("research policy final greedy mismatch")
        true_correct+=sum(a==b for a,b in zip(tokens[G.PREFIX:],G.truth(w)[G.PREFIX:],strict=True))
    emitted=len(worlds)*G.GENERATE
    if work["emitted_tokens"]!=emitted:
        raise AssertionError("emitted work conservation")
    tariff=F.tariff(work)
    cost=tariff/emitted
    return B.metric(20/(20+cost),0,tariff,
                    cost_per_token=cost,truth_accuracy=true_correct/emitted,
                    emitted_tokens=emitted,greedy_mismatches=0,
                    serialized_block_calls=work["serialized_block_calls"],
                    total_block_token_visits=work["verify_block_token_visits"]+work["draft_block_token_visits"],
                    controller_calls=work["controller_calls"],controller_dense_MACs=work["controller_dense_MACs"],
                    actions=dict(actions),work=dict(work),reference_work=reference_ledger(worlds))


def public_metrics(m):
    # Compact per-experiment observation, sufficient for calibration choice.
    return {k:m[k] for k in ("quality","unsafe","inference_work","cost_per_token",
                             "truth_accuracy","emitted_tokens","greedy_mismatches",
                             "serialized_block_calls","total_block_token_visits","controller_calls",
                             "controller_dense_MACs","actions")}


class Task:
    gym=GYM
    choices=CHOICES

    def __init__(self,role,index,cfg=None):
        if role not in ROLES or type(index) is not int or not 0<=index<4:
            raise ValueError("unsupported task")
        self.role,self.index=role,index
        self.root=task_root(role,index)
        self.cfg=dict(DEFAULT if cfg is None else cfg)
        validate(self.cfg)
        # Different model initialization per role/index, paired inside a task.
        self.seed=self.cfg["seeds"][0]+ROLES.index(role)*100+index
        self.sets=partitions(role,index,self.cfg)
        self.lineage={"fit":lineage_keys(sum([self.sets[s] for s in
                        ("base_train","adapter_train","controller_train")],[])),
                      "cal":lineage_keys(self.sets["cal"]),
                      "final":lineage_keys(self.sets["normal"]+self.sets["stress"])}
        base_cfg={k:v for k,v in self.cfg.items() if k in G.DEFAULT}
        self.target,t=G.fit_target("final_only",self.seed,self.sets["base_train"],base_cfg)
        self.target_hash=t["weights_hash"]
        self.target_receipt=t
        self.adapters={}
        self.adapter_receipts={}
        for mode in F.MODES:
            a,r=G.fit_adapter(self.target,self.seed,mode,F.RANK,self.sets["adapter_train"],base_cfg)
            a.requires_grad_(False)
            self.adapters[mode]=a
            self.adapter_receipts[mode]=r
        self.adapter_hashes={mode:r["weights_hash"] for mode,r in self.adapter_receipts.items()}
        self.refs={"cal":{w["root"]:G.serial(self.target,w["prefix"]) for w in self.sets["cal"]}}
        self.table=None
        self.cal_receipts={}

    def public(self):
        return {"schema":SCHEMA,"root":self.root,"gym":GYM,"role":self.role,
                "task":"Choose a lossless drafting policy to minimize deterministic decode work. Every output must match the frozen target.",
                "required_first":CHOICES[0],"choices":list(CHOICES),
                "choice_notes":"fixed_first/deep use one pass unless elastic2 or cycle_rlt4; trailing number is draft length. heuristic uses committed repetition; imitation and cost_regression fit a controller on separate training roots.",
                "limits":{"turns":B.MAX_TURNS,"experiments":B.MAX_FITS,"controller_updates":self.cfg["controller_steps"],
                          "controller_batch":self.cfg["controller_batch"],"target_loops":G.LOOPS,
                          "adapter_rank":F.RANK,"max_new_tokens":G.GENERATE},
                "objective":"quality=20/(20+cost_per_token); cost=4*serialized block calls + token-block visits, divided by emitted tokens. Higher quality is better. Controller overhead and offline labels separately recorded.",
                "rule":"Run required control first. Try up to two other choices. Submit highest CAL quality among safe artifacts, tie by lower inference work, train work, then earliest. Claim improved, equivalent or regressed relative to control CAL. Final outcomes are private until submission.",
                "setup":{"target_parameters":self.target_receipt["parameters"],"target_hash":self.target_hash,
                         "target_updates":self.target_receipt["updates"],
                         "adapter_hashes":self.adapter_hashes,"common_adapter_updates":self.cfg["adapter_steps"]}}

    def setup_receipt(self):
        return {"target":self.target_receipt,"adapters":self.adapter_receipts,
                "cal_reference_work":reference_ledger(self.sets["cal"])}

    def fit(self,choice):
        if choice not in CHOICES:
            raise ValueError("unknown policy")
        ctrl=None
        fit={}
        new_label_work=None
        if choice in F.LEARNED:
            if self.table is None:
                self.table=F.tables(self.target,self.adapters,self.sets["controller_train"])
                new_label_work=F.table_receipt(self.table)
            ctrl,fit=F.fit(self.seed,choice,self.table,self.cfg)
        descriptor=stable_model_descriptor(choice,ctrl,self.target_hash,self.adapter_hashes)
        info={"weights_hash":G.digest(descriptor),"target_hash":self.target_hash,
              "adapter_hashes":self.adapter_hashes,"choice":choice,
              "controller_fit":fit or None,"new_offline_label_work":new_label_work,
              "reused_label_table":choice in F.LEARNED and new_label_work is None}
        return descriptor,info,fit.get("example_exposures",0)

    def check_descriptor(self,model,choice):
        if model["choice"]!=choice or model["target_hash"]!=self.target_hash or model["adapter_hashes"]!=self.adapter_hashes:
            raise ValueError("policy artifact binding")
        if G.weight_hash(self.target)!=self.target_hash or any(G.weight_hash(a)!=self.adapter_hashes[m]
                                                               for m,a in self.adapters.items()):
            raise AssertionError("target or adapter mutated")

    def calibrate(self,model,choice):
        self.check_descriptor(model,choice)
        m=evaluate_policy(self.target,self.adapters,model,self.sets["cal"],self.refs["cal"])
        self.cal_receipts[choice]=m
        return public_metrics(m)

    def evaluate_final(self,model,choice):
        self.check_descriptor(model,choice)
        panels=[]
        for split in ("normal","stress"):
            if split not in self.refs:
                self.refs[split]={w["root"]:G.serial(self.target,w["prefix"]) for w in self.sets[split]}
            panels.append(evaluate_policy(self.target,self.adapters,model,self.sets[split],self.refs[split]))
        return B.metric(sum(p["quality"] for p in panels)/len(panels),
                        sum(p["unsafe"] for p in panels),sum(p["inference_work"] for p in panels),panels=panels)

    def persist_setup(self,out):
        entries={"target":G.save(out,"target.pt",self.target),"adapters":{}}
        for mode,a in self.adapters.items():
            entries["adapters"][mode]=G.save(out,f"{mode}-adapter.pt",a)
        return entries


def reference(s):
    for c in REFERENCE:
        if not s.act({"action":"run_experiment","choice":c})["ok"]:
            raise AssertionError("reference choice failed")
    a=s.best()
    s.act({"action":"submit","artifact_id":a["artifact_id"],
           "claim":B.relation(a["cal"],s.artifacts["a1"]["cal"])})
    return s.finish()


def record(t,s,r,actor):
    row=B.record(t,s,r,actor)
    row.pop("receipt_hash")
    row.update(fg09_schema=ROOT_SCHEMA,source_manifest=source_manifest(),
               config=t.cfg,setup=t.setup_receipt(),calibration_ledgers=t.cal_receipts)
    return {**row,"receipt_hash":G.digest(row)}


def suite(out,per_role=1,cfg=None):
    if type(per_role) is not int or not 1<=per_role<=4:
        raise ValueError("role root cap")
    cfg=dict(DEFAULT if cfg is None else cfg)
    validate(cfg)
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    cells=[]
    lineage={}
    trace=out/"fg09_research_rollouts.jsonl"
    with trace.open("w",encoding="utf-8") as f:
        for role in ROLES:
            for i in range(per_role):
                t=Task(role,i,cfg)
                keys=set().union(*t.lineage.values())
                if any(keys&x for x in lineage.values()):
                    raise ValueError("cross-task lineage overlap")
                lineage[role,i]=keys
                s=B.Session(t)
                r=reference(s)
                rec=record(t,s,r,"scripted_reference")
                cells.append({"role":role,"index":i,"root":t.root,"result":r,
                              "lineage":rec["lineage"],"receipt_hash":rec["receipt_hash"]})
                if role=="trace_train":
                    f.write(json.dumps(rec,sort_keys=True,allow_nan=False)+"\n")
    report={"schema":ROOT_SCHEMA,"status":"public_development_only","config":cfg,"per_role":per_role,
            "source_manifest":source_manifest(),"cells":cells,
            "trace_sha256":hashlib.sha256(trace.read_bytes()).hexdigest()}
    (out/"fg09_research_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def audit(out):
    torch.set_num_threads(1)
    report=json.loads((out/"fg09_research_report.json").read_text(encoding="utf-8"))
    cfg=report["config"]
    validate(cfg)
    if report["source_manifest"]!=source_manifest() or report["schema"]!=ROOT_SCHEMA:
        raise ValueError("source binding")
    n=report["per_role"]
    cells={(x["role"],x["index"]):x for x in report["cells"]}
    if len(cells)!=len(report["cells"]) or set(cells)!={(r,i) for r in ROLES for i in range(n)}:
        raise ValueError("coverage")
    path=out/"fg09_research_rollouts.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report["trace_sha256"]:
        raise ValueError("trace inventory")
    stored={}
    for line in path.read_text(encoding="utf-8").splitlines():
        r=json.loads(line)
        k=(r["role"],r["index"])
        if k in stored or k[0]!="trace_train" or k not in cells or r["training_ready"] is not False:
            raise ValueError("trace custody")
        stored[k]=r
    if set(stored)!={("trace_train",i) for i in range(n)}:
        raise ValueError("missing trace")
    seen=set()
    for key,c in cells.items():
        t=Task(*key,cfg)
        keys=set().union(*t.lineage.values())
        if seen&keys:
            raise ValueError("cross-task lineage")
        seen.update(keys)
        s=B.Session(t)
        r=reference(s)
        rec=record(t,s,r,"scripted_reference")
        if rec["receipt_hash"]!=c["receipt_hash"] or r!=c["result"] or rec["lineage"]!=c["lineage"]:
            raise ValueError("research experiment refit replay")
        if key in stored and rec!=stored[key]:
            raise ValueError("trace exact replay")
    return {"status":"audited_development_only","research_sessions":len(cells),
            "research_rollouts":len(stored),"greedy_mismatches":0}


def interactive(role,index,out):
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    t=Task(role,index)
    s=B.Session(t)
    print(json.dumps({"public":t.public(),"observation":s.observation()},allow_nan=False),flush=True)
    while not s.done:
        raw=sys.stdin.readline(65537)
        if not raw:
            s.done=True
            break
        try:
            a=json.loads(raw,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x))) if len(raw)<=65536 else None
            G.digest(a)
        except (ValueError,TypeError,RecursionError,OverflowError):
            a=None
        response=s.act(a)
        print(json.dumps({"ok":response["ok"],"error":response["error"],
                          "observation":response["observation"]},allow_nan=False),flush=True)
    r=s.finish()
    selected=s.save_selected(out/"selected.weights")
    if selected:
        r["artifact"]=selected
    r["setup_artifacts"]=t.persist_setup(out)
    rec=record(t,s,r,"external_unbound")
    (out/"episode.json").write_text(json.dumps(rec,indent=2,allow_nan=False),encoding="utf-8")


def selftest():
    cfg=dict(DEFAULT,base_steps=4,adapter_steps=4,batch=4,base_n=16,adapter_n=8,
             eval_n=4,controller_n=8,controller_steps=4,controller_batch=4)
    t=Task("development",0,cfg)
    s=B.Session(t)
    assert not s.act({"action":"run_experiment","choice":"cost_regression"})["ok"]
    assert s.act({"action":"run_experiment","choice":"fixed_first4"})["ok"]
    assert not s.act({"action":"run_experiment","choice":"fixed_first4"})["ok"]
    assert s.act({"action":"submit","artifact_id":"a1","claim":"equivalent"})["ok"]
    assert s.finish()["score"]==0 and s.invalid==2
    print("FG-09 bounded session, mandatory control and invalid-action checks passed")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path)
    p.add_argument("--audit",type=Path)
    p.add_argument("--session",nargs=3)
    p.add_argument("--selftest",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--per-role",type=int,default=1)
    a=p.parse_args()
    torch.set_num_threads(1)
    if a.selftest:
        selftest()
    if a.audit:
        print(json.dumps(audit(a.audit),sort_keys=True))
    elif a.session and a.out:
        g,role,index=a.session
        if g!=GYM:
            p.error("only GSF-10 is supported")
        interactive(role,int(index),a.out)
    elif a.out:
        cfg=dict(DEFAULT,base_steps=4,adapter_steps=4,batch=4,base_n=16,adapter_n=8,
                 eval_n=4,controller_n=8,controller_steps=4,controller_batch=4) if a.smoke else None
        r=suite(a.out,a.per_role,cfg)
        print(json.dumps({"status":r["status"],"sessions":len(r["cells"])}))
    elif not a.selftest:
        p.error("choose --selftest, --session, --out or --audit")


if __name__=="__main__":
    main()
