"""FG-07: offline exact prefix-DP supervision for online draft-budget selection."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import time

import torch
from torch import nn

from . import gym as G
from . import fg06_elastic as E

SCHEMA="hybrid.gsf.floodgates.fg07.v1"
OFFSET=30000
CONTROLLER_OFFSET=31000
RANK=4
HIDDEN=32
FEATURES=5*G.VOCAB+6
MODES=("first","deep")
ACTIONS=(("serial","serial",0),)+tuple(
    (mode,arm,k) for mode in MODES
    for arm,ks in (("single",(1,2,4)),("plain2",(2,4)),("cycle_rlt4",(4,)))
    for k in ks
POLICIES=("serial","fixed_first4","cal_fixed","heuristic","imitation","cost_regression","oracle_hindsight")
LEARNED=("imitation","cost_regression")
PANELS=("cal","dev","shift","trace_train")
DEFAULT=dict(G.DEFAULT,seeds=[137,163],controller_n=64,controller_steps=160,controller_batch=32)
SERIAL=ACTIONS.index(("serial","serial",0))
FIRST4=ACTIONS.index(("first","single",4))
FIRST2=ACTIONS.index(("first","single",2))


def validate(cfg):
    G.validate({k:v for k,v in cfg.items() if k not in ("controller_n","controller_steps","controller_batch")})
    for k,lo,hi in (("controller_n",8,128),("controller_steps",1,512),("controller_batch",1,128)):
        if type(cfg.get(k)) is not int or not lo<=cfg[k]<=hi:
            raise ValueError("invalid controller resource cap")


def manifest():
    here=Path(__file__).resolve().parent
    return {n:hashlib.sha256((here/n).read_bytes()).hexdigest()
            for n in ("gym.py","fg06_elastic.py","fg07_controller.py","FG07_PROTOCOL.md")}


def partitions(cfg):
    n={"base_train":cfg["base_n"],"adapter_train":cfg["adapter_n"]}
    sets={s:[G.world(s,OFFSET+i) for i in range(n.get(s,cfg["eval_n"]))] for s in G.SPLITS}
    sets["controller_train"]=[G.world("adapter_train",CONTROLLER_OFFSET+i)
                              for i in range(cfg["controller_n"])]
    seen=set()
    for ws in sets.values():
        keys={tuple(w["prefix"]) for w in ws}
        if len(keys)!=len(ws) or seen&keys:
            raise ValueError("prefix lineage overlap")
        seen.update(keys)
    return sets


def split_hashes(sets):
    return {s:G.digest([w["root"] for w in ws]) for s,ws in sets.items()}


def public_observation(tokens):
    generated=len(tokens)-G.PREFIX
    if not 0<=generated<G.GENERATE:
        raise ValueError("invalid committed prefix length")
    return {"committed_tokens":list(tokens),"generated":generated,
            "remaining":G.GENERATE-generated}


def features(obs):
    p=obs["committed_tokens"]
    flags=[float(len(p)>=2*d and p[-d:]==p[-2*d:-d]) for d in range(1,5)]
    f=[float(t==i) for t in [*p[-4:],p[0]] for i in range(G.VOCAB)]
    f+=flags+[obs["generated"]/G.GENERATE,obs["remaining"]/G.GENERATE]
    if len(f)!=FEATURES:
        raise AssertionError("feature dimension")
    return f


def tariff(work):
    return 4*work["serialized_block_calls"]+work["verify_block_token_visits"]+work["draft_block_token_visits"]


@torch.no_grad()
def execute(model,adapters,tokens,past,action):
    if type(action) is not int or not 0<=action<len(ACTIONS):
        raise ValueError("invalid action")
    mode,arm,nominal=ACTIONS[action]
    remaining=G.PREFIX+G.GENERATE-len(tokens)
    if remaining<1 or any(k.shape[1]!=len(tokens)-1 or v.shape!=k.shape for k,v in past):
        raise ValueError("invalid cache anchor")
    if action==SERIAL:
        z,newpast=model(torch.tensor([[tokens[-1]]]),past)
        emitted=[int(z[0,-1,-1].argmax())]
        work={"verify_calls":1,"verify_block_token_visits":G.LOOPS,
              "draft_calls":0,"draft_passes":0,"draft_block_token_visits":0,
              "draft_tokens":0,"accepted_prefix":0,"discarded_cache_positions":0,
              "logical_draft_prefix_kv_bytes":0,"serialized_block_calls":G.LOOPS,
              "peak_provisional_kv_bytes":G.cache_size(newpast),"emitted_tokens":1}
        receipt={"action":action,"nominal_length":0,"actual_length":0,"draft_trace":[],
                 "draft":[],"accepted_prefix":0,"emitted":emitted,"target_predictions":emitted,
                 "cache_before":len(tokens)-1,"cache_provisional":len(tokens),
                 "cache_after":len(tokens),"discarded_cache_positions":0,"work":work}
        return tokens+emitted,newpast,receipt
    k=min(nominal,remaining)
    proposal,passes=E.draft(model,adapters[mode],torch.tensor([[tokens[-1]]]),past,k,mode,arm)
    guesses=proposal[0].argmax(-1).tolist()
    newtokens,newpast,r=G.verify_step(model,tokens,past,guesses,remaining)
    work={"verify_calls":r["verify_calls"],"verify_block_token_visits":r["verify_block_token_visits"],
          "draft_calls":1,"draft_passes":len(passes),
          "draft_block_token_visits":sum(x["block_token_visits"] for x in passes),
          "draft_tokens":len(guesses),"accepted_prefix":r["accepted_prefix"],
          "discarded_cache_positions":r["discarded_cache_positions"],
          "logical_draft_prefix_kv_bytes":sum(x["logical_prefix_kv_bytes"] for x in passes),
          "serialized_block_calls":G.LOOPS+len(passes),
          "peak_provisional_kv_bytes":r["peak_provisional_kv_bytes"],"emitted_tokens":len(r["emitted"])}
    receipt={**r,"draft_block_token_visits":work["draft_block_token_visits"],
             "action":action,"nominal_length":nominal,"actual_length":k,
             "draft_trace":passes,"work":work}
    return newtokens,newpast,receipt


def transition(receipt):
    w=receipt["work"]
    return {"advance":len(receipt["emitted"]),"cost":tariff(w),
            "serialized_block_calls":w["serialized_block_calls"],
            "token_block_visits":w["verify_block_token_visits"]+w["draft_block_token_visits"]}


@torch.no_grad()
def make_table(model,adapters,w):
    """Evaluator-only counterfactuals on exact verified target prefix states."""
    ref=G.serial(model,w["prefix"])
    states=[]
    work=Counter()
    # G.serial performs an initial prefill, one cached target step and
    # one cold full-prefix forward at each position.
    work["reference_prefill_token_block_visits"]=(G.PREFIX-1)*G.LOOPS
    work["reference_serial_token_block_visits"]=G.GENERATE*G.LOOPS
    work["reference_cold_check_token_block_visits"]=sum((G.PREFIX+t)*G.LOOPS for t in range(G.GENERATE))
    for t in range(G.GENERATE):
        tokens=ref[:G.PREFIX+t]
        _,past=model(torch.tensor([tokens[:-1]]))
        work["table_prefix_build_token_block_visits"]+=(len(tokens)-1)*G.LOOPS
        arows=[]
        for action in range(len(ACTIONS)):
            newtokens,_,r=execute(model,adapters,tokens,past,action)
            if newtokens!=ref[:len(newtokens)]:
                raise AssertionError(f"counterfactual greedy mismatch {w['root']} {t}/{action}")
            a=transition(r)
            if not 1<=a["advance"]<=G.GENERATE-t:
                raise AssertionError("invalid DP transition")
            arows.append(a)
            work["counterfactual_trials"]+=1
            work["counterfactual_serialized_block_calls"]+=a["serialized_block_calls"]
            work["counterfactual_token_block_visits"]+=a["token_block_visits"]
        states.append({"observation":public_observation(tokens),"actions":arows})
    values=[0]*(G.GENERATE+1)
    for t in reversed(range(G.GENERATE)):
        q=[a["cost"]+values[t+a["advance"]] for a in states[t]["actions"]]
        best=min(range(len(q)),key=lambda a:(q[a],a))
        values[t]=q[best]
        states[t].update(q=q,best=best,value=values[t])
    return {"root":w["root"],"reference_output":ref[G.PREFIX:],
            "states":states,"optimal_cost":values[0],"offline_work":dict(work)}


def tables(model,adapters,worlds):
    return [make_table(model,adapters,w) for w in worlds]


def table_receipt(ts):
    work=Counter()
    for t in ts:
        work.update(t["offline_work"])
    return {"table_sha256":G.digest(ts),"roots":len(ts),"states":len(ts)*G.GENERATE,
            "offline_work":dict(work)}


def fixed_cost(table,action):
    pos=0
    total=0
    while pos<G.GENERATE:
        x=table["states"][pos]["actions"][action]
        total+=x["cost"]
        pos+=x["advance"]
    return total


def cal_selection(ts):
    costs=[sum(fixed_cost(t,a) for t in ts) for a in range(len(ACTIONS))]
    return {"fixed_action":min(range(len(costs)),key=lambda a:(costs[a],a)),
            "fixed_costs":costs}


class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Linear(FEATURES,HIDDEN)
        self.head=nn.Linear(HIDDEN,len(ACTIONS))

    def forward(self,x):
        return self.head(torch.tanh(self.encoder(x)))


def fit(seed,policy,ts,cfg):
    if policy not in LEARNED:
        raise ValueError(policy)
    torch.manual_seed(seed+9000)
    m=Controller()
    before=G.weight_hash(m)
    states=[s for t in ts for s in t["states"]]
    x=torch.tensor([features(s["observation"]) for s in states],dtype=torch.float32)
    labels=torch.tensor([s["best"] for s in states],dtype=torch.long)
    regrets=torch.tensor([[(q-s["value"])/100. for q in s["q"]] for s in states],dtype=torch.float32)
    rng=torch.Generator().manual_seed(seed+9100)
    opt=torch.optim.AdamW(m.parameters(),lr=.003)
    draws=[]
    for _ in range(cfg["controller_steps"]):
        idx=torch.randint(len(x),(cfg["controller_batch"],),generator=rng)
        draws.append(idx.tolist())
        z=m(x[idx])
        if policy=="imitation":
            loss=nn.functional.cross_entropy(z,labels[idx])
        else:
            loss=nn.functional.mse_loss(z,regrets[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval().requires_grad_(False)
    final=G.weight_hash(m)
    if final==before:
        raise AssertionError("controller unchanged")
    return m,{"initial_hash":before,"weights_hash":final,"parameters":sum(p.numel() for p in m.parameters()),
              "updates":cfg["controller_steps"],"example_exposures":cfg["controller_steps"]*cfg["controller_batch"],
              "batch_sha256":G.digest(draws),"training_table_sha256":G.digest(ts),
              "last_loss":float(loss.detach())}


@torch.no_grad()
def choose(obs,policy,controller=None,fixed=None,oracle=None):
    if policy=="serial":
        return SERIAL
    if policy=="fixed_first4":
        return FIRST4
    if policy=="cal_fixed":
        return fixed
    if policy=="heuristic":
        if obs["remaining"]==1:
            return SERIAL
        p=obs["committed_tokens"]
        return FIRST4 if p[-1]==p[-2] or p[-2:]==p[-4:-2] else FIRST2
    if policy=="oracle_hindsight":
        if oracle is None:
            raise ValueError("oracle must be evaluator-only")
        return oracle
    if policy not in LEARNED or controller is None:
        raise ValueError("invalid learned policy")
    z=controller(torch.tensor([features(obs)],dtype=torch.float32))[0]
    return int(z.argmax() if policy=="imitation" else z.argmin())


@torch.no_grad()
def rollout(model,adapters,w,table,policy,controller,fixed):
    tokens=list(w["prefix"])
    _,past=model(torch.tensor([tokens[:-1]]))
    ref=tokens+table["reference_output"]
    public=[]
    targets=[]
    work=Counter()
    total=0
    regret_sum=0
    start=time.perf_counter()
    while len(tokens)<G.PREFIX+G.GENERATE:
        obs=public_observation(tokens)
        t=obs["generated"]
        state=table["states"][t]
        if obs!=state["observation"]:
            raise AssertionError("reference-prefix binding")
        action=choose(obs,policy,controller,fixed,
                      oracle=state["best"] if policy=="oracle_hindsight" else None)
        tokens,past,r=execute(model,adapters,tokens,past,action)
        a=transition(r)
        if a!=state["actions"][action] or tokens!=ref[:len(tokens)]:
            raise AssertionError("online/counterfactual transition mismatch")
        local_regret=state["q"][action]-state["value"]
        regret_sum+=local_regret
        total+=a["cost"]
        public.append({"observation":obs,"action_index":action,"receipt":r})
        targets.append({"position":t,"q_cost_to_go":state["q"],"best_action":state["best"],
                        "optimal_cost_to_go":state["value"],"decision_regret":local_regret})
        for k,v in r["work"].items():
            work[k]=max(work[k],v) if k.startswith("peak_") else work[k]+v
        work["decisions"]+=1
        if policy in LEARNED:
            work["controller_calls"]+=1
            work["controller_dense_MACs"]+=FEATURES*HIDDEN+HIDDEN*len(ACTIONS)
    elapsed=time.perf_counter()-start
    if total!=tariff(work) or total-table["optimal_cost"]!=regret_sum or tokens!=ref:
        raise AssertionError("DP regret/work/greedy conservation")
    if policy=="oracle_hindsight" and regret_sum!=0:
        raise AssertionError("hindsight policy not optimal")
    out=tokens[G.PREFIX:]
    truth=G.truth(w)[G.PREFIX:]
    row={"public_prefix":list(w["prefix"]),"public_decisions":public,
         "evaluator_only":{"decision_targets":targets,"reference_output":table["reference_output"],
                           "truth_output":truth,"optimal_tariff":table["optimal_cost"]},
         "verified_output":out,"truth_correct_tokens":sum(a==b for a,b in zip(out,truth,strict=True)),
         "greedy_parity":True,"tariff":total,"oracle_regret":regret_sum,"work":dict(work)}
    return row,elapsed


def aggregate(rows):
    w=Counter()
    actions=Counter()
    regret=0
    cost=0
    optimal=0
    truth=0
    for r in rows:
        cost+=r["tariff"]
        regret+=r["oracle_regret"]
        optimal+=r["evaluator_only"]["optimal_tariff"]
        truth+=r["truth_correct_tokens"]
        actions.update(str(x["action_index"]) for x in r["public_decisions"])
        for k,v in r["work"].items():
            w[k]=max(w[k],v) if k.startswith("peak_") else w[k]+v
    n=len(rows)
    return {"n":n,"tariff":cost,"tariff_per_token":cost/(n*G.GENERATE),
            "oracle_tariff":optimal,"oracle_regret":regret,"regret_per_token":regret/(n*G.GENERATE),
            "truth_accuracy":truth/(n*G.GENERATE),"work":dict(w),"actions":dict(actions),
            "serialized_call_reduction":n*G.GENERATE*G.LOOPS/w["serialized_block_calls"],
            "block_work_ratio":(w["verify_block_token_visits"]+w["draft_block_token_visits"])/(n*G.GENERATE*G.LOOPS)}


def compare(rows,control):
    d=[b["tariff"]-a["tariff"] for a,b in zip(rows,control,strict=True)]
    return {"n":len(d),"tariff_saved":sum(d),"wins":sum(x>0 for x in d),
            "ties":sum(x==0 for x in d),"losses":sum(x<0 for x in d)}


def evaluate(model,adapters,worlds,ts,policy,controller,fixed):
    rows=[]
    elapsed=0.
    for w,t in zip(worlds,ts,strict=True):
        if w["root"]!=t["root"]:
            raise ValueError("root binding")
        row,sec=rollout(model,adapters,w,t,policy,controller,fixed)
        rows.append(row)
        elapsed+=sec
    return {**aggregate(rows),"diagnostic_seconds_including_host_receipts":elapsed},rows


def record(seed,policy,w,target_hash,adapter_hashes,controller_hash,fixed,row):
    r={"schema":SCHEMA+".rollout","split":"trace_train","root":w["root"],"index":w["index"],
       "seed":seed,"policy":policy,"target_weights_hash":target_hash,
       "adapter_weights_hashes":adapter_hashes,"controller_weights_hash":controller_hash,
       "cal_fixed_action":fixed,"uses_private_oracle":policy=="oracle_hindsight",
       **row,"training_ready":False}
    r["receipt_hash"]=G.digest(r)
    return r


def run(out,cfg=None):
    cfg=dict(DEFAULT if cfg is None else cfg)
    validate(cfg)
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    sets=partitions(cfg)
    report={"schema":SCHEMA,"status":"public_development_only","config":cfg,
            "source_manifest":manifest(),"split_hashes":split_hashes(sets),
            "actions":[list(a) for a in ACTIONS],"tariff":"4*serialized_block_calls+token_block_visits",
            "targets":[],"adapters":[],"controllers":[],"tables":[],"cells":[],"paired":[]}
    path=out/"fg07_rollouts.jsonl"
    with path.open("w",encoding="utf-8") as trace:
        for seed in cfg["seeds"]:
            model,t=G.fit_target("final_only",seed,sets["base_train"],cfg)
            t.update(seed=seed,**G.save(out,f"{seed}-target.pt",model))
            report["targets"].append(t)
            adapters={}
            ah={}
            for mode in MODES:
                a,info=G.fit_adapter(model,seed,mode,RANK,sets["adapter_train"],cfg)
                a.requires_grad_(False)
                info.update(seed=seed,mode=mode,rank=RANK,**G.save(out,f"{seed}-{mode}.pt",a))
                report["adapters"].append(info)
                adapters[mode]=a
                ah[mode]=info["weights_hash"]
            ts={s:tables(model,adapters,sets[s]) for s in ("controller_train",*PANELS)}
            cal=cal_selection(ts["cal"])
            fixed=cal["fixed_action"]
            report["tables"].append({"seed":seed,"panels":{s:table_receipt(xs) for s,xs in ts.items()},"cal":cal})
            controllers={}
            ids={}
            for policy in LEARNED:
                c,info=fit(seed,policy,ts["controller_train"],cfg)
                info.update(seed=seed,policy=policy,**G.save(out,f"{seed}-{policy}.pt",c))
                report["controllers"].append(info)
                controllers[policy]=c
                ids[policy]=info["weights_hash"]
            current=[c for c in report["controllers"] if c["seed"]==seed]
            if len({c["initial_hash"] for c in current})!=1 or len({c["batch_sha256"] for c in current})!=1:
                raise AssertionError("controller pairing")
            paired={}
            for policy in POLICIES:
                cell={"seed":seed,"policy":policy,"target_weights_hash":t["weights_hash"],
                      "adapter_weights_hashes":ah,"controller_weights_hash":ids.get(policy),
                      "cal_fixed_action":fixed,"panels":{}}
                paired[policy]={}
                for split in PANELS:
                    stats,rows=evaluate(model,adapters,sets[split],ts[split],policy,controllers.get(policy),fixed)
                    cell["panels"][split]=stats
                    paired[policy][split]=rows
                    if split=="trace_train":
                        for w,row in zip(sets[split],rows,strict=True):
                            r=record(seed,policy,w,t["weights_hash"],ah,ids.get(policy),fixed,row)
                            trace.write(json.dumps(r,sort_keys=True,allow_nan=False)+"\n")
                report["cells"].append(cell)
            for policy in POLICIES:
                if policy!="cal_fixed":
                    for split in PANELS:
                        report["paired"].append({"seed":seed,"policy":policy,"control":"cal_fixed","split":split,
                                    **compare(paired[policy][split],paired["cal_fixed"][split])})
            if G.weight_hash(model)!=t["weights_hash"] or any(G.weight_hash(a)!=ah[m] for m,a in adapters.items()):
                raise AssertionError("frozen weights changed")
    report["trace_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/"fg07_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def audit(out):
    torch.set_num_threads(1)
    report=json.loads((out/"fg07_report.json").read_text(encoding="utf-8"))
    cfg=report["config"]
    validate(cfg)
    sets=partitions(cfg)
    if (report["schema"]!=SCHEMA or report["source_manifest"]!=manifest()
        or report["split_hashes"]!=split_hashes(sets) or report["actions"]!=[list(a) for a in ACTIONS]):
        raise ValueError("source, split or action binding")
    def cells_for(name,fields,expected):
        rows={tuple(r[k] for k in fields):r for r in report[name]}
        if len(rows)!=len(report[name]) or set(rows)!=expected:
            raise ValueError(f"{name} coverage")
        return rows
    seeds=cfg["seeds"]
    tr=cells_for("targets",("seed",),{(s,) for s in seeds})
    ar=cells_for("adapters",("seed","mode"),set(itertools.product(seeds,MODES)))
    cr=cells_for("controllers",("seed","policy"),set(itertools.product(seeds,LEARNED)))
    tab=cells_for("tables",("seed",),{(s,) for s in seeds})
    cells=cells_for("cells",("seed","policy"),set(itertools.product(seeds,POLICIES)))
    targets={}
    adapters={}
    controllers={}
    trace_tables={}
    fixeds={}
    worlds={w["root"]:w for w in sets["trace_train"]}
    for seed in seeds:
        model=G.load(G.Target(),out,tr[seed,])
        targets[seed]=model
        th=G.weight_hash(model)
        aa={}
        for mode in MODES:
            r=ar[seed,mode]
            if r["target_hash"]!=th or r["rank"]!=RANK:
                raise ValueError("adapter binding")
            aa[mode]=G.load(G.DraftAdapter(RANK),out,r)
        adapters[seed]=aa
        ah={m:ar[seed,m]["weights_hash"] for m in MODES}
        # Audit reconstructs training, CAL and dedicated trace counterfactuals.
        ts={s:tables(model,aa,sets[s]) for s in ("controller_train","cal","trace_train")}
        for s,xs in ts.items():
            if table_receipt(xs)!=tab[seed,]["panels"][s]:
                raise ValueError("offline DP table replay")
        cal=cal_selection(ts["cal"])
        if cal!=tab[seed,]["cal"]:
            raise ValueError("CAL fixed selection")
        fixeds[seed]=cal["fixed_action"]
        trace_tables[seed]={t["root"]:t for t in ts["trace_train"]}
        for policy in LEARNED:
            r=cr[seed,policy]
            c,info=fit(seed,policy,ts["controller_train"],cfg)
            if any(r[k]!=v for k,v in info.items()):
                raise ValueError("controller refit replay")
            controllers[seed,policy]=G.load(Controller(),out,r)
            if G.weight_hash(c)!=G.weight_hash(controllers[seed,policy]):
                raise ValueError("controller weight replay")
        for policy in POLICIES:
            c=cells[seed,policy]
            if (c["target_weights_hash"]!=th or c["adapter_weights_hashes"]!=ah
                or c["cal_fixed_action"]!=fixeds[seed]
                or c["controller_weights_hash"]!=(cr[seed,policy]["weights_hash"] if policy in LEARNED else None)):
                raise ValueError("cell binding")
    path=out/"fg07_rollouts.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report["trace_sha256"]:
        raise ValueError("trace inventory")
    seen=set()
    rows={k:[] for k in cells}
    for line in path.read_text(encoding="utf-8").splitlines():
        r=json.loads(line)
        bound=r.pop("receipt_hash")
        seed,policy,root=r["seed"],r["policy"],r["root"]
        key=(seed,policy)
        ident=(key,root)
        if (G.digest(r)!=bound or r["schema"]!=SCHEMA+".rollout" or r["split"]!="trace_train"
            or r["training_ready"] is not False or key not in cells or root not in worlds or ident in seen):
            raise ValueError("trace custody")
        w=worlds[root]
        replay,_=rollout(targets[seed],adapters[seed],w,trace_tables[seed][root],
                         policy,controllers.get(key),fixeds[seed])
        c=cells[key]
        obj=record(seed,policy,w,c["target_weights_hash"],c["adapter_weights_hashes"],
                   c["controller_weights_hash"],fixeds[seed],replay)
        if obj["receipt_hash"]!=bound:
            raise ValueError("controller/greedy/DP decision replay")
        rows[key].append(replay)
        seen.add(ident)
    if len(seen)!=len(cells)*len(worlds):
        raise ValueError("missing traces")
    for key,rs in rows.items():
        m={k:v for k,v in cells[key]["panels"]["trace_train"].items()
           if k!="diagnostic_seconds_including_host_receipts"}
        if aggregate(rs)!=m:
            raise ValueError("trace aggregate mismatch")
        if key[1]!="cal_fixed":
            p={"seed":key[0],"policy":key[1],"control":"cal_fixed","split":"trace_train",
               **compare(rs,rows[key[0],"cal_fixed"])}
            if p not in report["paired"]:
                raise ValueError("paired comparison mismatch")
    return {"status":"audited_development_only","targets":len(tr),"adapters":len(ar),
            "controllers":len(cr),"cells":len(cells),"rollouts":len(seen),"greedy_mismatches":0}


def selftest():
    torch.set_num_threads(1)
    torch.manual_seed(137)
    m=G.Target().eval().requires_grad_(False)
    a={mode:G.DraftAdapter(RANK).eval().requires_grad_(False) for mode in MODES}
    w=G.world("dev",OFFSET)
    t=make_table(m,a,w)
    assert len(ACTIONS)==13 and len(t["states"])==G.GENERATE
    assert t["optimal_cost"]<=fixed_cost(t,SERIAL)==20*G.GENERATE
    assert all(s["value"]==min(s["q"]) for s in t["states"])
    for policy in ("serial","fixed_first4","cal_fixed","heuristic","oracle_hindsight"):
        r,_=rollout(m,a,w,t,policy,None,cal_selection([t])["fixed_action"])
        assert r["greedy_parity"] and r["tariff"]>=t["optimal_cost"]
        if policy=="oracle_hindsight":
            assert r["oracle_regret"]==0
    cfg=dict(DEFAULT,controller_steps=2,controller_batch=4)
    receipts=[]
    for policy in LEARNED:
        c,receipt=fit(137,policy,[t],cfg)
        r,_=rollout(m,a,w,t,policy,c,FIRST4)
        assert r["greedy_parity"] and r["work"]["controller_calls"]==r["work"]["decisions"]
        receipts.append(receipt)
    assert receipts[0]["initial_hash"]==receipts[1]["initial_hash"]
    assert receipts[0]["batch_sha256"]==receipts[1]["batch_sha256"]
    print("FG-07 exact prefix DP, lossless online decisions and paired controller fits passed")


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
        cfg=dict(DEFAULT,base_steps=4,adapter_steps=4,batch=4,base_n=16,adapter_n=8,
                 eval_n=4,seeds=[137],controller_n=8,controller_steps=4,controller_batch=4) if a.smoke else None
        r=run(a.out,cfg)
        print(json.dumps({"status":r["status"],"targets":len(r["targets"]),
                          "controllers":len(r["controllers"]),"cells":len(r["cells"])}))
    if a.audit:
        print(json.dumps(audit(a.audit),sort_keys=True))
    if not (a.selftest or a.out or a.audit):
        p.error("choose --selftest, --out or --audit")


if __name__=="__main__":
    main()
