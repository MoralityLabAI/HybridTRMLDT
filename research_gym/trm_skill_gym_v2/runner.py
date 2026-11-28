"""CPU checkpoint replay, matched-stratum candidate selection and controls.

Run from repository root. No paid API or GPU. Live compact Inky arm is opt-in.
The default is a small diagnostic; --count 128 --research-worlds 512 gives a larger replay.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from research_gym.trm_skill_gym import tasks as T, modules as M, train as TR
from research_gym.trm_skill_gym import dsl, campaign as C, skill_templates as S
from research_gym.trm_skill_gym.inky import InkyClient, InkyUnavailable
from .core import paired_rank, strict_selected, graph_certificate, research_values, eig, entropy, distill_rows

SOURCE = Path("experiments/trm_skill_gym_01/run")
ARMS = ("A_existing_skill", "B_feedforward", "C_gru", "D_trm_direct", "F_trm_rlt", "H_inky_engineered")
CAL = 1_100_000
TEST = 1_200_000
DEPTHS = (4, 8, 12)
GRAPH = ("retrieval", "deduction")


def hard_graph(skill, depth, root):
    """Chain with a disjoint distractor component, random slots and no label-correlated hints."""
    if skill not in GRAPH or not 1 <= depth <= T.N - 2:
        raise ValueError("bad graph depth")
    rng = np.random.default_rng([root, depth, GRAPH.index(skill)])
    order = rng.permutation(T.N)
    inst = T._empty()
    inst.mask[:] = 1
    inst.glob[0] = 1
    for i in range(1, depth + 1):
        inst.rel[order[i], order[i - 1]] = 1
    for i in range(depth + 2, T.N):
        inst.rel[order[i], order[i - 1]] = 1
    q = int(order[depth])
    if skill == "retrieval":
        inst.cand[q, 0] = inst.aux[q, 0] = 1
        inst.label[order[:depth]] = 1
        inst.cand[:, 1] = .5
        inst.cand[:, 10] = inst.rel.sum(1) / 4
        inst.cand[:, 11] = inst.rel.sum(0) / 4
    else:
        inst.label[order[:depth + 1:2]] = 1
        inst.label[order[depth + 1::2]] = 1
        inst.cand[:, 0] = .5
        inst.cand[:, 1] = inst.rel.sum(1) / 3
        inst.cand[:, 2] = inst.rel.sum(0) / 3
        inst.cand[:, 3] = inst.rel.sum(1) == 0
    return inst


def pools(skill, base, count, final=False):
    names = T.FAMILIES[skill]["dev"] + (T.FAMILIES[skill]["withheld"] if final else [])
    out = {f: T.stack([T.make_instance(skill, f, base+i) for i in range(count)]) for f in names}
    if skill in GRAPH:
        for depth in DEPTHS + ((14,) if final else ()):
            out[f"chain_{depth}"] = T.stack([hard_graph(skill, depth, base+i) for i in range(count)])
    return out


class ResetModule(M.SkillModule):
    """Coordinate causal probe, an input for a later BPD subspace study."""
    reset_dims = ()
    def visit(self, cand, batch, st, planned_visits):
        if st["u"] > 0 and self.reset_dims and "z" in st:
            z = st["z"].clone()
            z[..., list(self.reset_dims)] = self.z0[list(self.reset_dims)]
            st = dict(st, z=z)
        return super().visit(cand, batch, st, planned_visits)


def checkpoint(source, skill, arm, seed):
    if arm == ARMS[0]:
        return None, C.base_trm()
    ck = torch.load(source / "checkpoints" / f"{skill}__{arm}__s{seed}.pt",
                    map_location="cpu", weights_only=True)
    model = ResetModule(ck["config"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck["config"]


@torch.no_grad()
def evaluate(skill, model, cfg, b, visits=None):
    M.flops_reset()
    t0 = time.process_time()
    out = M.episode(model, skill, b, cfg, visits=visits, heuristic=model is None)
    work = M.flops_read()
    acc = out["exact"][-1].cpu().numpy()
    return acc, {"exact": float(acc.mean()), "flops": work/len(acc),
                 "calls": out["vcalls"] + 1, "cpu_s": time.process_time()-t0}


def cat_features(p):
    # Only public tensors. No task label, aux, verifier feedback or model outputs.
    mask = p["mask"].numpy()
    cand = p["cand"][..., :T.FS].numpy()
    denom = np.maximum(mask.sum(1, keepdims=True), 1)
    mean = (cand * mask[..., None]).sum(1)/denom
    var = (((cand - mean[:, None])**2)*mask[..., None]).sum(1)/denom
    return np.concatenate([np.ones((len(mask),1)), p["glob"].numpy(),
                           denom/T.N, p["rel"].numpy().sum((1,2))[:,None]/T.N**2,
                           mean, var], axis=1)


def ridge(x, y, penalty=1.0):
    return np.linalg.solve(x.T@x + penalty*np.eye(x.shape[1]), x.T@y)


def route(cal, test, cal_acc, test_acc, costs, arms, cap):
    # Fit per-arm accuracy predictors; routing sees only public instance features.
    keys = list(cal)
    x = np.concatenate([cat_features(cal[k]) for k in keys])
    y = np.stack([np.concatenate([cal_acc[a][k] for k in keys]) for a in arms], 1)
    beta = ridge(x, y)
    eligible = [i for i, a in enumerate(arms) if costs[a]["flops"] <= cap and costs[a]["calls"] <= 2]
    if not eligible:
        raise ValueError("no eligible route")
    fixed = eligible[int(np.argmax(y[:,eligible].mean(0)))]
    out = {}
    for k,b in test.items():
        xt = cat_features(b)
        scores = xt@beta
        chosen = np.array(eligible)[scores[:,eligible].argmax(1)]
        truth = np.stack([test_acc[a][k] for a in arms],1)
        out[k] = {"learned": float(truth[np.arange(len(xt)), chosen].mean()),
                  "fixed": float(truth[:,fixed].mean()),
                  "oracle": float(truth[:,eligible].max(1).mean()),
                  "selected": {arms[i]: int((chosen==i).sum()) for i in eligible},
                  "mean_model_flops": float(np.mean([costs[arms[i]]["flops"] for i in chosen])),
                  "router_dense_flops": 2*xt.shape[1]*len(arms)}
    return {"fixed_arm": arms[fixed], "strata": out, "cap_model_flops": cap}


def cert_report(skill, pool):
    # Compare public certificate against hidden labels only at grading time.
    results = {}
    for k,b in pool.items():
        rows = {}
        for depth in (1, 2, 4, 8, 16):
            known=wrong=work=total=solved=0
            for i in range(len(b["mask"])):
                active = torch.where(b["mask"][i] > 0)[0].tolist()
                index = {v:j for j,v in enumerate(active)}
                parents = [[index[j] for j in active if b["rel"][i,v,j] > 0] for v in active]
                q = next((index[v] for v in active if b["cand"][i,v,0] == 1), None) if skill=="retrieval" else None
                cert, w = graph_certificate(skill, parents, q, depth)
                lab = b["label"][i,active].tolist()
                known += sum(c != 0 for c in cert)
                wrong += sum(c != 0 and (c==1) != bool(l) for c,l in zip(cert,lab))
                solved += int(all(c != 0 for c in cert) and all((c==1)==bool(l) for c,l in zip(cert,lab)))
                total += len(active)
                work += w
            rows[str(depth)]={"coverage":known/total,"violations":wrong,"edge_exams":work,
                              "symbolic_exact":solved/len(b["mask"])}
        results[k]=rows
    return results


def reset_probe(skill, model, cfg, cal, test, seed):
    if cfg["core"] != "trm":
        raise ValueError("requires TRM")
    # One calibration family per stratum. Single-coordinate causal effects; freeze top quartile.
    def metrics(p, dims):
        model.reset_dims = tuple(dims)
        return {k:evaluate(skill, model, cfg, b)[1]["exact"] for k,b in p.items()}
    baseline = metrics(cal, ())
    scores=[]
    for j in range(model.H):
        m=metrics(cal,(j,))
        scores.append(sum(m[k]-baseline[k] for k in cal)/len(cal))
    rank = max(1, model.H//4)
    selected = sorted(range(model.H), key=lambda j:(-scores[j],j))[:rank]
    random = np.random.default_rng(seed+27).choice(model.H,rank,replace=False).tolist()
    results = {name:metrics(test,dims) for name,dims in
               (("keep",()),("full",range(model.H)),("selected",selected),("random",random))}
    model.reset_dims=()
    return {"selected_dims":selected,"random_dims":random,"cal_effect":scores,"test":results}


def replay(source, count, seeds, deadline):
    rows={}
    for seed in seeds:
        for skill in T.SKILLS:
            if time.monotonic() >= deadline:
                return rows
            cal=pools(skill,CAL,count)
            test=pools(skill,TEST,count,True)
            ca,ta,cm,tm={},{},{},{}
            for arm in ARMS:
                model,cfg=checkpoint(source,skill,arm,seed)
                ca[arm],cm[arm],ta[arm],tm[arm]={},{},{},{}
                for k,b in cal.items():
                    ca[arm][k],cm[arm][k]=evaluate(skill,model,cfg,b)
                for k,b in test.items():
                    ta[arm][k],tm[arm][k]=evaluate(skill,model,cfg,b)
            ranking=sorted(ARMS,key=lambda a:paired_rank(cm[a],cal),reverse=True)
            costs={a:{"flops":max(m["flops"] for m in cm[a].values()),
                      "calls":max(m["calls"] for m in cm[a].values())} for a in ARMS}
            # Flop cap from direct TRM, with equal verifier cap; report host CPU separately.
            row={"calibration":cm,"heldout":tm,"selected":ranking[0],"rank":ranking,
                 "router":route(cal,test,ca,ta,costs,ARMS,costs["D_trm_direct"]["flops"])}
            if skill in GRAPH:
                row["certificates"]=cert_report(skill,test)
            if skill in ("research","stopping","curation","repair"):
                model,cfg=checkpoint(source,skill,"F_trm_rlt",seed)
                row["reset"]=reset_probe(skill,model,cfg,cal,test,seed)
            rows[f"{skill}/s{seed}"]=row
    return rows


def research_world(root):
    r=np.random.default_rng(root)
    p=r.dirichlet(np.full(6,.7))
    likelihoods=r.uniform(.03,.97,(8,6))
    costs=r.uniform(0,.22,8)
    q1=research_values(p,likelihoods,costs,1)
    q2=research_values(p,likelihoods,costs,2)
    x=[]
    for i,l in enumerate(likelihoods):
        mu=float(p@l)
        x.append([1,q1[i],costs[i],mu,float(p@((l-mu)**2)),entropy(p),
                  max(q1[j] for j in q1 if j!=i),
                  float(np.max(np.abs(likelihoods-l),axis=1).mean())])
    return np.array(x),np.array(list(q1.values())),np.array(list(q2.values()))


def research_policy(n):
    train=[research_world(1_300_000+i) for i in range(n)]
    beta=ridge(np.concatenate([x for x,_,_ in train]),np.concatenate([q for _,_,q in train]))
    results={k:[] for k in ("one_step_eig","learned","oracle","variance")}
    regret={k:[] for k in results}
    for i in range(n):
        x,q1,q2=research_world(1_400_000+i)
        def choose(s):
            return int(np.argmax(s)) if max(s)>0 else -1
        choices={"one_step_eig":choose(q1),"learned":choose(x@beta),"oracle":choose(q2),
                 "variance":choose(8*x[:,4]-x[:,2])}
        best=max(0,max(q2))
        for k,a in choices.items():
            v=0. if a==-1 else float(q2[a])
            results[k].append(v)
            regret[k].append(best-v)
    return {k:{"mean_two_step_value":float(np.mean(v)),"mean_regret":float(np.mean(regret[k]))} for k,v in results.items()}


def leakage_control(count, steps):
    # Independent shuffled supervision with/without feedback; no in-loop LDT.
    result={}
    for skill in ("research","stopping"):
        train=T.stack([T.make_instance(skill,T.FAMILIES[skill]["dev"][i%2],1_500_000+i) for i in range(max(256,count))])
        test=T.stack([T.make_instance(skill,T.FAMILIES[skill]["dev"][i%2],1_600_000+i) for i in range(count)])
        for shuffle in (False,True):
            for calls in (0,1):
                cfg=dsl.validate({"core":"trm","verifier_visits":calls})
                model,info=TR.train_module(skill,cfg,steps,91,train,shuffled=shuffle)
                _,m=evaluate(skill,model,cfg,test)
                result[f"{skill}/shuffle={shuffle}/feedback={calls}"]={"train":info,"eval":m}
        # True majority-slot estimate on calibration labels; score on fresh test labels.
        slot=int(train["label"].sum(0).argmax())
        result[f"{skill}/majority"]={"slot":slot,"label_accuracy":float(test["label"][:,slot].mean())}
    return result


def live_inky(n):
    rows=[]
    # Same fresh public instances for both prompts. Equal cap and reasoning effort.
    for skill in ("research","stopping"):
        for i in range(n):
            fam=T.FAMILIES[skill]["dev"][i%2]
            inst=T.make_instance(skill,fam,1_700_000+i)
            payload=S.serialize_instance(skill,inst)
            for variant in ("original","compact"):
                system=("Return only a JSON object with selected integer indices. No prose."
                        if variant=="compact" else
                        "Read the skill and reason carefully. Return only the JSON object.")
                user=(S.TEMPLATES[skill]["objective"]+"\n"+S.TEMPLATES[skill]["action"]+
                      "\n"+payload if variant=="compact" else S.render(skill)+"\n"+payload)
                client=InkyClient(max_completion_tokens=2048,reasoning_effort="low",max_retries=0)
                rec={"skill":skill,"root":1_700_000+i,"variant":variant}
                try:
                    r=client.chat(system,user,temperature=0,seed=7)
                    rec.update(finish=r["finish_reason"],usage=r["usage"],seconds=r["seconds"])
                    selected=strict_selected(r["content"],set(np.flatnonzero(inst.mask)),True)
                    pred=torch.zeros(1,T.N)
                    pred[0,selected]=1
                    rec.update(status="ok",exact=float(T.exact(skill,T.stack([inst]),pred)[0]))
                except (InkyUnavailable,ValueError) as exc:
                    rec.update(status="invalid_or_unavailable",error=type(exc).__name__,exact=0)
                rows.append(rec)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=SOURCE)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--count",type=int,default=24)
    p.add_argument("--seeds",type=int,nargs="+",default=[0,1])
    p.add_argument("--research-worlds",type=int,default=64)
    p.add_argument("--shuffle-steps",type=int,default=0)
    p.add_argument("--hours",type=float,default=8)
    p.add_argument("--live-inky",type=int,default=0)
    args=p.parse_args()
    if min(args.count,args.research_worlds)<=0 or not 0<args.hours<=8 or args.live_inky<0:
        p.error("invalid budget")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    args.out.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+3600*args.hours
    archive=[json.loads(l) for l in (args.source/"search/archive.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    records=list(distill_rows(archive))
    (args.out/"distill.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in records),encoding="utf-8")
    results={"profile":{"count":args.count,"seeds":args.seeds,"research_worlds":args.research_worlds,
                        "mode":"frozen_checkpoint_diagnostic","source":str(args.source)},
             "distill_counts":{k:sum(r["split"]==k for r in records) for k in ("train","test")},
             "research":research_policy(args.research_worlds)}
    results["replay"]=replay(args.source,args.count,args.seeds,deadline)
    results["complete"]=len(results["replay"])==len(args.seeds)*len(T.SKILLS)
    C.write_json(args.out/"results.json",results)
    if args.shuffle_steps and time.monotonic()<deadline:
        results["leakage_control"]=leakage_control(args.count,args.shuffle_steps)
        C.write_json(args.out/"results.json",results)
    if args.live_inky and time.monotonic()<deadline:
        results["live_inky"]=live_inky(args.live_inky)
        C.write_json(args.out/"results.json",results)
    print(json.dumps({"complete":results["complete"],"replay_rows":len(results["replay"]),
                      "distill_counts":results["distill_counts"],"research":results["research"]},indent=2))


if __name__=="__main__":
    main()
