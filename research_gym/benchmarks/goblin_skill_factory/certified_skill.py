"""GSF-03: train a reusable repair policy against the native CCA-05 skill.

The native storyworld, certificate issuer/checker, and strict lattice repair
function remain the environment's authority. Neural policies only nominate.
All partitions are public development partitions, grouped by underlying state.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch import nn

from experiments.cca05_certified_trm_repair.src import semantics as M
from research_gym.core.certified_repair import bounded_recurrent_repair, execute_certified
from research_gym.core.hybrid import CandidateState
from research_gym.core.typed_soundness import SoundnessType
from research_gym.envs.coupled_storyworld import StoryState

SCHEMA = "hybrid.gsf.cca_repair.v1"
ACTIONS = tuple(M.ACTIONS)
N = len(ACTIONS)
FULL = frozenset(ACTIONS)
LEARNED = ("direct", "residual", "refresh", "rlt", "damp")
CONTROLS = ("static_refresh", "lexical", "ldt_scan", "oracle_scan")
ARMS = LEARNED + CONTROLS
SPLITS = ("train", "cal", "dev", "shift", "trace_train")
WIDTH_IN = 7 + N
PASSES = 4
BUDGETS = (1, 3, 5)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def first_replay_difference(actual, expected):
    """Report the first canonical JSON mismatch without changing replay rules."""
    def compare(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                return f"{path}: keys {sorted(set(a) ^ set(b))!r}"
            for key in sorted(a):
                delta = compare(a[key], b[key], f"{path}.{key}")
                if delta is not None:
                    return delta
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return f"{path}: length {len(a)} != {len(b)}"
            for i, (x, y) in enumerate(zip(a, b)):
                delta = compare(x, y, f"{path}[{i}]")
                if delta is not None:
                    return delta
        elif json.dumps(a, sort_keys=True, allow_nan=False) != json.dumps(b, sort_keys=True, allow_nan=False):
            return f"{path}: replay={a!r}, recorded={b!r}"
        return None
    # JSON round-trip normalizes tuple/list representation just as digest does.
    return compare(json.loads(json.dumps(actual, allow_nan=False)),
                   json.loads(json.dumps(expected, allow_nan=False)), "$")


def state_root(state):
    # Group every scenario, horizon, seed, mask and alternate policy together.
    return digest([SCHEMA, M.ENV_DIGEST, list(M.st(state))])


def partitions(train_n, eval_n):
    states = sorted({s for _, s in M.bench_domain()}, key=state_root)
    if not 1 <= train_n or not 1 <= eval_n or train_n + 4*eval_n > len(states):
        raise ValueError("invalid state counts")
    result = {}
    offset = 0
    for split in SPLITS:
        n = train_n if split == "train" else eval_n
        result[split] = states[offset:offset+n]
        offset += n
    return result


def packet(state, scenario, horizon, split):
    return {"schema": SCHEMA, "root": state_root(state), "state": list(M.st(state)),
            "scenario": scenario, "horizon": horizon, "split": split,
            "env": M.ENV_DIGEST}


def cases(states, split):
    # Train at horizon 3/6; CAL/dev at 6; shift to shorter, unseen horizon 2.
    horizons = (3, 6) if split == "train" else ((2,) if split == "shift" else (6,))
    return [packet(s, sc, k, split) for s in states for sc in M.SCENARIOS for k in horizons]


def unpack(p):
    return p["scenario"], StoryState(*p["state"]), p["horizon"]


def features(p, allowed):
    t,e,h,s = p["state"]
    return [t/3, e/4, h/4, s/5, p["horizon"]/6,
            float(p["scenario"] == M.SCENARIOS[0]),
            float(p["scenario"] == M.SCENARIOS[1])] + [float(a in allowed) for a in ACTIONS]


def training_arrays(packets):
    x,y,targets,valid = [],[],[],[]
    for p in packets:
        sc,s,k = unpack(p)
        safe = M.certified_set(sc,s,k)
        # Masked examples include reachable rejected-action removals and
        # arbitrary subset stress; nothing from other partitions enters fits.
        rng = random.Random(int(digest([p["root"],sc,k,"mask"]),16))
        bad = sorted(FULL-safe)
        reachable = FULL - frozenset(rng.sample(bad, rng.randrange(len(bad)+1)))
        stress = frozenset(rng.sample(list(ACTIONS), rng.randrange(1,N+1)))
        for allowed in (FULL, reachable, stress):
            if not allowed:
                continue
            pool = safe & allowed
            x.append(features(p, allowed))
            y.append([M.utility(sc,s,k,a) if a in pool else 0.0 for a in ACTIONS])
            best = M.oracle_action(sc,s,k,pool)
            targets.append(ACTIONS.index(best) if best is not None else 0)
            valid.append(best is not None)
    return (torch.tensor(x,dtype=torch.float32), torch.tensor(y,dtype=torch.float32),
            torch.tensor(targets,dtype=torch.long), torch.tensor(valid,dtype=torch.bool))


class RepairTRM(nn.Module):
    """Small tied recurrent MLP, with explicit candidate-mask observation."""
    def __init__(self,width,arm):
        super().__init__()
        if arm not in LEARNED:
            raise ValueError(arm)
        self.arm = arm
        self.encoder = nn.Linear(WIDTH_IN,width)
        self.cell = nn.Linear(2*width,width)
        self.head = nn.Linear(width,N)

    def forward(self,x):
        encoded = torch.tanh(self.encoder(x))
        h = encoded
        steps = 1 if self.arm == "direct" else PASSES
        outputs = []
        for i in range(steps):
            src = encoded if self.arm != "residual" or i == 0 else torch.zeros_like(encoded)
            delta = torch.tanh(self.cell(torch.cat((h,src),dim=-1)))
            alpha = 1.
            if self.arm == "rlt":
                alpha = 1/(steps-i)
            elif self.arm == "damp" and i != steps-1:
                alpha = sum(1/j for j in range(2,steps+1))/(steps-1)
            h = h + alpha*delta
            outputs.append(self.head(h))
        return torch.stack(outputs,dim=1)


def weight_hash(model):
    return digest({k:v.detach().tolist() for k,v in model.state_dict().items()})


def fit(arm,width,seed,arrays,steps,batch):
    torch.manual_seed(seed)
    model = RepairTRM(width,arm)
    before = weight_hash(model)
    opt = torch.optim.AdamW(model.parameters(),lr=.003)
    rng = torch.Generator().manual_seed(seed+103)
    x,y,targets,valid = arrays
    for _ in range(steps):
        idx = torch.randint(len(x),(batch,),generator=rng)
        logits = model(x[idx])
        loss = nn.functional.mse_loss(logits.sigmoid(),y[idx,None,:].expand_as(logits))
        # Same normalized objective at each recurrent visit. No CE for
        # impossible masks, which have only a zero-value regression target.
        eligible = valid[idx]
        if eligible.any():
            selected = logits[eligible]
            mask = x[idx][eligible,-N:].bool()
            selected = selected.masked_fill(~mask[:,None,:], -1e4)
            loss = loss + .25*nn.functional.cross_entropy(
                selected.flatten(0,1),targets[idx][eligible].repeat_interleave(logits.shape[1]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    after = weight_hash(model)
    if before == after:
        raise RuntimeError("unchanged weights")
    return model,{"initial_hash":before,"weights_hash":after,"updates":steps,
                  "parameters":sum(p.numel() for p in model.parameters()),
                  "train_cell_visits":steps*batch*(1 if arm=="direct" else PASSES),
                  "training_examples":len(x)}


@torch.no_grad()
def learned_scores(model,p,allowed):
    return model(torch.tensor([features(p,allowed)],dtype=torch.float32))[0].tolist()


def choose(scores,allowed):
    return max(allowed,key=lambda a:(scores[ACTIONS.index(a)],-ACTIONS.index(a)))


def episode(p,arm,model=None,budget=3):
    if arm not in ARMS or budget not in BUDGETS:
        raise ValueError("invalid arm or budget")
    sc,s,k = unpack(p)
    ctx = M.context(sc,s,k)
    domain = CandidateState({"action":FULL})
    visits = []
    counters = Counter()
    certificate_receipts = []
    initial_scores = None
    if arm == "static_refresh":
        initial_scores = learned_scores(model,p,FULL)
        counters["cell_visits"] += len(initial_scores)
        counters["proposer_queries"] += 1
    preferred = None
    prescan = False
    if arm in ("ldt_scan","oracle_scan"):
        prescan = True
        verify = M.CountingVerifier(sc,s,k)
        if arm == "ldt_scan":
            preferred = M.ldt_fallback(verify)
        else:
            pool = frozenset(a for a in ACTIONS if verify(a))
            preferred = M.oracle_action(sc,s,k,pool)
            counters["oracle_utility_evals"] += len(pool)
        counters["prescan_checks"] += verify.invocations
        counters["prescan_distinct"] += verify.distinct

    def propose(current):
        allowed = current.domains["action"]
        scores = None
        if arm in LEARNED:
            scores = learned_scores(model,p,allowed)
            counters["cell_visits"] += len(scores)
            counters["proposer_queries"] += 1
            action = choose(scores[-1],allowed)
        elif arm == "static_refresh":
            scores = initial_scores if not visits else None
            action = choose(initial_scores[-1],allowed)
        elif arm == "lexical":
            action = next(a for a in ACTIONS if a in allowed)
        else:
            if preferred not in allowed:
                raise AssertionError("prescan selected invalid action")
            action = preferred
        visits.append({"allowed":[a for a in ACTIONS if a in allowed],
                       "nomination":action,"scores":scores})
        return action

    def admissible(a):
        # Issuance performs its own admissibility test; certificate execution
        # independently replays the witness under the current binding.
        counters["issuer_checks"] += 1
        cert = M.issue(sc,s,k,a)
        if cert is None:
            certificate_receipts.append({"action":a,"certificate":None,"accepted":False})
            return False
        counters["certificate_checks"] += 1
        current = CandidateState({"action":frozenset(visits[-1]["allowed"])})
        accepted = execute_certified(ctx,a,cert,M.check_certificate,domain=current) == a
        certificate_receipts.append({"action":a,"certificate":cert.to_jsonable(),"accepted":accepted})
        if not accepted:
            raise AssertionError("issuer/checker parity failure")
        return True

    def eliminate(proposal,verifier_context):
        current = verifier_context["current"]
        removed = current.domains["action"] - proposal.proposed_state.domains["action"]
        if not removed:
            return SoundnessType.UNKNOWN
        verdicts = []
        for a in sorted(removed):
            counters["elimination_admissibility_checks"] += 1
            verdicts.append(M.admissible(sc,s,k,a))
        return SoundnessType.UNKNOWN if any(verdicts) else SoundnessType.ENV_SOUND_DEAD

    # A prescan may certify the empty set. It may then abstain without
    # nominating an arbitrary action. All other decisions use the native loop.
    result = bounded_recurrent_repair(domain,propose,admissible,eliminate,ctx,
                                     max_iterations=0 if prescan and preferred is None else budget)
    # Final grading is evaluator-owned and is not part of proposal features.
    safe = M.certified_set(sc,s,k)
    oracle = M.oracle_action(sc,s,k,safe)
    oracle_value = M.utility(sc,s,k,oracle) if oracle else 0.
    executed = result.action
    value = M.utility(sc,s,k,executed) if executed else 0.
    raw = visits[0]["nomination"] if visits else None
    if executed is not None and executed not in safe:
        raise AssertionError("unsafe execution")
    if any(b>=a for a,b in zip(result.ranks,result.ranks[1:])):
        raise AssertionError("nondecreasing rank")
    return {"visits":visits,"certificates":certificate_receipts,
            "outcome":asdict(result),"work":dict(counters),
            "grade":{"safe_actions":[a for a in ACTIONS if a in safe],
                     "oracle":oracle,"raw":raw,"raw_certified":raw in safe,
                     "viable":bool(safe),"executed":executed,"utility":value,
                     "oracle_value":oracle_value,"regret":oracle_value-value,
                     "repaired":raw is not None and raw not in safe and executed is not None,
                     "unsafe":executed is not None and executed not in safe}}


def metrics(rows):
    n = len(rows)
    grades = [r["grade"] for r in rows]
    work = Counter()
    for r in rows:
        work.update(r["work"])
    viable = sum(g["viable"] for g in grades)
    missed = sum(g["viable"] and g["executed"] is None for g in grades)
    return {"n":n,"viable":viable,"raw_certified":sum(g["raw_certified"] for g in grades),
            "executed":sum(g["executed"] is not None for g in grades),
            "missed_viable":missed,"repaired":sum(g["repaired"] for g in grades),
            "unsafe":sum(g["unsafe"] for g in grades),
            "mean_utility":sum(g["utility"] for g in grades)/n,
            "mean_regret":sum(g["regret"] for g in grades)/n,
            "nominations":sum(len(r["visits"]) for r in rows),
            "committed_eliminations":sum(len(r["outcome"]["ranks"])-1 for r in rows),
            "work":dict(work)}


def run(out,width=32,steps=160,batch=32,train_n=256,eval_n=64,seeds=(137,163)):
    if not (1<=width<=128 and 1<=steps<=1024 and 1<=batch<=128
            and 1<=len(seeds)<=8 and len(set(seeds))==len(seeds)):
        raise ValueError("resource contract")
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    parts = partitions(train_n,eval_n)
    sets = {s:cases(states,s) for s,states in parts.items()}
    arrays = training_arrays(sets["train"])
    cfg = dict(width=width,steps=steps,batch=batch,train_n=train_n,eval_n=eval_n,seeds=list(seeds))
    report = {"schema":SCHEMA,"status":"public_development_only","config":cfg,
              "env":M.ENV_DIGEST,"split_hashes":{s:digest([state_root(x) for x in states])
                                                for s,states in parts.items()},"cells":[]}
    trace_path = out/"repair_rollouts.jsonl"
    with trace_path.open("w",encoding="utf-8") as trace:
        for seed in seeds:
            models = {}
            for arm in ARMS:
                row = {"seed":seed,"arm":arm}
                model = None
                if arm in LEARNED:
                    model,receipt = fit(arm,width,seed,arrays,steps,batch)
                    models[arm] = model
                    artifact = out/f"{seed}-{arm}.pt"
                    torch.save(model.state_dict(),artifact)
                    row.update(receipt,artifact=artifact.name,
                               artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
                elif arm == "static_refresh":
                    model = models["refresh"]
                identity = weight_hash(model) if model else "rule:"+arm
                row["identity"] = identity
                for split in ("cal","dev","shift"):
                    row[split] = {}
                    for budget in BUDGETS:
                        start = time.process_time()
                        entries = [episode(p,arm,model,budget) for p in sets[split]]
                        row[split][str(budget)] = metrics(entries)
                        row[split][str(budget)]["cpu_seconds_including_grading"] = time.process_time()-start
                entries = []
                for p in sets["trace_train"]:
                    r = episode(p,arm,model,3)
                    entries.append(r)
                    g = r["grade"]
                    target_kind = ("correct_abstain" if not g["viable"] else
                                   "verified_optimal" if g["executed"] is not None and g["regret"] == 0. else "repair_target")
                    record = {"schema":SCHEMA+".rollout","split":"trace_train",
                              "root":p["root"],"seed":seed,"arm":arm,"identity":identity,
                              "public":p,"budget":3,"episode":r,"target_kind":target_kind,
                              "repair_target":g["oracle"],"training_ready":False}
                    record["receipt_hash"] = digest(record)
                    trace.write(json.dumps(record,sort_keys=True,allow_nan=False)+"\n")
                row["trace_train"] = metrics(entries)
                report["cells"].append(row)
    report["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    (out/"repair_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def audit(out):
    report = json.loads((out/"repair_report.json").read_text(encoding="utf-8"))
    cfg = report["config"]
    parts = partitions(cfg["train_n"],cfg["eval_n"])
    if report["env"] != M.ENV_DIGEST or report["split_hashes"] != {
            s:digest([state_root(x) for x in states]) for s,states in parts.items()}:
        raise ValueError("environment or split binding mismatch")
    cells = {(r["seed"],r["arm"]):r for r in report["cells"]}
    if len(cells)!=len(report["cells"]) or set(cells)!={(s,a) for s in cfg["seeds"] for a in ARMS}:
        raise ValueError("cell coverage")
    models = {}
    for key,c in cells.items():
        if c["arm"] in LEARNED:
            artifact = out/c["artifact"]
            if hashlib.sha256(artifact.read_bytes()).hexdigest()!=c["artifact_sha256"]:
                raise ValueError("artifact mismatch")
            m = RepairTRM(cfg["width"],c["arm"])
            m.load_state_dict(torch.load(artifact,map_location="cpu",weights_only=True))
            if weight_hash(m)!=c["identity"] or c["weights_hash"]!=c["identity"]:
                raise ValueError("weight identity")
            models[key] = m.eval()
    for seed in cfg["seeds"]:
        key = (seed,"static_refresh")
        models[key] = models[(seed,"refresh")]
        if cells[key]["identity"] != weight_hash(models[key]):
            raise ValueError("static checkpoint mismatch")
    traces = out/"repair_rollouts.jsonl"
    if hashlib.sha256(traces.read_bytes()).hexdigest()!=report["trace_sha256"]:
        raise ValueError("trace digest")
    expected = {digest(p):p for p in cases(parts["trace_train"],"trace_train")}
    seen = set()
    counts = Counter()
    for line in traces.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        bound = r.pop("receipt_hash")
        p = r["public"]
        key = (r["seed"],r["arm"])
        pk = digest(p)
        identity = (key,pk)
        if (digest(r)!=bound or r["schema"]!=SCHEMA+".rollout" or r["training_ready"] is not False
                or r["split"]!="trace_train" or pk not in expected or p!=expected[pk]
                or r["root"]!=p["root"] or key not in cells or identity in seen
                or r["identity"]!=cells[key]["identity"] or r["budget"]!=3):
            raise ValueError("receipt custody")
        seen.add(identity)
        replay = episode(p,r["arm"],models.get(key),3)
        if digest(replay)!=digest(r["episode"]):
            delta = first_replay_difference(replay, r["episode"])
            raise ValueError(f"policy and certificate replay: arm={r['arm']} seed={r['seed']} "
                             f"root={r['root']} {delta}")
        g = replay["grade"]
        kind = ("correct_abstain" if not g["viable"] else
                "verified_optimal" if g["executed"] is not None and g["regret"] == 0. else "repair_target")
        if r["target_kind"]!=kind or r["repair_target"]!=g["oracle"]:
            raise ValueError("training target mismatch")
        counts[kind] += 1
    if len(seen)!=len(expected)*len(cells):
        raise ValueError("missing rollouts")
    return {"status":"audited_development_only","fits":len(cfg["seeds"])*len(LEARNED),
            "cells":len(cells),"rollouts":len(seen),"targets":dict(counts)}


def selftest():
    assert first_replay_difference({"a": (1, 2)}, {"a": [1, 2]}) is None
    assert first_replay_difference({"visits": [{"scores": [0.1]}]},
                                   {"visits": [{"scores": [0.2]}]}).startswith("$.visits[0].scores[0]")
    assert first_replay_difference({"certificate": "x"}, {"certificate": "y"}).startswith("$.certificate")
    parts = partitions(32,8)
    assert sum(map(len,parts.values()))==len({state_root(s) for ss in parts.values() for s in ss})
    packets = cases(parts["train"],"train")
    arrays = training_arrays(packets)
    m,receipt = fit("refresh",8,137,arrays,3,8)
    assert receipt["weights_hash"]!=receipt["initial_hash"]
    for p in cases(parts["dev"],"dev"):
        for arm in ("refresh","static_refresh","lexical","ldt_scan","oracle_scan"):
            r = episode(p,arm,m if arm in ("refresh","static_refresh") else None,5)
            assert not r["grade"]["unsafe"]
            assert not r["grade"]["viable"] or r["grade"]["executed"] is not None
            if arm=="oracle_scan":
                assert r["grade"]["regret"]==0.
    # Real native stale certificate control: different horizon, same action.
    found = False
    for p in packets:
        sc,s,k = unpack(p)
        for a in ACTIONS:
            c = M.issue(sc,s,k,a)
            if c is not None:
                assert execute_certified(M.context(sc,s,k),a,c,M.check_certificate)==a
                assert execute_certified(M.context(sc,s,k-1),a,c,M.check_certificate) is None
                other = next(b for b in ACTIONS if b!=a)
                assert execute_certified(M.context(sc,s,k),other,c,M.check_certificate) is None
                assert execute_certified(M.context(sc,s,k),a,c,M.check_certificate,
                                         domain=CandidateState({"action":FULL-{a}})) is None
                found = True
                break
        if found:
            break
    assert found
    print("GSF-03 native certified repair selftest passed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",type=Path)
    p.add_argument("--audit",type=Path)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--selftest",action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.selftest:
        selftest()
    if a.out:
        kwargs = dict(width=8,steps=4,batch=8,train_n=32,eval_n=8,seeds=(137,)) if a.smoke else {}
        report = run(a.out,**kwargs)
        print(json.dumps({"cells":len(report["cells"]),"status":report["status"]}))
    if a.audit:
        print(json.dumps(audit(a.audit),sort_keys=True))
    if not (a.selftest or a.out or a.audit):
        p.error("choose --selftest, --out or --audit")


if __name__=="__main__":
    main()
