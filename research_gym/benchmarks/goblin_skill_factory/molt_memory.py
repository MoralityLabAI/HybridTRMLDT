"""GSF-02: delayed-query retention over the native exact scalar epoch bank.

CPU development fixture. The policy sees a distribution over possible edits and
queries, chooses two receipts, and receives an independently scored outcome.
ScalarBank remains host-owned. This is not neural KV-cache reuse.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import time
from fractions import Fraction
from pathlib import Path

import torch
from torch import nn

from research_gym.benchmarks.molt_hybrid.epoch import ScalarBank
from research_gym.benchmarks.molt_hybrid.prefix_memory import topological

SCHEMA = "hybrid.gsf.molt.v1"
N = 6
CAPACITY = 2
PASSES = 4
MODULUS = 17
SPLITS = ("train", "cal", "dev", "shift", "trace_train")
LEARNED = ("direct", "residual", "refresh", "rlt", "damp")
RULES = ("exact_expected", "query_frequency", "random")
MASKS = tuple(itertools.combinations(range(N), CAPACITY))
EVENT_COUNT = 4


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def rng_for(root, purpose):
    return random.Random(int(digest([root, purpose]), 16))


def world(split, seed, index):
    if split not in SPLITS:
        raise ValueError(split)
    root = digest([SCHEMA, split, seed, index])
    rng = rng_for(root, "public")
    p_edge = 0.62 if split == "shift" else 0.32
    parents = [tuple(j for j in range(i) if rng.random() < p_edge) for i in range(N)]
    inputs = tuple(rng.randrange(MODULUS) for _ in range(N))
    node = rng.randrange(N)
    edge_dst = rng.randrange(1, N)
    edge_src = rng.randrange(edge_dst)
    events = []
    for kind in ("noop", "input", "edge", "context"):
        weight = rng.randint(1, 7)
        if split == "shift" and kind in ("edge", "context"):
            weight += 4
        query = [rng.randint(1, 8) for _ in range(N)]
        query[rng.randrange(N)] += 8
        events.append({"kind": kind, "weight": weight, "query_weights": query,
                       "node": node, "new_input": (inputs[node] + rng.randint(1, 16)) % MODULUS,
                       "edge": [edge_src, edge_dst]})
    return {"schema": SCHEMA, "root": root, "split": split, "inputs": inputs,
            "parents": parents, "events": events}


def apply_edit(public, event):
    xs = list(public["inputs"])
    parents = [tuple(p) for p in public["parents"]]
    context = "epoch0"
    kind = event["kind"]
    if kind == "input":
        xs[event["node"]] = event["new_input"]
    elif kind == "edge":
        j, i = event["edge"]
        parents[i] = tuple(sorted(set(parents[i]) ^ {j}))
    elif kind == "context":
        context = "epoch1"
    elif kind != "noop":
        raise ValueError(kind)
    return tuple(xs), tuple(parents), context


def cold_query(inputs, parents, query):
    """Query-local exact recursion, separate from the host bank implementation."""
    memo = {}
    def compute(i):
        if i not in memo:
            memo[i] = (inputs[i] + sum(compute(j) for j in parents[i])) % MODULUS
        return memo[i]
    value = compute(query)
    return value, len(memo)


def independent_affected(public, inputs, parents, context):
    if context != "epoch0":
        return set(range(N))
    old = [tuple(p) for p in public["parents"]]
    affected = {i for i in range(N)
                if public["inputs"][i] != inputs[i] or old[i] != parents[i]}
    while True:
        extended = affected | {i for i in range(N)
                               if affected.intersection(old[i] + parents[i])}
        if extended == affected:
            return affected
        affected = extended


def expected_values(public):
    """Expected saved query-node evaluations for each retained entry."""
    values = [Fraction(0) for _ in range(N)]
    total = sum(e["weight"] for e in public["events"])
    for event in public["events"]:
        xs, parents, context = apply_edit(public, event)
        affected = independent_affected(public, xs, parents, context)
        for i, w in enumerate(event["query_weights"]):
            if i not in affected:
                _, cost = cold_query(xs, parents, i)
                values[i] += Fraction(event["weight"], total) * Fraction(
                    w, sum(event["query_weights"])) * cost
    return values


def best_mask(scores):
    return tuple(sorted(sorted(range(N), key=lambda i: (-scores[i], i))[:CAPACITY]))


def policy(public, arm):
    if arm == "exact_expected":
        return best_mask(expected_values(public))
    if arm == "query_frequency":
        total = sum(e["weight"] for e in public["events"])
        weights = [sum(Fraction(e["weight"], total) * Fraction(e["query_weights"][i],
                   sum(e["query_weights"])) for e in public["events"]) for i in range(N)]
        return best_mask(weights)
    if arm == "random":
        return tuple(sorted(rng_for(public["root"], "random_policy").sample(range(N), CAPACITY)))
    raise ValueError(arm)


def outcome(public):
    rng = rng_for(public["root"], "realized")
    events = public["events"]
    i = rng.choices(range(len(events)), weights=[e["weight"] for e in events], k=1)[0]
    q = rng.choices(range(N), weights=events[i]["query_weights"], k=1)[0]
    return i, q


def authorize_receipt(receipt, root, query, context, affected):
    """Check the host-issued receipt's scope before reading its payload."""
    if receipt is None:
        return False, "missing"
    if receipt["root"] != root or receipt["node"] != query:
        return False, "identity"
    if receipt["context"] != context or query in affected:
        return False, "stale"
    return True, "current"


def host_episode(public, mask, *, force_outcome=None):
    if tuple(mask) not in MASKS:
        raise ValueError("invalid retention mask")
    bank = ScalarBank(MODULUS)
    first = bank.evaluate(tuple(public["inputs"]), tuple(map(tuple, public["parents"])), "epoch0")
    # Only the selected receipts are visible to the acting retention policy.
    retained = {i: {"root": public["root"], "node": i, "context": "epoch0",
                    "value": first.values[i]} for i in mask}
    event_id, query = outcome(public) if force_outcome is None else force_outcome
    event = public["events"][event_id]
    xs, parents, context = apply_edit(public, event)
    current = bank.evaluate(xs, parents, context)
    affected = independent_affected(public, xs, parents, context)
    if set(current.recomputed) != affected or current.values != topological(xs, parents, MODULUS):
        raise AssertionError("native bank parity failure")
    exact, cold_cost = cold_query(xs, parents, query)
    if exact != current.values[query]:
        raise AssertionError("independent query mismatch")
    # Host binding, not coincidental value equality, authorizes reuse.
    hit, reason = authorize_receipt(retained.get(query), public["root"], query,
                                    context, affected)
    stale = reason == "stale"
    stale_rejected = stale and not hit
    if hit:
        answer, work, route = retained[query]["value"], 0, "reuse"
    else:
        answer, work = cold_query(xs, parents, query)
        route = "recompute"
    if answer != exact:
        raise AssertionError("wrong query answer")
    vals = expected_values(public)
    chosen = sum(vals[i] for i in mask)
    optimal = sum(vals[i] for i in best_mask(vals))
    return {"event": event_id, "query": query, "route": route, "answer": answer,
            "correct": True, "hit": hit, "stale": stale, "stale_rejected": stale_rejected,
            "authorization": reason,
            "cold_nodes": cold_cost, "query_nodes": work, "saved_nodes": cold_cost-work,
            "receipt_lookups": 1, "retained_entries": CAPACITY,
            "host_initial_nodes": len(first.recomputed),
            "host_refresh_nodes": len(current.recomputed), "host_verify_nodes": cold_cost,
            "expected_saved": str(chosen), "expected_regret": str(optimal-chosen),
            "target_mask": list(best_mask(vals))}


def features(public):
    x = [v / 16 for v in public["inputs"]]
    x += [float(j in public["parents"][i]) for i in range(N) for j in range(N)]
    total = sum(e["weight"] for e in public["events"])
    for e in public["events"]:
        x += [float(e["kind"] == kind) for kind in ("noop", "input", "edge", "context")]
        x += [e["weight"] / total]
        x += [w / sum(e["query_weights"]) for w in e["query_weights"]]
        x += [float(e["node"] == i) for i in range(N)]
        x += [e["new_input"] / 16]
        x += [float(e["edge"][0] == i) for i in range(N)]
        x += [float(e["edge"][1] == i) for i in range(N)]
    return x


WIDTH_IN = N + N*N + EVENT_COUNT * (4+1+N+N+1+N+N)


class Retention(nn.Module):
    def __init__(self, width, arm):
        super().__init__()
        self.arm = arm
        self.encoder = nn.Linear(WIDTH_IN, width)
        self.cell = nn.Linear(2*width, width)
        self.head = nn.Linear(width, N)

    def forward(self, x):
        encoded = torch.tanh(self.encoder(x))
        h = encoded
        total = 1 if self.arm == "direct" else PASSES
        predictions = []
        for visit in range(total):
            source = encoded if self.arm != "residual" or visit == 0 else torch.zeros_like(encoded)
            update = torch.tanh(self.cell(torch.cat((h, source), dim=-1)))
            alpha = 1.0
            if self.arm == "rlt":
                alpha = 1.0 / (total-visit)
            elif self.arm == "damp" and visit != total-1:
                alpha = sum(1.0/j for j in range(2, total+1))/(total-1)
            h = h + alpha*update
            predictions.append(self.head(h))
        return torch.stack(predictions, dim=1)


def weight_hash(model):
    return digest({k: v.detach().tolist() for k, v in model.state_dict().items()})


def fit(arm, width, seed, worlds, steps, batch):
    torch.manual_seed(seed)
    model = Retention(width, arm)
    before = weight_hash(model)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003)
    x = torch.tensor([features(w) for w in worlds], dtype=torch.float32)
    y = torch.tensor([[float(v)/N for v in expected_values(w)] for w in worlds], dtype=torch.float32)
    rng = torch.Generator().manual_seed(seed+103)
    for _ in range(steps):
        idx = torch.randint(len(worlds), (batch,), generator=rng)
        pred = model(x[idx])
        loss = nn.functional.mse_loss(pred, y[idx, None, :].expand_as(pred))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    after = weight_hash(model)
    if before == after:
        raise AssertionError("fit changed no weights")
    return model, {"initial_hash": before, "weights_hash": after, "updates": steps,
                   "params": sum(p.numel() for p in model.parameters()),
                   "train_cell_visits": steps*batch*(1 if arm == "direct" else PASSES)}


@torch.no_grad()
def propose(public, arm, model=None):
    if model is None:
        return [{"visit": 0, "mask": list(policy(public, arm))}]
    scores = model(torch.tensor([features(public)], dtype=torch.float32))[0].tolist()
    return [{"visit": i+1, "mask": list(best_mask(s)), "scores": s}
            for i, s in enumerate(scores)]


def evaluate(worlds, arm, model=None, trace=None, identity=None, seed=None):
    regrets = Fraction(0)
    expected = Fraction(0)
    totals = dict(saved_nodes=0, cold_nodes=0, query_nodes=0, hits=0, stale=0,
                  model_visits=0, host_initial_nodes=0, host_refresh_nodes=0,
                  host_verify_nodes=0, receipt_lookups=0)
    start = time.process_time()
    for index, w in enumerate(worlds):
        visits = propose(w, arm, model)
        receipt = host_episode(w, tuple(visits[-1]["mask"]))
        regrets += Fraction(receipt["expected_regret"])
        expected += Fraction(receipt["expected_saved"])
        for k in ("saved_nodes", "cold_nodes", "query_nodes", "host_initial_nodes",
                  "host_refresh_nodes", "host_verify_nodes", "receipt_lookups"):
            totals[k] += receipt[k]
        totals["hits"] += int(receipt["hit"])
        totals["stale"] += int(receipt["stale"])
        totals["model_visits"] += visits[-1]["visit"]
        if trace is not None:
            if w["split"] != "trace_train":
                raise ValueError("invalid trace split")
            row = {"schema": SCHEMA+".rollout", "split": "trace_train", "root": w["root"],
                   "seed": seed, "index": index, "arm": arm, "identity": identity,
                   "public": w, "visits": visits, "receipt": receipt,
                   "target_kind": "expected_optimal" if
                       Fraction(receipt["expected_regret"]) == 0 else "retention_repair",
                   "training_ready": False}
            row["receipt_hash"] = digest(row)
            trace.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
    return {"n": len(worlds), **totals, "mean_expected_regret": float(regrets/len(worlds)),
            "mean_expected_saved": float(expected/len(worlds)), "cpu_seconds": time.process_time()-start}


def run(out, width=32, steps=80, batch=32, train_n=256, eval_n=64, seeds=(137,163)):
    torch.set_num_threads(1)
    if not (1 <= width <= 128 and 1 <= steps <= 512 and 1 <= batch <= 128
            and 1 <= train_n <= 2048 and 1 <= eval_n <= 512
            and 1 <= len(seeds) <= 8 and len(set(seeds)) == len(seeds)):
        raise ValueError("budget out of range")
    out.mkdir(parents=True, exist_ok=False)
    cfg = {"width": width, "steps": steps, "batch": batch, "train_n": train_n,
           "eval_n": eval_n, "seeds": list(seeds)}
    report = {"schema": SCHEMA, "status": "public_development_only", "config": cfg, "cells": []}
    trace_path = out/"molt_rollouts.jsonl"
    with trace_path.open("w", encoding="utf-8") as trace:
        for seed in seeds:
            sets = {s: [world(s, seed, i) for i in range(train_n if s == "train" else eval_n)]
                    for s in SPLITS}
            seen = set()
            for split, worlds in sets.items():
                hashes = {digest({k:v for k,v in w.items() if k not in ("root","split")}) for w in worlds}
                if seen.intersection(hashes):
                    raise ValueError("duplicate public world across splits")
                seen.update(hashes)
            for arm in LEARNED+RULES:
                model, fit_info = fit(arm,width,seed,sets["train"],steps,batch) if arm in LEARNED else (None,{})
                row = {"seed": seed, "arm": arm, **fit_info}
                identity = "rule:"+arm
                if model is not None:
                    artifact = out/f"{seed}-{arm}.pt"
                    torch.save(model.state_dict(), artifact)
                    row.update(artifact=artifact.name, artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
                    identity = row["weights_hash"]
                for s in ("cal","dev","shift"):
                    row[s] = evaluate(sets[s],arm,model)
                row["trace_train"] = evaluate(sets["trace_train"],arm,model,trace,identity,seed)
                report["cells"].append(row)
    report["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    (out/"molt_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def audit(out):
    report = json.loads((out/"molt_report.json").read_text(encoding="utf-8"))
    cfg = report["config"]
    expected = {(c["seed"],c["arm"]):c for c in report["cells"]}
    if len(report["cells"]) != len(expected) or set(expected) != set(itertools.product(cfg["seeds"],LEARNED+RULES)):
        raise ValueError("incomplete tournament")
    models = {}
    for key,c in expected.items():
        if c["arm"] in LEARNED:
            path = out/c["artifact"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != c["artifact_sha256"]:
                raise ValueError("artifact hash mismatch")
            model = Retention(cfg["width"], c["arm"])
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            if weight_hash(model) != c["weights_hash"]:
                raise ValueError("weights mismatch")
            models[key] = model.eval()
    traces = out/"molt_rollouts.jsonl"
    if hashlib.sha256(traces.read_bytes()).hexdigest() != report["trace_sha256"]:
        raise ValueError("trace hash mismatch")
    seen = set()
    for line in traces.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        h = r.pop("receipt_hash")
        if digest(r) != h or r["training_ready"] is not False or r["split"] != "trace_train":
            raise ValueError("trace integrity or custody")
        key = (r["seed"],r["arm"])
        index = r["index"]
        if key not in expected or not 0 <= index < cfg["eval_n"] or (key,index) in seen:
            raise ValueError("unexpected trace identity")
        seen.add((key,index))
        w = world("trace_train",r["seed"],index)
        identity = expected[key].get("weights_hash","rule:"+r["arm"])
        if digest(w) != digest(r["public"]) or r["root"] != w["root"] or r["identity"] != identity:
            raise ValueError("public packet or model mismatch")
        replay = propose(w,r["arm"],models.get(key))
        if replay != r["visits"]:
            raise ValueError("model action replay mismatch")
        receipt = host_episode(w,tuple(replay[-1]["mask"]))
        kind = "expected_optimal" if Fraction(receipt["expected_regret"]) == 0 else "retention_repair"
        if receipt != r["receipt"] or r["target_kind"] != kind:
            raise ValueError("receipt mismatch")
    if len(seen) != len(expected)*cfg["eval_n"]:
        raise ValueError("missing traces")
    return {"status":"audited_development_only","cells":len(expected),"rollouts":len(seen)}


def selftest():
    sample = {"root": "r", "node": 2, "context": "epoch0", "value": 3}
    assert authorize_receipt(sample, "r", 2, "epoch0", set()) == (True, "current")
    assert authorize_receipt(sample, "r2", 2, "epoch0", set()) == (False, "identity")
    assert authorize_receipt(sample, "r", 3, "epoch0", set()) == (False, "identity")
    assert authorize_receipt(sample, "r", 2, "epoch1", set()) == (False, "stale")
    assert authorize_receipt(sample, "r", 2, "epoch0", {2}) == (False, "stale")
    for split in SPLITS:
        for i in range(12):
            w = world(split,137,i)
            assert len(features(w)) == WIDTH_IN
            optimal = policy(w,"exact_expected")
            for mask in MASKS:
                for e,q in itertools.product(range(EVENT_COUNT),range(N)):
                    r = host_episode(w,mask,force_outcome=(e,q))
                    assert r["correct"] and Fraction(r["expected_regret"]) >= 0
                    if r["stale"]:
                        assert r["stale_rejected"] and r["route"] == "recompute"
                    if w["events"][e]["kind"] == "context":
                        assert not r["hit"]
            assert Fraction(host_episode(w,optimal)["expected_regret"]) == 0
    print("GSF-02 native bank and retention selftest passed")


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
        kwargs = dict(width=8,steps=4,batch=8,train_n=24,eval_n=8,seeds=(137,)) if a.smoke else {}
        r = run(a.out,**kwargs)
        print(json.dumps({"cells":len(r["cells"]),"status":r["status"]}))
    if a.audit:
        print(json.dumps(audit(a.audit),sort_keys=True))
    if not (a.selftest or a.out or a.audit):
        p.error("choose --selftest, --out or --audit")


if __name__ == "__main__":
    main()
