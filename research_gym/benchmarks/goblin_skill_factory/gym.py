"""GSF-01: CPU-first synthetic skill-module training and matched loop proxies.

No native MoLT KV cache or full TRM reproduction is claimed. All scoring is
host-owned. The candidate features are public; exact targets are evaluator-owned.
Run as a module from the repository root. Requires PyTorch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch import nn

SCHEMA = "hybrid.gsf.v1"
FAMILIES = ("repair", "retention", "persona")
SPLITS = ("train", "cal", "dev", "shift")
ARMS = ("direct", "residual", "refresh", "rlt", "damp")
N = 4
ABSTAIN = N
FIELDS = ("cost", "gain", "authorized", "current", "verified", "preference",
          "scope_match", "critical_veto")
INPUT = N * len(FIELDS)
PASSES = 4


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def root_id(family, split, seed, index):
    return digest([SCHEMA, family, split, seed, index])


def case(family, split, seed, index):
    if family not in FAMILIES or split not in SPLITS:
        raise ValueError("unknown family or split")
    root = root_id(family, split, seed, index)
    rng = random.Random(int(root, 16))
    rows = []
    for _ in range(N):
        # Shift changes correlations and feasible-set frequency; same semantics.
        cost = rng.randint(1, 9)
        gain = rng.randint(0, 9)
        authorized = int(rng.random() < (0.55 if split == "shift" else 0.8))
        current = int(rng.random() < (0.50 if split == "shift" else 0.8))
        verified = int(rng.random() < 0.85)
        preference = rng.randint(0, 9)
        scope = int(rng.random() < 0.8)
        veto = int(rng.random() < (0.35 if split == "shift" else 0.1))
        if split == "shift" and veto:
            preference = max(preference, 8)
            gain = max(gain, 8)
        rows.append([cost, gain, authorized, current, verified, preference, scope, veto])
    return {"root": root, "family": family, "split": split, "rows": rows}


def permitted(family, row):
    cost, gain, authorized, current, verified, preference, scope, veto = row
    if family == "repair":
        return bool(authorized and current and verified and not veto and gain > 0)
    if family == "retention":
        return bool(current and verified and scope and not veto)
    if family == "persona":
        return bool(authorized and scope and not veto)
    raise ValueError(family)


def score_key(family, row):
    cost, gain, authorized, current, verified, preference, scope, veto = row
    if family == "repair":
        return gain - cost, gain, -cost
    if family == "retention":
        return gain, -cost
    if family == "persona":
        return preference, -cost
    raise ValueError(family)


def oracle(c):
    valid = [i for i, row in enumerate(c["rows"])
             if permitted(c["family"], row)]
    if not valid:
        return ABSTAIN
    return max(valid, key=lambda i: (score_key(c["family"], c["rows"][i]), -i))


def grade(c, action):
    if type(action) is not int or not 0 <= action <= ABSTAIN:
        raise ValueError("invalid action")
    valid = action == ABSTAIN or permitted(c["family"], c["rows"][action])
    guarded = action if valid else ABSTAIN
    gold = oracle(c)
    return {"gold": gold, "raw_correct": action == gold, "unsafe": not valid,
            "guarded_action": guarded, "guarded_correct": guarded == gold}


def dataset(family, split, seed, count):
    return [case(family, split, seed, i) for i in range(count)]


def tensors(cases):
    x = torch.tensor([c["rows"] for c in cases], dtype=torch.float32)
    x[:, :, [0, 1, 5]] /= 9.0
    return x.flatten(1), torch.tensor([oracle(c) for c in cases], dtype=torch.long)


class Skill(nn.Module):
    def __init__(self, width, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.encoder = nn.Linear(INPUT, width)
        self.cell = nn.Linear(2 * width, width)
        self.head = nn.Linear(width, N + 1)

    def start(self, x):
        encoded = torch.tanh(self.encoder(x))
        return encoded, encoded

    def step(self, h, encoded, visit, total):
        refresh = encoded if self.arm in ("refresh", "rlt", "damp") or visit == 0 else torch.zeros_like(encoded)
        proposal = h + torch.tanh(self.cell(torch.cat((h, refresh), dim=-1)))
        if self.arm == "rlt":
            alpha = 1.0 / (total - visit)
        elif self.arm == "damp":
            # Same mean nonterminal clock coefficients as RLT, terminal replacement.
            alpha = 1.0 if visit == total - 1 else sum(1.0 / j for j in range(2, total + 1)) / (total - 1)
        else:
            alpha = 1.0
        h = h + alpha * (proposal - h)
        return h, self.head(h)

    def forward(self, x, passes):
        encoded, h = self.start(x)
        logits = []
        for visit in range(passes):
            h, pred = self.step(h, encoded, visit, passes)
            logits.append(pred)
        return torch.stack(logits, dim=1)


def weight_hash(model):
    return digest({k: v.detach().cpu().tolist() for k, v in model.state_dict().items()})


def fit(arm, width, seed, train, steps, batch):
    torch.manual_seed(seed)
    model = Skill(width, arm)
    initial = weight_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    x, y = tensors(train)
    rng = torch.Generator().manual_seed(seed + 103)
    passes = 1 if arm == "direct" else PASSES
    for _ in range(steps):
        idx = torch.randint(len(train), (batch,), generator=rng)
        pred = model(x[idx], passes)
        # Fixed supervision at every executed visit, normalized over visits.
        loss = nn.functional.cross_entropy(pred.flatten(0, 1), y[idx].repeat_interleave(passes))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final = weight_hash(model)
    if initial == final:
        raise RuntimeError("fit changed no parameters")
    return model.eval(), {"initial_hash": initial, "weights_hash": final,
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "updates": steps, "train_cell_visits": steps * batch * passes}


@torch.no_grad()
def rollout(model, c, threshold=None):
    # No oracle call in the proposal path. Stop only after confidence criterion.
    x, _ = tensors([c])
    encoded, h = model.start(x)
    total = 1 if model.arm == "direct" else PASSES
    visits = []
    for visit in range(total):
        h, logits = model.step(h, encoded, visit, total)
        probs = logits.softmax(dim=-1)[0]
        action = int(probs.argmax())
        confidence = float(probs[action])
        visits.append({"visit": visit + 1, "action": action, "confidence": confidence})
        if threshold is not None and confidence >= threshold:
            break
    return visits


@torch.no_grad()
def evaluate(model, cases, threshold=None):
    results = []
    start = time.process_time()
    for c in cases:
        visits = rollout(model, c, threshold)
        receipt = grade(c, visits[-1]["action"])
        results.append({"case": c, "visits": visits, "receipt": receipt})
    n = len(results)
    return {"n": n, "raw_accuracy": sum(r["receipt"]["raw_correct"] for r in results) / n,
            "guarded_accuracy": sum(r["receipt"]["guarded_correct"] for r in results) / n,
            "unsafe_rate": sum(r["receipt"]["unsafe"] for r in results) / n,
            "cell_visits": sum(len(r["visits"]) for r in results),
            "cpu_seconds": time.process_time() - start}, results


def calibrate(model, cal):
    fixed, _ = evaluate(model, cal)
    candidates = []
    # Select using CAL only; exact raw score and invalid-rate guardrails.
    for threshold in (0.5, 0.65, 0.8, 0.9, 0.95, 1.01):
        m, _ = evaluate(model, cal, threshold)
        if (m["raw_accuracy"] >= fixed["raw_accuracy"] - 0.01
                and m["unsafe_rate"] <= fixed["unsafe_rate"]):
            candidates.append((m["cell_visits"], -m["raw_accuracy"], threshold))
    return min(candidates)[2]


def export_train(path, family, arm, seed, weights_hash, results):
    # Development-only teacher training roots, never CAL/dev/shift.
    with path.open("a", encoding="utf-8") as f:
        for r in results:
            if r["case"]["split"] != "train" or r["case"]["family"] != family:
                raise ValueError("invalid rollout export")
            c = r["case"]
            if grade(c, r["visits"][-1]["action"]) != r["receipt"]:
                raise ValueError("receipt mismatch")
            record = {"schema": SCHEMA + ".rollout", "family": family, "root": c["root"],
                      "split": "train", "arm": arm, "seed": seed,
                      "weights_hash": weights_hash, "public": c,
                      "visits": r["visits"], "receipt": r["receipt"],
                      "training_ready": False}
            record["receipt_hash"] = digest(record)
            f.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def run(out, width=32, steps=100, batch=32, train_n=512, eval_n=128, seeds=(137, 163)):
    if not (1 <= width <= 128 and 1 <= steps <= 1024 and 1 <= batch <= 128
            and 1 <= train_n <= 4096 and 1 <= eval_n <= 1024
            and 1 <= len(seeds) <= 8 and len(set(seeds)) == len(seeds)):
        raise ValueError("budget outside CPU development contract")
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=False)
    traces = out / "train_rollouts.jsonl"
    report = {"schema": SCHEMA, "status": "public_development_only",
              "config": {"width": width, "steps": steps, "batch": batch,
                         "train_n": train_n, "eval_n": eval_n, "seeds": list(seeds)},
              "cells": []}
    for family in FAMILIES:
        for seed in seeds:
            sets = {s: dataset(family, s, seed, train_n if s == "train" else eval_n)
                    for s in SPLITS}
            for arm in ARMS:
                model, fit_receipt = fit(arm, width, seed, sets["train"], steps, batch)
                row = {"family": family, "seed": seed, "arm": arm, **fit_receipt,
                       "root_hashes": {s: digest([c["root"] for c in cs]) for s, cs in sets.items()}}
                threshold = calibrate(model, sets["cal"]) if arm != "direct" else None
                row["threshold"] = threshold
                for split in ("cal", "dev", "shift"):
                    row[split], _ = evaluate(model, sets[split])
                    if threshold is not None:
                        row[split + "_adaptive"], _ = evaluate(model, sets[split], threshold)
                _, train_rows = evaluate(model, sets["train"])
                export_train(traces, family, arm, seed, fit_receipt["weights_hash"], train_rows)
                report["cells"].append(row)
    report["train_rollouts_sha256"] = hashlib.sha256(traces.read_bytes()).hexdigest()
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def selftest():
    for family in FAMILIES:
        roots = set()
        for split in SPLITS:
            cs = dataset(family, split, 137, 20)
            assert not roots.intersection(c["root"] for c in cs)
            roots.update(c["root"] for c in cs)
            for c in cs:
                assert grade(c, oracle(c))["raw_correct"]
                for i in range(N):
                    if not permitted(family, c["rows"][i]):
                        assert grade(c, i)["unsafe"]
                        assert grade(c, i)["guarded_action"] == ABSTAIN
        train = dataset(family, "train", 137, 16)
        for arm in ARMS:
            m, receipt = fit(arm, 8, 137, train, 2, 4)
            assert receipt["initial_hash"] != receipt["weights_hash"]
            assert len(rollout(m, train[0])) == (1 if arm == "direct" else PASSES)
            assert len(rollout(m, train[0], 0.0)) == 1
    print("GSF-01 selftest passed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.selftest:
        selftest()
    if args.out:
        kwargs = dict(width=8, steps=4, batch=8, train_n=32, eval_n=16, seeds=(137,)) if args.smoke else {}
        report = run(args.out, **kwargs)
        print(json.dumps({"cells": len(report["cells"]), "status": report["status"]}))
    elif not args.selftest:
        p.error("provide --out or --selftest")


if __name__ == "__main__":
    main()
