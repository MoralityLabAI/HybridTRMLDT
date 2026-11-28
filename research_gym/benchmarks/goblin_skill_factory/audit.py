"""Independent development receipt audit for GSF-01 outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def root(family, split, seed, index):
    return digest(["hybrid.gsf.v1", family, split, seed, index])


def independent_grade(public, action):
    family = public["family"]
    candidates = []
    allowed = set()
    for i, (cost, gain, auth, live, checked, pref, scope, veto) in enumerate(public["rows"]):
        if family == "repair":
            accept = auth == live == checked == 1 and veto == 0 and gain > cost
            rank = (gain - cost, gain, -cost)
        elif family == "retention":
            accept = live == checked == scope == 1 and veto == 0
            rank = (gain, -cost)
        elif family == "persona":
            accept = auth == scope == 1 and veto == 0
            rank = (pref, -cost)
        else:
            raise ValueError("unknown family")
        if accept:
            allowed.add(i)
            candidates.append((rank, -i))
    if type(action) is not int or action not in range(5):
        raise ValueError("invalid action")
    gold = -sorted(candidates)[-1][1] if candidates else 4
    unsafe = action != 4 and action not in allowed
    guarded = 4 if unsafe else action
    return {"gold": gold, "raw_correct": action == gold, "unsafe": unsafe,
            "guarded_action": guarded, "guarded_correct": guarded == gold}


def audit(out):
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    cfg = report["config"]
    expected = {}
    seen = Counter()
    for c in report["cells"]:
        key = (c["family"], c["seed"], c["arm"])
        if key in expected:
            raise ValueError("duplicate fit")
        expected[key] = c
        for split in ("train", "cal", "dev", "shift", "trace_train"):
            n = cfg["train_n"] if split == "train" else cfg["eval_n"]
            wanted = digest([root(c["family"], split, c["seed"], i) for i in range(n)])
            if c["root_hashes"][split] != wanted:
                raise ValueError("split binding mismatch")
    for meta in report["meta_repair"]:
        base = expected[(meta["family"], meta["seed"], "refresh")]
        for c in meta["candidates"]:
            if c["initial_hash"] != base["weights_hash"]:
                raise ValueError("meta-repair checkpoint mismatch")
            key = (meta["family"], meta["seed"], "meta_" + c["arm"])
            if key in expected:
                raise ValueError("duplicate repair fit")
            expected[key] = c
        chosen = max(meta["candidates"], key=lambda c: (
            c["cal"]["raw_accuracy"], -c["cal"]["unsafe_rate"], c["arm"] == "refresh"))
        if chosen["arm"] != meta["selected_by_cal"]:
            raise ValueError("non-CAL selection")
    for c in expected.values():
        artifact = out / c["artifact"]
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != c["artifact_sha256"]:
            raise ValueError("weight artifact mismatch")
    traces = out / "train_rollouts.jsonl"
    if hashlib.sha256(traces.read_bytes()).hexdigest() != report["train_rollouts_sha256"]:
        raise ValueError("rollout inventory mismatch")
    allowed_roots = {(fam, seed): {root(fam, "trace_train", seed, i)
                                  for i in range(cfg["eval_n"])}
                     for fam, seed, _ in expected}
    outcomes = Counter()
    for line in traces.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        bound = r.pop("receipt_hash")
        if digest(r) != bound:
            raise ValueError("receipt digest mismatch")
        key = (r["family"], r["seed"], r["arm"])
        c = r["public"]
        if (r["schema"] != "hybrid.gsf.v1.rollout" or r["split"] != "trace_train"
                or c["split"] != "trace_train" or c["root"] != r["root"]
                or c["family"] != r["family"] or r["root"] not in allowed_roots[key[:2]]
                or r["training_ready"] is not False):
            raise ValueError("rollout custody mismatch")
        if expected[key]["weights_hash"] != r["weights_hash"]:
            raise ValueError("weight identity mismatch")
        visits = r["visits"]
        count = 1 if r["arm"] == "direct" else 4
        if [v["visit"] for v in visits] != list(range(1, count + 1)):
            raise ValueError("visit accounting mismatch")
        if any(not 0 <= v["confidence"] <= 1 or type(v["action"]) is not int
               or v["action"] not in range(5) for v in visits):
            raise ValueError("invalid visit")
        verdict = independent_grade(c, visits[-1]["action"])
        if verdict != r["receipt"] or verdict["gold"] != r["repair_target"]:
            raise ValueError("independent grader mismatch")
        if r["target_kind"] != ("verified_success" if verdict["raw_correct"] else "failure_repair"):
            raise ValueError("target kind mismatch")
        seen[(key, r["root"])] += 1
        outcomes[r["target_kind"]] += 1
    if any(seen[(key, rid)] != 1 for key in expected for rid in allowed_roots[key[:2]]):
        raise ValueError("missing or duplicate rollout")
    return {"status": "audited_development_only", "fits": len(expected),
            "rollouts": sum(outcomes.values()), "outcomes": dict(outcomes)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    print(json.dumps(audit(p.parse_args().out), sort_keys=True))
