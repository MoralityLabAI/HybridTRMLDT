"""Utility-constrained persona evolution for JINN-GYM-02.

* Abstention-aware diagnostics: justified abstention, excessive caution, unsupported action,
  successful recovery (and failed recovery), per task and per batch.
* Five matched mutation arms, each producing at most one candidate per persona per round:
  random valid mutation, deterministic abstention repair, heuristic coordinate search,
  Inky-generated patch, learned mutation selection.
* Feasibility-first, utility-constrained acceptance with a 2-point noninferiority margin
  against neutral Inky on the same batch.
* Typed recombination (rubric feature, action ranking, LDT parameter exchange).
* Pareto archive over fidelity, utility, calibration, compute cost and abstention quality.
"""
from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from research_gym.jinn_gym_01 import host, metrics
from research_gym.jinn_gym_01.heads import VetoedHead, compositional_tagsets, train_numpy
from research_gym.jinn_gym_01.module import HeadScorer, PersonaModule, decide
from research_gym.jinn_gym_01.schema import PATCH_OPS, SpecError, apply_patch, canonical, spec_hash, validate
from research_gym.jinn_gym_01.tags import ABSTAIN, METHOD_TAGS, PERSONA_TAGS, PHASES, SIGNATURES
from research_gym.jinn_gym_01._vendor.ldt_experiment import AdditiveCandidateHead

MARGIN = 0.02          # preregistered utility noninferiority margin (normalized utility)
FID_STEP = 0.02        # minimum fidelity gain for a feasible parent
UTIL_STEP = 0.02       # minimum utility gain (Pareto move or feasibility restoration)
ARMS = ("random", "abstention", "heuristic", "inky", "learned")
COST_UNITS = {"low": 1.0, "medium": 2.0, "high": 3.0}
HEAD_EPOCHS = 30
N_COMP = 700


# --------------------------------------------------------------------------- modules
class ModuleFactory:
    """Compile a spec and train its head on development-family verified examples."""

    def __init__(self, dev_tasks, workdir):
        self.work = workdir
        rows_tags, seen = [], set()
        for t in dev_tasks:
            for a in t.actions:
                k = tuple(sorted(a.tags))
                if k not in seen:
                    seen.add(k)
                    rows_tags.append(list(k))
        self.task_tagsets = rows_tags
        self.cache: dict[str, dict] = {}

    def build(self, persona: str, spec: dict, head_override: dict | None = None) -> dict:
        h = spec_hash(spec) + (":" + _hhash(head_override) if head_override else "")
        if h in self.cache:
            return self.cache[h]
        sh = spec_hash(spec)
        rubric = PersonaModule.compile(spec, "g", self.work / persona / sh[:12])
        lane = lambda tags: rubric.scorer(tags)[1]  # noqa: E731
        if head_override:
            base = AdditiveCandidateHead.from_jsonable(head_override)
        else:
            tagsets = self.task_tagsets + compositional_tagsets(_seed("comp02", persona, sh), N_COMP)
            rows = [{"tags": ts, "split": "train", "labels": {"persona": {"lane": lane(ts)}}} for ts in tagsets]
            base = train_numpy(rows, seed=_seed("head02", persona, sh) % (2 ** 31), epochs=HEAD_EPOCHS)
        head = VetoedHead(base, frozenset(spec["critical_tags"]))
        out = {"persona": persona, "hash": h, "spec_hash": sh, "spec": spec, "rubric": rubric,
               "learned": rubric.with_scorer(HeadScorer(head)), "head": head}
        self.cache[h] = out
        return out


def _seed(*parts) -> int:
    import hashlib
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def _hhash(head_json: dict | None) -> str:
    import hashlib
    return hashlib.sha256(canonical(head_json).encode()).hexdigest()[:16] if head_json else ""


# --------------------------------------------------------------------------- evaluation
def neutral_row(task, persona, parsed) -> dict:
    g = host.execute(task, parsed["choice"])
    d = {"executed": g["executed"], "proposal": parsed["choice"], "intervened": g["intervened"],
         "proposal_unauthorized": parsed["choice"] is not None and parsed["choice"] not in host.permitted(task),
         "confidence": parsed["confidence"] if parsed["confidence"] is not None else 0.5,
         "parse_ok": parsed["parsed"] and parsed["choice"] is not None}
    return metrics.outcome_record(task, persona, "A", d, None)


def module_row(task, persona, arm, prior, mods) -> dict:
    d = decide(task, prior, mods)
    d["proposal"] = d["proposal_unconstrained"]
    d["intervened"] = d["proposal_unconstrained"] != d["executed"] and d["proposal_unauthorized"]
    r = metrics.outcome_record(task, persona, arm, d, prior, d["persona_u"])
    ex = d["executed"]
    r["cost_units"] = COST_UNITS[task.action(ex).cost] if ex else 0.0
    r["failure_reported"] = bool(task.hidden["domain_hidden"].get("failure_reported")) if "domain_hidden" in task.hidden else False
    r["template"] = task.hidden.get("template")
    r["abstain_reason"] = d["abstain_reason"]
    return r


def batch_stats(rows: Sequence[dict], neutral_rows: Sequence[dict]) -> dict:
    s = metrics.summarize(rows)
    ua = float(np.mean([r["normalized_utility"] for r in neutral_rows])) if neutral_rows else None
    conf = [(r["confidence"], float(r["optimal"])) for r in rows if r["confidence"] is not None]
    return {"fidelity": s.get("fidelity") if s.get("fidelity") is not None else 0.0,
            "n_sig": s.get("n_signature_available", 0), "utility": s["normalized_utility"],
            "utility_neutral": ua, "gap": s["normalized_utility"] - ua if ua is not None else None,
            "brier": float(np.mean([(c - o) ** 2 for c, o in conf])) if conf else None,
            "cost": float(np.mean([r.get("cost_units", 0.0) for r in rows])), "abstain_f1": s.get("abstain_f1"),
            "abstain_rate": s["abstain_rate"], "n": len(rows)}


def accept(parent: dict, child: dict) -> tuple[bool, str]:
    """Feasibility-first utility-constrained selection (PREREGISTRATION section 'Selection')."""
    feasible_p = parent["gap"] >= -MARGIN
    if feasible_p:
        if child["gap"] < -MARGIN:
            return False, "violates_noninferiority"
        if child["fidelity"] >= parent["fidelity"] + FID_STEP:
            return True, "fidelity_gain_feasible"
        if child["fidelity"] >= parent["fidelity"] and child["utility"] >= parent["utility"] + UTIL_STEP:
            return True, "pareto_utility_gain"
        return False, "no_gain"
    if child["gap"] >= parent["gap"] + UTIL_STEP and child["fidelity"] >= parent["fidelity"] - 0.05:
        return True, "feasibility_restoration"
    return False, "parent_infeasible_no_restoration"


# --------------------------------------------------------------------------- diagnostics
def task_codes(r: dict) -> list[str]:
    c = []
    if r["abstained"] and r["abstain_optimal"]:
        c.append("justified_abstention")
    if r["abstained"] and not r["abstain_optimal"]:
        c.append("excessive_caution")
    if not r["abstained"] and r["abstain_optimal"]:
        c.append("unsupported_action")
    if r.get("failure_reported") and not r["abstained"]:
        c.append("successful_recovery" if r["optimal"] else "failed_recovery")
    if r["signature_available"] and not r["signature_hit"]:
        c.append("missed_signature")
    if r["best_utility"] - r["utility"] > 0.1:
        c.append("suboptimal_utility")
    if r["signature_hit"] and r["best_utility"] - r["utility"] > 0.1:
        c.append("signature_cost")
    return c


def batch_diagnostics(rows, neutral_rows) -> dict:
    from collections import Counter
    codes = Counter(c for r in rows for c in task_codes(r))
    reasons = Counter(r.get("abstain_reason") for r in rows if r["abstained"])
    return {"codes": dict(codes), "abstain_reasons": {str(k): v for k, v in reasons.items()},
            "stats": batch_stats(rows, neutral_rows)}


# --------------------------------------------------------------------------- operators
def random_mutation(spec: dict, rng: random.Random, persona: str, tries: int = 30) -> list | None:
    for _ in range(tries):
        f = rng.choice(spec["preference_features"])
        kind = rng.choice(["set_weight", "add_tag", "remove_tag", "set_abstain_threshold", "set_evidence",
                           "add_transition", "remove_transition", "set_ranking", "set_abstain_if_no_ranked_action"])
        if kind == "set_weight":
            op = {"op": kind, "feature": f["name"], "weight": round(min(10, max(0.1, f["weight"] * rng.uniform(0.5, 1.6))), 3)}
        elif kind == "add_tag":
            op = {"op": kind, "feature": f["name"], "tag": rng.choice(PERSONA_TAGS), "sign": rng.choice("+-")}
        elif kind == "remove_tag":
            tags = [t for g in spec["preference_features"] for t in g["positive_tags"] + g["negative_tags"]]
            if not tags:
                continue
            op = {"op": kind, "tag": rng.choice(tags)}
        elif kind == "set_abstain_threshold":
            op = {"op": kind, "value": round(rng.uniform(0, 0.8), 3)}
        elif kind == "set_evidence":
            field = rng.choice(["min_evidence_items", "abstain_on_missing_inputs", "require_failure_log_for_repair"])
            op = {"op": kind, "field": field, "value": rng.randint(0, 3) if field == "min_evidence_items" else rng.random() < 0.5}
        elif kind == "add_transition":
            op = {"op": kind, "from": rng.choice(spec["lattice"]["states"]), "to": rng.choice(PHASES)}
        elif kind == "remove_transition":
            if not spec["lattice"]["transitions"]:
                continue
            a, b = rng.choice(spec["lattice"]["transitions"])
            op = {"op": kind, "from": a, "to": b}
        elif kind == "set_ranking":
            r = list(spec["action_ranking"])
            i, j = rng.randrange(len(r)), rng.randrange(len(r))
            r[i], r[j] = r[j], r[i]
            op = {"op": kind, "ranking": r}
        else:
            op = {"op": kind, "value": not spec["abstention"]["abstain_if_no_ranked_action"]}
        try:
            child = apply_patch(spec, [op])
        except SpecError:
            continue
        if not validate(child, persona) and spec_hash(child) != spec_hash(spec):
            return [op]
    return None


def abstention_repair(spec: dict, diag: dict) -> tuple[list | None, str]:
    """Deterministic abstention-aware repair. Returns (ops, rule)."""
    c = diag["codes"]
    E, U = c.get("excessive_caution", 0), c.get("unsupported_action", 0)
    R, FR = c.get("successful_recovery", 0), c.get("failed_recovery", 0)
    ab, ev = spec["abstention"], spec["evidence_requirements"]
    if E > U and E >= 2:
        if ab["abstain_if_no_ranked_action"]:
            return [{"op": "set_abstain_if_no_ranked_action", "value": False}], "relax:no_ranked_action_off"
        if ev["min_evidence_items"] > 1:
            return [{"op": "set_evidence", "field": "min_evidence_items", "value": ev["min_evidence_items"] - 1}], "relax:evidence_min_down"
        if ab["min_train_score"] > 0.0:
            return [{"op": "set_abstain_threshold", "value": round(max(0.0, ab["min_train_score"] - 0.1), 3)}], "relax:threshold_down"
        return None, "relax:exhausted"
    if U > E and U >= 2:
        if not ev["abstain_on_missing_inputs"]:
            return [{"op": "set_evidence", "field": "abstain_on_missing_inputs", "value": True}], "tighten:missing_inputs_on"
        return [{"op": "set_abstain_threshold", "value": round(min(0.9, ab["min_train_score"] + 0.1), 3)}], "tighten:threshold_up"
    if FR > R:
        tr = spec["lattice"]["transitions"]
        for pair in (["diagnose", "repair"], ["diagnose", "test"], ["orient", "diagnose"]):
            if pair not in tr:
                return [{"op": "add_transition", "from": pair[0], "to": pair[1]}], "recovery:add_" + "_".join(pair)
        if ev["require_failure_log_for_repair"]:
            return [{"op": "set_evidence", "field": "require_failure_log_for_repair", "value": False}], "recovery:log_req_off"
    return None, "justified_or_balanced:no_op"


def heuristic_mutation(spec: dict, persona: str, step: int) -> list | None:
    """Fixed coordinate schedule aimed at the registered signature."""
    sig = sorted(SIGNATURES[persona])
    feats = sorted(spec["preference_features"], key=lambda f: -f["weight"])
    top = feats[0]
    moves = []
    for t in sig:
        if not any(t in f["positive_tags"] for f in spec["preference_features"]):
            moves.append([{"op": "add_tag", "feature": top["name"], "tag": t, "sign": "+"}])
    holder = next((f for f in spec["preference_features"] if set(f["positive_tags"]) & set(sig)), top)
    moves.append([{"op": "set_weight", "feature": holder["name"], "weight": round(min(10, holder["weight"] * 1.3), 3)}])
    moves.append([{"op": "set_abstain_threshold", "value": round(max(0, spec["abstention"]["min_train_score"] - 0.05), 3)}])
    moves.append([{"op": "set_abstain_threshold", "value": round(min(0.9, spec["abstention"]["min_train_score"] + 0.05), 3)}])
    phases = sorted({METHOD_TAGS[t] for t in sig})
    for src in ("orient", "test", "diagnose"):
        for ph in phases:
            if [src, ph] not in spec["lattice"]["transitions"]:
                moves.append([{"op": "add_transition", "from": src, "to": ph}])
    if not any("high_cost" in f["negative_tags"] for f in spec["preference_features"]):
        moves.append([{"op": "add_tag", "feature": feats[-1]["name"], "tag": "high_cost", "sign": "-"}])
    for k in range(len(moves)):
        ops = moves[(step + k) % len(moves)]
        try:
            child = apply_patch(spec, ops)
        except SpecError:
            continue
        if not validate(child, persona) and spec_hash(child) != spec_hash(spec):
            return ops
    return None


def op_features(ops: list, diag: dict) -> np.ndarray:
    """Context features for the learned mutation selector."""
    kinds = [o.get("op") for o in ops or []]
    v = [float(any(k == p for k in kinds)) for p in PATCH_OPS]
    st, c = diag["stats"], diag["codes"]
    n = max(1, st["n"])
    v += [st["fidelity"], st["gap"] if st["gap"] is not None else 0.0, st["abstain_rate"],
          c.get("excessive_caution", 0) / n, c.get("unsupported_action", 0) / n, c.get("missed_signature", 0) / n,
          c.get("failed_recovery", 0) / n]
    sign_down = sum(1 for o in ops or [] if o.get("op") == "set_abstain_threshold")
    v.append(float(sign_down))
    return np.array(v)


class LearnedSelector:
    """Online L2 logistic regression: P(accept | op features, batch context)."""

    def __init__(self):
        self.X, self.y = [], []
        self.w = None

    def add(self, x: np.ndarray, accepted: bool):
        self.X.append(x)
        self.y.append(float(accepted))

    def fit(self, l2: float = 1.0, iters: int = 300):
        if len(self.y) < 8 or len(set(self.y)) < 2:
            self.w = None
            return
        X = np.c_[np.ones(len(self.X)), np.array(self.X)]
        y = np.array(self.y)
        w = np.zeros(X.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-X @ w))
            g = X.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]] / len(y)
            w -= 0.5 * g
        self.w = w

    def score(self, x: np.ndarray) -> float:
        if self.w is None:
            return 0.5
        return float(1 / (1 + np.exp(-(self.w[0] + x @ self.w[1:]))))


# --------------------------------------------------------------------------- recombination
def recombine_feature(recipient: dict, donor: dict, persona: str) -> dict | None:
    """Recipient gains the donor's top-weighted feature (tags already used by the recipient are dropped)."""
    s = copy.deepcopy(recipient)
    d = max(donor["preference_features"], key=lambda f: f["weight"])
    used = {t for f in s["preference_features"] for t in f["positive_tags"] + f["negative_tags"]}
    name = ("x_" + d["name"])[:32]
    if any(f["name"] == name for f in s["preference_features"]) or len(s["preference_features"]) >= 10:
        return None
    s["preference_features"].append({"name": name, "weight": d["weight"],
                                     "positive_tags": [t for t in d["positive_tags"] if t not in used and t not in s["critical_tags"]],
                                     "negative_tags": [t for t in d["negative_tags"] if t not in used]})
    return s if not validate(s, persona) else None


def recombine_ranking(recipient: dict, donor: dict, persona: str) -> dict | None:
    s = copy.deepcopy(recipient)
    merged, seen = [], set()
    for a, b in zip(recipient["action_ranking"] + [None] * 20, donor["action_ranking"] + [None] * 20):
        for t in (a, b):
            if t and t not in seen:
                seen.add(t)
                merged.append(t)
    s["action_ranking"] = merged[: len(METHOD_TAGS)]
    return s if not validate(s, persona) else None


def recombine_params(rec_head, don_head, donor_persona: str) -> dict:
    """LDT parameter exchange: average the donor-signature tag columns of the two heads."""
    rb = rec_head.base if hasattr(rec_head, "base") else rec_head
    db = don_head.base if hasattr(don_head, "base") else don_head
    names = list(rb.feature_names)
    W = np.array(rb.weights)
    Wd = np.array(db.weights)
    for t in SIGNATURES[donor_persona]:
        j = names.index(t)
        W[:, j] = 0.5 * (W[:, j] + Wd[:, j])
    v = rb.to_jsonable()
    v["weights"] = W.tolist()
    return v


# --------------------------------------------------------------------------- Pareto archive
OBJ = (("fidelity", 1), ("utility", 1), ("brier", -1), ("cost", -1), ("abstain_f1", 1))


def dominates(a: dict, b: dict) -> bool:
    ge, gt = True, False
    for k, s in OBJ:
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            continue
        if s * x < s * y - 1e-12:
            ge = False
        if s * x > s * y + 1e-12:
            gt = True
    return ge and gt


def pareto_front(items: list[dict]) -> list[dict]:
    return [a for a in items if not any(dominates(b["panel"], a["panel"]) for b in items if b is not a)]


def freeze_choice(items: list[dict]) -> dict:
    """Max panel fidelity subject to panel gap >= -MARGIN; ties by utility; else max utility."""
    feas = [i for i in items if i["panel"]["gap"] is not None and i["panel"]["gap"] >= -MARGIN]
    if feas:
        return max(feas, key=lambda i: (i["panel"]["fidelity"], i["panel"]["utility"]))
    return max(items, key=lambda i: (i["panel"]["utility"], i["panel"]["fidelity"]))
