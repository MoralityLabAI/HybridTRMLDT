"""Independent audit of the JINN-GYM-01 run (outputs/run_20261128).

    python -m research_gym.jinn_gym_02.audit01 [RUN_DIR] [OUT_DIR]

The audit does not trust the stored outcome rows:

1. It regenerates every held-out task from its id and checks the digest against the
   DataGoblin decision records.
2. It rebuilds every frozen and initial module from its specification with the GYM-01
   training recipe and checks the rebuilt head against the committed head.json, and the
   compiled rubric against the committed rubric.metta.
3. It re-parses every raw Inky response from responses.jsonl, recomputes the decisions of
   arms A, B, C, D, E, F, G, C0 and D0, and compares them with the stored rows.
4. It recomputes the summary metrics and the paired contrasts with separate NumPy code
   (a vectorized bootstrap and a different seed), at root (task) level and at domain
   (task-family) level.
5. It classifies every accepted mutation and decomposes the evolution gain D - D0 into
   abstention repair and everything else, using counterfactual modules.
6. It checks for a winner's curse by evaluating each accepted (parent, child) pair on the
   held-out tasks.
7. It reruns the boundary audit and compares the host code hash with registration.
"""
from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from research_gym.jinn_gym_01 import host, metrics
from research_gym.jinn_gym_01 import tasks as T
from research_gym.jinn_gym_01.heads import VetoedHead, load_head, train_numpy, training_rows
from research_gym.jinn_gym_01.module import HeadScorer, PersonaModule, decide, prior_from_response
from research_gym.jinn_gym_01.prompts import parse_decision
from research_gym.jinn_gym_01.runner import CFG as CFG01, seed_of
from research_gym.jinn_gym_01.schema import spec_hash
from research_gym.jinn_gym_01.tags import PERSONAS, SHUFFLE, SIGNATURES, ABSTAIN

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "experiments" / "jinn_gym_01" / "outputs" / "run_20261128"
ABSTENTION_FIELDS = ("evidence_requirements", "abstention")
ABSTENTION_OPS = {"set_abstain_threshold", "set_evidence", "set_abstain_if_no_ranked_action"}
PREFERENCE_OPS = {"set_weight", "add_tag", "remove_tag", "set_ranking", "add_feature", "add_critical", "remove_critical"}
LATTICE_OPS = {"add_transition", "remove_transition"}


def read_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


class Builder:
    """The GYM-01 module recipe (runner.Gym.build_modules), re-implemented for the audit."""

    def __init__(self, workdir: Path):
        self.work = workdir
        self.train_tasks = [t for t in T.suite("train", 0, CFG01["train_tasks_per_domain"]) if T.verify_task(t)]
        self.cache = {}

    def build(self, persona: str, spec: dict) -> dict:
        h = spec_hash(spec)
        if h in self.cache:
            return self.cache[h]
        rubric = PersonaModule.compile(spec, "audit", self.work / persona / h[:12])
        lane = lambda tags: rubric.scorer(tags)[1]  # noqa: E731
        rows = training_rows(lane, self.train_tasks, seed=seed_of("comp", persona, h))
        head = VetoedHead(train_numpy(rows, seed=seed_of("head", persona, h) % (2 ** 31)), frozenset(spec["critical_tags"]))
        out = {"hash": h, "spec": spec, "rubric": rubric, "learned": rubric.with_scorer(HeadScorer(head)), "head": head}
        self.cache[h] = out
        return out


def head_diff(a, b_json: dict) -> float:
    aw = np.array(a.base.weights if hasattr(a, "base") else a.weights)
    ab = np.array(a.base.bias if hasattr(a, "base") else a.bias)
    return float(max(np.abs(aw - np.array(b_json["weights"])).max(), np.abs(ab - np.array(b_json["bias"])).max()))


# ------------------------------------------------------------------ independent metrics
def ind_summary(rows: list) -> dict:
    sig = [r for r in rows if r["signature_available"]]
    util = np.array([r["utility"] / r["best_utility"] for r in rows])
    ab = np.array([r["executed"] == ABSTAIN for r in rows])
    opt_ab = np.array([r["abstain_optimal"] for r in rows])
    tp, fp, fn = int((ab & opt_ab).sum()), int((ab & ~opt_ab).sum()), int((~ab & opt_ab).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    return {"n": len(rows), "fidelity": float(np.mean([r["signature_hit"] for r in sig])) if sig else None,
            "normalized_utility": float(util.mean()), "abstain_rate": float(ab.mean()), "abstain_f1": f1}


def bootstrap_diff(cells: list, cluster_key, n_boot: int = 4000, seed: int = 7) -> dict:
    """cells: (cluster, diff). Cluster bootstrap of the mean difference (vectorized)."""
    if not cells:
        return {"n": 0}
    clusters = sorted({c for c, _ in cells})
    idx = {c: i for i, c in enumerate(clusters)}
    sums = np.zeros(len(clusters))
    cnts = np.zeros(len(clusters))
    for c, d in cells:
        sums[idx[c]] += d
        cnts[idx[c]] += 1
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
    boot = sums[draws].sum(1) / cnts[draws].sum(1)
    est = sums.sum() / cnts.sum()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"diff": float(est), "ci95": [float(lo), float(hi)], "n_cells": int(cnts.sum()), "n_clusters": len(clusters)}


def contrast(rows: list, a: str, b: str, metric: str, level: str) -> dict:
    A = {(r["task_id"], r["persona"]): r for r in rows if r["arm"] == a}
    B = {(r["task_id"], r["persona"]): r for r in rows if r["arm"] == b}
    cells = []
    for k in sorted(set(A) & set(B)):
        if metric == "fidelity":
            if not A[k]["signature_available"]:
                continue
            d = float(A[k]["signature_hit"]) - float(B[k]["signature_hit"])
        else:
            d = A[k]["utility"] / A[k]["best_utility"] - B[k]["utility"] / B[k]["best_utility"]
        cluster = k[0] if level == "root" else A[k]["domain"]
        cells.append((cluster, d))
    return bootstrap_diff(cells, None)


# ------------------------------------------------------------------ audit
def audit(run: Path = RUN, out: Path | None = None) -> dict:
    out = out or ROOT / "experiments" / "jinn_gym_02" / "audit_gym01"
    out.mkdir(parents=True, exist_ok=True)
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    stored = read_jsonl(run / "final_records.jsonl")
    dec = read_jsonl(run / "decision_records.jsonl")
    resp = {r["key"]: r for r in read_jsonl(run / "responses.jsonl")}
    lineage = read_jsonl(run / "lineage.jsonl")
    its = read_jsonl(run / "stageC_iterations.jsonl")
    res: dict = {"run": str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run)}

    # 1. tasks
    digests = {}
    for r in dec:
        digests.setdefault(r["root"], r["task_digest"])
    task_ids = list(dict.fromkeys(r["task_id"] for r in stored))
    tasks = {}
    mism = 0
    for tid in task_ids:
        dom = next(r["domain"] for r in stored if r["task_id"] == tid)
        t = T.make_task(dom, 1, int(tid.rsplit("-", 1)[1]), "heldout")
        tasks[tid] = t
        mism += t.digest() != digests.get(tid)
        assert T.verify_task(t)
    res["tasks"] = {"n": len(tasks), "digest_mismatches": mism}

    # 2. modules
    work = Path(tempfile.mkdtemp(prefix="audit01_"))
    B = Builder(work)
    frozen, initial = {}, {}
    rebuild = {}
    for p in PERSONAS:
        fz = json.loads((run / "frozen" / f"{p}.json").read_text(encoding="utf-8"))
        fspec = fz["spec"]
        assert spec_hash(fspec) == fz["spec_hash"]
        ispec = json.loads((run / "specs" / f"{state['personas'][p]['initial']}.json").read_text(encoding="utf-8"))
        frozen[p], initial[p] = B.build(p, fspec), B.build(p, ispec)
        for tag, m in (("frozen", frozen[p]), ("initial", initial[p])):
            d = run / "modules" / p / m["hash"][:12]
            hj = json.loads((d / "head.json").read_text(encoding="utf-8"))
            def body(path):  # the comment and constitution lines carry the version label only
                return [l for l in Path(path).read_text(encoding="utf-8").splitlines()
                        if not l.startswith(";") and not l.startswith("(constitution")]
            rub_same = body(d / "rubric.metta") == body(m["rubric"].files["rubric"])
            rebuild[f"{p}:{tag}"] = {"max_abs_head_diff": head_diff(m["head"], hj), "rubric_equal_except_version_line": rub_same}
            # Use the committed head for decisions (what the run actually used).
            m["learned"] = m["rubric"].with_scorer(HeadScorer(load_head(hj)))
    res["module_rebuild"] = {"max_abs_head_diff": max(v["max_abs_head_diff"] for v in rebuild.values()),
                             "all_rubrics_equal": all(v["rubric_equal_except_version_line"] for v in rebuild.values()),
                             "per_module": rebuild}

    # 3. recompute decisions
    def parsed(key, t):
        e = resp.get(key)
        if e is None:
            return None, None
        pr = parse_decision(t, e["text"])
        return pr, prior_from_response(t, pr if pr["parsed"] else None)

    def direct(t, pr):
        g = host.execute(t, pr["choice"])
        return {"executed": g["executed"], "proposal": pr["choice"], "intervened": g["intervened"],
                "proposal_unauthorized": pr["choice"] is not None and pr["choice"] not in host.permitted(t),
                "confidence": pr["confidence"] if pr["confidence"] is not None else 0.5}

    def mod(t, p, arm, prior, mods):
        d = decide(t, prior, mods)
        d["proposal"] = d["proposal_unconstrained"]
        d["intervened"] = d["proposal_unconstrained"] != d["executed"] and d["proposal_unauthorized"]
        return metrics.outcome_record(t, p, arm, d, prior, d["persona_u"])

    rec, missing = [], Counter()
    mem_keys = defaultdict(list)
    for k in resp:
        if k.startswith("memory|"):
            _, tid, p, _ = k.split("|")
            mem_keys[(tid, p)].append(k)
    for tid, t in tasks.items():
        npr, nprior = parsed(f"neutral|{tid}", t)
        for p in PERSONAS:
            f, i = frozen[p], initial[p]
            cpr, cprior = parsed(f"card|{tid}|{p}|{f['hash']}", t)
            mk = mem_keys.get((tid, p), [])
            mpr, mprior = parsed(mk[0], t) if len(mk) == 1 else (None, None)
            if cpr is None:
                missing["card"] += 1
            if mpr is None:
                missing["memory"] += 1
            rec.append(metrics.outcome_record(t, p, "A", direct(t, npr), None))
            if cpr is not None:
                rec.append(metrics.outcome_record(t, p, "B", direct(t, cpr), None))
                rec.append(mod(t, p, "E", cprior, [f["learned"]]))
            if mpr is not None:
                rec.append(mod(t, p, "F", mprior, [f["learned"]]))
            rec.append(mod(t, p, "C", nprior, [f["rubric"]]))
            rec.append(mod(t, p, "D", nprior, [f["learned"]]))
            rec.append(mod(t, p, "G", nprior, [frozen[SHUFFLE[p]]["learned"]]))
            rec.append(mod(t, p, "C0", nprior, [i["rubric"]]))
            rec.append(mod(t, p, "D0", nprior, [i["learned"]]))
    S = {(r["task_id"], r["persona"], r["arm"]): r for r in stored}
    agree = Counter()
    disagree = Counter()
    for r in rec:
        s = S.get((r["task_id"], r["persona"], r["arm"]))
        if s is None:
            disagree[(r["arm"], "missing_stored")] += 1
            continue
        same = s["executed"] == r["executed"] and abs(s["utility"] - r["utility"]) < 1e-12
        (agree if same else disagree)[r["arm"]] += 1
    res["recompute"] = {"rows_recomputed": len(rec), "rows_stored": len(stored), "agree_by_arm": dict(agree),
                        "disagree_by_arm": {str(k): v for k, v in disagree.items()}, "missing_responses": dict(missing)}

    # 4. independent metrics and contrasts
    res["metrics_independent"] = {a: ind_summary([r for r in rec if r["arm"] == a]) for a in sorted({r["arm"] for r in rec})}
    res["metrics_stored"] = {a: {k: v for k, v in metrics.summarize([r for r in stored if r["arm"] == a]).items()
                                 if k in ("fidelity", "normalized_utility", "abstain_rate", "abstain_f1")}
                             for a in sorted({r["arm"] for r in stored})}
    C = []
    for a, b, m in (("D", "A", "fidelity"), ("D", "A", "utility"), ("D", "B", "fidelity"), ("D", "C", "fidelity"),
                    ("D", "G", "fidelity"), ("D", "D0", "fidelity"), ("D", "D0", "utility"), ("F", "D", "utility"),
                    ("E", "D", "fidelity"), ("C", "C0", "utility")):
        for level in ("root", "domain"):
            C.append({"a": a, "b": b, "metric": m, "level": level, **contrast(rec, a, b, m, level)})
    res["contrasts_independent"] = C

    # 5. lineage and abstention decomposition
    acc = [r for r in its if r["accepted"]]
    op_class = Counter()
    per_rev = Counter()
    for r in acc:
        kinds = {("abstention" if o.get("op") in ABSTENTION_OPS else "preference" if o.get("op") in PREFERENCE_OPS
                  else "lattice" if o.get("op") in LATTICE_OPS else "other") for o in r["ops"]}
        for o in r["ops"]:
            op_class[o.get("op")] += 1
        per_rev["+".join(sorted(kinds))] += 1
    first_abst = {}
    for r in acc:
        if r["persona"] not in first_abst and any(o.get("op") in ABSTENTION_OPS for o in r["ops"]):
            first_abst[r["persona"]] = r["iteration"]
    res["lineage"] = {"accepted": len(acc), "rejected": len(its) - len(acc), "ops": dict(op_class),
                      "revision_kinds": dict(per_rev), "first_abstention_revision_iteration": first_abst,
                      "nodes": len(lineage)}

    def hybrid(base: dict, donor: dict) -> dict:
        s = copy.deepcopy(base)
        for k in ABSTENTION_FIELDS:
            s[k] = copy.deepcopy(donor[k])
        return s

    decomp_rows = []
    for tid, t in tasks.items():
        _, nprior = parsed(f"neutral|{tid}", t)
        for p in PERSONAS:
            f, i = frozen[p], initial[p]
            i_abst = B.build(p, hybrid(i["spec"], f["spec"]))  # initial + frozen abstention fields
            f_noab = B.build(p, hybrid(f["spec"], i["spec"]))  # frozen with initial abstention fields
            for arm, m in (("D0", i["learned"]), ("D0+abst", i_abst["learned"]), ("D-abst", f_noab["learned"]),
                           ("D", f["learned"])):
                decomp_rows.append(mod(t, p, arm, nprior, [m]))
    dsum = {a: ind_summary([r for r in decomp_rows if r["arm"] == a]) for a in ("D0", "D0+abst", "D-abst", "D")}

    def shap(metric):
        g = lambda a: dsum[a][metric]  # noqa: E731
        total = g("D") - g("D0")
        ab = 0.5 * ((g("D0+abst") - g("D0")) + (g("D") - g("D-abst")))
        return {"total": total, "abstention": ab, "rest": total - ab,
                "abstention_share": ab / total if abs(total) > 1e-12 else None}
    res["abstention_decomposition"] = {"arms": dsum, "fidelity": shap("fidelity"),
                                       "normalized_utility": shap("normalized_utility"),
                                       "abstain_f1": shap("abstain_f1"),
                                       "contrasts": [{"a": a, "b": b, "metric": m, **contrast(decomp_rows, a, b, m, "root")}
                                                     for a, b, m in (("D0+abst", "D0", "utility"), ("D0+abst", "D0", "fidelity"),
                                                                     ("D", "D0+abst", "fidelity"), ("D", "D0+abst", "utility"))]}

    # 6. winner's curse on held-out
    lin_acc = [r for r in lineage if r["accepted"] and r.get("parent")]
    wc = []
    for r in lin_acc:
        p = r["persona"]
        try:
            ps = json.loads((run / "specs" / f"{r['parent']}.json").read_text(encoding="utf-8"))
            cs = json.loads((run / "specs" / f"{r['child']}.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        pm, cm = B.build(p, ps), B.build(p, cs)
        rp, rc = [], []
        for tid, t in tasks.items():
            _, nprior = parsed(f"neutral|{tid}", t)
            rp.append(mod(t, p, "P", nprior, [pm["learned"]]))
            rc.append(mod(t, p, "Q", nprior, [cm["learned"]]))
        wc.append({"persona": p, "iteration": r["iteration"], "selection_delta_J": r["delta_J"],
                   "heldout_delta_J": metrics.objective(rc) - metrics.objective(rp)})
    sel = np.array([w["selection_delta_J"] for w in wc])
    hel = np.array([w["heldout_delta_J"] for w in wc])
    res["winners_curse"] = {"n": len(wc), "mean_selection_delta_J": float(sel.mean()), "mean_heldout_delta_J": float(hel.mean()),
                            "share_heldout_positive": float((hel > 0).mean()), "share_heldout_negative": float((hel < 0).mean()),
                            "pearson_r": float(np.corrcoef(sel, hel)[0, 1]) if len(wc) > 2 else None, "items": wc}

    # 7. boundary
    reg = json.loads((ROOT / "experiments" / "jinn_gym_01" / "registration.json").read_text(encoding="utf-8"))
    fns = {f"{k}:{p}": (lambda t, m=frozen[p][k]: decide(t, prior_from_response(t, None), [m])) for p in PERSONAS for k in ("rubric", "learned")}
    res["boundary"] = {"audit": host.boundary_audit(list(tasks.values()), fns),
                       "host_code_hash_now": host.host_code_hash(), "host_code_hash_registered": reg["host_code_hash"],
                       "host_code_unchanged": host.host_code_hash() == reg["host_code_hash"],
                       "unauthorized_executions_recomputed": sum(r["unauthorized_execution"] for r in rec)}
    (out / "audit.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8", newline="\n")
    return res


if __name__ == "__main__":
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else RUN
    r = audit(run)
    print(json.dumps({k: r[k] for k in ("tasks", "recompute", "abstention_decomposition", "boundary")}, indent=1, default=str)[:6000])
