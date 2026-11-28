"""JINN-GYM-02 results (results.json, results.md) and DataGoblin exports.

    python -m research_gym.jinn_gym_02.report OUT_DIR
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from research_gym.jinn_gym_01 import metrics
from research_gym.jinn_gym_01.inky import MODEL
from research_gym.jinn_gym_01.tags import PERSONAS

from .audit01 import bootstrap_diff
from .evolve import ARMS, MARGIN

SCHEMA = "datagoblin.jinn_gym_02.{kind}.v1"


def _read(p: Path) -> list:
    if not p.exists():
        return []
    out = []
    for l in p.read_text(encoding="utf-8").splitlines():
        if l.strip():
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


def persona_contrast(rows, a, b, metric, level):
    A = {(r["task_id"], r["persona"]): r for r in rows if r["arm"] == a}
    B = {(r["task_id"], r["persona"]): r for r in rows if r["arm"] == b}
    cells = []
    for k in sorted(set(A) & set(B)):
        if metric == "fidelity":
            if not A[k]["signature_available"]:
                continue
            d = float(A[k]["signature_hit"]) - float(B[k]["signature_hit"])
        else:
            d = A[k]["normalized_utility"] - B[k]["normalized_utility"]
        cells.append((k[0] if level == "root" else (A[k]["domain"], A[k].get("template")), d))
    return bootstrap_diff(cells, None)


def system_contrast(rr, a, b, level):
    cells = []
    for r in rr:
        ua = r["uniform"] if a == "uniform" else (r["option_utility"]["inky"] if a == "neutral" else r["utility"][a])
        ub = r["uniform"] if b == "uniform" else (r["option_utility"]["inky"] if b == "neutral" else r["utility"][b])
        cells.append((r["task_id"] if level == "root" else (r["domain"], r["template"]), ua - ub))
    return bootstrap_diff(cells, None)


def build_report(out: Path) -> dict:
    out = Path(out)
    st = json.loads((out / "state.json").read_text(encoding="utf-8")) if (out / "state.json").exists() else {}
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8")) if (out / "manifest.json").exists() else {}
    muts = _read(out / "mutations.jsonl")
    arch = _read(out / "archive.jsonl")
    rec = _read(out / "recombination.jsonl")
    hr = _read(out / "heldout_rows.jsonl")
    rr = _read(out / "routing_rows.jsonl")
    resp = _read(out / "responses.jsonl")
    res = {"experiment": "JINN-GYM-02", "dry_run": man.get("dry_run"), "small": man.get("small"), "model": man.get("model"),
           "lean_version": man.get("lean_version"), "phase_reached": st.get("phase"), "stop_reason": st.get("stop_reason"),
           "runtime_hours": ((st.get("ended_at") or 0) - (st.get("started_at") or 0)) / 3600, "resumes": st.get("resumes"),
           "bank_sizes": st.get("bank_sizes"), "rerun_check": st.get("rerun_check"), "cpu_seconds": st.get("cpu_seconds"),
           "ram_peak_increment_bytes": st.get("ram_peak_increment_bytes"),
           "calls": {"total": len(resp), "by_purpose": dict(Counter(r.get("purpose") for r in resp)),
                     "finish_length": sum(r.get("finish_reason") == "length" for r in resp),
                     "empty_text": sum(not (r.get("text") or "").strip() for r in resp),
                     "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in resp),
                     "completion_tokens": sum(r.get("completion_tokens") or 0 for r in resp)},
           "paid_spend_usd": 0.0}
    # ---- mutation arms
    per_arm = {}
    for a in ARMS:
        ms = [m for m in muts if m["arm"] == a]
        acc = [m for m in ms if m["accepted"]]
        per_arm[a] = {"candidates": len(ms), "accepted": len(acc),
                      "reasons": dict(Counter(m["reason"].split(":")[0] for m in ms)),
                      "mean_delta_fidelity_accepted": float(np.mean([m["child_stats"]["fidelity"] - m["parent_stats"]["fidelity"] for m in acc])) if acc else None,
                      "mean_delta_gap_accepted": float(np.mean([m["child_stats"]["gap"] - m["parent_stats"]["gap"] for m in acc])) if acc else None}
    rules = Counter(m["rule"] for m in muts if m["arm"] == "abstention")
    codes = Counter()
    for m in muts:
        if m["arm"] == "abstention":
            for k, v in m["diagnostics"]["codes"].items():
                codes[k] += v
    res["mutation"] = {"rounds": st.get("round"), "rounds_available": st.get("mutation_rounds_available"), "per_arm": per_arm,
                       "abstention_rules": dict(rules), "abstention_arm_diagnostic_codes": dict(codes),
                       "boundary_audit_failures": sum(1 for m in muts if m.get("boundary_audit") and not m["boundary_audit"]["passed"])}
    # ---- recombination
    rv = [x for x in rec if x.get("valid")]
    by_op = defaultdict(list)
    for x in rv:
        by_op[x["operator"]].append(x)
    res["recombination"] = {op: {"n": len(v), **{k: float(np.mean([x[k] for x in v])) for k in ("transfer", "interference", "complementarity")},
                                 "share_positive_complementarity": float(np.mean([x["complementarity"] > 0 for x in v]))}
                            for op, v in by_op.items()}
    res["recombination"]["invalid"] = sum(1 for x in rec if not x.get("valid"))
    # ---- archive / freeze
    front = json.loads((out / "pareto_front.json").read_text(encoding="utf-8")) if (out / "pareto_front.json").exists() else {}
    frozen = {}
    for p in PERSONAS:
        f = out / "frozen" / f"{p}.json"
        if f.exists():
            z = json.loads(f.read_text(encoding="utf-8"))
            frozen[p] = {"source": z["source"], "panel": z["panel"], "hash": z["hash"]}
    res["archive"] = {"variants": len(arch), "pareto_front_sizes": {p: len(v) for p, v in front.items()}, "frozen": frozen}
    # ---- held-out persona level
    if hr:
        arms = sorted({r["arm"] for r in hr})
        res["heldout_persona"] = {"n_tasks": len({r["task_id"] for r in hr}),
                                  "by_arm": {a: metrics.summarize([r for r in hr if r["arm"] == a]) for a in arms},
                                  "by_arm_persona": {a: {p: metrics.summarize([r for r in hr if r["arm"] == a and r["persona"] == p])
                                                         for p in PERSONAS} for a in ("A", "FINAL", "ROOT")}}
        C = []
        pairs = [("FINAL", "A", "normalized_utility"), ("FINAL", "A", "fidelity"), ("FINAL", "ROOT", "normalized_utility"),
                 ("FINAL", "ROOT", "fidelity"), ("ROOT", "A", "normalized_utility"), ("ROOT", "A", "fidelity"),
                 ("FINAL", "FINAL_C", "fidelity")]
        pairs += [(f"ARM_{a}", "ARM_random", m) for a in ARMS if a != "random" for m in ("fidelity", "normalized_utility")]
        pairs += [(f"ARM_{a}", "A", "normalized_utility") for a in ARMS]
        for a, b, m in pairs:
            for lv in ("root", "template"):
                C.append({"a": a, "b": b, "metric": m, "level": lv, **persona_contrast(hr, a, b, m, lv)})
        res["heldout_persona"]["contrasts"] = C
        res["heldout_persona"]["noninferiority_by_persona"] = {
            p: persona_contrast([r for r in hr if r["persona"] == p], "FINAL", "A", "normalized_utility", "root") for p in PERSONAS}
    # ---- held-out system level (routing)
    if rr:
        keys = sorted(rr[0]["utility"])
        res["heldout_routing"] = {"n_tasks": len(rr),
                                  "mean_utility": {k: float(np.mean([r["utility"][k] for r in rr])) for k in keys},
                                  "neutral": float(np.mean([r["option_utility"]["inky"] for r in rr])),
                                  "uniform": float(np.mean([r["uniform"] for r in rr])),
                                  "choice_counts": {k: dict(Counter(r["choices"][k] for r in rr)) for k in keys},
                                  "unauthorized_executions": sum(v for r in rr for v in r["unauthorized"].values())}
        C = []
        for a, b in (("trm_k4", "neutral"), ("trm_k4", "static_best"), ("trm_k4", "domain_best"), ("trm_k4", "rule"),
                     ("trm_k4", "inky_route"), ("trm_k4", "trm_k1"), ("trm_k4", "uniform"), ("static_best", "neutral"),
                     ("domain_best", "neutral"), ("inky_route", "neutral"), ("oracle", "neutral"), ("trm_k1", "neutral")):
            if (a in keys or a in ("neutral", "uniform")) and (b in keys or b in ("neutral", "uniform")):
                for lv in ("root", "template"):
                    C.append({"a": a, "b": b, "level": lv, **system_contrast(rr, a, b, lv)})
        res["heldout_routing"]["contrasts"] = C
        res["heldout_routing"]["by_domain"] = {d: {k: float(np.mean([r["utility"][k] for r in rr if r["domain"] == d])) for k in keys}
                                               | {"neutral": float(np.mean([r["option_utility"]["inky"] for r in rr if r["domain"] == d]))}
                                               for d in sorted({r["domain"] for r in rr})}
    for name in ("router.json", "selfmodel_eval.json", "compact.json", "cca_audit_freeze.json"):
        if (out / name).exists():
            j = json.loads((out / name).read_text(encoding="utf-8"))
            if name == "router.json":
                j = {k: v for k, v in j.items() if not k.startswith("trm_")} | {k: {"best_val_routed_utility": v["best_val_routed_utility"]} for k, v in j.items() if k.startswith("trm_")}
            res[name.replace(".json", "")] = j
    res["unauthorized_executions_total"] = (sum(r["unauthorized_execution"] for r in hr) +
                                            (res.get("heldout_routing") or {}).get("unauthorized_executions", 0))
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8", newline="\n")
    (out / "results.md").write_text(render(res), encoding="utf-8", newline="\n")
    export(out, st, muts, rec, rr, resp, res)
    return res


def _f(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render(res) -> str:
    L = [f"# JINN-GYM-02 results{' (DRY RUN, fake model)' if res.get('dry_run') else ''}", "",
         f"Model `{res.get('model')}`; {res.get('lean_version')}. Phase {res.get('phase_reached')}; stop {res.get('stop_reason')}; "
         f"{_f(res.get('runtime_hours'), 2)} h; calls {res['calls']['total']} ({res['calls']['by_purpose']}); "
         f"cap hits {res['calls']['finish_length']}, empty {res['calls']['empty_text']}; paid spend $0.", ""]
    hr = res.get("heldout_routing")
    if hr:
        L += ["## Routing on held-out (system level, normalized verified utility)", "",
              f"Neutral Inky {_f(hr['neutral'])}; uniform over modules {_f(hr['uniform'])}; " +
              "; ".join(f"{k} {_f(v)}" for k, v in hr["mean_utility"].items()), "",
              "| contrast | level | diff | 95% CI | clusters |", "|---|---|---|---|---|"]
        for c in hr["contrasts"]:
            if c.get("n_cells"):
                L.append(f"| {c['a']} - {c['b']} | {c['level']} | {_f(c['diff'])} | [{_f(c['ci95'][0])}, {_f(c['ci95'][1])}] | {c['n_clusters']} |")
    hp = res.get("heldout_persona")
    if hp:
        L += ["", "## Persona level on held-out", "", "| arm | fidelity | norm. utility | abstain F1 | Brier | unauth. |", "|---|---|---|---|---|---|"]
        for a, s in hp["by_arm"].items():
            L.append(f"| {a} | {_f(s['fidelity'])} | {_f(s['normalized_utility'])} | {_f(s['abstain_f1'])} | {_f(s['brier'])} | {s['unauthorized_executions']} |")
        L += ["", "| contrast | metric | level | diff | 95% CI |", "|---|---|---|---|---|"]
        for c in hp["contrasts"]:
            if c.get("n_cells"):
                L.append(f"| {c['a']} - {c['b']} | {c['metric']} | {c['level']} | {_f(c['diff'])} | [{_f(c['ci95'][0])}, {_f(c['ci95'][1])}] |")
    m = res.get("mutation") or {}
    L += ["", "## Mutation arms", "", f"Rounds {m.get('rounds')} of {m.get('rounds_available')}.", "",
          "| arm | candidates | accepted | mean dF accepted | mean d(gap) accepted | reasons |", "|---|---|---|---|---|---|"]
    for a, v in (m.get("per_arm") or {}).items():
        L.append(f"| {a} | {v['candidates']} | {v['accepted']} | {_f(v['mean_delta_fidelity_accepted'])} | {_f(v['mean_delta_gap_accepted'])} | {v['reasons']} |")
    L += ["", f"Abstention-repair rules fired: {m.get('abstention_rules')}", "", "## Recombination", "", json.dumps(res.get("recombination"), indent=1)]
    L += ["", "## Self-model, compactification, boundary", "",
          "See results.json keys selfmodel_eval, compact, cca_audit_freeze.",
          f"Unauthorized executions (held-out persona rows and routing): {res.get('unauthorized_executions_total')}."]
    return "\n".join(L) + "\n"


def export(out: Path, st, muts, rec, rr, resp, res):
    ex = out / "export"
    ex.mkdir(exist_ok=True)
    hashes_digest = json.loads((out / "manifest.json").read_text(encoding="utf-8")).get("code_digest") if (out / "manifest.json").exists() else None

    def w(name, rows):
        with (ex / name).open("w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    base = {"source_model": MODEL, "provider_execution_attested": False, "source_hashes_digest": hashes_digest}
    acc = [m for m in muts if m["accepted"]]
    rej = [m for m in muts if not m["accepted"]]

    def mrow(m, kind):
        return {"schema": SCHEMA.format(kind=kind), "id": f"{m['round']}|{m['persona']}|{m['arm']}|{m['parent'][:12]}", "kind": kind,
                "root": m["persona"], "family": "selection/f2", "split": "train" if m["accepted"] else "failed",
                "input": {"persona": m["persona"], "arm": m["arm"], "parent_spec_hash": m["parent"], "diagnostics": m["diagnostics"],
                          "batch": m["batch"]},
                "target": {"ops": m["ops"], "child_spec_hash": m.get("child")},
                "verification": {"parent": m["parent_stats"], "child": m.get("child_stats"), "accepted": m["accepted"],
                                 "reason": m["reason"], "boundary_audit": m.get("boundary_audit")},
                "loss": "sft_on_target" if m["accepted"] else "none", "inky_text": m.get("inky_text"), **base}
    w("mutations_accepted.jsonl", [mrow(m, "persona_mutation") for m in acc])
    w("mutations_rejected.jsonl", [mrow(m, "persona_mutation_rejected") for m in rej])
    fh = [{"schema": SCHEMA.format(kind="failed_hypothesis"), "kind": "rejected_mutation", "persona": m["persona"], "arm": m["arm"],
           "hypothesis": (m.get("inky_text") or json.dumps(m["ops"]))[:1500], "refuted_by": m["reason"],
           "evidence": {"parent": m["parent_stats"], "child": m.get("child_stats")}, **base} for m in rej]
    sm = res.get("selfmodel_eval") or {}
    for p, per in (sm.get("elicited") or {}).items():
        for d, v in per.items():
            for k, e in (v.get("abs_error") or {}).items():
                if e is not None and e > 0.15:
                    fh.append({"schema": SCHEMA.format(kind="failed_hypothesis"), "kind": "self_prediction", "persona": p, "domain": d,
                               "metric": k, "predicted": v["predicted"][k], "observed": v["observed"][k], "abs_error": e, **base})
    w("failed_hypotheses.jsonl", fh)
    w("repair_decisions.jsonl", [{"schema": SCHEMA.format(kind="repair_decision"), "persona": m["persona"], "round": m["round"],
                                  "rule": m["rule"], "ops": m["ops"], "diagnostics": m["diagnostics"]["codes"],
                                  "abstain_reasons": m["diagnostics"]["abstain_reasons"], "accepted": m["accepted"],
                                  "reason": m["reason"], **base} for m in muts if m["arm"] == "abstention"])
    vf = []
    for name in ("dev", "sel", "held"):
        for d in _read(out / "bank" / f"{name}.jsonl"):
            h = d["hidden"]["domain_hidden"]
            rec_ = h.get("receipt") or h.get("receipts")
            vf.append({"schema": SCHEMA.format(kind="verifier_feedback"), "task_id": d["task_id"], "domain": d["domain"],
                       "family": d["family"], "template": d["hidden"].get("template"), "utilities": d["utilities"],
                       "receipt": rec_, **base})
    w("verifier_feedback.jsonl", vf)
    w("routing_decisions.jsonl", [{"schema": SCHEMA.format(kind="routing_decision"), "task_id": r["task_id"], "domain": r["domain"],
                                   "features": r["features"], "choices": r["choices"], "utility": r["utility"],
                                   "option_utility": r["option_utility"], **base} for r in rr])
    (ex / "resource_usage.json").write_text(json.dumps({"calls": res["calls"], "cpu_seconds": st.get("cpu_seconds"),
                                                        "ram_peak_increment_bytes": st.get("ram_peak_increment_bytes"),
                                                        "runtime_hours": res.get("runtime_hours"), "paid_spend_usd": 0.0}, indent=1),
                                            encoding="utf-8", newline="\n")


if __name__ == "__main__":
    r = build_report(Path(sys.argv[1]))
    print(json.dumps({k: r.get(k) for k in ("phase_reached", "stop_reason", "calls")}, indent=1, default=str))
