"""Build results.json and results.md from a JINN-GYM-01 output directory.

    python -m research_gym.jinn_gym_01.report OUT_DIR
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from . import metrics
from .tags import PERSONAS

CONTRASTS = (  # (arm_a, arm_b, metric): registered primary and secondary contrasts
    ("D", "A", "fidelity"), ("D", "A", "normalized_utility"),
    ("C", "A", "fidelity"), ("B", "A", "fidelity"), ("B", "A", "normalized_utility"),
    ("D", "B", "fidelity"), ("D", "C", "fidelity"), ("E", "B", "fidelity"), ("E", "D", "fidelity"),
    ("F", "D", "normalized_utility"), ("F", "D", "fidelity"),
    ("D", "G", "fidelity"), ("D", "D0", "fidelity"), ("D", "D0", "normalized_utility"),
    ("C", "C0", "fidelity"), ("C", "C0", "normalized_utility"),
)


def _read(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def arm_block(rows: list) -> dict:
    arms = sorted({r["arm"] for r in rows})
    out = {"by_arm": {}, "by_arm_persona": {}, "divergence": {}, "contrasts": []}
    for a in arms:
        ra = [r for r in rows if r["arm"] == a]
        out["by_arm"][a] = metrics.summarize(ra)
        out["divergence"][a] = metrics.divergence(ra)
        out["by_arm_persona"][a] = {p: metrics.summarize([r for r in ra if r["persona"] == p])
                                    for p in PERSONAS if any(r["persona"] == p for r in ra)}
    for a, b, m in CONTRASTS:
        if a in arms and b in arms:
            out["contrasts"].append({"a": a, "b": b, "metric": m, **metrics.paired(rows, a, b, m)})
    return out


def build_report(out: Path) -> dict:
    out = Path(out)
    state = json.loads((out / "state.json").read_text(encoding="utf-8")) if (out / "state.json").exists() else {}
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8")) if (out / "manifest.json").exists() else {}
    a_att = _read(out / "stageA_attempts.jsonl")
    b_rows = _read(out / "stageB_records.jsonl")
    c_its = _read(out / "stageC_iterations.jsonl")
    comps = _read(out / "compositions.jsonl")
    f_rows = _read(out / "final_records.jsonl")
    res = {"experiment": "JINN-GYM-01", "dry_run": manifest.get("dry_run"), "model": manifest.get("model"),
           "stop_reason": state.get("stop_reason"), "phase_reached": state.get("phase"),
           "runtime_hours": ((state.get("ended_at") or 0) - (state.get("started_at") or 0)) / 3600,
           "usage": state.get("usage"), "ram_peak_increment_bytes": state.get("ram_peak_increment_bytes"),
           "resumes": state.get("resumes")}
    resp = _read(out / "responses.jsonl")
    res["calls_from_responses"] = {"total": len(resp),
                                   "by_purpose": dict(Counter(r.get("purpose") for r in resp)),
                                   "finish_length": sum(r.get("finish_reason") == "length" for r in resp),
                                   "empty_text": sum(not (r.get("text") or "").strip() for r in resp)}
    per =defaultdict(lambda: {"attempts": 0, "valid": False, "errors": Counter()})
    for r in a_att:
        d = per[r["persona"]]
        d["attempts"] += 1
        d["valid"] = d["valid"] or r["valid"]
        for e in r["errors"]:
            d["errors"][e.split(":")[0][:60]] += 1
    res["stage_a"] = {p: {"attempts": v["attempts"], "accepted": v["valid"],
                          "error_kinds": dict(v["errors"].most_common(5))} for p, v in per.items()}
    res["stage_b"] = {"n_tasks": len({r["task_id"] for r in b_rows}), **arm_block(b_rows)} if b_rows else {"n_tasks": 0}
    fails = Counter((r["failure"] or "accepted").split(":")[0] for r in c_its)
    per_c = {}
    for p in PERSONAS:
        rs = [r for r in c_its if r["persona"] == p]
        if rs:
            per_c[p] = {"iterations": len(rs), "accepted": sum(r["accepted"] for r in rs),
                        "generation": (state.get("personas", {}).get(p) or {}).get("generation"),
                        "mean_delta_J_accepted": metrics._mean([r["child_J"] - r["parent_J"] for r in rs
                                                                if r["accepted"] and r["child_J"] is not None])}
    code_counts = Counter(c for r in c_its for c in r["codes"])
    res["stage_c"] = {"iterations": len(c_its), "outcomes": dict(fails), "per_persona": per_c,
                      "diagnostic_codes": dict(code_counts),
                      "inky_card_parse_rate": metrics._mean([float(r["inky_parse_ok"]) for r in c_its])}
    res["compositions"] = {"n": len(comps), "cooperating": sum(c["cooperates"] for c in comps),
                           "all_boundary_audits_passed": all(c["boundary_audit"]["passed"] for c in comps),
                           "items": [{"pair": c["pair"], "retention_p": c["retention_p"], "retention_q": c["retention_q"],
                                      "utility_comp": c["utility_comp"], "utility_solo_max": c["utility_solo_max"],
                                      "cooperates": c["cooperates"]} for c in comps]}
    res["final"] = {"n_tasks": len({r["task_id"] for r in f_rows}), **arm_block(f_rows)} if f_rows else {"n_tasks": 0}
    preds = {}
    for p in PERSONAS:
        fz = out / "frozen" / f"{p}.json"
        if fz.exists() and f_rows:
            spec = json.loads(fz.read_text(encoding="utf-8"))["spec"]
            preds[p] = metrics.check_predictions(spec, [r for r in f_rows if r["persona"] == p and r["arm"] == "C"])
    res["predictions"] = preds
    for name in ("cca_audit_stageA.json", "cca_audit_freeze.json"):
        if (out / name).exists():
            res[name.replace(".json", "")] = json.loads((out / name).read_text(encoding="utf-8"))
    res["unauthorized_executions_total"] = sum(r["unauthorized_execution"] for r in b_rows + f_rows)
    res["records"] = {k: len(_read(out / f"{k}.jsonl")) for k in
                      ("training_records", "failed_trajectories", "decision_records", "lineage", "responses")}
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8", newline="\n")
    (out / "results.md").write_text(render_md(res), encoding="utf-8", newline="\n")
    return res


def _f(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render_md(res: dict) -> str:
    L = [f"# JINN-GYM-01 run results{' (DRY RUN, fake model)' if res.get('dry_run') else ''}", "",
         f"Model: `{res.get('model')}`. Phase reached: {res.get('phase_reached')}. Stop reason: {res.get('stop_reason')}. "
         f"Runtime: {_f(res.get('runtime_hours'), 2)} h.", ""]
    u = res.get("usage") or {}
    cr = res.get("calls_from_responses") or {}
    L.append(f"Successful Inky calls recorded in responses.jsonl (all processes): {cr.get('total')} "
             f"(by purpose {cr.get('by_purpose')}); {cr.get('finish_length')} ended at the completion cap, "
             f"{cr.get('empty_text')} with empty text.")
    L.append(f"Client usage counters (last process only for this run): {u.get('requests')} ok, {u.get('failed_requests')} failed attempts, "
             f"{u.get('rate_limited')} rate-limited; prompt tokens {u.get('prompt_tokens')}, completion tokens "
             f"{u.get('completion_tokens')}; paid spend ${u.get('paid_spend_usd')}. Peak process-tree RAM increment: "
             f"{_f((res.get('ram_peak_increment_bytes') or 0) / 2**20, 1)} MiB.")
    L += ["", "## Stage A", "", "| persona | attempts | accepted | error kinds |", "|---|---|---|---|"]
    for p, v in res.get("stage_a", {}).items():
        L.append(f"| {p} | {v['attempts']} | {v['accepted']} | {v['error_kinds'] or ''} |")
    for title, key in (("Stage B (development panel, initial personas)", "stage_b"),
                       ("Final held-out evaluation (frozen personas, held-out families)", "final")):
        blk = res.get(key) or {}
        L += ["", f"## {title}", "", f"Tasks: {blk.get('n_tasks', 0)}."]
        if not blk.get("by_arm"):
            continue
        L += ["", "| arm | n | fidelity | norm. utility | optimal | Brier | ECE | abstain F1 | interv. | unauth. exec | info ret. | disagreement | JS bits |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for a, s in blk["by_arm"].items():
            dv = blk["divergence"].get(a, {})
            L.append(f"| {a} | {s['n']} | {_f(s['fidelity'])} | {_f(s['normalized_utility'])} | {_f(s['optimal_rate'])} | "
                     f"{_f(s['brier'])} | {_f(s['ece'])} | {_f(s['abstain_f1'])} | {_f(s['intervention_rate'])} | "
                     f"{s['unauthorized_executions']} | {_f(s['info_retention'])} | {_f(dv.get('pairwise_disagreement'))} | "
                     f"{_f(dv.get('pairwise_js_bits'))} |")
        L += ["", "| contrast | metric | diff | 95% CI (task bootstrap) | cells |", "|---|---|---|---|---|"]
        for c in blk["contrasts"]:
            if c.get("n_cells"):
                L.append(f"| {c['a']} - {c['b']} | {c['metric']} | {_f(c['diff'])} | [{_f(c['ci95'][0])}, {_f(c['ci95'][1])}] | {c['n_cells']} |")
    sc = res.get("stage_c", {})
    L += ["", "## Stage C", "", f"Iterations: {sc.get('iterations')}. Outcomes: {sc.get('outcomes')}. "
          f"Card-call parse rate: {_f(sc.get('inky_card_parse_rate'))}.", "",
          "| persona | iterations | accepted | final generation | mean dJ of accepted |", "|---|---|---|---|---|"]
    for p, v in sc.get("per_persona", {}).items():
        L.append(f"| {p} | {v['iterations']} | {v['accepted']} | {v['generation']} | {_f(v['mean_delta_J_accepted'])} |")
    co = res.get("compositions", {})
    L += ["", f"Compositions: {co.get('n')} tested, {co.get('cooperating')} met the cooperation criterion; "
          f"all boundary audits passed: {co.get('all_boundary_audits_passed')}."]
    L += ["", "## Falsifiable predictions (frozen specs, arm C, held-out)", ""]
    for p, ps in res.get("predictions", {}).items():
        for x in ps:
            L.append(f"- {p} `{x['id']}`: {x['metric']} {x['comparator']} {x['threshold']} on {x['domain']}: "
                     f"observed {_f(x['observed'])} (n={x['n']}) -> {'held' if x['held'] else ('failed' if x['held'] is False else 'untestable')}")
    fz = res.get("cca_audit_freeze") or {}
    L += ["", "## Boundary", "", f"Unauthorized executions across Stage B and final: {res.get('unauthorized_executions_total')}.",
          f"Freeze audit: {json.dumps(fz.get('audit'))}; host code unchanged: {fz.get('host_code_unchanged')}."]
    L += ["", f"Records: {res.get('records')}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    print(json.dumps(build_report(Path(sys.argv[1]))["final"].get("by_arm", {}), indent=1)[:4000])
