"""Final GYM-02 report: gates, summary.json, receipts.json, REPORT.md, skill registry.

    python -m research_gym.trm_skill_gym_v3.final_report --campaign experiments/trm_skill_gym_02/full --out experiments/trm_skill_gym_02

Every stage is reported as one of: established (gate passed), failed hypothesis
(gate evaluated and not passed), exploratory (descriptive or post hoc), or
unfinished (not run / cut by the deadline).  Missing results are never imputed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def rj(p: Path) -> Optional[Dict]:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def rjl(p: Path) -> List[Dict]:
    try:
        return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    except OSError:
        return []


def sha_lf(p: Path) -> str:
    b = Path(p).read_bytes()
    if Path(p).suffix in (".json", ".jsonl", ".md", ".py", ".log", ".txt"):
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def f3(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


# ------------------------------------------------------------------ stage gates
def race(c: Path) -> Dict:
    h = (rj(c / "race" / "heldout.json") or {}).get("results", {})
    fz = rj(c / "race" / "frozen.json") or {}
    out = {"skills": {}, "unfinished": []}
    for skill in ("retrieval", "deduction", "repair", "curation", "research", "stopping"):
        r = h.get(skill)
        if skill not in fz:
            out["unfinished"].append(skill)
            continue
        if not r or r.get("status") != "ok":
            out["skills"][skill] = {"status": (r or {}).get("status", "not_tested")}
            continue
        md = r["macro_selected"] - r["macro_baseline"]
        retained = r["selected"] == r["baseline"]
        gate = (not retained) and md >= 0.02 and r["worst_stratum_selected"] >= r["worst_stratum_baseline"] - 0.02
        pc = r["per_candidate"]
        best_any = max(pc, key=lambda n: np.mean([v["exact"] for v in pc[n].values()]))
        out["skills"][skill] = {
            "selected": r["selected"], "baseline": r["baseline"], "macro_selected": r["macro_selected"],
            "macro_baseline": r["macro_baseline"], "macro_delta": md, "worst_selected": r["worst_stratum_selected"],
            "worst_baseline": r["worst_stratum_baseline"], "decision": "incumbent_retained" if retained else ("promoted" if gate else "not_promoted"),
            "eligible": fz[skill]["eligible"], "incomplete": fz[skill]["incomplete"],
            "best_test_macro_any_candidate_descriptive": best_any,
            "macro_by_candidate": {n: float(np.mean([v["exact"] for v in pc[n].values()])) for n in pc},
            "worst_by_candidate": {n: float(min(v["exact"] for v in pc[n].values())) for n in pc},
            "flops_by_candidate": {n: float(np.mean([v["flops"] for v in pc[n].values()])) for n in pc},
            "inference_cpu_s_per_instance_by_candidate": {n: float(np.mean([v["inference_cpu_s_per_instance"] for v in pc[n].values()])) for n in pc},
            "host_ops_per_instance_by_candidate": {n: float(np.mean([v["host_ops_per_instance"] for v in pc[n].values()])) for n in pc},
            "per_stratum": {n: {f: v["exact"] for f, v in pc[n].items()} for n in pc},
            "contrast": r["contrast"]}
    return out


def hybrid(c: Path) -> Dict:
    res = rj(c / "hybrid" / "results.json") or {}
    if not res:
        return {"status": "unfinished"}
    arms = ("threshold_exact", "cert_const", "neural_only", "hybrid", "hybrid_adaptive")
    out = {"skills": {}}
    viol_total, oblig = 0, 0
    for key, job in res.items():
        if job.get("status") not in ("ok", "incomplete") or "test" not in job:
            out["skills"].setdefault(key.split("/")[0], {}).setdefault("jobs_not_ok", []).append(key)
            continue
        skill = key.split("/")[0]
        rec = out["skills"].setdefault(skill, {"seeds": [], "macro": {a: [] for a in arms}, "by_mode": {}, "coverage": [],
                                                "exact_on_unresolved": {a: [] for a in arms}, "neural_instances": {a: [] for a in arms},
                                                "host_ops": {a: [] for a in arms}, "flops": {a: [] for a in arms}})
        rec["seeds"].append(key.split("/")[1])
        noisy = [s for s in job["test"] if not s.startswith("complete")]
        for a in arms:
            rec["macro"][a].append(float(np.mean([job["test"][s][a]["exact"] for s in noisy])))
            eu = [job["test"][s][a]["exact_on_unresolved"] for s in noisy if job["test"][s][a].get("exact_on_unresolved") is not None]
            rec["exact_on_unresolved"][a].append(float(np.mean(eu)) if eu else None)
            rec["neural_instances"][a].append(float(np.mean([job["test"][s][a]["neural_instances"] for s in job["test"]])))
            rec["host_ops"][a].append(float(np.mean([job["test"][s][a]["host_edge_examinations"] for s in job["test"]])))
            rec["flops"][a].append(float(np.mean([job["test"][s][a]["model_flops"] for s in job["test"]])))
        for mode in ("noisy", "incomplete", "complete"):
            ss = [s for s in job["test"] if s.startswith(mode)]
            rec["by_mode"].setdefault(mode, {a: [] for a in arms})
            for a in arms:
                rec["by_mode"][mode][a].append(float(np.mean([job["test"][s][a]["exact"] for s in ss])))
        rec["coverage"].append(float(np.mean([job["test"][s]["hybrid"]["coverage"] for s in noisy])))
        for s in job["test"].values():
            for a in arms:
                viol_total += s[a]["certificate_violations"] if a in ("cert_const", "hybrid", "hybrid_adaptive") else 0
        for ob in job.get("proof_obligations", []):
            if ob.get("checked"):
                oblig += 1
                viol_total += ob["violations"]
    for skill, rec in out["skills"].items():
        if "macro" not in rec:
            continue
        m = {a: float(np.mean(v)) for a, v in rec["macro"].items()}
        rec["macro_mean"] = m
        rec["gate_pass"] = m["hybrid"] >= max(m["cert_const"], m["neural_only"]) + 0.02
        rec["by_mode_mean"] = {mode: {a: float(np.mean(v)) for a, v in d.items()} for mode, d in rec["by_mode"].items()}
    out["certificate_violations_total"] = viol_total
    out["proof_obligations_checked"] = oblig
    out["gate_pass_any"] = viol_total == 0 and any(r.get("gate_pass") for r in out["skills"].values())
    return out


def router(c: Path) -> Dict:
    r = rj(c / "router" / "results.json")
    if not r:
        return {"status": "unfinished"}
    res = r["results"]
    fixed = res.get("fixed_neural", {})
    gate = {}
    for mode in ("learned", "confidence_gated"):
        m = res.get(mode, {})
        if m.get("status") != "ok":
            continue
        d = m.get("delta_vs_fixed_neural", {})
        gate[mode] = bool(d.get("mean", -1) >= 0.02 and d.get("ci95", [0])[0] > 0 and m["metered_cost"] <= fixed.get("metered_cost", 0) + 1e-12)
    return {"status": "ok", "modes": {k: {kk: v.get(kk) for kk in ("utility", "all_correct", "stage_correct", "metered_cost",
                                                                  "experts_executed_per_episode", "model_flops_per_episode",
                                                                  "host_ops_per_episode", "delta_vs_fixed_neural", "stage_expert_counts", "realizable")}
                                      for k, v in res.items() if v.get("status") == "ok"},
            "gate": gate, "gate_pass_any": any(gate.values()), "frozen": r.get("frozen")}


def distill(c: Path) -> Dict:
    r = rj(c / "distill" / "results.json")
    if not r:
        return {"status": "unfinished"}
    return {"status": "ok", "selection": r.get("selection"), "pilot_reproduction": r.get("pilot_reproduction"),
            "paired_bootstrap": r.get("paired_bootstrap"), "n_records": r.get("n_records"), "caveats": r.get("caveats"),
            "schemes": {s: {rg: {m: {k: v.get(k) for k in ("mae_pp", "pair_accuracy")} for m, v in blk["models"].items()}
                            for rg, blk in d.get("rungs", {}).items()} for s, d in r.get("schemes", {}).items()}}


def research(c: Path) -> Dict:
    r = rj(c / "research" / "results.json")
    if not r:
        return {"status": "unfinished"}
    agg = r.get("aggregate", {})
    return {"status": "ok", "incomplete_jobs": r.get("incomplete_jobs"), "baselines": agg.get("baselines"),
            "learned": agg.get("learned"), "surprise_revision": agg.get("surprise_revision"),
            "planner_status": r.get("planner_status"), "groups": r.get("groups"), "model_configs": r.get("model_configs")}


def bpd(c: Path) -> Dict:
    r = rj(c / "bpd" / "results.json")
    if not r:
        return {"status": "unfinished"}
    agg = r.get("aggregate", {})
    gate = {}
    for skill, a in agg.items():
        ms = a.get("methods", {})
        f, g, h = ms.get("F"), ms.get("G"), ms.get("H")
        if not f:
            continue
        cis = f.get("ci95_by_seed") or []
        gate[skill] = bool(f["delta_vs_A_mean"] >= 0.02 and all(ci[0] > 0 for ci in cis) and
                           all(f["macro_mean"] >= x["macro_mean"] + 0.02 for x in (g, h) if x))
    return {"status": "ok", "complete": r.get("complete"), "incomplete_jobs": r.get("incomplete_jobs"), "bpd_commit": r.get("bpd_commit"),
            "aggregate": agg, "gate": gate, "gate_pass_any": any(gate.values())}


def inky(c: Path) -> Dict:
    pc = rjl(c / "inky_lab" / "prompt_comparison.jsonl")
    prompts = {}
    for x in pc:
        k = f"{x['skill']}/{x['variant']}/cap{x['cap']}"
        d = prompts.setdefault(k, {"n": 0, "exact": 0.0, "truncated": 0, "malformed_or_invalid": 0, "api_failure": 0, "completion_tokens": 0})
        d["n"] += 1
        d["exact"] += x.get("exact", 0)
        d["truncated"] += x["status"] == "truncated"
        d["malformed_or_invalid"] += x["status"] == "malformed_or_invalid"
        d["api_failure"] += x["status"] == "api_failure"
        d["completion_tokens"] += (x.get("usage") or {}).get("completion_tokens", 0) or 0
    for d in prompts.values():
        d["exact"] /= max(d["n"], 1)
        d["completion_tokens"] /= max(d["n"], 1)
    props = rjl(c / "inky_lab" / "proposals.jsonl")
    latest = {}
    for r in props:
        latest[(r["arm"], r["skill"], r["round"])] = r
    arms = {}
    for (arm, skill, k), r in latest.items():
        a = arms.setdefault(arm, {"n": 0, "status": {}})
        a["n"] += 1
        a["status"][r.get("status")] = a["status"].get(r.get("status"), 0) + 1
    meas = rjl(c / "inky_lab" / "measured.jsonl")
    for m in meas:
        a = arms.setdefault(m["arm"], {"n": 0, "status": {}})
        a.setdefault("improvements", []).append(m["improvement"])
        if m.get("prediction_error") is not None:
            a.setdefault("prediction_errors", []).append(m["prediction_error"])
    for a in arms.values():
        if a.get("improvements"):
            a["mean_improvement"] = float(np.mean(a["improvements"]))
            a["n_measured"] = len(a["improvements"])
            a["n_improved"] = int(sum(x > 0 for x in a["improvements"]))
        if a.get("prediction_errors"):
            a["prediction_mae"] = float(np.mean(np.abs(a["prediction_errors"])))
            a["prediction_bias"] = float(np.mean(a["prediction_errors"]))
    return {"prompt_comparison": prompts, "proposal_arms": arms, "measured": bool(meas)}


def audit_runner(c: Path) -> Dict:
    r = rj(c / "audit_runner" / "results.json")
    if not r:
        return {"status": "unfinished"}
    rows = {k: v for k, v in (r.get("leakage") or r).items() if "shuffle=" in k or "majority" in k} if isinstance(r, dict) else {}
    return {"status": "ok", "rows": rows}


# ------------------------------------------------------------------ main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, default=Path("experiments/trm_skill_gym_02/full"))
    p.add_argument("--out", type=Path, default=Path("experiments/trm_skill_gym_02"))
    a = p.parse_args(argv)
    c = a.campaign
    S = {"race": race(c), "hybrid": hybrid(c), "router": router(c), "distill": distill(c), "research": research(c),
         "bpd": bpd(c), "inky": inky(c), "shuffle_audit": audit_runner(c), "audit": rj(c / "audit.json"),
         "queue": rj(c / "queue_state.json")}
    files = sorted(p for p in c.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl", ".log", ".md") and "specs" not in p.parts)
    receipts = {"campaign_files_sha256_lf": {str(p.relative_to(c)).replace("\\", "/"): sha_lf(p) for p in files},
                "source_sha256_lf": {str(p).replace("\\", "/"): sha_lf(p) for p in sorted(Path("research_gym/trm_skill_gym_v3").glob("*.py"))
                                     + sorted(Path("research_gym/baby_goblin").glob("*.py"))},
                "preregistration_sha256_lf": sha_lf(Path("experiments/trm_skill_gym_02/PREREGISTRATION_FULL.md")),
                "replays": (S["audit"] or {}).get("receipt_replay_race")}
    (a.out / "summary.json").write_text(json.dumps(S, indent=1, sort_keys=True, default=float) + "\n", encoding="utf-8", newline="\n")
    (a.out / "receipts.json").write_text(json.dumps(receipts, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"race": {k: v.get("decision") for k, v in S["race"]["skills"].items()}, "race_unfinished": S["race"]["unfinished"],
                      "hybrid_gate": S["hybrid"].get("gate_pass_any"), "router_gate": S["router"].get("gate"),
                      "bpd_gate": S["bpd"].get("gate"), "research": S["research"].get("status")}, default=float))


if __name__ == "__main__":
    main()
