"""Registered MOLT-GOBLIN-03 contrasts on hidden roots, distillation retention, costs, mechanisms."""
from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics as stats
from collections import defaultdict
from pathlib import Path

from research_gym.molt_goblin_02.analysis import pareto, seed_ci

CONTRASTS = {
    "P1 source decomposition (vs A)": [(x, "A_single") for x in ("B_hrm", "C_mix", "D_tied", "H_sparse_st", "I_anneal", "J_sequential")],
    "P2 matched-cost controls (B vs)": [("B_hrm", x) for x in ("E_param", "G_flop", "F_visit", "A_cpu")],
    "P3 structure (B vs)": [("B_hrm", x) for x in ("C_mix", "D_tied", "J_sequential")],
    "P4 native sparse routing": [("H_sparse_st", "B_hrm"), ("I_anneal", "B_hrm"), ("H_sparse_st", "A_single"), ("I_anneal", "A_single")],
    "P5 loop forcing (all - last)": [(x, f"{x}_last") for x in ("A_single", "B_hrm", "C_mix", "H_sparse_st")],
    "P6 distillation": [("K_distill", "A_single"), ("K_distill", "B_hrm"), ("K_win", "A_single")],
}


def load(paths) -> list[dict]:
    rows = []
    for p in paths:
        op = gzip.open if str(p).endswith(".gz") else open
        with op(p, "rt", encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def index(runs):
    by = defaultdict(dict)
    for r in runs:
        by[(r["arm"], r["family"])][r["seed"]] = r
    return by


def paired(by, a, b, fam):
    A, B = by.get((a, fam), {}), by.get((b, fam), {})
    common = sorted(set(A) & set(B))
    if len(common) < 2:
        return None
    d = [A[s]["evals"]["dev"]["exact"] - B[s]["evals"]["dev"]["exact"] for s in common]
    return seed_ci(d)


def retention(by, fams, student="K_distill", teacher="B_hrm", ref="A_single", reps=4000, seed=0):
    """(student - ref) / (teacher - ref): per family and pooled, bootstrap over seeds."""
    out = {}
    rng = random.Random(seed)
    for fam in fams + ["pooled"]:
        fl = fams if fam == "pooled" else [fam]
        seeds = sorted(set.intersection(*[set(by.get((x, f), {})) for x in (student, teacher, ref) for f in fl])) if fl else []
        if len(seeds) < 2:
            continue

        def ratio(ss):
            num = sum(by[(student, f)][s]["evals"]["dev"]["exact"] - by[(ref, f)][s]["evals"]["dev"]["exact"] for f in fl for s in ss)
            den = sum(by[(teacher, f)][s]["evals"]["dev"]["exact"] - by[(ref, f)][s]["evals"]["dev"]["exact"] for f in fl for s in ss)
            return num / den if abs(den) > 1e-9 else float("nan")
        boots = sorted(r for r in (ratio([rng.choice(seeds) for _ in seeds]) for _ in range(reps)) if r == r)
        gap = stats.mean(by[(teacher, f)][s]["evals"]["dev"]["exact"] - by[(ref, f)][s]["evals"]["dev"]["exact"] for f in fl for s in seeds)
        out[fam] = {"retained_fraction": ratio(seeds), "ci95": [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]] if boots else None,
                    "teacher_gap": gap, "n_seeds": len(seeds)}
    return out


def table(by, arms, fams):
    t = {}
    for arm in arms:
        for fam in fams:
            cell = by.get((arm, fam))
            if not cell:
                continue
            rs = list(cell.values())
            ev = [r["evals"] for r in rs]
            row = {"n_seeds": len(rs), "exact": stats.mean(e["dev"]["exact"] for e in ev),
                   "ood": {k: stats.mean(e[k]["exact"] for e in ev) for k in ev[0] if k.startswith("ood") and "@" not in k},
                   "ood_2x_loops": {k: stats.mean(e[k]["exact"] for e in ev) for k in ev[0] if "@deep" in k},
                   "exact_by_loop": [stats.mean(x) for x in zip(*(e["dev"]["exact_by_loop"] for e in ev))],
                   "params": rs[0]["params"], "visits": rs[0]["logical_visits"],
                   "flops": stats.mean(r["cost"]["flops_per_example"] for r in rs),
                   "flops_hard": stats.mean(r["cost"]["dispatch_hard"]["flops_per_example"] for r in rs) if "dispatch_hard" in rs[0]["cost"] else None,
                   "latency_ms": stats.median(r["cost"]["latency_ms_per_batch"] for r in rs),
                   "latency_hard_ms": stats.median(r["cost"]["dispatch_hard"]["latency_ms_per_batch"] for r in rs) if "dispatch_hard" in rs[0]["cost"] else None,
                   "train_cpu_s": stats.mean(r["train_cpu_s"] for r in rs),
                   "peak_rss_mb": max(r["peak_rss_bytes"] for r in rs) / 2 ** 20,
                   "rss_increment_mb": max(r["rss_increment_bytes"] for r in rs) / 2 ** 20,
                   "interventions": {k.split("|")[1]: stats.mean(e[k]["exact"] for e in ev) for k in ev[0] if "|" in k}}
            for k in ("routing_entropy", "routing_loop_mi"):
                v = [e["dev"][k] for e in ev if e["dev"].get(k) is not None]
                if v:
                    row[k] = stats.mean(v)
            hu = [e["dev"]["hard_usage"] for e in ev if e["dev"].get("hard_usage")]
            if hu:
                row["hard_usage"] = [stats.mean(x) for x in zip(*hu)]
            mech = [r["mech"] for r in rs if r.get("mech")]
            if mech:
                agg = {}
                for k in mech[0]:
                    vals = [m[k] for m in mech if k in m]
                    if all(isinstance(v, (int, float)) for v in vals):
                        agg[k] = stats.mean(vals)
                    elif all(isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v) for v in vals):
                        agg[k] = [stats.mean(x) for x in zip(*vals)]
                row["mech"] = agg
            t[f"{arm}|{fam}"] = row
    return t


def analyse(runs, multitask=None):
    by = index(runs)
    arms = sorted({r["arm"] for r in runs})
    fams = sorted({r["family"] for r in runs})
    full = [f for f in fams]
    t = table(by, arms, fams)
    contrasts = {}
    for title, pairs in CONTRASTS.items():
        rows = {}
        for a, b in pairs:
            if a not in arms or b not in arms:
                continue
            rows[f"{a} - {b}"] = {f: paired(by, a, b, f) for f in fams}
        contrasts[title] = rows
    inky = [a for a in arms if a.startswith("inky_")]
    contrasts["P7 Inky finalists"] = {f"{a} - {b}": {f: paired(by, a, b, f) for f in fams} for a in inky for b in ("A_single", "B_hrm")}
    main_arms = [a for a in arms if not a.endswith("_last")]
    points = {}
    for a in main_arms:
        rows = [t[f"{a}|{f}"] for f in full if f"{a}|{f}" in t]
        if len(rows) == len(full):
            fl = [r["flops_hard"] or r["flops"] for r in rows]
            points[a] = {"quality": stats.mean(r["exact"] for r in rows), "cpu_s": stats.mean(r["flops_hard"] or r["flops"] for r in rows),
                         "bytes": stats.mean(r["latency_hard_ms"] or r["latency_ms"] for r in rows),
                         "train_cpu_s": stats.mean(r["train_cpu_s"] for r in rows), "params": rows[0]["params"],
                         "visits": rows[0]["visits"], "inference_flops": stats.mean(fl)}
    front = pareto(points) if points else []   # (quality up, inference FLOPs down, latency down)
    res = {"arms": arms, "families": fams, "table": t, "contrasts": contrasts, "points": points, "pareto_front": front,
           "retention_K_of_B": retention(by, [f for f in fams if f != "local_map3"]),
           "retention_K_win": retention(by, [f for f in fams if f != "local_map3"], student="K_win",
                                        teacher=next((a for a in arms if a.startswith("T_")), "B_hrm"))}
    if multitask:
        mt = defaultdict(list)
        for r in multitask:
            mt[r["arm"]].append(r)
        res["multitask_routing"] = {a: {"family_module_mi_nats": stats.mean(r["family_module_mi_nats"] for r in rs if "family_module_mi_nats" in r),
                                        "max_mi_nats": rs[0].get("max_mi_nats"), "n_seeds": len(rs),
                                        "per_family": {f: {"exact": stats.mean(r["families"][f]["exact"] for r in rs),
                                                           "hard_usage": [stats.mean(x) for x in zip(*(r["families"][f].get("hard_usage", [0]) for r in rs))]}
                                                       for f in rs[0]["families"]}} for a, rs in mt.items()}
    return res


def _ci(v):
    if v is None:
        return "—"
    return f"{v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" if v.get("ci95") else f"{v['mean']:+.3f}"


def markdown(res) -> str:
    fams = res["families"]
    t = res["table"]
    out = ["### Exact on hidden roots (mean over seeds)", "", "| arm | " + " | ".join(fams) + " | params | visits | inf. MFLOP/ex | train CPU s |",
           "|---|" + "---:|" * (len(fams) + 4)]
    for a in res["arms"]:
        cells = [t.get(f"{a}|{f}") for f in fams]
        any_row = next(c for c in cells if c)
        out.append(f"| {a} | " + " | ".join("—" if c is None else f"{c['exact']:.3f}" for c in cells)
                   + f" | {any_row['params']} | {any_row['visits']} | {(any_row['flops_hard'] or any_row['flops']) / 1e6:.2f} | "
                   f"{stats.mean(c['train_cpu_s'] for c in cells if c):.1f} |")
    for title, rows in res["contrasts"].items():
        if not rows:
            continue
        out += ["", f"### {title}", "", "| contrast | " + " | ".join(fams) + " |", "|---|" + "---|" * len(fams)]
        for name, per in rows.items():
            out.append(f"| {name} | " + " | ".join(_ci(per.get(f)) for f in fams) + " |")
    for key in ("retention_K_of_B", "retention_K_win"):
        r = res.get(key) or {}
        if r:
            out += ["", f"### {key} (student at 4 visits vs teacher, bootstrap over seeds)", "", "| family | retained | CI95 | teacher gap | seeds |", "|---|---:|---|---:|---:|"]
            for f, v in r.items():
                ci = "—" if not v["ci95"] else f"[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]"
                out.append(f"| {f} | {v['retained_fraction']:+.2f} | {ci} | {v['teacher_gap']:+.3f} | {v['n_seeds']} |")
    out += ["", "Pareto front over (mean exact ↑, inference FLOPs ↓, latency ↓): " + ", ".join(f"`{a}`" for a in res["pareto_front"])]
    if res.get("multitask_routing"):
        out += ["", "### Task-conditional routing (one model on the four principal families)", "",
                "| arm | MI(family; module) nats | max | per-family exact |", "|---|---:|---:|---|"]
        for a, v in res["multitask_routing"].items():
            out.append(f"| {a} | {v['family_module_mi_nats']:.3f} | {v['max_mi_nats']:.3f} | "
                       + ", ".join(f"{f}={x['exact']:.3f}" for f, x in v["per_family"].items()) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--multitask", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    mt = load([a.multitask]) if a.multitask and a.multitask.exists() else None
    res = analyse(load(a.runs), mt)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    md = markdown(res)
    (a.out / "analysis.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
