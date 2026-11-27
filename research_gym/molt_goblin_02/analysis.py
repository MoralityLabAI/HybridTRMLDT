"""Paired comparisons, Pareto frontier and mechanistic summaries from frozen final runs."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as stats
from collections import defaultdict
from pathlib import Path

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def seed_ci(diffs: list[float]) -> dict:
    n = len(diffs)
    m = sum(diffs) / n
    if n < 2:
        return {"n_seeds": n, "mean": m, "ci95": None}
    sd = stats.stdev(diffs)
    h = T95.get(n - 1, 1.96) * sd / math.sqrt(n)
    return {"n_seeds": n, "mean": m, "sd": sd, "ci95": [m - h, m + h]}


def root_bootstrap(a_rows: list[list[int]], b_rows: list[list[int]], reps: int = 2000, seed: int = 0) -> list[float]:
    """Paired bootstrap over roots within seed (conditional on the trained checkpoints)."""
    rng = random.Random(seed)
    diffs = [[x - y for x, y in zip(a, b)] for a, b in zip(a_rows, b_rows)]
    n = len(diffs[0])
    out = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        out.append(sum(sum(d[i] for i in idx) / n for d in diffs) / len(diffs))
    out.sort()
    return [out[int(0.025 * reps)], out[int(0.975 * reps)]]


def pareto(points: dict[str, dict], keys=("quality", "cpu_s", "bytes")) -> list[str]:
    names = list(points)
    front = []
    for a in names:
        pa = points[a]
        dominated = False
        for b in names:
            if a == b:
                continue
            pb = points[b]
            ge = pb["quality"] >= pa["quality"] and pb["cpu_s"] <= pa["cpu_s"] and pb["bytes"] <= pa["bytes"]
            gt = pb["quality"] > pa["quality"] or pb["cpu_s"] < pa["cpu_s"] or pb["bytes"] < pa["bytes"]
            if ge and gt:
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front


def analyse(runs: list[dict], reference: str = "prefix") -> dict:
    by = defaultdict(dict)            # (arm, family) -> seed -> run
    for r in runs:
        by[(r["arm"], r["family"])][r["seed"]] = r
    arms = sorted({r["arm"] for r in runs})
    families = sorted({r["family"] for r in runs})
    table, paired = {}, {}
    for arm in arms:
        for fam in families:
            cell = by.get((arm, fam), {})
            if not cell:
                continue
            ev = [c["evals"] for c in cell.values()]
            row = {"seeds": sorted(cell), "exact": stats.mean(e["dev"]["exact"] for e in ev),
                   "ood": {k: stats.mean(e[k]["exact"] for e in ev) for k in ev[0] if k.startswith("ood")},
                   "visits": stats.mean(e["dev"]["mean_block_visits"] for e in ev),
                   "success_per_visit": stats.mean(e["dev"]["exact"] / max(e["dev"]["mean_block_visits"], 1e-9) for e in ev),
                   "train_cpu_s": stats.mean(c["train_cpu_s"] for c in cell.values()),
                   "success_per_cpu_s": stats.mean(c["summary"]["success_per_cpu_s"] for c in cell.values()),
                   "bytes": stats.mean(e["dev"]["retained_bytes_per_example"] for e in ev),
                   "peak_rss_mb": max(c["peak_rss_bytes"] for c in cell.values()) / 2 ** 20,
                   "params": next(iter(cell.values()))["params"],
                   "exact_by_loop": [stats.mean(x) for x in zip(*(e["dev"]["exact_by_loop"] for e in ev))],
                   "loop_fixes": stats.mean(e["dev"].get("loop_fixes") or 0 for e in ev),
                   "loop_breaks": stats.mean(e["dev"].get("loop_breaks") or 0 for e in ev),
                   "redundant_after_correct": stats.mean(e["dev"].get("redundant_after_correct") or 0 for e in ev)}
            iv = defaultdict(list)
            for e in ev:
                for k, v in e.items():
                    if "|" in k:
                        iv[k.split("|")[1]].append(v["exact"])
                    if "@deep" in k:
                        iv["deep_" + k.split("@")[0]].append(v["exact"])
            row["interventions"] = {k: stats.mean(v) for k, v in iv.items()}
            for k in ("routing_entropy", "routing_loop_mi", "halt_brier", "halt_ece"):
                vals = [e["dev"].get(k) for e in ev if e["dev"].get(k) is not None]
                if vals:
                    row[k] = stats.mean(vals)
            halt = [e.get("dev|halt") for e in ev if e.get("dev|halt")]
            if halt:
                row["halt_exact"] = stats.mean(h["exact"] for h in halt)
                row["halt_visits"] = stats.mean(h["mean_block_visits"] for h in halt)
                row["halt_brier"] = stats.mean(h.get("halt_brier", float("nan")) for h in halt)
            sub = [c.get("subspace") for c in cell.values() if c.get("subspace")]
            if sub:
                row["subspace"] = sub[0]
            table[f"{arm}|{fam}"] = row
            ref = by.get((reference, fam), {})
            common = sorted(set(cell) & set(ref))
            if arm != reference and common:
                d = [cell[s]["evals"]["dev"]["exact"] - ref[s]["evals"]["dev"]["exact"] for s in common]
                res = seed_ci(d)
                res["root_bootstrap_ci95"] = root_bootstrap([cell[s]["evals"]["dev"]["per_example"] for s in common],
                                                            [ref[s]["evals"]["dev"]["per_example"] for s in common])
                paired[f"{arm}|{fam}"] = res
    points = {}
    for arm in arms:
        rows = [table[f"{arm}|{f}"] for f in families if f"{arm}|{f}" in table]
        if len(rows) == len(families):
            points[arm] = {"quality": stats.mean(r["exact"] for r in rows), "cpu_s": stats.mean(r["train_cpu_s"] for r in rows),
                           "bytes": stats.mean(r["bytes"] for r in rows), "visits": stats.mean(r["visits"] for r in rows),
                           "ood": stats.mean(stats.mean(r["ood"].values()) for r in rows if r["ood"])}
    return {"reference": reference, "arms": arms, "families": families, "table": table, "paired_vs_reference": paired,
            "points": points, "pareto_front": pareto(points) if points else []}


def to_markdown(res: dict, freeze: dict | None = None) -> str:
    fams = res["families"]
    lines = ["| arm | " + " | ".join(fams) + " | mean | cpu s | bytes/ex |", "|---|" + "---:|" * (len(fams) + 3)]
    for arm in res["arms"]:
        cells = [res["table"].get(f"{arm}|{f}", {}).get("exact") for f in fams]
        p = res["points"].get(arm, {})
        lines.append(f"| {arm} | " + " | ".join("—" if c is None else f"{c:.3f}" for c in cells) +
                     f" | {p.get('quality', float('nan')):.3f} | {p.get('cpu_s', float('nan')):.1f} | {p.get('bytes', float('nan')):.0f} |")
    lines += ["", f"Paired difference vs `{res['reference']}` (seed-level mean, t 95% CI; root bootstrap CI conditional on checkpoints):", "",
              "| arm | family | mean diff | seed CI95 | root CI95 |", "|---|---|---:|---|---|"]
    for k, v in sorted(res["paired_vs_reference"].items()):
        arm, fam = k.split("|")
        ci = "—" if v["ci95"] is None else f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
        rb = v["root_bootstrap_ci95"]
        lines.append(f"| {arm} | {fam} | {v['mean']:+.3f} | {ci} | [{rb[0]:+.3f}, {rb[1]:+.3f}] |")
    lines += ["", "Pareto front over (mean exact ↑, train CPU s ↓, retained bytes ↓): " + ", ".join(f"`{a}`" for a in res["pareto_front"])]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = analyse([r for p in a.runs for r in load(p)])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (a.out / "analysis.md").write_text(to_markdown(res) + "\n", encoding="utf-8")
    print(to_markdown(res))


if __name__ == "__main__":
    main()
