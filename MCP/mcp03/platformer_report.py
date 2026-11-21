"""Matched CPU retrieval report for independently generated platformer roots."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty percentile")
    xs = sorted(values)
    p = q * (len(xs) - 1)
    low = int(p)
    high = min(len(xs) - 1, low + 1)
    return xs[low] + (xs[high] - xs[low]) * (p - low)


def paired_ci(diffs: list[float], seed: int = 412, reps: int = 2000) -> list[float]:
    if not diffs:
        raise ValueError("empty paired cohort")
    rng = random.Random(seed)
    n = len(diffs)
    sample_means = [mean([diffs[rng.randrange(n)] for _ in range(n)])
                    for _ in range(reps)]
    return [round(percentile(sample_means, q), 6) for q in (0.025, 0.975)]


def summarize(episodes: list[dict], *, baseline: str, experimental: str,
              budget: int = 1024, max_calls: int = 24) -> dict:
    by_policy: dict[str, dict[str, dict]] = {}
    for e in episodes:
        if e["max_tool_calls"] != max_calls or str(budget) not in e["packets"]:
            raise ValueError("unmatched call or packet budget")
        if e["arm"] != "search":
            raise ValueError("unmatched environment arm")
        name, iid = e["policy_name"], e["instance_id"]
        if iid in by_policy.setdefault(name, {}):
            raise ValueError("duplicate policy/instance episode")
        by_policy[name][iid] = e
    if baseline not in by_policy or experimental not in by_policy:
        raise ValueError("missing baseline or experimental arm")
    roots = set(by_policy[baseline])
    if roots != set(by_policy[experimental]):
        raise ValueError("baseline and experiment differ in assigned instances")
    if any(set(records) != roots for records in by_policy.values()):
        raise ValueError("every arm must use precisely the same set of test roots")
    ordered = sorted(roots)
    def value(e: dict, field: str) -> float:
        return e["packets"][str(budget)]["grade"][field]
    groups = {}
    for name, cases in sorted(by_policy.items()):
        group = [cases[k] for k in ordered]
        groups[name] = {
            "assigned": len(group),
            "file_recall_mean": round(mean([value(g, "file_recall") for g in group]), 6),
            "symbol_recall_mean": round(mean([value(g, "symbol_recall") for g in group]), 6),
            "all_files_hit": sum(value(g, "all_files") for g in group),
            "all_symbols_hit": sum(value(g, "all_symbols") for g in group),
            "packet_tokens_mean": round(mean([g["packets"][str(budget)]["tokens"] for g in group]), 3),
            "admitted_symbol_bodies_mean": round(mean([g["packets"][str(budget)]["snippets"] for g in group]), 3),
            "read_symbol_actions_mean": round(mean([g["compute"]["read_symbol_actions"] for g in group]), 3),
            "read_candidate_steps_mean": round(mean([g["compute"]["read_candidate_steps"] for g in group]), 3),
            "router_flops_mean": round(mean([g["compute"]["flops"] for g in group]), 1),
            "tool_result_tokens_mean": round(mean([g["compute"]["result_tokens"] for g in group]), 2),
            "calls_mean": round(mean([g["tool_calls"] for g in group]), 3),
            "time_seconds_mean": round(mean([g["elapsed_seconds"] for g in group]), 3),
        }
    old, new = by_policy[baseline], by_policy[experimental]
    sym_diffs = [value(new[k], "symbol_recall") - value(old[k], "symbol_recall") for k in ordered]
    file_diffs = [value(new[k], "file_recall") - value(old[k], "file_recall") for k in ordered]
    ci = paired_ci(sym_diffs)
    same = all(d == 0 for d in sym_diffs)
    result = {
        "experiment": "MCP-03-RU01-platformer",
        "scope": "fresh generated repository roots; retrieval only; no SWE patch grading",
        "assigned_roots": len(ordered),
        "matched_call_budget": max_calls,
        "matched_packet_budget": budget,
        "groups": groups,
        "paired": {
            "baseline": baseline, "experimental": experimental,
            "symbol_recall_delta": round(mean(sym_diffs), 6),
            "symbol_delta_bootstrap_ci95": ci,
            "file_recall_delta": round(mean(file_diffs), 6),
            "improved": sum(d > 0 for d in sym_diffs),
            "unchanged": sum(d == 0 for d in sym_diffs),
            "regressed": sum(d < 0 for d in sym_diffs),
        },
        "gates": {
            "symbol_delta_ci_lower_above_zero": ci[0] > 0 and not same,
            "file_recall_delta_at_least_minus_0p02": mean(file_diffs) >= -0.02,
        },
        "instance_ids": ordered
    }
    result["gates"]["promote_retrieval"] = all(result["gates"].values())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", default="looped_p4_d64_binary")
    parser.add_argument("--experimental", default="looped_p4_d64_readaware")
    args = parser.parse_args(argv)
    episodes = [json.loads(s) for s in (args.run / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    report = summarize(episodes, baseline=args.baseline, experimental=args.experimental)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("experiment", "assigned_roots", "paired", "gates")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
