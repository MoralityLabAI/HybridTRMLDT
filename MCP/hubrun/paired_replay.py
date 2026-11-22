"""Score an exact three-arm native MCP-05 replay, rejecting unmatched task allocations.

This report reads native verifiers traces and relay receipts. It never claims
an agent solved a task from a packet-only or no-inference preflight. The source
traces are evaluated by the existing hubrun summarizer.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .summarize import load_episodes, summarize_run, load_calls


ARMS = ("raw", "mcp04", "mcp05")
TASKSETS = {"docedit": (8, "docedit"), "er-t1": (5, "event-recon")}


def build_report(runs_dir: Path, calls_path: Path, mode: str) -> dict[str, Any]:
    if mode not in ("smoke", "full"):
        raise ValueError("mode must be smoke or full")
    if not calls_path.is_file():
        raise ValueError(f"missing relay call log: {calls_path}")
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for line in calls_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("call log row must be object")
        by_tag[str(row.get("tag", ""))].append(row)

    totals = load_calls(calls_path)
    results: dict[str, Any] = {"mode": mode, "tasksets": {}, "valid_comparison": True,
                               "errors": [], "limitations": [
                                   "Live model outcomes come from native verifiers traces only.",
                                   "Relay logs do not expose tool actions; repeated file reads and stale edit sites require separate tool-trace extraction.",
                                   "A paired task ID controls task selection, not sampling stochasticity.",
                               ]}
    for taskset, (full_count, _source) in TASKSETS.items():
        expected = 1 if mode == "smoke" else full_count
        grouped: dict[str, Any] = {}
        episodes_by_arm: dict[str, dict[str, dict]] = {}
        for arm in ARMS:
            tag = f"mcp05-{mode}-{taskset}-{arm}"
            path = runs_dir / tag
            episode_list = load_episodes(path, tag) if path.is_dir() else []
            cases = {str(e["task"]): e for e in episode_list}
            issues: list[str] = []
            if len(episode_list) != expected:
                issues.append(f"{len(episode_list)} episodes, expected {expected}")
            if len(cases) != len(episode_list):
                issues.append("duplicate task IDs")
            if not (path / "DONE").is_file():
                issues.append("DONE marker absent")
            call_rows = by_tag.get(tag, [])
            if not call_rows:
                issues.append("no relay calls recorded")
            parity_failures = sum(1 for r in call_rows if r.get("preflight_matches_provider") is False)
            parity_missing = sum(
                1 for r in call_rows if r.get("status") == 200
                and r.get("preflight_matches_provider") is not True
                and arm in ("mcp04", "mcp05"))
            if parity_failures or parity_missing:
                issues.append(f"provider token parity false={parity_failures}, missing={parity_missing}")
            call_errors = sum(1 for r in call_rows if int(r.get("status") or 0) >= 400)
            compacted = [r.get("compaction") or {} for r in call_rows if (r.get("compaction") or {}).get("compacted")]
            stats = summarize_run(tag, episode_list, totals.get(tag, {}))
            stats.update({
                "complete": not issues,
                "issues": issues,
                "provider_parity_failures": parity_failures,
                "provider_parity_missing": parity_missing,
                "http_error_calls": call_errors,
                "attention_ranked_read_atoms": sum(int(c.get("attention_ranked_reads") or 0) for c in compacted),
                "pinned_negative_atoms": sum(int(c.get("pinned_negative_atoms") or 0) for c in compacted),
            })
            grouped[arm] = stats
            episodes_by_arm[arm] = cases
            for issue in issues:
                results["errors"].append(f"{tag}: {issue}")
        common = set.intersection(*(set(episodes_by_arm[a]) for a in ARMS))
        union = set.union(*(set(episodes_by_arm[a]) for a in ARMS))
        matched = len(common) == expected and common == union
        if not matched:
            results["errors"].append(
                f"{taskset}: task IDs differ or incomplete (matched {len(common)}/{expected}, union {len(union)})")
        pairs = {}
        for arm in ("mcp04", "mcp05"):
            changes = {
                t: round(episodes_by_arm[arm][t]["score"] - episodes_by_arm["raw"][t]["score"], 5)
                for t in sorted(common)
            }
            pairs[f"{arm}_vs_raw"] = {
                "n_matched": len(changes),
                "mean_delta": round(sum(changes.values()) / len(changes), 5) if changes else None,
                "better": sum(v > 0 for v in changes.values()),
                "worse": sum(v < 0 for v in changes.values()),
                "ties": sum(v == 0 for v in changes.values()),
                "per_task_delta": changes,
            }
        changes = {
            t: round(episodes_by_arm["mcp05"][t]["score"] - episodes_by_arm["mcp04"][t]["score"], 5)
            for t in sorted(common)
        }
        pairs["mcp05_vs_mcp04"] = {
            "n_matched": len(changes),
            "mean_delta": round(sum(changes.values()) / len(changes), 5) if changes else None,
            "better": sum(v > 0 for v in changes.values()),
            "worse": sum(v < 0 for v in changes.values()),
            "ties": sum(v == 0 for v in changes.values()),
            "per_task_delta": changes,
        }
        results["tasksets"][taskset] = {
            "expected_cases": expected, "paired_task_ids": sorted(common), "matched": matched,
            "arms": grouped, "pairs": pairs,
        }
    results["valid_comparison"] = not results["errors"]
    results["promotion_gate"] = {
        "valid": results["valid_comparison"],
        "task_success": (results["valid_comparison"]
                         and all(results["tasksets"][t]["arms"]["mcp05"]["solved"] >=
                                 results["tasksets"][t]["arms"]["mcp04"]["solved"]
                                 for t in TASKSETS)),
        "no_mcp05_http_errors": (results["valid_comparison"]
                                 and all(results["tasksets"][t]["arms"]["mcp05"]["http_error_calls"] == 0
                                         for t in TASKSETS)),
        "requires_native_review": True,
    }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--calls", type=Path, required=True)
    parser.add_argument("--mode", required=True, choices=("smoke", "full"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = build_report(args.runs, args.calls, args.mode)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"invalid replay: {exc}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "valid_comparison": receipt["valid_comparison"],
        "errors": receipt["errors"],
        "promotion_gate": receipt["promotion_gate"],
        "tasksets": {name: {"matched": item["matched"],
                             "solved": {arm: item["arms"][arm]["solved"] for arm in ARMS},
                             "mcp05_minus_mcp04": item["pairs"]["mcp05_vs_mcp04"]["mean_delta"]}
                     for name, item in receipt["tasksets"].items()},
        "report": str(args.out),
    }, indent=2))
    return 0 if receipt["valid_comparison"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
