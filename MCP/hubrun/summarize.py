"""Summarize hub runs: scores per run, provider tokens per run, and raw-versus-goblin pairs.

Reads ``<runs>/<tag>/`` written by ``vf-eval`` (``results.jsonl``) or the
verifiers v1 ``eval`` CLI (``traces.jsonl``) and the relay's call log, and
writes ``summary.json`` and ``summary.md``.

Score per episode: the environment's reward (the weighted sum of v1 reward
scores), except where ``PRIMARY_METRIC`` names a metric because the reward
was disabled (mcp-sqlite runs without its paid API judge, so its score is the
taskset's ``exact_match``). Runs of private environments (``PRIVATE_PREFIXES``)
report aggregates only, without per-task rows.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

PRIMARY_METRIC = {"sqlite": "exact_match"}
PRIVATE_PREFIXES = ("sw-", "jbm-")
BRACKETS = re.compile(r"^[<\[\(\s`'\"]+|[>\]\)\s`'\".]+$")


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def wilson(successes: float, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _latest(paths: list[Path]) -> Path | None:
    paths = [p for p in paths if p.stat().st_size > 0]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def load_episodes(run_dir: Path, tag: str) -> list[dict[str, Any]]:
    metric = next((m for k, m in PRIMARY_METRIC.items() if tag.startswith(k)), None)
    traces = _latest(list(run_dir.rglob("traces.jsonl")))
    episodes = []
    if traces is not None:
        for line in traces.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            for tr in row.get("traces", []):
                rewards = tr.get("rewards") or {}
                metrics = tr.get("metrics") or {}
                reward = sum(float(v.get("score", 0.0)) * float(v.get("weight", 0.0)) for v in rewards.values())
                episodes.append({"task": row["task"]["data"]["name"],
                                 "score": float(metrics.get(metric, 0.0)) if metric else reward,
                                 "reward": reward, "ok": bool(tr.get("ok")), "stop": tr.get("stop_condition"),
                                 "errors": len(tr.get("errors") or []) + len(row.get("errors") or []),
                                 "turns": metrics.get("turns"),
                                 "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))}})
        return episodes
    results = _latest(list(run_dir.rglob("results.jsonl")))
    if results is None:
        return episodes
    for line in results.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        metrics = {k: v for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                   and k not in ("example_id", "reward")}
        metrics.update({k: v for k, v in (row.get("metrics") or {}).items() if isinstance(v, (int, float))})
        reward = float(row.get("reward") or 0.0)
        ep = {"task": str(row.get("example_id")), "score": float(metrics.get(metric, 0.0)) if metric else reward,
              "reward": reward, "ok": not row.get("error"), "stop": row.get("stop_condition"),
              "errors": int(bool(row.get("error"))), "turns": row.get("num_turns"), "metrics": metrics}
        if tag.startswith("sw-constitution"):
            comp = row.get("completion") or []
            text = comp[-1].get("content", "") if comp and isinstance(comp[-1], dict) else str(comp)
            first = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
            ep["metrics"]["bracket_stripped_choice_match"] = float(
                BRACKETS.sub("", first) == str(row.get("answer", "")).strip())
        episodes.append(ep)
    return episodes


def load_calls(log: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    if not log.exists():
        return out
    for line in log.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        a = out[r["tag"]]
        a["calls"] += 1
        p = r.get("prompt_tokens") or 0
        a["prompt_tokens"] += p
        a["prompt_peak"] = max(a["prompt_peak"], p)
        a["completion_tokens"] += r.get("completion_tokens") or 0
        a["cached_tokens"] += r.get("cached_tokens") or 0
        a["latency_s"] += r.get("latency_s") or 0.0
        comp = r.get("compaction") or {}
        a["compacted_calls"] += int(bool(comp.get("compacted")))
        a["capped_messages"] += comp.get("capped_messages") or 0
        a["context_length_stops"] += int(r.get("finish_reason") == "length")
        a["http_errors"] += int((r.get("status") or 200) >= 400)
        a["context_overflows"] += int((r.get("status") or 200) >= 400 and "exceed_context" in (r.get("error") or ""))
    return out


def summarize_run(tag: str, episodes: list[dict], calls: dict[str, float]) -> dict[str, Any]:
    n = len(episodes)
    scores = [e["score"] for e in episodes]
    solved = sum(1 for s in scores if s >= 0.999)
    metric_keys = sorted({k for e in episodes for k in e["metrics"]})
    metric_means = {k: _mean([float(e["metrics"][k]) for e in episodes if k in e["metrics"]]) for k in metric_keys}
    out = {"tag": tag, "episodes": n, "mean_score": _mean(scores), "solved": solved, "solved_wilson95": wilson(solved, n),
           "episodes_with_errors": sum(1 for e in episodes if e["errors"]),
           "mean_turns": _mean([float(e["turns"]) for e in episodes if e.get("turns") is not None]),
           "metric_means": metric_means,
           "provider": {k: round(v, 3) for k, v in calls.items()}}
    if calls.get("calls") and n:
        out["provider"]["prompt_tokens_per_episode"] = round(calls["prompt_tokens"] / n, 1)
        out["provider"]["completion_tokens_per_episode"] = round(calls["completion_tokens"] / n, 1)
        if solved:
            out["provider"]["total_tokens_per_solved"] = round(
                (calls["prompt_tokens"] + calls["completion_tokens"]) / solved, 1)
    if not tag.startswith(PRIVATE_PREFIXES):
        out["per_task"] = {e["task"]: round(e["score"], 4) for e in episodes}
    return out


def pairs(runs: dict[str, dict], episodes: dict[str, list[dict]]) -> dict[str, dict]:
    """Raw against each packet arm on shared tasks: ``<stem>`` for goblin, ``<stem>-late`` for late compaction."""
    out = {}
    for tag in runs:
        if not tag.endswith("-raw"):
            continue
        stem = tag[: -len("-raw")]
        for suffix, key in (("goblin", stem), ("late", f"{stem}-late")):
            gob = f"{stem}-{suffix}"
            if gob in runs:
                out[key] = _pair(runs, episodes, tag, gob)
    return out


def _pair(runs: dict[str, dict], episodes: dict[str, list[dict]], tag: str, gob: str) -> dict:
    """Paired comparison of run ``tag`` (raw) and run ``gob`` (a packet arm); ``goblin_*`` keys name the packet arm."""
    raw_by = {e["task"]: e["score"] for e in episodes[tag]}
    gob_by = {e["task"]: e["score"] for e in episodes[gob]}
    common = sorted(set(raw_by) & set(gob_by))
    diffs = [gob_by[t] - raw_by[t] for t in common]
    r, g = runs[tag]["provider"], runs[gob]["provider"]
    return {"packet_arm": gob, "paired_tasks": len(common),
            "raw_mean": _mean([raw_by[t] for t in common]), "goblin_mean": _mean([gob_by[t] for t in common]),
            "mean_diff_goblin_minus_raw": _mean(diffs),
            "goblin_better": sum(1 for d in diffs if d > 1e-9), "raw_better": sum(1 for d in diffs if d < -1e-9),
            "prompt_tokens_ratio_goblin_over_raw": (round(g.get("prompt_tokens", 0) / r["prompt_tokens"], 3)
                                                    if r.get("prompt_tokens") else None),
            "prompt_peak_raw": r.get("prompt_peak"), "prompt_peak_goblin": g.get("prompt_peak"),
            "context_length_stops_raw": r.get("context_length_stops"),
            "context_length_stops_goblin": g.get("context_length_stops"),
            "context_overflows_raw": r.get("context_overflows"),
            "context_overflows_goblin": g.get("context_overflows"),
            "episode_errors_raw": runs[tag]["episodes_with_errors"],
            "episode_errors_goblin": runs[gob]["episodes_with_errors"]}


def markdown(summary: dict) -> str:
    lines = ["| run | episodes | mean score | solved (95% CI) | episodes with errors | prompt tok/episode | "
             "completion tok/episode | peak prompt | compacted calls | context overflows |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for tag, s in summary["runs"].items():
        p = s["provider"]
        ci = s["solved_wilson95"]
        lines.append(f"| {tag} | {s['episodes']} | {s['mean_score']} | {s['solved']} "
                     f"({ci[0]:.2f}-{ci[1]:.2f}) | {s['episodes_with_errors']} | {p.get('prompt_tokens_per_episode')} | "
                     f"{p.get('completion_tokens_per_episode')} | {int(p.get('prompt_peak', 0))} | "
                     f"{int(p.get('compacted_calls', 0))} | {int(p.get('context_overflows', 0))} |"
                     if ci else f"| {tag} | 0 | | | | | | | | |")
    if summary["pairs"]:
        lines += ["", "| pair | tasks | raw mean | goblin mean | goblin better / raw better | "
                      "prompt tokens goblin/raw | peak raw / goblin | context overflows raw / goblin | "
                      "episodes with errors raw / goblin |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for stem, p in summary["pairs"].items():
            lines.append(f"| {stem} | {p['paired_tasks']} | {p['raw_mean']} | {p['goblin_mean']} | "
                         f"{p['goblin_better']} / {p['raw_better']} | {p['prompt_tokens_ratio_goblin_over_raw']} | "
                         f"{int(p['prompt_peak_raw'] or 0)} / {int(p['prompt_peak_goblin'] or 0)} | "
                         f"{int(p['context_overflows_raw'] or 0)} / {int(p['context_overflows_goblin'] or 0)} | "
                         f"{p['episode_errors_raw']} / {p['episode_errors_goblin']} |")
    for tag, reason in summary.get("skipped", {}).items():
        lines.append(f"\nSkipped `{tag}`: {reason}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--calls", required=True, nargs="+", help="one or more relay call logs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-prefix", action="append", default=None,
                    help="skip run directories with this prefix (repeatable; default: smoke)")
    args = ap.parse_args()
    args.skip_prefix = args.skip_prefix or ["smoke"]
    calls: dict = {}
    for log in args.calls:
        calls.update(load_calls(Path(log)))
    runs, episodes, skipped = {}, {}, {}
    for run_dir in sorted(Path(args.runs).iterdir()):
        tag = run_dir.name
        if not run_dir.is_dir() or tag.startswith(tuple(args.skip_prefix)):
            continue
        done = run_dir / "DONE"
        if done.exists() and done.read_text(encoding="utf-8").startswith("skipped"):
            skipped[tag] = done.read_text(encoding="utf-8").strip()
            continue
        episodes[tag] = load_episodes(run_dir, tag)
        runs[tag] = summarize_run(tag, episodes[tag], calls.get(tag, {}))
        runs[tag]["finished"] = (run_dir / "DONE").exists()
    summary = {"runs": runs, "pairs": pairs(runs, episodes), "skipped": skipped}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")
    (out / "summary.md").write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))


if __name__ == "__main__":
    main()
