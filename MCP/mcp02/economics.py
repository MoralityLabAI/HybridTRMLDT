"""Failure-inclusive retrieval economics with explicit, labelled price assumptions.

Cost per episode is a calculation, not a measured bill:

    downstream  = handoff_bytes / bytes_per_token * frontier input price
    local       = router seconds * local device price per second
    tools       = MCP tool calls * tool-call price
    verification= LDT metadata checks * check price

Prices live in ``configs/economics.json``. Handoff tokens use an explicitly
named byte-per-token estimator; exact handoff bytes are reported separately.
The primary metric divides the summed cost of every assigned episode
(failures included) by the number of accepted episodes.
"""
from __future__ import annotations

import math
import random
from typing import Any


def wilson(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def episode_cost(episode: dict, prices: dict, device: str) -> dict:
    tokens_est = episode["handoff_utf8_bytes"] / prices["bytes_per_token_estimate"]
    downstream = tokens_est * prices["frontier_input_usd_per_mtok"] / 1e6
    rate = prices["local_gpu_usd_per_hour"] if device == "cuda" else prices["local_cpu_usd_per_hour"]
    local_seconds = episode["compute"]["router_seconds"]  # LLM episodes record inference seconds here
    local = local_seconds * rate / 3600.0
    tools = episode["tool_calls"] * prices["tool_call_usd"]
    verify = episode["environment_stats"].get("ldt_metadata_checks", 0) * prices["verification_check_usd"]
    # Counterfactual: a frontier model that read every tool result itself.
    direct = episode["compute"]["result_bytes"] / prices["bytes_per_token_estimate"] \
        * prices["frontier_input_usd_per_mtok"] / 1e6
    return {"downstream_usd": downstream, "local_usd": local, "tools_usd": tools, "verification_usd": verify,
            "total_usd": downstream + local + tools + verify, "handoff_tokens_estimate": tokens_est,
            "counterfactual_frontier_reads_results_usd": direct}


def paired_difference(a: list[float], b: list[float], seed: int = 0, rounds: int = 2000) -> dict:
    """Mean of a-b over paired episodes with a percentile bootstrap 95% interval."""
    if not a or len(a) != len(b):
        return {"n": len(a), "mean": None, "ci95": None}
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(rounds))
    return {"n": len(diffs), "mean": sum(diffs) / len(diffs),
            "ci95": [means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]],
            "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs)}


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(q * len(values)) - 1)]


def summarize_group(episodes: list[dict], prices: dict, device: str) -> dict:
    n = len(episodes)
    accepted = sum(e["accepted"] for e in episodes)
    costs = [episode_cost(e, prices, device) for e in episodes]
    total = {k: sum(c[k] for c in costs) for k in costs[0]} if costs else {}
    decisions = sum(e["compute"]["decisions"] for e in episodes)
    failures: dict[str, int] = {}
    for e in episodes:
        if not e["accepted"]:
            key = e["failure"] or ("handoff_budget" if e["handoff_budget_exceeded"] else
                                   "+".join(e["grade"]["reasons"]) or "unknown")
            failures[key] = failures.get(key, 0) + 1
    def per_success(value):
        return value / accepted if accepted else None
    flops = sum(e["compute"]["flops"] for e in episodes)
    return {
        "episodes": n, "accepted": accepted, "success_rate": accepted / n if n else None,
        "success_ci95_wilson": wilson(accepted, n),
        "evidence_recall_mean": sum(e["grade"]["evidence_recall"] for e in episodes) / n if n else None,
        "answer_correct": sum(e["grade"]["answer_correct"] for e in episodes),
        "abstentions_submitted": sum(bool(e["submission"].get("abstain")) for e in episodes),
        "abstained_on_failure": sum(e["abstained_on_failure"] for e in episodes),
        "failures": failures,
        "retries": {"revalidation_reads": sum(e["composer"]["revalidations"] for e in episodes),
                    "rejected_reads": sum(e["composer"]["rejections"] for e in episodes),
                    "tool_errors": sum(e["environment_stats"]["tool_errors"] for e in episodes)},
        "tool_calls_mean": sum(e["tool_calls"] for e in episodes) / n if n else None,
        "tool_calls_per_success": per_success(sum(e["tool_calls"] for e in episodes)),
        "refine_calls": sum(e["environment_stats"]["refine_calls"] for e in episodes),
        "ldt_excluded": sum(e["environment_stats"]["ldt_excluded"] for e in episodes),
        "ldt_rejected": sum(e["environment_stats"]["ldt_rejected"] for e in episodes),
        "ldt_metadata_checks": sum(e["environment_stats"]["ldt_metadata_checks"] for e in episodes),
        "certified_exclusions_containing_required": sum(
            e["private_checks"]["certified_exclusions_containing_required"] for e in episodes),
        "handoff_bytes_mean": sum(e["handoff_utf8_bytes"] for e in episodes) / n if n else None,
        "handoff_bytes_max": max((e["handoff_utf8_bytes"] for e in episodes), default=None),
        "handoff_budget_exceeded": sum(e["handoff_budget_exceeded"] for e in episodes),
        "tool_result_bytes_mean": sum(e["compute"]["result_bytes"] for e in episodes) / n if n else None,
        "router_decisions": decisions,
        "router_flops_total": flops, "router_flops_per_success": per_success(flops),
        "router_tokens_total": sum(e["compute"]["router_tokens"] for e in episodes),
        "candidates_scored": sum(e["compute"]["candidates_scored"] for e in episodes),
        "router_seconds_total": sum(e["compute"]["router_seconds"] for e in episodes),
        "router_ms_per_decision": (1000 * sum(e["compute"]["router_seconds"] for e in episodes) / decisions
                                   if decisions else None),
        "episode_seconds_p50": _pct([e["elapsed_seconds"] for e in episodes], 0.5),
        "episode_seconds_p95": _pct([e["elapsed_seconds"] for e in episodes], 0.95),
        "cost_total_usd": total, "cost_per_success_usd": (total.get("total_usd", 0) / accepted) if accepted else None,
        "cost_breakdown_per_success_usd": {k: v / accepted for k, v in total.items()} if accepted else None,
    }
