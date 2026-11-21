"""Failure-inclusive accounting; byte counts never stand in for model tokens."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def byte_size(value: Any) -> int:
    return len(canonical_json(value).encode("utf-8"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def aggregate_usage(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep partial observations explicit, including failed model requests."""
    result: dict[str, Any] = {"model_calls": len(calls)}
    for name in ("prompt_tokens", "completion_tokens", "cached_tokens"):
        values = [call.get("usage", {}).get(name) for call in calls]
        known = [value for value in values if type(value) is int and value >= 0]
        result[name] = sum(known) if calls and len(known) == len(calls) else None
        result[f"{name}_known_sum"] = sum(known)
        result[f"{name}_missing_calls"] = len(calls) - len(known)
    result["usage_complete"] = bool(calls) and all(
        result[f"{name}_missing_calls"] == 0 for name in ("prompt_tokens", "completion_tokens")
    )
    return result


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Every episode contributes to spend and latency, even when it fails."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        key = (episode["controller"], episode["architecture"], episode["catalog_size"], episode["depth"])
        groups[key].append(episode)
    rows: list[dict[str, Any]] = []
    for (controller, architecture, catalog_size, depth), group in sorted(groups.items()):
        accepted = sum(bool(e["accepted"]) for e in group)
        correct = sum(bool(e.get("grade", {}).get("success")) for e in group)
        calls = [call for e in group for call in e.get("model_calls", [])]
        usage = aggregate_usage(calls)
        latencies = sorted(e["elapsed_seconds"] for e in group)
        row: dict[str, Any] = {
            "controller": controller, "architecture": architecture,
            "catalog_size": catalog_size, "depth": depth,
            "episodes": len(group), "accepted": accepted, "correct": correct,
            "success_rate": correct / len(group), "accepted_rate": accepted / len(group),
            "failures": len(group) - accepted,
            "tool_calls": sum(e["tool_calls"] for e in group),
            "catalog_utf8_bytes": sum(e["catalog_utf8_bytes"] for e in group),
            "model_input_utf8_bytes": sum(e["model_input_utf8_bytes"] for e in group),
            "reference_context_utf8_bytes": sum(e.get("reference_context_utf8_bytes", 0) for e in group),
            "tool_result_utf8_bytes": sum(e["tool_result_utf8_bytes"] for e in group),
            "model_observation_utf8_bytes": sum(e.get("model_observation_utf8_bytes", 0) for e in group),
            "handoff_utf8_bytes": sum(e["handoff_utf8_bytes"] for e in group),
            "handoff_budget_exceeded": sum(e["handoff_budget_exceeded"] for e in group),
            "elapsed_seconds": sum(latencies),
            "p95_episode_seconds": latencies[math.ceil(0.95 * len(latencies)) - 1],
            "local_inference_seconds": sum(e["local_inference_seconds"] for e in group),
            "transport_replay_verified": all(e.get("transport_replay_verified", False) for e in group),
            "usage": usage,
            "environment_stats": {name: sum(e.get("environment_stats", {}).get(name, 0) for e in group)
                                  for name in sorted({key for e in group for key in e.get("environment_stats", {})})},
        }
        for name in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            total = usage[name]
            row[f"{name}_per_accepted_task"] = total / accepted if accepted and total is not None else None
        row["model_input_utf8_bytes_per_accepted_task"] = row["model_input_utf8_bytes"] / accepted if accepted else None
        rows.append(row)
    return {"episodes": len(episodes), "arms": rows, "token_source": "local server usage, when reported",
            "byte_source": "exact UTF-8 JSON serialization; separate from token counts",
            "p95_method": "nearest rank", "failure_accounting": "all attempted episodes included"}
