"""Offline packet-marginal supervision for the MCP-03 read-aware experiment.

This module must only be imported from dataset collection or tests. At inference,
the router receives the public observation and a trained checkpoint, never the
CodeEnv private judge or the oracle labels.

Read targets measure the actual marginal value of admitting a symbol to a
bounded packet. Expand targets are an explicitly discounted reachability proxy,
not exact multi-step counterfactual VOI. They are zero with one call remaining.
"""
from __future__ import annotations

import math

from .agent import build_packet


def packet_utility_targets(env, obs: dict, labels: list[int], reads: list[tuple[float, str]],
                           seen: dict[str, float], *, packet_budget: int,
                           remaining_calls: int) -> list[float]:
    """Private training targets aligned to obs['candidates'].

    A read gets credit only when its symbol body is actually admitted by the
    existing packet composer and raises private gold recall at this budget.
    Gold counts rescale fractional recall to reward a newly covered target
    equally in cases with different numbers of gold symbols. Token cost is
    measured from the same named tokenizer as the actual packet.

    Expand is a delayed-action reachability label with fixed discount 0.30,
    conditional on enough calls remaining and unresolved gold symbols. Future
    work should replace this proxy with rollout-backed k-step marginal VOI.
    """
    if packet_budget <= 0 or remaining_calls < 0:
        raise ValueError("packet_budget must be positive and remaining_calls nonnegative")
    cands = obs["candidates"]
    if len(cands) != len(labels):
        raise ValueError("candidate/label count mismatch")
    base_packet = build_packet(env, reads, seen, packet_budget)
    base_grade = env.grade(base_packet)
    already_read = {uri for _score, uri in reads}
    targets: list[float] = []

    for item, label in zip(cands, labels):
        if item["type"] != "doc":
            value = (0.30 if label and remaining_calls >= 2
                     and not base_grade["all_symbols"] else 0.0)
            targets.append(value)
            continue

        uri = item.get("meta", {}).get("uri")
        if uri in already_read or uri not in env.symbols:
            targets.append(0.0)
            continue

        candidate_packet = build_packet(env, reads + [(1.0, uri)], seen, packet_budget)
        admitted = next((s for s in candidate_packet["snippets"] if s["uri"] == uri), None)
        if admitted is None:
            targets.append(0.0)
            continue
        grade = env.grade(candidate_packet)
        symbol_gain = max(0.0, grade["symbol_recall"] - base_grade["symbol_recall"])
        file_gain = max(0.0, grade["file_recall"] - base_grade["file_recall"])
        gain = 0.8 * symbol_gain * grade["gold_symbols"] + 0.2 * file_gain * grade["gold_files"]
        token_cost = max(1, admitted["tokens"])
        targets.append(round(min(1.0, gain * 7.0 / math.log2(16.0 + token_cost)), 6))
    return targets
