"""Snapshot-only RLM invocation gate and all-in utility accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from research_gym.benchmarks.rlm_evidence_acquisition import (
    ACQUISITION_CONTRACT_HASH,
    EXACT_MECHANICS,
    architecture_query_sequence,
    build_initial_snapshot,
)
from research_gym.benchmarks.rlm_hybrid_neighborhood import LongContextControlTask, canonical_sha256


GATE_ID = "three_way_module_disagreement_v0_1"
ARCHITECTURES = (
    "fixed_no_query",
    "gated_always_exact",
    "gated_deterministic_voi",
    "ungated_rlm",
    "gated_rlm",
    "gated_forced_failure",
)
API_ARCHITECTURES = ("ungated_rlm", "gated_rlm")


@dataclass(frozen=True)
class ProviderCostSpec:
    token_utility_per_1k: float
    latency_utility_per_second: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProviderCostSpec":
        return cls(
            token_utility_per_1k=float(value["token_utility_per_1k"]),
            latency_utility_per_second=float(value["latency_utility_per_second"]),
        )


def gate_features(
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = build_initial_snapshot(task, trained_row)
    return {
        "snapshot_id": snapshot["snapshot_id"],
        "distinct_top_count": int(snapshot["agreement"]["distinct_top_count"]),
        "all_equal": bool(snapshot["agreement"]["all_equal"]),
        "proxy_confidence": float(snapshot["modules"]["proxy_trm"]["confidence"]),
        "trained_confidence": float(
            snapshot["modules"]["trained_control_trm"]["confidence"]
        ),
    }


def should_invoke_rlm(
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
) -> bool:
    """Freeze the prior-derived gate: invoke only on three-way disagreement."""

    return gate_features(task, trained_row)["distinct_top_count"] == 3


def provider_utility_cost(
    usage: Mapping[str, Any],
    elapsed_seconds: float,
    cost: ProviderCostSpec,
) -> float:
    return (
        cost.token_utility_per_1k * float(usage.get("total_tokens", 0)) / 1000.0
        + cost.latency_utility_per_second * float(elapsed_seconds)
    )


def all_in_utility(
    evidence_net_utility: float,
    usage: Mapping[str, Any],
    elapsed_seconds: float,
    cost: ProviderCostSpec,
) -> float:
    return float(evidence_net_utility) - provider_utility_cost(usage, elapsed_seconds, cost)


def local_sequence(
    architecture_id: str,
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
) -> tuple[str, ...]:
    invoke = should_invoke_rlm(task, trained_row)
    if architecture_id in {"fixed_no_query", "gated_forced_failure"} or not invoke:
        return ()
    if architecture_id == "gated_always_exact":
        return (EXACT_MECHANICS,)
    if architecture_id == "gated_deterministic_voi":
        return architecture_query_sequence("mesh_deterministic_voi", task, trained_row)
    raise ValueError(f"architecture has no local sequence: {architecture_id}")


def architecture_hashes(cost_regimes: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    specifications = {
        "fixed_no_query": ("local", "never_invoke"),
        "gated_always_exact": ("local", GATE_ID, "exact"),
        "gated_deterministic_voi": ("local", GATE_ID, "snapshot_voi"),
        "ungated_rlm": ("official_rlm", "always_invoke"),
        "gated_rlm": ("official_rlm", GATE_ID),
        "gated_forced_failure": ("control", GATE_ID, "fixed_fallback"),
    }
    return {
        architecture: canonical_sha256(
            (
                ACQUISITION_CONTRACT_HASH,
                specifications[architecture],
                dict(cost_regimes),
            )
        )
        for architecture in ARCHITECTURES
    }
