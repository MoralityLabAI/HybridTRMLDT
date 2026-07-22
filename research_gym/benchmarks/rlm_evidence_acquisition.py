"""Typed evidence acquisition for an action-opaque RLM control mesh."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from math import exp
import re
from typing import Any, Mapping, Sequence

from research_gym.benchmarks.rlm_hybrid_neighborhood import (
    LongContextControlTask,
    canonical_sha256,
    ldt_action,
    rank_actions,
)


INDEPENDENT_PROBE = "independent_probe"
RECEIPT_ATTESTATION = "receipt_attestation"
EXACT_MECHANICS = "exact_mechanics"
COUNTERFACTUAL_ROLLOUT = "counterfactual_rollout"
QUERY_IDS = (
    INDEPENDENT_PROBE,
    RECEIPT_ATTESTATION,
    EXACT_MECHANICS,
    COUNTERFACTUAL_ROLLOUT,
)
QUERY_COSTS = {
    INDEPENDENT_PROBE: 0.01,
    RECEIPT_ATTESTATION: 0.02,
    EXACT_MECHANICS: 0.04,
    COUNTERFACTUAL_ROLLOUT: 0.04,
}
QUERY_DESCRIPTIONS = {
    INDEPENDENT_PROBE: "Independent calibrated support over opaque candidates; soft evidence only.",
    RECEIPT_ATTESTATION: "Verify whether proxy and trained proposal receipts bind the current case.",
    EXACT_MECHANICS: "Return the exact-admissible opaque candidate set.",
    COUNTERFACTUAL_ROLLOUT: "Return calibrated noisy value bands over opaque candidates.",
}
CHANNEL_WEIGHTS = {
    "ldt": 1.0,
    "proxy": 1.0,
    "trained": 1.0,
    INDEPENDENT_PROBE: 1.0,
    COUNTERFACTUAL_ROLLOUT: 2.0,
}
MAX_QUERIES = 2
ACQUISITION_CONTRACT = {
    "version": "rlm_evidence_acquisition_mesh_v0",
    "max_queries": MAX_QUERIES,
    "query_costs": QUERY_COSTS,
    "visible": (
        "opaque_candidate_references",
        "module_rank_confidence_and_provenance",
        "query_catalog",
        "acquired_evidence_receipts",
    ),
    "hidden": (
        "task_text",
        "task_family",
        "action_tokens",
        "exact_utility",
        "optimal_action",
    ),
    "authority": {
        "rlm": "select_query_or_stop",
        "host": "execute_query_and_deterministically_arbitrate",
        "executor": "verify_and_commit_or_fixed_consensus_fallback",
    },
}
ACQUISITION_CONTRACT_HASH = canonical_sha256(ACQUISITION_CONTRACT)
ARCHITECTURES = (
    "mesh_fixed_no_query",
    "mesh_always_exact",
    "mesh_exact_then_rollout",
    "mesh_random_two_query",
    "mesh_deterministic_voi",
    "rlm_adaptive_two_query",
    "rlm_forced_failure",
)
API_ARCHITECTURES = ("rlm_adaptive_two_query",)


def architecture_hashes() -> dict[str, str]:
    specifications = {
        "mesh_fixed_no_query": ("local", ()),
        "mesh_always_exact": ("local", (EXACT_MECHANICS,)),
        "mesh_exact_then_rollout": ("local", (EXACT_MECHANICS, COUNTERFACTUAL_ROLLOUT)),
        "mesh_random_two_query": ("local", "seeded_query_count_matched_two_query"),
        "mesh_deterministic_voi": ("local", "snapshot_only_voi_v0"),
        "rlm_adaptive_two_query": ("official_rlm", "two_round_text_decision"),
        "rlm_forced_failure": ("control", "fixed_no_query_fallback"),
    }
    return {
        name: canonical_sha256(
            (ACQUISITION_CONTRACT_HASH, specifications[name], CHANNEL_WEIGHTS)
        )
        for name in ARCHITECTURES
    }


@dataclass(frozen=True)
class AcquisitionOutcome:
    executed_action: str
    baseline_action: str
    query_sequence: tuple[str, ...]
    receipts: tuple[Mapping[str, Any], ...]
    raw_utility: float
    query_cost: float
    net_utility: float
    action_changed: bool
    fallback: bool
    decision_reason: str

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "executed_action": self.executed_action,
            "baseline_action": self.baseline_action,
            "query_sequence": list(self.query_sequence),
            "receipts": [dict(value) for value in self.receipts],
            "raw_utility": self.raw_utility,
            "query_cost": self.query_cost,
            "net_utility": self.net_utility,
            "action_changed": self.action_changed,
            "fallback": self.fallback,
            "decision_reason": self.decision_reason,
        }


def candidate_references(task: LongContextControlTask) -> dict[str, str]:
    references = {
        action: "candidate_"
        + canonical_sha256(("rlm-acquisition-candidate-v0", task.prompt_sha256, action))[:12]
        for action in task.candidates
    }
    if len(set(references.values())) != len(references):
        raise RuntimeError("opaque acquisition candidate-reference collision")
    return references


def _normalize_scores(task: LongContextControlTask, values: Mapping[str, float]) -> dict[str, float]:
    selected = [float(values[action]) for action in task.candidates]
    low, high = min(selected), max(selected)
    width = max(1e-9, high - low)
    return {action: (float(values[action]) - low) / width for action in task.candidates}


def _hash_noise(*values: object) -> float:
    raw = int(canonical_sha256(values)[:12], 16) / float(16**12 - 1)
    return 2.0 * raw - 1.0


def _rank_confidence(task: LongContextControlTask, scores: Mapping[str, float]) -> tuple[str, float]:
    ranked = rank_actions(scores, task.candidates)
    ordered = sorted((float(scores[action]) for action in task.candidates), reverse=True)
    return ranked[0], 1.0 / (1.0 + exp(-(ordered[0] - ordered[1])))


def fixed_consensus_action(
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
) -> tuple[str, bool]:
    """Resolve raw module consensus, then enforce exact safety at the executor."""

    proxy_top = rank_actions(task.proxy_scores, task.candidates)[0]
    trained_top = rank_actions(
        {str(key): float(value) for key, value in trained_row["scores"].items()},
        task.candidates,
    )[0]
    ldt_top = rank_actions(task.ldt_scores, task.candidates)[0]
    votes = Counter((proxy_top, trained_top, ldt_top))
    selected = min(
        votes,
        key=lambda action: (-votes[action], action != ldt_top, task.candidates.index(action)),
    )
    if selected in task.exact_allowed:
        return selected, False
    return ldt_action(task), True


def build_initial_snapshot(
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
) -> dict[str, Any]:
    references = candidate_references(task)
    proxy_top, proxy_confidence = _rank_confidence(task, task.proxy_scores)
    trained_scores = {str(key): float(value) for key, value in trained_row["scores"].items()}
    trained_top, trained_confidence = _rank_confidence(task, trained_scores)
    ldt_top, _ = _rank_confidence(task, task.ldt_scores)
    material = {
        "contract_hash": ACQUISITION_CONTRACT_HASH,
        "case_ref": canonical_sha256(("rlm-acquisition-case-v0", task.prompt_sha256))[:20],
        "candidate_refs": sorted(references.values()),
        "modules": {
            "proxy_trm": {
                "top_ref": references[proxy_top],
                "confidence": round(proxy_confidence, 8),
                "provenance": "model_sound_proxy",
                "receipt_ref": canonical_sha256((task.task_id, "proxy", task.proxy_scores))[:20],
            },
            "trained_control_trm": {
                "top_ref": references[trained_top],
                "confidence": round(trained_confidence, 8),
                "provenance": str(trained_row["claimed_provenance"]),
                "receipt_ref": str(trained_row["proposal_receipt_ref"]),
            },
            "exact_ldt": {
                "top_ref": references[ldt_top],
                "provenance": "environment_policy_prior",
            },
        },
        "agreement": {
            "distinct_top_count": len({proxy_top, trained_top, ldt_top}),
            "all_equal": len({proxy_top, trained_top, ldt_top}) == 1,
        },
        "query_catalog": [
            {
                "query_id": query_id,
                "cost": QUERY_COSTS[query_id],
                "description": QUERY_DESCRIPTIONS[query_id],
            }
            for query_id in QUERY_IDS
        ],
    }
    return {"snapshot_id": canonical_sha256(material), **material}


def acquisition_context(
    snapshot: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    queried: Sequence[str],
) -> dict[str, Any]:
    remaining = [query_id for query_id in QUERY_IDS if query_id not in queried]
    return {
        "protocol": "rlm_evidence_acquisition_mesh_v0",
        "snapshot": dict(snapshot),
        "acquired_evidence": [dict(value) for value in receipts],
        "query_count": len(queried),
        "queries_remaining": MAX_QUERIES - len(queried),
        "available_query_ids": remaining,
        "allowed_decisions": [*remaining, "STOP"],
    }


def context_privacy_violations(
    task: LongContextControlTask,
    context: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return direct task/action disclosures in a provider-bound context."""

    serialized = json.dumps(context, sort_keys=True, separators=(",", ":"))
    violations = []
    if task.transcript and task.transcript in serialized:
        violations.append("task_text")
    if task.family and task.family in serialized:
        violations.append("task_family")
    for action in task.candidates:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(action)}(?![A-Za-z0-9_])", serialized):
            violations.append(f"action_token:{action}")
    forbidden_keys = {
        "action",
        "executed_action",
        "exact_utility",
        "family",
        "optimal_action",
        "task_id",
        "task_text",
        "transcript",
        "utilities",
    }

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key) in forbidden_keys:
                    violations.append(f"forbidden_key:{key}")
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(context)
    return tuple(sorted(set(violations)))


def random_two_query_sequence(task: LongContextControlTask) -> tuple[str, str]:
    ordered = sorted(
        QUERY_IDS,
        key=lambda query_id: canonical_sha256(
            ("rlm-acquisition-random-v0", task.task_id, query_id)
        ),
    )
    return ordered[0], ordered[1]


def deterministic_voi_sequence(
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
) -> tuple[str, ...]:
    snapshot = build_initial_snapshot(task, trained_row)
    modules = snapshot["modules"]
    if snapshot["agreement"]["all_equal"]:
        return (EXACT_MECHANICS,)
    model_confidence = max(
        float(modules["proxy_trm"]["confidence"]),
        float(modules["trained_control_trm"]["confidence"]),
    )
    if model_confidence >= 0.80:
        return RECEIPT_ATTESTATION, EXACT_MECHANICS
    return INDEPENDENT_PROBE, EXACT_MECHANICS


def architecture_query_sequence(
    architecture_id: str,
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
) -> tuple[str, ...]:
    if architecture_id in {"mesh_fixed_no_query", "rlm_forced_failure"}:
        return ()
    if architecture_id == "mesh_always_exact":
        return (EXACT_MECHANICS,)
    if architecture_id == "mesh_exact_then_rollout":
        return EXACT_MECHANICS, COUNTERFACTUAL_ROLLOUT
    if architecture_id == "mesh_random_two_query":
        return random_two_query_sequence(task)
    if architecture_id == "mesh_deterministic_voi":
        return deterministic_voi_sequence(task, trained_row)
    raise ValueError(f"architecture does not have a static acquisition sequence: {architecture_id}")


def make_evidence_receipt(
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
    truth: Mapping[str, Any],
    query_id: str,
) -> dict[str, Any]:
    if query_id not in QUERY_IDS:
        raise ValueError(f"unknown evidence query: {query_id}")
    references = candidate_references(task)
    if query_id == INDEPENDENT_PROBE:
        normalized = _normalize_scores(task, task.public_action_scores)
        payload = {
            "support": {
                references[action]: round(
                    min(1.0, max(0.0, normalized[action] + 0.35 * _hash_noise(task.task_id, action, query_id))),
                    6,
                )
                for action in task.candidates
            },
            "calibration_error_bound": 0.35,
            "soundness": "model_sound_independent",
        }
    elif query_id == RECEIPT_ATTESTATION:
        payload = {
            "module_bindings": {
                "proxy_trm": "current" if truth["proxy_current"] else "stale",
                "trained_control_trm": "current" if truth["trained_current"] else "stale",
            },
            "soundness": "identity_attested",
        }
    elif query_id == EXACT_MECHANICS:
        payload = {
            "exact_admissible_refs": sorted(references[action] for action in task.exact_allowed),
            "soundness": "environment_sound",
        }
    else:
        payload = {
            "value_bands": {
                references[action]: {
                    "lower": round(max(0.0, task.utilities[action] + 0.08 * _hash_noise(task.task_id, action, query_id) - 0.08), 6),
                    "center": round(min(1.0, max(0.0, task.utilities[action] + 0.08 * _hash_noise(task.task_id, action, query_id))), 6),
                    "upper": round(min(1.0, task.utilities[action] + 0.08 * _hash_noise(task.task_id, action, query_id) + 0.08), 6),
                }
                for action in task.candidates
            },
            "calibration_error_bound": 0.08,
            "soundness": "counterfactual_model_sound",
        }
    material = {
        "query_id": query_id,
        "case_ref": canonical_sha256(("rlm-acquisition-case-v0", task.prompt_sha256))[:20],
        "context_sha256": task.prompt_sha256,
        "cost": QUERY_COSTS[query_id],
        "payload": payload,
    }
    return {**material, "receipt_sha256": canonical_sha256(material)}


def verify_evidence_receipt(
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
    truth: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> bool:
    query_id = str(receipt.get("query_id", ""))
    if query_id not in QUERY_IDS:
        return False
    return dict(receipt) == make_evidence_receipt(task, trained_row, truth, query_id)


def valid_query_sequences() -> tuple[tuple[str, ...], ...]:
    output: list[tuple[str, ...]] = [()]
    output.extend((query_id,) for query_id in QUERY_IDS)
    output.extend((first, second) for first in QUERY_IDS for second in QUERY_IDS if first != second)
    return tuple(output)


def execute_acquisition(
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
    truth: Mapping[str, Any],
    query_sequence: Sequence[str],
) -> AcquisitionOutcome:
    sequence = tuple(str(value) for value in query_sequence)
    if len(sequence) > MAX_QUERIES or len(set(sequence)) != len(sequence):
        raise ValueError("acquisition sequence exceeds budget or repeats a query")
    if any(value not in QUERY_IDS for value in sequence):
        raise ValueError("acquisition sequence contains an unknown query")
    baseline, baseline_fallback = fixed_consensus_action(task, trained_row)
    receipts = tuple(make_evidence_receipt(task, trained_row, truth, value) for value in sequence)
    if not all(verify_evidence_receipt(task, trained_row, truth, receipt) for receipt in receipts):
        raise AssertionError("internally generated evidence receipt did not replay")
    receipt_by_query = {str(value["query_id"]): value for value in receipts}
    selected = baseline
    reason = "fixed_consensus_no_exact_acquisition"
    fallback = baseline_fallback
    if EXACT_MECHANICS in receipt_by_query:
        fallback = False
        normalized_channels: list[tuple[float, dict[str, float]]] = [
            (CHANNEL_WEIGHTS["ldt"], _normalize_scores(task, task.ldt_scores))
        ]
        bindings = {
            "proxy_trm": "unverified",
            "trained_control_trm": "unverified",
        }
        if RECEIPT_ATTESTATION in receipt_by_query:
            bindings = dict(receipt_by_query[RECEIPT_ATTESTATION]["payload"]["module_bindings"])
        if bindings["proxy_trm"] != "stale":
            normalized_channels.append(
                (CHANNEL_WEIGHTS["proxy"], _normalize_scores(task, task.proxy_scores))
            )
        if bindings["trained_control_trm"] != "stale":
            normalized_channels.append(
                (
                    CHANNEL_WEIGHTS["trained"],
                    _normalize_scores(
                        task,
                        {str(key): float(value) for key, value in trained_row["scores"].items()},
                    ),
                )
            )
        references = candidate_references(task)
        if INDEPENDENT_PROBE in receipt_by_query:
            support = receipt_by_query[INDEPENDENT_PROBE]["payload"]["support"]
            normalized_channels.append(
                (
                    CHANNEL_WEIGHTS[INDEPENDENT_PROBE],
                    {action: float(support[references[action]]) for action in task.candidates},
                )
            )
        if COUNTERFACTUAL_ROLLOUT in receipt_by_query:
            bands = receipt_by_query[COUNTERFACTUAL_ROLLOUT]["payload"]["value_bands"]
            normalized_channels.append(
                (
                    CHANNEL_WEIGHTS[COUNTERFACTUAL_ROLLOUT],
                    {action: float(bands[references[action]]["center"]) for action in task.candidates},
                )
            )
        exact_refs = set(receipt_by_query[EXACT_MECHANICS]["payload"]["exact_admissible_refs"])
        allowed = [action for action in task.candidates if references[action] in exact_refs]
        score = {
            action: sum(weight * channel[action] for weight, channel in normalized_channels)
            / sum(weight for weight, _ in normalized_channels)
            for action in task.candidates
        }
        ldt_order = rank_actions(task.ldt_scores, task.candidates)
        selected = min(allowed, key=lambda action: (-score[action], ldt_order.index(action), task.candidates.index(action)))
        reason = "deterministic_acquired_evidence_arbiter"
    if selected not in task.exact_allowed:
        selected = baseline
        fallback = True
        reason = "exact_executor_fixed_consensus_fallback"
    raw_utility = float(task.utilities[selected])
    query_cost = sum(QUERY_COSTS[value] for value in sequence)
    return AcquisitionOutcome(
        executed_action=selected,
        baseline_action=baseline,
        query_sequence=sequence,
        receipts=receipts,
        raw_utility=raw_utility,
        query_cost=query_cost,
        net_utility=raw_utility - query_cost,
        action_changed=selected != baseline,
        fallback=fallback,
        decision_reason=reason,
    )


def execute_failure_fallback(
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
    truth: Mapping[str, Any],
    executed_queries: Sequence[str],
) -> AcquisitionOutcome:
    """Preserve fixed-consensus utility after a controller failure, charging spent queries."""

    sequence = tuple(str(value) for value in executed_queries)
    if len(sequence) > MAX_QUERIES or len(set(sequence)) != len(sequence):
        raise ValueError("failure fallback sequence exceeds budget or repeats a query")
    if any(value not in QUERY_IDS for value in sequence):
        raise ValueError("failure fallback sequence contains an unknown query")
    baseline = execute_acquisition(task, trained_row, truth, ())
    receipts = tuple(make_evidence_receipt(task, trained_row, truth, value) for value in sequence)
    query_cost = sum(QUERY_COSTS[value] for value in sequence)
    return AcquisitionOutcome(
        executed_action=baseline.executed_action,
        baseline_action=baseline.baseline_action,
        query_sequence=sequence,
        receipts=receipts,
        raw_utility=baseline.raw_utility,
        query_cost=query_cost,
        net_utility=baseline.raw_utility - query_cost,
        action_changed=False,
        fallback=True,
        decision_reason="rlm_failure_fixed_consensus_fallback",
    )


def sequence_oracle(
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
    truth: Mapping[str, Any],
) -> AcquisitionOutcome:
    outcomes = [
        execute_acquisition(task, trained_row, truth, sequence)
        for sequence in valid_query_sequences()
    ]
    order = {query_id: index for index, query_id in enumerate(QUERY_IDS)}
    return min(
        outcomes,
        key=lambda value: (
            -value.net_utility,
            value.query_cost,
            len(value.query_sequence),
            tuple(order[item] for item in value.query_sequence),
        ),
    )


def parse_acquisition_decision(response: str, available_queries: Sequence[str]) -> str | None:
    stripped = response.strip()
    allowed = set(available_queries)
    if stripped.upper() == "STOP":
        return "STOP"
    if stripped in allowed:
        return stripped
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    decision = payload.get("decision")
    if decision == "stop" and set(payload) == {"decision"}:
        return "STOP"
    if decision != "query" or set(payload) != {"decision", "query_id"}:
        return None
    query_id = payload.get("query_id")
    return str(query_id) if query_id in allowed else None
