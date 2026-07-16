"""Paired resilience benchmark for AIRIS/DAS forecast authorization."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from statistics import mean
from typing import Mapping, Sequence

from research_gym.adapters.airis_das import (
    airis_observation,
    embedded_forecast,
    resolve_airis_decision,
    trusted_rule_sha256,
)


TOPOLOGY_ONLY = "topology_only"
INTEGRITY_SEALED = "integrity_sealed"
CLEAN = "clean"
SCENARIOS = (
    CLEAN,
    "protocol_shift",
    "unseen_context",
    "sequence_substitution",
    "confidence_inflation",
    "support_inflation",
    "route_override",
    "authority_override",
    "unknown_rule_id",
    "retrieval_outage",
)


def _set_outcome_fields(match: dict[str, object], **updates: str) -> None:
    fields = dict(match.get("outcome_fields") or {})
    fields.update(updates)
    outcome = ";".join(
        f"{key}={fields[key]}"
        for key in ("selected_sequence", "control_route", "authority")
    )
    predicts = dict(match.get("predicts") or {})
    predicts["outcome"] = outcome
    match["predicts"] = predicts
    match["outcome_fields"] = fields


def _base_forecast(
    row: Mapping[str, object], protocol_sha256: str, rules: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    return embedded_forecast(airis_observation(row, protocol_sha256), rules, limit=3)


def _challenge_cases(
    row: Mapping[str, object], protocol_sha256: str, rules: Sequence[Mapping[str, object]]
) -> list[tuple[str, dict[str, object], dict[str, object]]]:
    clean_row = deepcopy(dict(row))
    clean_forecast = _base_forecast(clean_row, protocol_sha256, rules)
    cases = [(CLEAN, clean_row, clean_forecast)]

    shifted_protocol = f"shifted-{protocol_sha256}"
    cases.append(
        (
            "protocol_shift",
            deepcopy(clean_row),
            _base_forecast(clean_row, shifted_protocol, rules),
        )
    )

    unseen_row = deepcopy(clean_row)
    unseen_row["context"] = f"unseen:{row.get('context')}"
    cases.append(
        (
            "unseen_context",
            unseen_row,
            _base_forecast(unseen_row, protocol_sha256, rules),
        )
    )

    substituted = deepcopy(clean_forecast)
    top = substituted["matches"][0]
    candidates = clean_row["candidate_outcomes"]
    fallback = clean_row["selections"]["control_math"]
    alternate = next(name for name in sorted(candidates) if name != fallback)
    _set_outcome_fields(top, selected_sequence=str(alternate))
    cases.append(("sequence_substitution", deepcopy(clean_row), substituted))

    confidence = deepcopy(clean_forecast)
    top = confidence["matches"][0]
    current_confidence = float(top["confidence"])
    top["confidence"] = 0.999 if current_confidence != 0.999 else 0.998
    cases.append(("confidence_inflation", deepcopy(clean_row), confidence))

    support = deepcopy(clean_forecast)
    support["matches"][0]["support"] = int(support["matches"][0]["support"]) + 100
    cases.append(("support_inflation", deepcopy(clean_row), support))

    route = deepcopy(clean_forecast)
    top = route["matches"][0]
    current_route = str(top["outcome_fields"]["control_route"])
    replacement_route = "global_signed" if current_route == "local_section" else "local_section"
    _set_outcome_fields(top, control_route=replacement_route)
    cases.append(("route_override", deepcopy(clean_row), route))

    authority = deepcopy(clean_forecast)
    _set_outcome_fields(authority["matches"][0], authority="airis")
    cases.append(("authority_override", deepcopy(clean_row), authority))

    unknown = deepcopy(clean_forecast)
    unknown["matches"][0]["rule_id"] = f"unknown-{unknown['matches'][0]['rule_id']}"
    cases.append(("unknown_rule_id", deepcopy(clean_row), unknown))

    outage = deepcopy(clean_forecast)
    outage.update(
        {
            "match_count": 0,
            "matches": [],
            "best_prediction": None,
            "best_rule_id": None,
        }
    )
    cases.append(("retrieval_outage", deepcopy(clean_row), outage))
    return cases


def _decision_metrics(
    row: Mapping[str, object], decision: Mapping[str, object]
) -> dict[str, float | bool]:
    candidates = row["candidate_outcomes"]
    selections = row["selections"]
    selected = str(decision["selected_sequence"])
    fallback = str(selections["control_math"])
    selected_utility = float(candidates[selected]["utility"])
    fallback_utility = float(candidates[fallback]["utility"])
    return {
        "accepted": bool(decision["accepted"]),
        "fallback": not bool(decision["accepted"]),
        "control_parity": selected == fallback,
        "changed_from_control": selected != fallback,
        "utility": selected_utility,
        "regret_vs_control": fallback_utility - selected_utility,
    }


def _controller_summary(
    trials: Sequence[Mapping[str, object]], controller: str
) -> dict[str, float | int]:
    clean = [row for row in trials if row["scenario"] == CLEAN]
    attacks = [row for row in trials if row["scenario"] != CLEAN]
    clean_metrics = [row[controller]["metrics"] for row in clean]
    attack_metrics = [row[controller]["metrics"] for row in attacks]
    return {
        "clean_trials": len(clean),
        "attack_trials": len(attacks),
        "clean_acceptance_rate": mean(float(item["accepted"]) for item in clean_metrics),
        "attack_acceptance_rate": mean(float(item["accepted"]) for item in attack_metrics),
        "attack_rejection_rate": mean(1.0 - float(item["accepted"]) for item in attack_metrics),
        "attack_fallback_rate": mean(float(item["fallback"]) for item in attack_metrics),
        "attack_control_parity_rate": mean(
            float(item["control_parity"]) for item in attack_metrics
        ),
        "attack_changed_from_control_rate": mean(
            float(item["changed_from_control"]) for item in attack_metrics
        ),
        "mean_attack_regret_vs_control": mean(
            float(item["regret_vs_control"]) for item in attack_metrics
        ),
        "max_attack_regret_vs_control": max(
            float(item["regret_vs_control"]) for item in attack_metrics
        ),
    }


def _scenario_summary(trials: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in trials:
        grouped[str(row["scenario"])].append(row)
    summary = {}
    for scenario in SCENARIOS:
        rows = grouped[scenario]
        item: dict[str, object] = {"trials": len(rows), "expected_accept": scenario == CLEAN}
        for controller in (TOPOLOGY_ONLY, INTEGRITY_SEALED):
            metrics = [row[controller]["metrics"] for row in rows]
            reasons = Counter(
                reason
                for row in rows
                for reason in row[controller]["decision"]["reasons"]
            )
            item[controller] = {
                "acceptance_rate": mean(float(value["accepted"]) for value in metrics),
                "fallback_rate": mean(float(value["fallback"]) for value in metrics),
                "control_parity_rate": mean(
                    float(value["control_parity"]) for value in metrics
                ),
                "changed_from_control_rate": mean(
                    float(value["changed_from_control"]) for value in metrics
                ),
                "mean_regret_vs_control": mean(
                    float(value["regret_vs_control"]) for value in metrics
                ),
                "reasons": dict(sorted(reasons.items())),
            }
        summary[scenario] = item
    return summary


def _family_summary(trials: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in trials:
        if row["scenario"] != CLEAN:
            grouped[str(row["family"])].append(row)
    return {
        family: {
            "attack_trials": len(rows),
            "topology_only_acceptance_rate": mean(
                float(row[TOPOLOGY_ONLY]["metrics"]["accepted"]) for row in rows
            ),
            "integrity_sealed_acceptance_rate": mean(
                float(row[INTEGRITY_SEALED]["metrics"]["accepted"]) for row in rows
            ),
            "integrity_sealed_fallback_rate": mean(
                float(row[INTEGRITY_SEALED]["metrics"]["fallback"]) for row in rows
            ),
        }
        for family, rows in sorted(grouped.items())
    }


def run_airis_resilience_benchmark(
    payload: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    ruleset: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    protocol_sha256 = str(payload.get("protocol_sha256") or "")
    rules = ruleset.get("rules")
    if not protocol_sha256 or not isinstance(rules, list):
        raise ValueError("benchmark protocol and AIRIS rules are required")
    registry = trusted_rule_sha256(rules)
    trials = []
    for source_row in rows:
        for scenario, challenge_row, forecast in _challenge_cases(
            source_row, protocol_sha256, rules
        ):
            topology_decision = resolve_airis_decision(
                challenge_row,
                forecast,
                expected_protocol_sha256=protocol_sha256,
            ).to_jsonable()
            integrity_decision = resolve_airis_decision(
                challenge_row,
                forecast,
                expected_protocol_sha256=protocol_sha256,
                trusted_rules=registry,
            ).to_jsonable()
            trials.append(
                {
                    "episode_id": source_row.get("episode_id"),
                    "family": source_row.get("family"),
                    "context": source_row.get("context"),
                    "scenario": scenario,
                    "expected_accept": scenario == CLEAN,
                    TOPOLOGY_ONLY: {
                        "decision": topology_decision,
                        "metrics": _decision_metrics(challenge_row, topology_decision),
                    },
                    INTEGRITY_SEALED: {
                        "decision": integrity_decision,
                        "metrics": _decision_metrics(challenge_row, integrity_decision),
                    },
                }
            )

    topology_summary = _controller_summary(trials, TOPOLOGY_ONLY)
    integrity_summary = _controller_summary(trials, INTEGRITY_SEALED)
    result = {
        "schema": "hybrid_airis_das_resilience_v1",
        "protocol_sha256": protocol_sha256,
        "episode_count": len(rows),
        "scenario_count": len(SCENARIOS),
        "trial_count": len(trials),
        "attack_trial_count": len(rows) * (len(SCENARIOS) - 1),
        "controllers": {
            TOPOLOGY_ONLY: topology_summary,
            INTEGRITY_SEALED: integrity_summary,
        },
        "integrity_effect": {
            "attack_acceptance_rate_delta": (
                float(integrity_summary["attack_acceptance_rate"])
                - float(topology_summary["attack_acceptance_rate"])
            ),
            "attack_changed_from_control_rate_delta": (
                float(integrity_summary["attack_changed_from_control_rate"])
                - float(topology_summary["attack_changed_from_control_rate"])
            ),
        },
        "scenarios": _scenario_summary(trials),
        "families": _family_summary(trials),
        "claim_boundary": (
            "Deterministic paired fault-injection over sealed AIRIS/DAS replay receipts. "
            "It measures authorization resilience, not adversarial robustness of a learned AIRIS model "
            "or native distributed DAS."
        ),
    }
    return result, trials


def resilience_markdown(result: Mapping[str, object]) -> str:
    controllers = result["controllers"]
    scenarios = result["scenarios"]
    lines = [
        "# AIRIS/DAS Forecast Integrity Resilience",
        "",
        f"Episodes: `{result['episode_count']}`",
        f"Paired scenarios per episode: `{result['scenario_count']}`",
        f"Total trials: `{result['trial_count']}`",
        "",
        "| Controller | Clean acceptance | Negative-control acceptance | Negative-control fallback | Changed from control | Max regret |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for controller in (TOPOLOGY_ONLY, INTEGRITY_SEALED):
        metrics = controllers[controller]
        lines.append(
            f"| `{controller}` | {float(metrics['clean_acceptance_rate']):.3f} | "
            f"{float(metrics['attack_acceptance_rate']):.3f} | "
            f"{float(metrics['attack_fallback_rate']):.3f} | "
            f"{float(metrics['attack_changed_from_control_rate']):.3f} | "
            f"{float(metrics['max_attack_regret_vs_control']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Scenario Breakdown",
            "",
            "| Scenario | Topology-only accept | Integrity-sealed accept | Integrity fallback |",
            "|---|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        item = scenarios[scenario]
        lines.append(
            f"| `{scenario}` | {float(item[TOPOLOGY_ONLY]['acceptance_rate']):.3f} | "
            f"{float(item[INTEGRITY_SEALED]['acceptance_rate']):.3f} | "
            f"{float(item[INTEGRITY_SEALED]['fallback_rate']):.3f} |"
        )
    lines.extend(["", f"Claim boundary: {result['claim_boundary']}", ""])
    return "\n".join(lines)
