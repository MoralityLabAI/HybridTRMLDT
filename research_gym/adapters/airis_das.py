from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence
import urllib.error
import urllib.request


AIRIS_RULE_SCHEMA = "airis_storyworld_rule_v1"
AIRIS_RULESET_SCHEMA = "airis_storyworld_ruleset_v1"
AIRIS_FORECAST_SCHEMA = "airis_das_forecast_v1"
AIRIS_BRIDGE_SCHEMA = "hybrid_airis_das_bridge_v1"
AIRIS_SEQUENCER = "airis_das"
CONTROL_SEQUENCER = "control_math"
VALID_TOPOLOGY_ROUTES = frozenset({"global_signed", "local_section"})
VALID_SKILL_SEQUENCES = frozenset(
    {"proposal_only", "deduction_only", "typed_propose_certify"}
)


class AirisDasError(RuntimeError):
    pass


def _family_for_context(context: str) -> str:
    prefix, _, suffix = context.partition(":")
    if prefix == "story":
        return "story_secret" if suffix == "secret_ending" else "story_moral"
    return prefix


def _risk_bucket(value: object) -> str:
    bound = float(value)
    if bound <= 0.5:
        return "bounded"
    if bound <= 1.0:
        return "review"
    if bound < 2.0:
        return "section"
    return "blocked"


def _parse_outcome(value: object) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in str(value or "").split(";"):
        if "=" not in item:
            continue
        key, field_value = item.split("=", 1)
        fields[key.strip()] = field_value.strip()
    return fields


def airis_rule_sha256(rule: Mapping[str, object]) -> str:
    predicts = rule.get("predicts") if isinstance(rule.get("predicts"), Mapping) else {}
    material = {
        "rule_id": str(rule.get("rule_id") or ""),
        "confidence": float(rule.get("confidence") or 0.0),
        "support": int(rule.get("support") or 0),
        "counterexample_count": int(rule.get("counterexample_count") or 0),
        "preconditions": sorted(str(value) for value in rule.get("preconditions", [])),
        "predicts": dict(predicts),
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def trusted_rule_sha256(rules: Sequence[Mapping[str, object]]) -> dict[str, str]:
    registry: dict[str, str] = {}
    for rule in rules:
        rule_id = str(rule.get("rule_id") or "")
        if not rule_id:
            raise ValueError("AIRIS rule is missing rule_id")
        if rule_id in registry:
            raise ValueError(f"duplicate AIRIS rule_id: {rule_id}")
        registry[rule_id] = airis_rule_sha256(rule)
    return registry


def _condition_features(
    *,
    protocol_sha256: str,
    family: str,
    context: str,
    global_sequence: str,
    topology_route: str,
    orientation_reversal: bool,
    signed_control_loss_upper_bound: object,
) -> list[str]:
    return sorted(
        {
            "observation_kind=hybrid_sequencer_choice",
            f"protocol_sha256={protocol_sha256}",
            f"family={family}",
            f"context={context}",
            f"global_sequence={global_sequence}",
            f"topology_route={topology_route}",
            f"orientation_reversal={str(orientation_reversal).lower()}",
            f"signed_control_risk={_risk_bucket(signed_control_loss_upper_bound)}",
        }
    )


def airis_condition_features(
    *,
    protocol_sha256: str,
    family: str,
    context: str,
    global_sequence: str,
    topology_route: str,
    orientation_reversal: bool,
    signed_control_loss_upper_bound: object,
) -> list[str]:
    return _condition_features(
        protocol_sha256=protocol_sha256,
        family=family,
        context=context,
        global_sequence=global_sequence,
        topology_route=topology_route,
        orientation_reversal=orientation_reversal,
        signed_control_loss_upper_bound=signed_control_loss_upper_bound,
    )


def build_airis_ruleset(payload: Mapping[str, object]) -> dict[str, object]:
    contexts = payload.get("contexts")
    fit = payload.get("fit_receipt")
    if not isinstance(contexts, Mapping) or not isinstance(fit, Mapping):
        raise ValueError("benchmark payload must contain contexts and fit_receipt")
    protocol_sha256 = str(payload.get("protocol_sha256") or "")
    if not protocol_sha256:
        raise ValueError("benchmark payload is missing protocol_sha256")

    rules = []
    for context, raw_plan in sorted(contexts.items()):
        if not isinstance(raw_plan, Mapping):
            raise ValueError(f"invalid context plan: {context}")
        route = str(raw_plan.get("route") or "local_section")
        if route not in VALID_TOPOLOGY_ROUTES:
            raise ValueError(f"unsupported topology route for {context}: {route}")
        global_sequence = str(raw_plan.get("global_sequence") or "")
        local_sequence = str(raw_plan.get("local_sequence") or "")
        selected_sequence = global_sequence if route == "global_signed" else local_sequence
        if selected_sequence not in VALID_SKILL_SEQUENCES:
            raise ValueError(f"unsupported skill sequence for {context}: {selected_sequence}")
        score_source = fit.get("global_scores") if route == "global_signed" else raw_plan.get("local_scores")
        scores = {
            str(key): float(value)
            for key, value in (score_source.items() if isinstance(score_source, Mapping) else [])
        }
        ordered_scores = sorted(scores.values(), reverse=True)
        margin = ordered_scores[0] - ordered_scores[1] if len(ordered_scores) > 1 else 0.0
        confidence = max(0.0, min(1.0, 0.5 + margin))
        family = _family_for_context(str(context))
        features = _condition_features(
            protocol_sha256=protocol_sha256,
            family=family,
            context=str(context),
            global_sequence=global_sequence,
            topology_route=route,
            orientation_reversal=bool(raw_plan.get("orientation_reversal")),
            signed_control_loss_upper_bound=raw_plan.get("signed_control_loss_upper_bound", 2.0),
        )
        rule_basis = "|".join([protocol_sha256, str(context), route, selected_sequence, *features])
        rule_id = f"hybrid_airis_rule_{sha256(rule_basis.encode('utf-8')).hexdigest()[:12]}"
        rules.append(
            {
                "schema": AIRIS_RULE_SCHEMA,
                "rule_id": rule_id,
                "preconditions": features,
                "predicts": {
                    "outcome": (
                        f"selected_sequence={selected_sequence};control_route={route};"
                        "authority=topology_membrane"
                    ),
                    "value_signs": {
                        "calibration_margin": "positive" if margin > 1e-12 else "neutral",
                    },
                },
                "confidence": confidence,
                "support": int(raw_plan.get("calibration_count") or 0),
                "counterexample_count": 0,
                "counterexamples": [],
                "metadata": {
                    "source": "hybrid_calibration_context_plan",
                    "protocol_sha256": protocol_sha256,
                    "calibration_sha256": fit.get("calibration_sha256"),
                    "context": context,
                    "family": family,
                    "score_margin": margin,
                    "counterexample_status": "not_available_from_aggregate_plan",
                },
            }
        )

    registry = trusted_rule_sha256(rules)
    return {
        "schema": AIRIS_RULESET_SCHEMA,
        "summary": {
            "schema": AIRIS_BRIDGE_SCHEMA,
            "rule_count": len(rules),
            "source": "calibration_context_plans_only",
            "protocol_sha256": protocol_sha256,
            "calibration_sha256": fit.get("calibration_sha256"),
            "claim_boundary": (
                "AIRIS-compatible retrieval rules over frozen hybrid sequencer plans; "
                "not AIRIS causal-learning performance."
            ),
        },
        "trusted_rule_sha256": registry,
        "rules": rules,
    }


def airis_observation(row: Mapping[str, object], protocol_sha256: str) -> dict[str, object]:
    selections = row.get("selections")
    topology = row.get("topology")
    if not isinstance(selections, Mapping) or not isinstance(topology, Mapping):
        raise ValueError("sequencer row must contain selections and topology")
    return {
        "schema": "hybrid_airis_observation_v1",
        "episode_id": row.get("episode_id"),
        "condition_features": _condition_features(
            protocol_sha256=protocol_sha256,
            family=str(row.get("family") or ""),
            context=str(row.get("context") or ""),
            global_sequence=str(selections.get("global_signed") or ""),
            topology_route=str(topology.get("route") or ""),
            orientation_reversal=bool(topology.get("orientation_reversal")),
            signed_control_loss_upper_bound=topology.get("signed_control_loss_upper_bound", 2.0),
        ),
    }


def embedded_forecast(
    observation: Mapping[str, object],
    rules: Sequence[Mapping[str, object]],
    *,
    limit: int = 10,
) -> dict[str, object]:
    observed = {str(value) for value in observation.get("condition_features", [])}
    matches = []
    for rule in rules:
        preconditions = {str(value) for value in rule.get("preconditions", [])}
        if not preconditions:
            continue
        matched = sorted(preconditions & observed)
        if not matched:
            continue
        missing = sorted(preconditions - observed)
        confidence = float(rule.get("confidence") or 0.0)
        support = int(rule.get("support") or 0)
        score = (len(matched) / len(preconditions)) + (confidence * 0.25) + (min(support, 20) / 100)
        if not missing:
            score += 1.0
        predicts = rule.get("predicts") if isinstance(rule.get("predicts"), Mapping) else {}
        matches.append(
            {
                "rule_id": rule.get("rule_id"),
                "confidence": confidence,
                "support": support,
                "counterexample_count": int(rule.get("counterexample_count") or 0),
                "preconditions": sorted(preconditions),
                "predicts": dict(predicts),
                "outcome_fields": _parse_outcome(predicts.get("outcome")),
                "matched_preconditions": matched,
                "missing_preconditions": missing,
                "match_kind": "subset" if not missing else "overlap",
                "score": round(score, 6),
            }
        )
    matches.sort(key=lambda item: (-float(item["score"]), -int(item["support"]), str(item["rule_id"])))
    top = matches[:limit]
    return {
        "schema": AIRIS_FORECAST_SCHEMA,
        "observation_features": sorted(observed),
        "match_count": len(matches),
        "matches": top,
        "best_prediction": top[0]["predicts"] if top else None,
        "best_rule_id": top[0]["rule_id"] if top else None,
    }


@dataclass(frozen=True)
class AirisDasHttpClient:
    server: str = "http://127.0.0.1:8788"
    timeout_seconds: float = 30.0

    def _request(self, path: str, payload: Mapping[str, object] | None = None) -> dict[str, object]:
        body = None if payload is None else json.dumps(payload, sort_keys=True).encode("utf-8")
        request = urllib.request.Request(
            f"{self.server.rstrip('/')}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise AirisDasError(f"AIRIS/DAS request failed: {path}: {exc}") from exc
        if not isinstance(result, dict):
            raise AirisDasError(f"AIRIS/DAS returned a non-object response: {path}")
        return result

    def summary(self) -> dict[str, object]:
        return self._request("/api/summary")

    def forecast(self, observation: Mapping[str, object], *, limit: int = 10) -> dict[str, object]:
        payload = dict(observation)
        payload["limit"] = limit
        return self._request("/api/forecast", payload)


@dataclass(frozen=True)
class AirisTopologyDecision:
    proposed_sequence: str | None
    selected_sequence: str
    fallback_sequence: str
    accepted: bool
    changed_from_control: bool
    rule_id: str | None
    rule_sha256: str | None
    reasons: tuple[str, ...]

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


def resolve_airis_decision(
    row: Mapping[str, object],
    forecast: Mapping[str, object],
    *,
    expected_protocol_sha256: str,
    trusted_rules: Mapping[str, str] | None = None,
    min_confidence: float = 0.5,
    min_support: int = 2,
) -> AirisTopologyDecision:
    if not expected_protocol_sha256:
        raise ValueError("expected_protocol_sha256 must be non-empty")
    selections = row.get("selections")
    topology = row.get("topology")
    candidates = row.get("candidate_outcomes")
    if not isinstance(selections, Mapping) or not isinstance(topology, Mapping) or not isinstance(candidates, Mapping):
        raise ValueError("sequencer row is missing selections, topology, or candidate outcomes")
    fallback = str(selections[CONTROL_SEQUENCER])
    matches = forecast.get("matches")
    top = matches[0] if isinstance(matches, list) and matches and isinstance(matches[0], Mapping) else None
    reasons = []
    proposed: str | None = None
    rule_id: str | None = None
    rule_sha256: str | None = None
    if top is None:
        reasons.append("no_airis_match")
    else:
        rule_id = str(top.get("rule_id") or "") or None
        predicts = top.get("predicts") if isinstance(top.get("predicts"), Mapping) else {}
        outcome_fields = _parse_outcome(predicts.get("outcome"))
        supplied_outcome_fields = top.get("outcome_fields")
        if isinstance(supplied_outcome_fields, Mapping):
            supplied = {str(key): str(value) for key, value in supplied_outcome_fields.items()}
            if supplied != outcome_fields:
                reasons.append("forecast_outcome_mismatch")
        rule_sha256 = airis_rule_sha256(top)
        if trusted_rules is not None:
            expected_rule_sha256 = trusted_rules.get(rule_id or "")
            if expected_rule_sha256 is None:
                reasons.append("untrusted_rule")
            elif rule_sha256 != expected_rule_sha256:
                reasons.append("rule_integrity_mismatch")
        proposed = str(outcome_fields.get("selected_sequence") or "") or None
        rule_route = str(outcome_fields.get("control_route") or "")
        authority = str(outcome_fields.get("authority") or "")
        matched_preconditions = {str(value) for value in top.get("matched_preconditions", [])}
        if str(top.get("match_kind")) != "subset" or top.get("missing_preconditions"):
            reasons.append("partial_airis_match")
        if float(top.get("confidence") or 0.0) < min_confidence:
            reasons.append("low_airis_confidence")
        if int(top.get("support") or 0) < min_support:
            reasons.append("low_airis_support")
        if proposed not in candidates:
            reasons.append("unknown_skill_sequence")
        if authority != "topology_membrane":
            reasons.append("invalid_authority_claim")
        protocol_feature = f"protocol_sha256={expected_protocol_sha256}"
        if protocol_feature not in matched_preconditions:
            reasons.append("protocol_not_bound")
        topology_route = str(topology.get("route") or "")
        if topology_route not in VALID_TOPOLOGY_ROUTES:
            reasons.append("unsupported_topology_route")
        if rule_route not in VALID_TOPOLOGY_ROUTES:
            reasons.append("unsupported_airis_route")
        if rule_route != topology_route:
            reasons.append("topology_route_mismatch")
        context_feature = f"context={row.get('context')}"
        if context_feature not in matched_preconditions:
            reasons.append("context_not_bound")
        if bool(topology.get("orientation_reversal")) and rule_route != "local_section":
            reasons.append("orientation_requires_local_section")

    accepted = top is not None and not reasons and proposed is not None
    selected = proposed if accepted and proposed is not None else fallback
    return AirisTopologyDecision(
        proposed_sequence=proposed,
        selected_sequence=selected,
        fallback_sequence=fallback,
        accepted=accepted,
        changed_from_control=selected != fallback,
        rule_id=rule_id,
        rule_sha256=rule_sha256,
        reasons=tuple(reasons),
    )


def _macro_metrics(rows: Sequence[Mapping[str, object]], sequencer: str) -> dict[str, float]:
    by_family: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_family.setdefault(str(row.get("family")), []).append(row)
    family_metrics = []
    for family_rows in by_family.values():
        outcomes = []
        for row in family_rows:
            selected = row.get("selected_outcomes")
            if not isinstance(selected, Mapping) or not isinstance(selected.get(sequencer), Mapping):
                raise ValueError(f"missing selected outcome for {sequencer}")
            outcomes.append(selected[sequencer])
        family_metrics.append(
            (
                mean(float(outcome["utility"]) for outcome in outcomes),
                mean(float(outcome["correct"]) for outcome in outcomes),
                mean(float(outcome["cost"]) for outcome in outcomes),
            )
        )
    return {
        "macro_utility": mean(item[0] for item in family_metrics),
        "macro_accuracy": mean(item[1] for item in family_metrics),
        "macro_cost": mean(item[2] for item in family_metrics),
    }


def apply_airis_bridge(
    payload: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    ruleset: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rules = ruleset.get("rules")
    if not isinstance(rules, list):
        raise ValueError("AIRIS ruleset must contain rules")
    protocol_sha256 = str(payload.get("protocol_sha256") or "")
    registry = trusted_rule_sha256(rules)
    declared_registry = ruleset.get("trusted_rule_sha256")
    if isinstance(declared_registry, Mapping) and dict(declared_registry) != registry:
        raise ValueError("AIRIS ruleset integrity registry does not match its rules")
    bridged_rows = []
    receipts = []
    for raw_row in rows:
        row = deepcopy(dict(raw_row))
        observation = airis_observation(row, protocol_sha256)
        forecast = embedded_forecast(observation, rules, limit=3)
        decision = resolve_airis_decision(
            row,
            forecast,
            expected_protocol_sha256=protocol_sha256,
            trusted_rules=registry,
        )
        selections = row["selections"]
        selected_outcomes = row["selected_outcomes"]
        candidates = row["candidate_outcomes"]
        selections[AIRIS_SEQUENCER] = decision.selected_sequence
        selected_outcomes[AIRIS_SEQUENCER] = candidates[decision.selected_sequence]
        receipt = {
            "episode_id": row.get("episode_id"),
            "observation": observation,
            "forecast": forecast,
            "decision": decision.to_jsonable(),
            "trusted_rule_sha256": registry.get(decision.rule_id or ""),
        }
        row["airis_das"] = receipt
        receipts.append(receipt)
        bridged_rows.append(row)

    accepted = sum(bool(item["decision"]["accepted"]) for item in receipts)
    covered = sum(bool(item["forecast"]["match_count"]) for item in receipts)
    changed = sum(bool(item["decision"]["changed_from_control"]) for item in receipts)
    parity = sum(
        item["decision"]["selected_sequence"] == item["decision"]["fallback_sequence"]
        for item in receipts
    )
    airis_metrics = _macro_metrics(bridged_rows, AIRIS_SEQUENCER)
    control_metrics = _macro_metrics(bridged_rows, CONTROL_SEQUENCER)
    count = len(bridged_rows)
    summary = {
        "schema": AIRIS_BRIDGE_SCHEMA,
        "protocol_sha256": protocol_sha256,
        "calibration_sha256": payload.get("fit_receipt", {}).get("calibration_sha256")
        if isinstance(payload.get("fit_receipt"), Mapping)
        else None,
        "episode_count": count,
        "rule_count": len(rules),
        "forecast_coverage": covered / count if count else 0.0,
        "topology_acceptance_rate": accepted / count if count else 0.0,
        "fallback_rate": (count - accepted) / count if count else 0.0,
        "control_parity_rate": parity / count if count else 0.0,
        "changed_from_control_rate": changed / count if count else 0.0,
        "airis_das": airis_metrics,
        "control_math": control_metrics,
        "macro_utility_delta": airis_metrics["macro_utility"] - control_metrics["macro_utility"],
        "claim_boundary": (
            "Conformance and replay benchmark for calibration-derived AIRIS rules in a DAS-shaped index. "
            "It does not measure independent AIRIS causal learning or native distributed DAS performance."
        ),
    }
    return summary, bridged_rows


def bridge_markdown(summary: Mapping[str, object]) -> str:
    airis = summary["airis_das"]
    control = summary["control_math"]
    return "\n".join(
        [
            "# DAS/AIRIS Hybrid Sequencer Bridge",
            "",
            f"Episodes: `{summary['episode_count']}`",
            f"Calibration-derived AIRIS rules: `{summary['rule_count']}`",
            f"Forecast coverage: `{float(summary['forecast_coverage']):.3f}`",
            f"Topology acceptance rate: `{float(summary['topology_acceptance_rate']):.3f}`",
            f"Fail-closed fallback rate: `{float(summary['fallback_rate']):.3f}`",
            f"Control-math parity rate: `{float(summary['control_parity_rate']):.3f}`",
            "",
            "| Sequencer | Macro utility | Macro accuracy | Macro cost |",
            "|---|---:|---:|---:|",
            f"| `airis_das` | {float(airis['macro_utility']):.4f} | "
            f"{float(airis['macro_accuracy']):.4f} | {float(airis['macro_cost']):.3f} |",
            f"| `control_math` | {float(control['macro_utility']):.4f} | "
            f"{float(control['macro_accuracy']):.4f} | {float(control['macro_cost']):.3f} |",
            "",
            f"Claim boundary: {summary['claim_boundary']}",
            "",
        ]
    )


def load_jsonl(path: str) -> list[dict[str, object]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def dump_jsonl(rows: Iterable[Mapping[str, object]]) -> str:
    return "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
