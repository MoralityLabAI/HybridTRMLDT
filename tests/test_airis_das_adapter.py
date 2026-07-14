from copy import deepcopy

import pytest

from research_gym.adapters.airis_das import (
    AIRIS_RULE_SCHEMA,
    AIRIS_RULESET_SCHEMA,
    AIRIS_SEQUENCER,
    airis_observation,
    apply_airis_bridge,
    build_airis_ruleset,
    embedded_forecast,
    resolve_airis_decision,
)


def _payload() -> dict[str, object]:
    return {
        "protocol_sha256": "protocol-123",
        "fit_receipt": {
            "calibration_sha256": "calibration-456",
            "global_scores": {
                "proposal_only": 0.7,
                "deduction_only": 0.6,
                "typed_propose_certify": 0.9,
            },
        },
        "contexts": {
            "story:secret_ending": {
                "route": "local_section",
                "global_sequence": "typed_propose_certify",
                "local_sequence": "proposal_only",
                "orientation_reversal": True,
                "signed_control_loss_upper_bound": 1.2,
                "calibration_count": 12,
                "local_scores": {
                    "proposal_only": 0.9,
                    "deduction_only": 0.5,
                    "typed_propose_certify": 0.7,
                },
            }
        },
    }


def _row() -> dict[str, object]:
    candidates = {
        "proposal_only": {
            "answer": "secret",
            "correct": True,
            "utility": 0.8,
            "cost": 2.0,
            "constraint_violation": False,
        },
        "deduction_only": {
            "answer": "blocked",
            "correct": False,
            "utility": -0.1,
            "cost": 1.0,
            "constraint_violation": False,
        },
        "typed_propose_certify": {
            "answer": "safe",
            "correct": True,
            "utility": 0.7,
            "cost": 3.0,
            "constraint_violation": False,
        },
    }
    return {
        "episode_id": "story-secret-eval-1",
        "family": "story_secret",
        "context": "story:secret_ending",
        "prompt": "Find the secret ending",
        "expected_answer": "secret",
        "candidate_outcomes": candidates,
        "selections": {
            "global_signed": "typed_propose_certify",
            "control_math": "proposal_only",
        },
        "selected_outcomes": {"control_math": candidates["proposal_only"]},
        "topology": {
            "route": "local_section",
            "orientation_reversal": True,
            "signed_control_loss_upper_bound": 1.2,
        },
    }


def test_ruleset_uses_airis_contract_and_calibration_receipts():
    payload = _payload()
    ruleset = build_airis_ruleset(payload)
    rule = ruleset["rules"][0]

    assert ruleset["schema"] == AIRIS_RULESET_SCHEMA
    assert ruleset["summary"]["source"] == "calibration_context_plans_only"
    assert rule["schema"] == AIRIS_RULE_SCHEMA
    assert rule["support"] == 12
    assert "protocol_sha256=protocol-123" in rule["preconditions"]
    assert "context=story:secret_ending" in rule["preconditions"]
    assert rule["predicts"]["outcome"].startswith("selected_sequence=proposal_only")


def test_exact_forecast_is_accepted_by_topology_membrane():
    payload = _payload()
    row = _row()
    rules = build_airis_ruleset(payload)["rules"]
    observation = airis_observation(row, payload["protocol_sha256"])
    forecast = embedded_forecast(observation, rules)

    decision = resolve_airis_decision(
        row, forecast, expected_protocol_sha256=payload["protocol_sha256"]
    )

    assert forecast["matches"][0]["match_kind"] == "subset"
    assert decision.accepted
    assert decision.selected_sequence == "proposal_only"
    assert not decision.changed_from_control


def test_partial_or_stale_forecast_falls_back_to_control_math():
    payload = _payload()
    row = _row()
    rules = build_airis_ruleset(payload)["rules"]
    observation = airis_observation(row, "stale-protocol")
    forecast = embedded_forecast(observation, rules)

    decision = resolve_airis_decision(
        row, forecast, expected_protocol_sha256=payload["protocol_sha256"]
    )

    assert not decision.accepted
    assert "partial_airis_match" in decision.reasons
    assert "protocol_not_bound" in decision.reasons
    assert decision.selected_sequence == row["selections"]["control_math"]


def test_route_tampering_is_rejected():
    payload = _payload()
    row = _row()
    rules = build_airis_ruleset(payload)["rules"]
    observation = airis_observation(row, payload["protocol_sha256"])
    forecast = embedded_forecast(observation, rules)
    tampered = deepcopy(forecast)
    tampered["matches"][0]["outcome_fields"]["control_route"] = "global_signed"

    decision = resolve_airis_decision(
        row, tampered, expected_protocol_sha256=payload["protocol_sha256"]
    )

    assert not decision.accepted
    assert "topology_route_mismatch" in decision.reasons
    assert "orientation_requires_local_section" in decision.reasons


def test_unknown_route_is_rejected_during_rule_export():
    payload = _payload()
    payload["contexts"]["story:secret_ending"]["route"] = "confidence_override"

    with pytest.raises(ValueError, match="unsupported topology route"):
        build_airis_ruleset(payload)


def test_forecast_cannot_claim_execution_authority():
    payload = _payload()
    row = _row()
    rules = build_airis_ruleset(payload)["rules"]
    forecast = embedded_forecast(
        airis_observation(row, payload["protocol_sha256"]), rules
    )
    forecast["matches"][0]["outcome_fields"]["authority"] = "airis"

    decision = resolve_airis_decision(
        row, forecast, expected_protocol_sha256=payload["protocol_sha256"]
    )

    assert not decision.accepted
    assert "invalid_authority_claim" in decision.reasons


def test_bridge_adds_sequencer_metrics_and_sealed_receipt():
    payload = _payload()
    row = _row()
    ruleset = build_airis_ruleset(payload)

    summary, rows = apply_airis_bridge(payload, [row], ruleset)

    assert summary["episode_count"] == 1
    assert summary["topology_acceptance_rate"] == 1.0
    assert summary["control_parity_rate"] == 1.0
    assert summary["macro_utility_delta"] == 0.0
    assert rows[0]["selections"][AIRIS_SEQUENCER] == "proposal_only"
    assert rows[0]["airis_das"]["decision"]["accepted"]
