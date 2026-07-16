from research_gym.adapters.airis_das import build_airis_ruleset
from research_gym.benchmarks.airis_resilience_bench import (
    INTEGRITY_SEALED,
    TOPOLOGY_ONLY,
    run_airis_resilience_benchmark,
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


def test_integrity_seal_rejects_all_controlled_attacks():
    payload = _payload()
    ruleset = build_airis_ruleset(payload)

    result, trials = run_airis_resilience_benchmark(payload, [_row()], ruleset)

    assert result["trial_count"] == len(trials) == 10
    assert result["controllers"][TOPOLOGY_ONLY]["clean_acceptance_rate"] == 1.0
    assert result["controllers"][TOPOLOGY_ONLY]["attack_acceptance_rate"] == 4 / 9
    assert result["controllers"][INTEGRITY_SEALED]["clean_acceptance_rate"] == 1.0
    assert result["controllers"][INTEGRITY_SEALED]["attack_acceptance_rate"] == 0.0
    assert result["controllers"][INTEGRITY_SEALED]["attack_fallback_rate"] == 1.0
    assert result["scenarios"]["clean"][TOPOLOGY_ONLY]["fallback_rate"] == 0.0
    assert result["scenarios"]["confidence_inflation"][TOPOLOGY_ONLY]["fallback_rate"] == 0.0
    assert result["scenarios"]["confidence_inflation"][TOPOLOGY_ONLY]["control_parity_rate"] == 1.0


def test_integrity_reasons_distinguish_substitution_and_unknown_rule():
    payload = _payload()
    result, _ = run_airis_resilience_benchmark(
        payload, [_row()], build_airis_ruleset(payload)
    )

    substitution = result["scenarios"]["sequence_substitution"]
    unknown = result["scenarios"]["unknown_rule_id"]

    assert substitution[TOPOLOGY_ONLY]["acceptance_rate"] == 1.0
    assert substitution[INTEGRITY_SEALED]["acceptance_rate"] == 0.0
    assert substitution[INTEGRITY_SEALED]["reasons"]["rule_integrity_mismatch"] == 1
    assert unknown[INTEGRITY_SEALED]["reasons"]["untrusted_rule"] == 1
