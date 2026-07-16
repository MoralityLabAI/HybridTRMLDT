import pytest

from research_gym.benchmarks.airis_induction_bench import (
    episodes_sha256,
    evaluate_induced_airis,
    induce_airis_ruleset,
)
from research_gym.benchmarks.sequencer_control_bench import (
    BenchmarkEpisode,
    CandidateOutcome,
)


def _outcome(utility: float, *, correct: bool = True) -> CandidateOutcome:
    return CandidateOutcome(
        correct=correct,
        utility=utility,
        cost=1.0,
        constraint_violation=False,
        answer=str(utility),
    )


def _episode(index: int, winner: str) -> BenchmarkEpisode:
    utilities = {
        "proposal_only": 0.6,
        "deduction_only": 0.5,
        "typed_propose_certify": 0.4,
    }
    utilities[winner] = 1.0
    return BenchmarkEpisode(
        episode_id=f"cal-{index}",
        family="story_secret",
        context="story:secret_ending",
        split="calibration",
        prompt="task",
        expected_answer="answer",
        outcomes={name: _outcome(value) for name, value in utilities.items()},
    )


def _protocol() -> dict[str, object]:
    return {
        "study_id": "test_induction",
        "primary_confidence_threshold": 0.5,
        "minimum_support": 2,
        "sensitivity_thresholds": [0.0, 0.5, 0.8, 1.0],
        "rule_target": "episode_utility_winner",
        "tie_break_order": [
            "typed_propose_certify",
            "deduction_only",
            "proposal_only",
        ],
        "claim_boundary": "test",
    }


def _payload(calibration: list[BenchmarkEpisode]) -> dict[str, object]:
    return {
        "protocol_sha256": "protocol-123",
        "eval_sha256": "eval-456",
        "fit_receipt": {"calibration_sha256": episodes_sha256(calibration)},
        "contexts": {
            "story:secret_ending": {
                "route": "local_section",
                "global_sequence": "typed_propose_certify",
                "orientation_reversal": True,
                "signed_control_loss_upper_bound": 1.2,
            }
        },
    }


def _eval_row(index: int, oracle: str, control: str) -> dict[str, object]:
    utilities = {
        "proposal_only": 0.5,
        "deduction_only": 0.4,
        "typed_propose_certify": 0.6,
    }
    utilities[oracle] = 1.0
    candidates = {
        name: _outcome(value).to_jsonable() for name, value in utilities.items()
    }
    return {
        "episode_id": f"eval-{index}",
        "family": "story_secret",
        "context": "story:secret_ending",
        "candidate_outcomes": candidates,
        "selections": {
            "global_signed": "typed_propose_certify",
            "control_math": control,
        },
        "selected_outcomes": {"control_math": candidates[control]},
        "topology": {
            "route": "local_section",
            "orientation_reversal": True,
            "signed_control_loss_upper_bound": 1.2,
        },
    }


def test_induction_uses_calibration_majority_and_emits_counterexample():
    calibration = [
        _episode(0, "proposal_only"),
        _episode(1, "proposal_only"),
        _episode(2, "proposal_only"),
        _episode(3, "deduction_only"),
    ]

    ruleset = induce_airis_ruleset(_payload(calibration), calibration, _protocol())
    rule = ruleset["rules"][0]

    assert rule["metadata"]["source"] == "calibration_episode_outcome_induction"
    assert rule["confidence"] == 0.75
    assert rule["support"] == 3
    assert rule["counterexample_count"] == 1
    assert rule["counterexamples"][0]["row_id"] == "cal-3"
    assert rule["predicts"]["outcome"].startswith("selected_sequence=proposal_only")


def test_guarded_evaluation_demotes_intact_wrong_rule_at_high_threshold():
    calibration = [
        _episode(0, "proposal_only"),
        _episode(1, "proposal_only"),
        _episode(2, "proposal_only"),
        _episode(3, "deduction_only"),
    ]
    payload = _payload(calibration)
    protocol = _protocol()
    ruleset = induce_airis_ruleset(payload, calibration, protocol)
    rows = [
        _eval_row(0, "proposal_only", "proposal_only"),
        _eval_row(1, "deduction_only", "typed_propose_certify"),
    ]

    result, _ = evaluate_induced_airis(payload, rows, ruleset, protocol)
    high_threshold = next(
        item
        for item in result["threshold_sensitivity"]
        if item["confidence_threshold"] == 0.8
    )

    assert result["raw_airis"]["oracle_label_accuracy"] == 0.5
    assert result["guarded_airis"]["acceptance_rate"] == 1.0
    assert result["guarded_airis"]["harmful_change_count"] == 1
    assert result["guarded_airis"]["intact_wrong_rule_count"] == 1
    assert result["guarded_airis"]["raw_below_control_count"] == 1
    assert result["raw_airis"]["label_precision_by_family"]["story_secret"] == {
        "eval_count": 2,
        "heldout_rule_precision": 0.5,
        "wrong_rule_count": 1,
    }
    assert high_threshold["acceptance_rate"] == 0.0
    assert high_threshold["fallback_save_rate"] == 1.0
    assert high_threshold["intact_wrong_rule_acceptance_rate"] == 0.0


def test_induction_rejects_calibration_hash_mismatch():
    calibration = [_episode(0, "proposal_only"), _episode(1, "deduction_only")]
    payload = _payload(calibration)
    payload["fit_receipt"]["calibration_sha256"] = "wrong"

    with pytest.raises(ValueError, match="do not match"):
        induce_airis_ruleset(payload, calibration, _protocol())
