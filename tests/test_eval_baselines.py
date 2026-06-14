from research_gym.core.frames import Frame
from research_gym.core.hybrid import CandidateState, LatticeProposal, certify_and_apply
from research_gym.core.typed_soundness import SoundnessType
from research_gym.eval.baselines import (
    conflict_precision_recall,
    exact_match,
    membrane_decision_accuracy_from_results,
    summarize_common_frames,
    typed_attribution_accuracy,
)


def test_exact_match_scores_aligned_sequences():
    metric = exact_match([1, 2, 3], [1, 0, 3], name="toy")

    assert metric.name == "toy"
    assert metric.numerator == 2
    assert metric.denominator == 3
    assert metric.value == 2 / 3


def test_conflict_precision_recall():
    precision, recall = conflict_precision_recall(
        expected=[True, True, False, False],
        predicted=[True, False, True, False],
    )

    assert precision.numerator == 1
    assert precision.denominator == 2
    assert recall.numerator == 1
    assert recall.denominator == 2


def test_typed_attribution_accuracy():
    metric = typed_attribution_accuracy(
        expected=["env_sound_dead", "unknown"],
        predicted=["env_sound_dead", "live"],
    )

    assert metric.numerator == 1
    assert metric.denominator == 2


def test_membrane_decision_accuracy_from_results():
    current = CandidateState({"slot": frozenset({"a", "b"})})
    accepted = certify_and_apply(
        current,
        LatticeProposal(
            proposed_state=CandidateState({"slot": frozenset({"a"})}),
            soundness=SoundnessType.ENV_SOUND_DEAD,
        ),
    )
    rejected = certify_and_apply(
        current,
        LatticeProposal(
            proposed_state=CandidateState({"slot": frozenset({"a"})}),
            soundness=SoundnessType.MODEL_SOUND_DEAD,
        ),
    )

    metric = membrane_decision_accuracy_from_results([accepted, rejected])

    assert metric.numerator == 2
    assert metric.denominator == 2


def test_summarize_common_frames():
    frames = [
        Frame(
            id="d1",
            family="deduction",
            source="unit",
            input_state={"slot": ["a", "b"]},
            operation={"rule": "prune"},
            output_state={"slot": ["a"]},
            soundness_type=SoundnessType.ENV_SOUND_DEAD,
            label="conflict",
        ),
        Frame(
            id="e1",
            family="execution",
            source="unit",
            input_state={"x": 1},
            operation={"rule": "step"},
            output_state={"x": 2},
            soundness_type=SoundnessType.UNKNOWN,
            label="passed",
        ),
    ]

    metrics = {metric.name: metric for metric in summarize_common_frames(frames)}

    assert metrics["deduction_frame_exact_match"].value == 1.0
    assert metrics["transition_frame_exact_match"].value == 1.0
    assert metrics["conflict_precision"].value == 1.0
    assert metrics["conflict_recall"].value == 1.0
    assert metrics["typed_attribution_accuracy"].value == 1.0
