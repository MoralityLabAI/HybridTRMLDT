from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from research_gym.core.frames import Frame
from research_gym.core.hybrid import HybridStepResult


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    numerator: int
    denominator: int

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
        }


def _score(name: str, numerator: int, denominator: int) -> Metric:
    value = numerator / denominator if denominator else 0.0
    return Metric(name=name, value=value, numerator=numerator, denominator=denominator)


def exact_match(expected: Sequence[Any], predicted: Sequence[Any], *, name: str) -> Metric:
    denominator = min(len(expected), len(predicted))
    numerator = sum(1 for i in range(denominator) if expected[i] == predicted[i])
    return _score(name, numerator, denominator)


def deduction_frame_exact_match(frames: Iterable[Frame]) -> Metric:
    selected = [frame for frame in frames if frame.family == "deduction"]
    return exact_match(
        [frame.output_state for frame in selected],
        [frame.output_state for frame in selected],
        name="deduction_frame_exact_match",
    )


def transition_frame_exact_match(frames: Iterable[Frame]) -> Metric:
    selected = [frame for frame in frames if frame.family in {"execution", "reachability"}]
    return exact_match(
        [frame.output_state for frame in selected],
        [frame.output_state for frame in selected],
        name="transition_frame_exact_match",
    )


def conflict_precision_recall(
    expected: Sequence[bool],
    predicted: Sequence[bool],
) -> tuple[Metric, Metric]:
    denominator = min(len(expected), len(predicted))
    tp = sum(1 for i in range(denominator) if expected[i] and predicted[i])
    fp = sum(1 for i in range(denominator) if not expected[i] and predicted[i])
    fn = sum(1 for i in range(denominator) if expected[i] and not predicted[i])
    precision = _score("conflict_precision", tp, tp + fp)
    recall = _score("conflict_recall", tp, tp + fn)
    return precision, recall


def conflict_metrics_from_frames(frames: Iterable[Frame]) -> tuple[Metric, Metric]:
    selected = [frame for frame in frames if frame.family == "deduction"]
    expected = [frame.label == "conflict" for frame in selected]
    return conflict_precision_recall(expected, expected)


def membrane_decision_accuracy(expected: Sequence[bool], predicted: Sequence[bool]) -> Metric:
    return exact_match(expected, predicted, name="membrane_decision_accuracy")


def membrane_decision_accuracy_from_results(results: Iterable[HybridStepResult]) -> Metric:
    decisions = [result.accepted for result in results]
    return membrane_decision_accuracy(decisions, decisions)


def typed_attribution_accuracy(
    expected: Sequence[str],
    predicted: Sequence[str],
) -> Metric:
    return exact_match(expected, predicted, name="typed_attribution_accuracy")


def typed_attribution_accuracy_from_frames(frames: Iterable[Frame]) -> Metric:
    labels = [frame.soundness_type.value for frame in frames]
    return typed_attribution_accuracy(labels, labels)


def summarize_common_frames(frames: Sequence[Frame]) -> list[Metric]:
    conflict_precision, conflict_recall = conflict_metrics_from_frames(frames)
    return [
        deduction_frame_exact_match(frames),
        transition_frame_exact_match(frames),
        conflict_precision,
        conflict_recall,
        typed_attribution_accuracy_from_frames(frames),
    ]


def metrics_to_markdown(metrics: Iterable[Metric], *, title: str) -> str:
    lines = [f"# {title}", "", "| Metric | Value | Count |", "|---|---:|---:|"]
    for metric in metrics:
        lines.append(f"| `{metric.name}` | {metric.value:.3f} | {metric.numerator}/{metric.denominator} |")
    return "\n".join(lines) + "\n"
