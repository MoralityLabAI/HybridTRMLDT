"""Evaluation metrics and paired uncertainty for architecture discovery."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Mapping, Sequence

from .tasks import TaskExample


@dataclass(frozen=True)
class TaskMetrics:
    macro_exact: float
    by_family: Mapping[str, float]
    correct_by_family: Mapping[str, int]
    total_by_family: Mapping[str, int]
    sudoku_full_puzzle_exact: float | None
    routing_macro_accuracy: float | None


def evaluate_predictions(
    examples: Sequence[TaskExample], predictions: Mapping[str, int]
) -> TaskMetrics:
    correct: dict[str, int] = {}
    total: dict[str, int] = {}
    sudoku: dict[str, list[bool]] = {}
    routing_correct: dict[str, int] = {}
    routing_total: dict[str, int] = {}
    for example in examples:
        if example.example_id not in predictions:
            raise ValueError(f"missing prediction for {example.example_id}")
        family = example.family
        matched = int(predictions[example.example_id]) == example.target_token
        correct[family] = correct.get(family, 0) + int(matched)
        total[family] = total.get(family, 0) + 1
        if family == "sudoku":
            sudoku.setdefault(str(example.metadata["puzzle_id"]), []).append(matched)
        if family == "routing":
            env = str(example.metadata["env_id"])
            routing_correct[env] = routing_correct.get(env, 0) + int(matched)
            routing_total[env] = routing_total.get(env, 0) + 1
    by_family = {family: correct[family] / total[family] for family in sorted(total)}
    sudoku_exact = (
        sum(all(values) for values in sudoku.values()) / len(sudoku) if sudoku else None
    )
    routing_macro = (
        sum(routing_correct[env] / routing_total[env] for env in routing_total)
        / len(routing_total)
        if routing_total
        else None
    )
    return TaskMetrics(
        macro_exact=sum(by_family.values()) / len(by_family),
        by_family=by_family,
        correct_by_family=correct,
        total_by_family=total,
        sudoku_full_puzzle_exact=sudoku_exact,
        routing_macro_accuracy=routing_macro,
    )


def hierarchical_paired_bootstrap(
    paired_seed_deltas: Mapping[int, Sequence[float]],
    *,
    resamples: int = 10_000,
    seed: int = 971,
) -> tuple[float, float, float]:
    if not paired_seed_deltas:
        raise ValueError("paired bootstrap needs seed deltas")
    rng = random.Random(seed)
    seeds = sorted(paired_seed_deltas)
    samples: list[float] = []
    for _ in range(resamples):
        selected_seeds = [rng.choice(seeds) for _ in seeds]
        values: list[float] = []
        for selected in selected_seeds:
            deltas = paired_seed_deltas[selected]
            if not deltas:
                raise ValueError("each seed needs paired example deltas")
            values.extend(rng.choice(deltas) for _ in deltas)
        samples.append(sum(values) / len(values))
    samples.sort()
    mean = sum(samples) / len(samples)
    lower = samples[int(0.025 * (len(samples) - 1))]
    upper = samples[int(0.975 * (len(samples) - 1))]
    return mean, lower, upper


def holm_thresholds(test_count: int, alpha: float = 0.05) -> tuple[float, ...]:
    if test_count <= 0:
        raise ValueError("test_count must be positive")
    return tuple(alpha / (test_count - index) for index in range(test_count))
