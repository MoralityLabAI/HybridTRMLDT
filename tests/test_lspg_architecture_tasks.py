from __future__ import annotations

from pathlib import Path

from research_gym.architecture_discovery.metrics import (
    evaluate_predictions,
    hierarchical_paired_bootstrap,
    holm_thresholds,
)
from research_gym.architecture_discovery.tasks import (
    ANSWER_TOKEN,
    build_task_bundle,
    read_task_bundle,
    write_task_bundle,
)


def _tiny_counts():
    return {
        family: {"train": 6, "calibration": 3, "evaluation": 3}
        for family in (
            "pointer_chase",
            "modular_recurrence",
            "rewrite_normalization",
            "sudoku",
        )
    }


def test_task_bundle_is_deterministic_and_split_disjoint() -> None:
    first = build_task_bundle(seed=17, counts=_tiny_counts(), routing_max_per_env=6)
    second = build_task_bundle(seed=17, counts=_tiny_counts(), routing_max_per_env=6)

    assert first.manifest() == second.manifest()
    assert all(example.tokens[-1] == ANSWER_TOKEN for example in first.examples)
    assert {example.family for example in first.examples} == {
        "pointer_chase",
        "modular_recurrence",
        "rewrite_normalization",
        "sudoku",
        "routing",
    }
    first.validate()


def test_task_bundle_round_trip_and_tamper_hash(tmp_path: Path) -> None:
    bundle = build_task_bundle(seed=19, counts=_tiny_counts(), routing_max_per_env=6)
    write_task_bundle(bundle, tmp_path)

    loaded = read_task_bundle(tmp_path)
    assert loaded.manifest()["bundle_hash"] == bundle.manifest()["bundle_hash"]

    train = tmp_path / "train.jsonl"
    train.write_bytes(train.read_bytes() + b"\n")
    try:
        read_task_bundle(tmp_path)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("tampered task shard was accepted")


def test_metrics_are_macro_weighted_and_bootstrap_is_deterministic() -> None:
    bundle = build_task_bundle(seed=23, counts=_tiny_counts(), routing_max_per_env=6)
    evaluation = bundle.split("evaluation")
    predictions = {example.example_id: example.target_token for example in evaluation}
    metrics = evaluate_predictions(evaluation, predictions)

    assert metrics.macro_exact == 1.0
    assert metrics.routing_macro_accuracy == 1.0
    assert metrics.sudoku_full_puzzle_exact == 1.0
    assert hierarchical_paired_bootstrap({1: [1.0, 0.0], 2: [0.5, 0.5]}, resamples=50) == hierarchical_paired_bootstrap(
        {1: [1.0, 0.0], 2: [0.5, 0.5]}, resamples=50
    )
    assert holm_thresholds(2) == (0.025, 0.05)
