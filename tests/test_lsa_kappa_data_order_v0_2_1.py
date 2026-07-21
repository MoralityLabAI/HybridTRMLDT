from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from research_gym.integrity import canonical_file_sha256
from research_gym.scripts.bench_loop_schedule_algebra_v0_1 import signed_permutation_batch
from research_gym.scripts.bench_loop_schedule_kappa_data_order_v0_2_1 import (
    DEFAULT_CONFIG,
    REGISTRATION,
    baseline_replay_gate,
    classify_order_trace,
)


def _trace(kappas: list[float]) -> list[dict[str, float | int]]:
    return [
        {"exposure": exposure, "kappa": kappa}
        for exposure, kappa in zip((0, 512, 1024, 2048, 4096), kappas)
    ]


def test_registration_hashes_config_before_decoupled_outcomes() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    assert canonical_file_sha256(DEFAULT_CONFIG) == registration["config_sha256"]
    assert registration["new_seed_decoupled_outcomes_observed"] is False


def test_data_seed_changes_inputs_without_changing_task_transform() -> None:
    left_inputs, left_targets = signed_permutation_batch(
        103, 4, 16, torch.device("cpu"), stream=2, data_seed=211
    )
    repeat_inputs, repeat_targets = signed_permutation_batch(
        103, 4, 16, torch.device("cpu"), stream=2, data_seed=211
    )
    right_inputs, right_targets = signed_permutation_batch(
        103, 4, 16, torch.device("cpu"), stream=2, data_seed=223
    )

    assert torch.equal(left_inputs, repeat_inputs)
    assert torch.equal(left_targets, repeat_targets)
    assert not torch.equal(left_inputs, right_inputs)
    assert not torch.equal(left_targets, right_targets)


def test_order_classifier_uses_registered_drop_and_rebound() -> None:
    present = classify_order_trace(_trace([1.0, 1.0, 1.0, 0.7, 1.1]), 0.1)
    absent = classify_order_trace(_trace([1.0, 1.0, 1.0, 0.95, 1.0]), 0.1)
    assert present["classification"] == "e2048_trough_and_recovery"
    assert absent["classification"] == "registered_shape_absent"


def test_baseline_gate_rejects_kappa_drift() -> None:
    new = _trace([1.0, 1.1, 1.2, 0.8, 1.3])
    parent = [
        {**row, "regime": "tied", "rounds": 64, "seed": 103}
        for row in new
        if row["exposure"] != 512
    ]
    assert baseline_replay_gate(new, parent, tolerance=1e-12)["passed"]
    parent[2]["kappa"] = 1.21
    assert not baseline_replay_gate(new, parent, tolerance=1e-12)["passed"]


def test_config_discloses_post_outcome_seed_selection() -> None:
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert "selected after parent outcomes" in config["selection_disclosure"]
    assert config["seed_channels"]["fresh_data_order_seeds"] == [211, 223, 227]
    assert config["registered_predictions"]["minimum_persistent_fresh_orders"] == 2
