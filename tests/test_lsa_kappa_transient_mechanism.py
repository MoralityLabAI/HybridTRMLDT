from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_gym.analysis.lsa_kappa_transient_mechanism import (
    classify_interference,
    decompose_estimate,
    interference_ratio,
    training_location,
)


ROOT = Path(__file__).resolve().parents[1]


def test_interference_ratio_distinguishes_alignment_and_cancellation() -> None:
    assert interference_ratio(2**0.5, [1.0, 1.0]) == pytest.approx(1.0)
    assert classify_interference(interference_ratio(2.0, [1.0, 1.0])) == "net_constructive"
    assert classify_interference(interference_ratio(0.0, [1.0, 1.0])) == "net_destructive"


def test_stored_r128_trough_reconstructs_and_is_sensitivity_cancellation() -> None:
    path = (
        ROOT
        / "experiments"
        / "loop_schedule_kappa_transient_v0_2_recovery1"
        / "combined_records.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    trough = [
        decompose_estimate(row["estimate"])
        for row in rows
        if row.get("regime") == "tied" and row.get("exposure") == 2048
    ]

    assert len(trough) == 3
    assert all(row["sensitivity_interference_class"] == "net_destructive" for row in trough)
    assert all(row["gradient_interference_class"] == "net_constructive" for row in trough)
    assert min(row["reconstructed_kappa"] for row in trough) > 0.0


def test_same_exposure_maps_to_different_training_streams_across_depth() -> None:
    locations = {
        rounds: training_location(2048, batch_size=8, rounds=rounds)
        for rounds in (16, 32, 64, 128)
    }
    assert locations[64] == {"optimizer_step": 4, "training_stream": 3}
    assert locations[128] == {"optimizer_step": 2, "training_stream": 1}
    assert len({entry["training_stream"] for entry in locations.values()}) == 4
