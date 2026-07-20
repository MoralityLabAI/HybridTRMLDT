from __future__ import annotations

import json
from pathlib import Path

from research_gym.architecture_discovery.calibration import (
    CalibrationMeasurement,
    calibration_token_visit_budget,
    select_budget_profile,
)


ROOT = Path(__file__).resolve().parents[1]


def _measurement(scale: str, seconds: float) -> CalibrationMeasurement:
    return CalibrationMeasurement(
        scale_rung=scale,
        mean_step_seconds=seconds,
        measured_step_count=30,
        peak_memory_bytes=1,
        peak_ram_mb=1.0,
        peak_vram_mb=1.0,
        peak_io_mb_s=1.0,
        result_sha256=scale,
        resource_receipt_sha256=scale,
    )


def _policy():
    return json.loads(
        (ROOT / "configs/lsa/architecture_promotion_policy_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_calibration_budget_is_exactly_fifty_l8_optimizer_steps() -> None:
    assert calibration_token_visit_budget(
        effective_batch_size=32, sequence_length=64
    ) == 819_200


def test_profile_selector_chooses_largest_profile_that_fits() -> None:
    selected = select_budget_profile(
        _policy(),
        [_measurement("S0", 0.1), _measurement("S1", 0.2), _measurement("S2", 0.4)],
        effective_batch_size=32,
        sequence_length=64,
    )

    assert selected["status"] == "selected"
    assert selected["selected_profile"] == "full"
    assert not selected["task_outcomes_observed"]


def test_profile_selector_seals_construction_failure_when_minimum_does_not_fit() -> None:
    selected = select_budget_profile(
        _policy(),
        [_measurement("S0", 60.0), _measurement("S1", 60.0), _measurement("S2", 60.0)],
        effective_batch_size=32,
        sequence_length=64,
    )

    assert selected["status"] == "construction_failure"
    assert selected["selected_profile"] is None
