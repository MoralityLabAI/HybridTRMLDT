from __future__ import annotations

import json
import math
from pathlib import Path

from research_gym.integrity import canonical_file_sha256
from research_gym.scripts.bench_loop_schedule_kappa_transient_v0_2 import (
    DEFAULT_CONFIG,
    REGISTRATION,
    _power_exponent,
    classify_timing,
    load_registered_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_registration_precedes_r128_outcomes_and_hashes_config() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    assert canonical_file_sha256(DEFAULT_CONFIG) == registration["config_sha256"]
    assert registration["new_r128_outcomes_observed"] is False
    assert registration["registration_status"] == "frozen_before_any_r128_outcome"


def test_registered_predictions_diverge_at_r128() -> None:
    config, _, _ = load_registered_config(DEFAULT_CONFIG)
    frozen = config["frozen_training"]
    predictions = config["registered_predictions"]
    quantum = frozen["batch_size"] * frozen["rounds"][0]

    assert predictions["exposure_pinned"]["trough_exposure"] // quantum == 2
    assert predictions["step_pinned"]["trough_exposure"] // quantum == 4
    assert predictions["recovery"]["evaluation_exposure"] // quantum == 8


def test_timing_classifier_separates_exposure_and_step_pinning() -> None:
    exposure = classify_timing({1024: 1.0, 2048: 0.8, 4096: 1.1, 8192: 1.2}, 0.1)
    step = classify_timing({1024: 1.0, 2048: 1.2, 4096: 0.8, 8192: 1.1}, 0.1)
    flat = classify_timing({1024: 1.0, 2048: 0.96, 4096: 1.02, 8192: 1.04}, 0.1)

    assert exposure["classification"] == "exposure_pinned"
    assert step["classification"] == "step_pinned"
    assert flat["classification"] == "timing_unresolved"


def test_three_point_stress_exponent_recovers_power_law() -> None:
    ratios = {32: 4.0, 64: 16.0, 128: 64.0}
    assert math.isclose(_power_exponent(ratios), 2.0, abs_tol=1e-12)
