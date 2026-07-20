from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from research_gym.analysis.lsa_kappa_surface import (
    classify_surface,
    compare_surface_models,
    curvature_intervals,
    curvature_onset,
    fit_surface,
    predict_surface,
    score_untied_control,
)
from research_gym.scripts.bench_loop_schedule_kappa_surface import (
    DEFAULT_CONFIG,
    REGISTRATION,
    validate,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_surface_config_rehashes_registration() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    assert hashlib.sha256(DEFAULT_CONFIG.read_bytes()).hexdigest() == registration["config_sha256"]
    assert registration["nonterminal_r32_r64_outcomes_observed"] is False


def test_measurement_grid_is_reachable_without_changing_batch_size() -> None:
    frozen = _config()["frozen_training"]
    for rounds in frozen["rounds"]:
        quantum = frozen["batch_size"] * rounds
        assert all(exposure % quantum == 0 for exposure in frozen["measurement_exposures"])
    assert 256 not in frozen["measurement_exposures"]
    assert 256 % (frozen["batch_size"] * 64) != 0


def _change_point_cells() -> list[dict[str, float | int]]:
    cells = []
    for rounds in (16, 32, 64):
        x = math.log2(rounds / 16)
        for exposure in (0, 512, 1024, 2048, 4096):
            u = math.log2(1 + exposure / 512)
            i64 = 1.0 if rounds == 64 else 0.0
            log_kappa = 0.1 + 0.2 * x + 0.05 * u + 0.08 * x * u - 0.5 * i64 * u
            cells.append({"rounds": rounds, "exposure": exposure, "kappa": math.exp(log_kappa)})
    return cells


def test_registered_change_point_model_wins_on_its_own_surface() -> None:
    result = compare_surface_models(_change_point_cells(), _config())
    assert result["winner"] == "r64_change_point"
    assert result["aicc_advantage"] >= 4.0
    assert result["blocked_cv_rmse_ratio"] <= 0.9


def test_surface_prediction_round_trips_registered_model() -> None:
    cells = _change_point_cells()
    fit = fit_surface("r64_change_point", cells)
    expected = next(cell["kappa"] for cell in cells if cell["rounds"] == 64 and cell["exposure"] == 2048)
    assert predict_surface(fit, 64, 2048) == pytest.approx(expected, rel=1e-10)


def test_curvature_onset_excludes_known_terminal_endpoint() -> None:
    config = _config()
    seeds = config["frozen_training"]["seeds"]
    values = {}
    curvature_by_exposure = {0: 0.0, 512: -0.05, 1024: -0.35, 2048: -0.45, 4096: -0.8}
    for exposure, curvature in curvature_by_exposure.items():
        for rounds in (16, 32, 64):
            base = 1.0
            if rounds == 64:
                base = math.exp(curvature)
            values[(rounds, exposure)] = {seed: base for seed in seeds}
    intervals = curvature_intervals(values, config)
    assert curvature_onset(intervals, config) == 1024
    terminal_only = copy.deepcopy(values)
    for exposure in (512, 1024, 2048):
        terminal_only[(64, exposure)] = {seed: 1.0 for seed in seeds}
    assert curvature_onset(curvature_intervals(terminal_only, config), config) is None


def test_untied_control_uses_practical_equivalence_not_exact_zero() -> None:
    cells = []
    for exposure in (0, 512, 1024, 2048, 4096):
        for rounds in (16, 32, 64):
            cells.append(
                {
                    "rounds": rounds,
                    "exposure": exposure,
                    "kappa": math.exp(0.07 * math.log(rounds)),
                }
            )
    result = score_untied_control(cells, _config())
    assert result["passed"] is True
    assert result["equivalent_checkpoints"] == 5
    assert all(row["gamma"] == pytest.approx(0.07) for row in result["fits"])


def test_integrity_failures_dominate_scientific_classification() -> None:
    assert classify_surface(
        winner="r64_change_point",
        onset=1024,
        gradient_supported=True,
        replication_passed=False,
        untied_passed=True,
    ) == "instrument_drift"
    assert classify_surface(
        winner="r64_change_point",
        onset=1024,
        gradient_supported=True,
        replication_passed=True,
        untied_passed=False,
    ) == "control_failure"


def test_validation_refuses_existing_registered_outcome(tmp_path: Path) -> None:
    config = _config()
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    assert validate(config, registration["config_sha256"], tmp_path)["measurement_count"] == 90
    (tmp_path / "surface_result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="outcome already exists"):
        validate(config, registration["config_sha256"], tmp_path)


def test_wrapper_contains_required_hard_caps_and_pid_cleanup() -> None:
    wrapper = (ROOT / "scripts" / "run_lsa_kappa_surface_phase.ps1").read_text(encoding="utf-8")
    for marker in (
        "ProcessMemory",
        "CpuHardCap",
        "sustained_io_cap_exceeded",
        "vram_cap_exceeded",
        "aggregate_gpu_hour_cap_exceeded",
        "lingering_owned_process",
        "foreign_gpu_compute_process_present",
    ):
        assert marker in wrapper
