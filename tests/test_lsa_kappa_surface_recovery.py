from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from research_gym.analysis.lsa_kappa_surface import compare_surface_models
from research_gym.scripts.bench_loop_schedule_kappa_surface import _train_surface_cell
from research_gym.scripts.bench_loop_schedule_kappa_surface_recovery import (
    DEFAULT_CONFIG,
    REGISTRATION,
    _admitted_partial_records,
    load_registered_config,
    validate,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_recovery_config_rehashes_post_partial_registration() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    assert hashlib.sha256(DEFAULT_CONFIG.read_bytes()).hexdigest() == registration["config_sha256"]
    assert registration["new_outcomes_observed"] is False
    assert registration["attempt_1_debit_seconds"] == 1204.1


def test_partial_admission_excludes_incomplete_r64_cell() -> None:
    records = _admitted_partial_records(_config())
    assert len(records) == 30
    assert {row["rounds"] for row in records} == {16, 32}
    assert all(row["regime"] == "tied" for row in records)
    assert all("R64" not in row["record_id"] for row in records)


def test_recovery_validation_reports_only_six_admitted_cells(tmp_path: Path) -> None:
    config, config_hash, _ = load_registered_config(DEFAULT_CONFIG)
    result = validate(config, config_hash, tmp_path)
    assert result["admitted_measurements"] == 30
    assert result["admitted_cells"] == 6
    assert result["incomplete_r64_cell_admitted"] is False


def test_reduced_surface_still_identifies_registered_change_model() -> None:
    cells = []
    for rounds in (16, 32, 64):
        x = math.log2(rounds / 16)
        for exposure in (0, 1024, 2048, 4096):
            u = math.log2(1 + exposure / 512)
            i64 = 1.0 if rounds == 64 else 0.0
            response = 0.2 + 0.1 * x + 0.05 * u + 0.07 * x * u - 0.6 * i64 * u
            cells.append({"rounds": rounds, "exposure": exposure, "kappa": math.exp(response)})
    assert compare_surface_models(cells, _config())["winner"] == "r64_change_point"


def test_gradient_only_cell_skips_kappa_probe(tmp_path: Path) -> None:
    config, _, parent = load_registered_config(DEFAULT_CONFIG)
    tiny = json.loads(json.dumps(config))
    tiny["frozen_training"].update(
        {
            "batch_size": 2,
            "measurement_batch_size": 2,
            "progress_target": 4,
            "checkpoint_exposures": [],
        }
    )
    records, training = _train_surface_cell(
        tiny,
        parent,
        rounds=2,
        regime="untied",
        seed=101,
        output_dir=tmp_path,
        event_path=tmp_path / "events.jsonl",
        kappa_exposures=[],
        gradient_exposures=[4],
        checkpoint_exposures=[],
    )
    assert training["nonfinite"] is False
    assert len(records) == 1
    assert records[0]["kind"] == "gradient_surface"
    assert "kappa" not in records[0]
    assert records[0]["interval_gradient"]["steps"] == 1


def test_wrapper_debits_external_attempt_and_supports_recovery_phases() -> None:
    wrapper = (ROOT / "scripts" / "run_lsa_kappa_surface_phase.ps1").read_text(encoding="utf-8")
    for marker in (
        "ExternalPriorElapsedSeconds",
        "r64_tied",
        "untied_gradients",
        '"-m", $Module',
    ):
        assert marker in wrapper
