from __future__ import annotations

import hashlib
import json
from pathlib import Path

import research_gym.scripts.bench_loop_schedule_kappa_surface as surface_runner
from research_gym.scripts.bench_loop_schedule_kappa_surface_recovery2 import (
    DEFAULT_CONFIG,
    REGISTRATION,
    load_registered_config,
    validate,
)


ROOT = Path(__file__).resolve().parents[1]


def test_recovery2_config_rehashes_pacing_registration() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    assert hashlib.sha256(DEFAULT_CONFIG.read_bytes()).hexdigest() == registration["config_sha256"]
    assert registration["scientific_change"] == "none"
    assert registration["checkpoint_pacing_seconds"] == 2.0


def test_recovery2_validation_rejects_all_failed_untied_records(tmp_path: Path) -> None:
    config, config_hash, recovery1, _ = load_registered_config(DEFAULT_CONFIG)
    result = validate(config, config_hash, recovery1, tmp_path)
    assert result["admitted_partial_measurements"] == 30
    assert result["failed_untied_records_admitted"] == 0
    assert result["checkpoint_pacing_seconds"] == 2.0
    assert result["debit_before_recovery2_seconds"] == 1909.626


def test_checkpoint_pacing_occurs_after_durable_checkpoint(monkeypatch, tmp_path: Path) -> None:
    config, _, _, parent = load_registered_config(DEFAULT_CONFIG)
    tiny = json.loads(json.dumps(config))
    tiny["frozen_training"].update(
        {
            "batch_size": 2,
            "measurement_batch_size": 2,
            "progress_target": 4,
            "checkpoint_exposures": [4],
        }
    )
    order: list[str] = []

    def fake_checkpoint(*args, **kwargs):
        order.append("checkpoint")
        return {"path": "test.pt", "sha256": "0" * 64, "bytes": 1}

    def fake_sleep(seconds: float) -> None:
        order.append(f"sleep:{seconds}")

    monkeypatch.setattr(surface_runner, "_save_checkpoint", fake_checkpoint)
    monkeypatch.setattr(surface_runner.time, "sleep", fake_sleep)
    records, training = surface_runner._train_surface_cell(
        tiny,
        parent,
        rounds=2,
        regime="untied",
        seed=101,
        output_dir=tmp_path,
        event_path=tmp_path / "events.jsonl",
        kappa_exposures=[],
        gradient_exposures=[4],
        checkpoint_exposures=[4],
        checkpoint_pacing_seconds=2.0,
    )
    assert training["nonfinite"] is False
    assert len(records) == 1
    assert order == ["checkpoint", "sleep:2.0"]


def test_wrapper_exposes_paced_phase_without_changing_io_cap() -> None:
    wrapper = (ROOT / "scripts" / "run_lsa_kappa_surface_phase.ps1").read_text(encoding="utf-8")
    assert "untied_gradients_paced" in wrapper
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert config["resources"]["io_abort_mb_s"] == 50
