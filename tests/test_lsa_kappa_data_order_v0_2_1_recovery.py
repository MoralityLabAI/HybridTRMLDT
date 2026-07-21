from __future__ import annotations

import json
from pathlib import Path

from research_gym.integrity import canonical_file_sha256, verify_file_sha256
from research_gym.scripts.bench_loop_schedule_kappa_data_order_v0_2_1_recovery import (
    DEFAULT_CONFIG,
    REGISTRATION,
    admitted_records,
    load_registered_config,
)


def test_recovery_registration_hashes_config_before_aggregate_branch() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    assert canonical_file_sha256(DEFAULT_CONFIG) == registration["config_sha256"]
    assert registration["aggregate_branch_computed"] is False
    assert registration["new_training_cells"] == 0


def test_recovery_admits_exact_complete_seed_decoupled_grid() -> None:
    config, _ = load_registered_config(DEFAULT_CONFIG)
    records = admitted_records(config)
    assert len(records) == len({row["record_id"] for row in records}) == 20
    assert {row["seed_channels"]["data_order_seed"] for row in records} == {
        103,
        211,
        223,
        227,
    }
    assert all(row["seed_channels"]["model_seed"] == 103 for row in records)


def test_sealed_recovery_receipt_and_registered_outcome() -> None:
    root = Path(__file__).resolve().parents[1]
    receipt_path = (
        root / "data" / "benchmarks" / "lsa_kappa_data_order_v0_2_1_recovery1_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    result_path = root / receipt["result_path"]
    records_path = root / receipt["records_path"]
    resource_path = root / receipt["resource_path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert receipt["status"] == "sealed"
    assert receipt["scientific_status"] == "persists_under_data_order_reseed"
    assert receipt["fresh_persistence_count"] == 2
    assert receipt["new_training_cells"] == 0
    assert verify_file_sha256(result_path, receipt["result_sha256"])
    assert verify_file_sha256(records_path, receipt["records_sha256"])
    assert verify_file_sha256(resource_path, receipt["resource_sha256"])
    assert result["summary"]["baseline_replay"]["maximum_absolute_kappa_difference"] == 0.0
    assert sum(
        row["sensitivity_interference_class"] == "net_destructive"
        for row in result["summary"]["e2048_cancellation"].values()
    ) == 4


def test_report_figure_renders_from_sealed_result() -> None:
    from research_gym.scripts.report_lsa_kappa_data_order_v0_2_1 import render_svg

    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (
            root
            / "experiments"
            / "loop_schedule_kappa_data_order_v0_2_1_recovery1"
            / "recovery_result.json"
        ).read_text(encoding="utf-8")
    )
    svg = render_svg(result)
    assert "E2048 cancellation recurs" in svg
    assert "Registered outcome: 2/3 fresh orders pass" in svg
