from __future__ import annotations

import json
from pathlib import Path

from research_gym.integrity import canonical_file_sha256, verify_file_sha256
from research_gym.scripts.bench_loop_schedule_kappa_splice_v0_2_2 import (
    DEFAULT_CONFIG,
    REGISTRATION,
    classify_splice,
    continuation_streams,
    load_registered_config,
)


def test_registration_hashes_config_before_splice_outcomes() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    assert canonical_file_sha256(DEFAULT_CONFIG) == registration["config_sha256"]
    assert registration["new_splice_outcomes_observed"] is False
    assert registration["new_prefix_training"] == 0
    config, config_hash, _ = load_registered_config(DEFAULT_CONFIG)
    assert config_hash == registration["config_sha256"]
    assert len(config["splice_arms"]) + 1 == 5


def test_continuation_streams_start_after_checkpoint() -> None:
    assert continuation_streams(2048, 4096, quantum=512) == [4, 5, 6, 7]
    assert continuation_streams(4096, 8192, quantum=512) == list(range(8, 16))


def test_splice_classifier_preserves_registered_asymmetry() -> None:
    assert classify_splice(True, True, False) == "suffix_controlled_exit"
    assert classify_splice(False, False, True) == "prehistory_controlled_exit"
    assert classify_splice(True, False, False) == "mixed_state_suffix_control"


def test_sealed_splice_receipt_and_outcome() -> None:
    root = Path(__file__).resolve().parents[1]
    receipt_path = root / "data" / "benchmarks" / "lsa_kappa_splice_v0_2_2_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    result_path = root / receipt["result_path"]
    records_path = root / receipt["records_path"]
    resource_path = root / receipt["resource_path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert receipt["status"] == "sealed"
    assert receipt["splice_classification"] == "mixed_state_suffix_control"
    assert receipt["censoring_status"] == "delayed_recovery_by_E5120"
    assert verify_file_sha256(result_path, receipt["result_sha256"])
    assert verify_file_sha256(records_path, receipt["records_sha256"])
    assert verify_file_sha256(resource_path, receipt["resource_sha256"])
    assert result["summary"]["integrity_replay"]["tensor_exact_model_and_optimizer"]
    assert result["summary"]["recovery"]["S211_D223"]["recovered_by_endpoint"]
    assert not result["summary"]["recovery"]["S211_D227"]["recovered_by_endpoint"]
    assert result["summary"]["recovery"]["S223_D211"]["recovered_by_endpoint"]


def test_report_figure_renders_sealed_mixed_result() -> None:
    from research_gym.scripts.report_lsa_kappa_splice_v0_2_2 import render_svg

    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "experiments" / "loop_schedule_kappa_splice_v0_2_2" / "splice_result.json").read_text(encoding="utf-8")
    )
    records = [
        json.loads(line)
        for line in (root / "experiments" / "loop_schedule_kappa_splice_v0_2_2" / "splice_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    source = [
        json.loads(line)
        for line in (root / "experiments" / "loop_schedule_kappa_data_order_v0_2_1" / "data_order_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    svg = render_svg(result, records, source)
    assert "state-by-suffix mixed" in svg
    assert "D211 delayed recovery: E5120" in svg
