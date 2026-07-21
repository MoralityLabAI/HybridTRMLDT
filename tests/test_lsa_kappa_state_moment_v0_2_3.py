from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_gym.integrity import canonical_file_sha256, verify_file_sha256
from research_gym.scripts.bench_loop_schedule_kappa_state_moment_v0_2_3 import (
    DEFAULT_CONFIG,
    REGISTRATION,
    classify_factorial,
    factorial_effects,
    load_registered_config,
)


def _cells(response) -> list[dict[str, float | int]]:
    return [
        {
            "model_weight_seed": model_seed,
            "optimizer_moment_seed": optimizer_seed,
            "suffix_seed": suffix_seed,
            "endpoint_log_change": response(model_seed, optimizer_seed, suffix_seed),
        }
        for model_seed in (211, 223)
        for optimizer_seed in (211, 223)
        for suffix_seed in (211, 223)
    ]


def test_registration_hashes_config_before_state_moment_outcomes() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    assert canonical_file_sha256(DEFAULT_CONFIG) == registration["config_sha256"]
    assert registration["new_state_moment_outcomes_observed"] is False
    assert registration["new_prefix_training"] == 0
    config, config_hash, _ = load_registered_config(DEFAULT_CONFIG)
    assert config_hash == registration["config_sha256"]
    assert len(config["new_swap_cells"]) == 4
    assert len(config["imported_control_cells"]) == 4


def test_factorial_effects_recover_known_main_and_interaction_terms() -> None:
    level = {211: -1.0, 223: 1.0}

    def response(model_seed: int, optimizer_seed: int, suffix_seed: int) -> float:
        model = level[model_seed]
        optimizer = level[optimizer_seed]
        suffix = level[suffix_seed]
        return 0.2 + 0.5 * model + 0.25 * optimizer * suffix

    effects = factorial_effects(_cells(response))
    assert effects["model_weight"] == pytest.approx(1.0)
    assert effects["optimizer_moment_x_suffix"] == pytest.approx(0.5)
    assert effects["optimizer_moment"] == pytest.approx(0.0)
    assert effects["three_way_interaction"] == pytest.approx(0.0)


def test_factorial_classifier_applies_registered_dominance_ratio() -> None:
    effects = {
        "model_weight": 0.6,
        "optimizer_moment": 0.1,
        "suffix": 0.2,
        "model_weight_x_optimizer_moment": 0.0,
        "model_weight_x_suffix": 0.0,
        "optimizer_moment_x_suffix": 0.0,
        "three_way_interaction": 0.0,
    }
    result = classify_factorial(effects, dominance_ratio=1.5)
    assert result["classification"] == "model_weight_dominant"
    effects["suffix"] = 0.5
    result = classify_factorial(effects, dominance_ratio=1.5)
    assert result["classification"] == "factorial_unresolved"


def test_factorial_analysis_rejects_missing_or_duplicate_cells() -> None:
    cells = _cells(lambda *_: 0.0)
    with pytest.raises(ValueError, match="exactly once"):
        factorial_effects(cells[:-1])
    with pytest.raises(ValueError, match="exactly once"):
        factorial_effects([*cells[:-1], cells[0]])


def test_sealed_recovery_reproduces_interrupted_attempt() -> None:
    root = Path(__file__).resolve().parents[1]
    interrupted = root / "experiments" / "loop_schedule_kappa_state_moment_v0_2_3"
    recovery = root / "experiments" / "loop_schedule_kappa_state_moment_v0_2_3_recovery1"
    receipt = json.loads(
        (root / "data" / "benchmarks" / "lsa_kappa_state_moment_v0_2_3_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    result = json.loads((root / receipt["result_path"]).read_text(encoding="utf-8"))
    resource = json.loads((root / receipt["resource_path"]).read_text(encoding="utf-8"))

    assert receipt["status"] == "sealed"
    assert receipt["factorial_classification"]["classification"] == "factorial_unresolved"
    assert verify_file_sha256(root / receipt["result_path"], receipt["result_sha256"])
    assert verify_file_sha256(root / receipt["records_path"], receipt["records_sha256"])
    assert verify_file_sha256(root / receipt["resource_path"], receipt["resource_sha256"])
    assert resource["status"] == "completed"
    assert resource["cleanup_passed"]
    assert not resource["lingering_owned_process"]
    assert not resource["lingering_owned_gpu_app"]
    assert canonical_file_sha256(interrupted / "state_moment_records.jsonl") == canonical_file_sha256(
        recovery / "state_moment_records.jsonl"
    )

    first = json.loads((interrupted / "state_moment_result.json").read_text(encoding="utf-8"))

    def normalized(summary):
        value = json.loads(json.dumps(summary))
        for cell in value["new_cell_execution"]:
            cell["terminal_checkpoint"]["path"] = Path(cell["terminal_checkpoint"]["path"]).name
        return value

    assert normalized(first["summary"]) == normalized(result["summary"])
    for checkpoint in sorted((interrupted / "checkpoints").glob("*.pt")):
        assert checkpoint.read_bytes() == (recovery / "checkpoints" / checkpoint.name).read_bytes()


def test_sealed_result_preserves_registered_interaction_outcome() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (
            root
            / "experiments"
            / "loop_schedule_kappa_state_moment_v0_2_3_recovery1"
            / "state_moment_result.json"
        ).read_text(encoding="utf-8")
    )
    summary = result["summary"]
    effects = summary["factorial_effects"]
    assert effects["three_way_interaction"] == pytest.approx(0.5280058034008847)
    assert effects["model_weight"] == pytest.approx(0.5031478988768489)
    assert summary["factorial_classification"]["largest_to_runner_up_ratio"] == pytest.approx(
        1.049404766629305
    )
    cells = {
        (cell["model_weight_seed"], cell["optimizer_moment_seed"], cell["suffix_seed"]): cell
        for cell in summary["cells"]
    }
    assert not cells[(211, 211, 211)]["recovered_by_endpoint"]
    assert cells[(211, 211, 223)]["recovered_by_endpoint"]
    assert cells[(211, 223, 211)]["recovered_by_endpoint"]
    assert not cells[(211, 223, 223)]["recovered_by_endpoint"]


def test_report_figure_renders_sealed_factorial_result() -> None:
    from research_gym.scripts.report_lsa_kappa_state_moment_v0_2_3 import render_svg

    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (
            root
            / "experiments"
            / "loop_schedule_kappa_state_moment_v0_2_3_recovery1"
            / "state_moment_result.json"
        ).read_text(encoding="utf-8")
    )
    svg = render_svg(result)
    assert "Rebound is routed by checkpoint-component interactions" in svg
    assert "factorial_unresolved" in svg
    assert "|M x O x D| / |M| = 1.049" in svg
