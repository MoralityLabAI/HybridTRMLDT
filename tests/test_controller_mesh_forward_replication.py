import json
from pathlib import Path

import pytest


pytest.importorskip("torch", exc_type=ImportError)

from research_gym.analysis.controller_mesh_forward_replication import (
    generate_replication_genomes,
    paired_bootstrap_increment,
    split_replication_genomes,
    validate_frozen_config,
)
from research_gym.analysis.controller_mesh_sheaf import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]


def _registration() -> dict[str, object]:
    return json.loads(
        (
            ROOT / "configs/controller_mesh_sheaf_forward_replication_v2.json"
        ).read_text(encoding="utf-8")
    )


def test_replication_config_freezes_balanced_whole_family_holdout():
    registration = _registration()
    genomes = generate_replication_genomes(registration)
    discovery, heldout = split_replication_genomes(genomes, registration)

    assert validate_frozen_config(registration) == registration["frozen_config_sha256"]
    assert (len(genomes), len(discovery), len(heldout)) == (128, 64, 64)
    assert {genome.fallback_mode for genome in discovery} == {"identical", "ldt"}
    assert {genome.fallback_mode for genome in heldout} == {
        "safe_ldt",
        "correction_infused",
    }
    assert {genome.evidence_mode for genome in heldout} == {
        "none",
        "soft",
        "exact",
        "dual",
    }
    assert len({genome.confidence_threshold for genome in heldout}) == 8

    registration["incremental_gate"]["minimum_rho_delta"] = -1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_frozen_config(registration)


def test_paired_bootstrap_detects_incremental_rank_signal():
    outcomes = [float(value) for value in range(16)]
    spectral = list(outcomes)
    categorical = [0.0] * len(outcomes)

    result = paired_bootstrap_increment(
        spectral,
        categorical,
        outcomes,
        iterations=500,
        seed=71,
    )

    assert result["spectral_rho"] == pytest.approx(1.0)
    assert result["categorical_rho"] == pytest.approx(0.0)
    assert result["rho_delta"] == pytest.approx(1.0)
    assert result["lower_95"] > 0.0


def test_whole_family_split_has_no_fallback_leakage():
    registration = _registration()
    discovery, heldout = split_replication_genomes(
        generate_replication_genomes(registration), registration
    )

    discovery_fallbacks = {genome.fallback_mode for genome in discovery}
    heldout_fallbacks = {genome.fallback_mode for genome in heldout}
    assert discovery_fallbacks.isdisjoint(heldout_fallbacks)
    assert {
        (genome.evidence_mode, genome.confidence_threshold) for genome in discovery
    } == {
        (genome.evidence_mode, genome.confidence_threshold) for genome in heldout
    }


def test_frozen_replication_receipts_and_negative_incremental_result():
    result_path = (
        ROOT / "data/benchmarks/controller_mesh_sheaf_forward_replication_v2.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    receipts = result["stage_receipts"]

    analysis_receipt = result.pop("analysis_receipt_sha256")
    assert canonical_sha256(result) == analysis_receipt

    prediction_material = {
        "config_sha256": result["config_sha256"],
        "geometry_receipt_sha256": receipts["calibration_geometry_sha256"],
        "spectral_predictor": result["spectral_predictor"],
        "categorical_predictor": result["categorical_predictor"],
        "predictions": result["predictions"],
    }
    assert "outcome" not in json.dumps(prediction_material).lower()
    assert canonical_sha256(prediction_material) == receipts["dual_prediction_sha256"]
    assert canonical_sha256(result["discovery_outcomes"]) == receipts[
        "discovery_outcomes_sha256"
    ]
    assert canonical_sha256(result["heldout_outcomes"]) == receipts[
        "heldout_outcomes_sha256"
    ]
    assert receipts["ordering"] == [
        "calibration_geometry_sealed",
        "discovery_outcomes_revealed",
        "spectral_and_categorical_predictions_sealed",
        "whole_family_heldout_outcomes_revealed",
    ]

    split = result["genome_split"]
    assert set(split["discovery_fallback_families"]).isdisjoint(
        split["heldout_fallback_families"]
    )
    assert len(split["discovery"]) == len(split["heldout"]) == 64

    replication = result["replication_result"]
    paired = replication["paired_bootstrap"]
    assert paired["spectral_rho"] == pytest.approx(0.5679323252698698)
    assert paired["categorical_rho"] == pytest.approx(0.5980281389551314)
    assert paired["rho_delta"] == pytest.approx(-0.030095813685261597)
    assert paired["lower_95"] == pytest.approx(-0.3172842235429739)
    assert paired["upper_95"] == pytest.approx(0.24597084275309866)
    assert replication["top_k_uplift_delta"] == pytest.approx(
        -0.02342122395833357
    )
    assert replication["gate_passed"] is False
    assert replication["gate_checks"] == {
        "spectral_rho": True,
        "rho_delta": False,
        "bootstrap_lower": False,
        "matched_null": True,
        "top_k_uplift_delta": False,
    }
