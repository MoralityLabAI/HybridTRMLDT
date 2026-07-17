import json
from dataclasses import replace
from pathlib import Path

import pytest


torch = pytest.importorskip("torch", exc_type=ImportError)

from research_gym.analysis.controller_mesh_forward import (
    ControllerGenome,
    FEATURE_NAMES,
    analyze_genome_geometry,
    calibration_message_matrices,
    fit_correction_model,
    fit_ridge_predictor,
    generate_genomes,
    route_genome,
    split_genomes,
    validate_frozen_config,
)
from research_gym.analysis.controller_mesh_sheaf import canonical_sha256
from research_gym.envs.control_tasks import generate_control_tasks


ROOT = Path(__file__).resolve().parents[1]


class PoisonUtilities(dict):
    def __getitem__(self, key):
        raise AssertionError("spectral construction read a forbidden utility")


def _registration() -> dict[str, object]:
    return json.loads(
        (ROOT / "configs/controller_mesh_sheaf_forward_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_forward_config_and_stratified_genome_split_are_frozen():
    config = _registration()
    genomes = generate_genomes()
    discovery, heldout = split_genomes(genomes)

    assert validate_frozen_config(config) == config["frozen_config_sha256"]
    assert (len(genomes), len(discovery), len(heldout)) == (64, 48, 16)
    assert {genome.evidence_mode for genome in heldout} == {
        "none",
        "soft",
        "exact",
        "dual",
    }
    assert {genome.fallback_mode for genome in heldout} == {
        "identical",
        "ldt",
        "safe_ldt",
        "correction_infused",
    }

    config["forward_gate"]["minimum_spearman_rho"] = 0.0
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_frozen_config(config)


def test_correction_infused_fallback_is_calibration_fit_and_safe():
    tasks = generate_control_tasks(n_train=8, n_eval=2, seed=41)
    calibration = [task for task in tasks if task.split == "train"]
    model = fit_correction_model(
        calibration, candidate_alphas=(0.0, 0.5, 1.0)
    )
    genome = ControllerGenome("exact", 0.75, "correction_infused")

    traces = [route_genome(task, genome, model) for task in calibration]

    assert set(model.alpha_by_skill) == {task.skill for task in calibration}
    assert all(
        trace.fallback_action in task.environment_allowed
        for task, trace in zip(calibration, traces)
    )


def test_calibration_spectra_cannot_read_task_utilities():
    tasks = generate_control_tasks(n_train=4, n_eval=1, seed=43)
    calibration = [task for task in tasks if task.split == "train"]
    correction = fit_correction_model(
        calibration, candidate_alphas=(0.0, 0.5, 1.0)
    )
    poisoned = [replace(task, utilities=PoisonUtilities()) for task in calibration]
    genome = ControllerGenome("dual", 0.35, "safe_ldt")

    matrices = calibration_message_matrices(poisoned, genome, correction)

    assert set(matrices) == {"proposer", "evidence", "gate", "fallback", "executor"}
    assert all(matrix.shape[0] == len(poisoned) for matrix in matrices.values())


def test_intervention_and_stability_laplacians_are_separate():
    config = _registration()
    tasks = generate_control_tasks(n_train=4, n_eval=1, seed=47)
    calibration = [task for task in tasks if task.split == "train"]
    correction = fit_correction_model(
        calibration, candidate_alphas=(0.0, 0.5, 1.0)
    )

    geometry, null = analyze_genome_geometry(
        calibration,
        ControllerGenome("exact", 0.1, "correction_infused"),
        correction,
        construction=config["construction"],
        null_replicates=3,
        null_seed=51,
    )

    assert set(geometry["features"]) == set(FEATURE_NAMES)
    assert len(null) == 3
    assert geometry["intervention_spectrum"] != geometry["stability_spectrum"]
    assert geometry["intervention_spectrum"]["global_section_rank"] >= 1
    assert geometry["stability_spectrum"]["global_section_rank"] >= 1


def test_ridge_predictor_recovers_monotone_spectral_target():
    rows = [
        {name: float(index if position == 0 else 0.0) for position, name in enumerate(FEATURE_NAMES)}
        for index in range(1, 7)
    ]
    targets = [2.0 * index + 1.0 for index in range(1, 7)]

    predictor = fit_ridge_predictor(rows, targets, ridge=1e-6)
    predictions = [predictor.predict(row) for row in rows]

    assert predictions == sorted(predictions)
    assert max(abs(left - right) for left, right in zip(predictions, targets)) < 1e-4


def test_frozen_forward_result_preserves_stage_receipts_and_gate():
    result = json.loads(
        (ROOT / "data/benchmarks/controller_mesh_sheaf_forward_v1.json").read_text(
            encoding="utf-8"
        )
    )
    receipt = result.pop("analysis_receipt_sha256")
    prediction_material = {
        "config_sha256": result["config_sha256"],
        "geometry_receipt_sha256": result["stage_receipts"][
            "calibration_geometry_sha256"
        ],
        "predictor": result["predictor"],
        "predictions": result["predictions"],
    }

    assert canonical_sha256(result) == receipt
    assert canonical_sha256(prediction_material) == result["stage_receipts"][
        "heldout_prediction_sha256"
    ]
    assert "outcome" not in json.dumps(prediction_material).lower()
    assert canonical_sha256(result["discovery_outcomes"]) == result["stage_receipts"][
        "discovery_outcomes_sha256"
    ]
    assert canonical_sha256(result["heldout_outcomes"]) == result["stage_receipts"][
        "heldout_outcomes_sha256"
    ]
    assert set(result["genome_split"]["discovery"]).isdisjoint(
        result["genome_split"]["heldout"]
    )
    assert result["forward_result"]["gate_passed"] is True
    assert result["forward_result"]["heldout_spearman_rho"] == pytest.approx(
        0.7049302865
    )
    assert result["forward_result"]["matched_null_one_sided_p"] == pytest.approx(
        1 / 129
    )
    assert result["forward_result"]["top_k"]["uplift_vs_heldout_mean"] == pytest.approx(
        0.1475607639
    )
    assert result["posthoc_diagnostics"]["categorical_architecture_baseline"][
        "heldout_spearman_rho"
    ] == pytest.approx(0.6725535301)
