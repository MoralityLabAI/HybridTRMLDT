"""Independent whole-family replication of spectral controller prediction."""

from __future__ import annotations

from hashlib import sha256
import json
import math
import random
from statistics import mean
from typing import Mapping, Sequence

from research_gym.analysis.controller_mesh_forward import (
    EVIDENCE_MODES,
    FALLBACK_MODES,
    FEATURE_NAMES,
    INTERVENTION_EDGES,
    STABILITY_EDGES,
    ControllerGenome,
    _architecture_features,
    _task_corpus_sha256,
    _top_k_metrics,
    analyze_genome_geometry,
    evaluate_genome_outcomes,
    fit_correction_model,
    fit_ridge_predictor,
)
from research_gym.analysis.controller_mesh_sheaf import (
    canonical_sha256,
    spearman_correlation,
)
from research_gym.envs.control_tasks import generate_control_tasks


def frozen_config_sha256(config: Mapping[str, object]) -> str:
    material = {key: value for key, value in config.items() if key != "frozen_config_sha256"}
    return canonical_sha256(material)


def validate_frozen_config(config: Mapping[str, object]) -> str:
    expected = str(config.get("frozen_config_sha256") or "")
    actual = frozen_config_sha256(config)
    if not expected or expected != actual:
        raise ValueError(
            "controller-mesh replication config hash mismatch: "
            f"expected={expected or '<missing>'} actual={actual}"
        )
    genome = config.get("genome_grid")
    if not isinstance(genome, Mapping):
        raise ValueError("genome_grid is required")
    if tuple(genome.get("evidence_modes", ())) != EVIDENCE_MODES:
        raise ValueError("evidence axis does not match the registered replication")
    if tuple(genome.get("fallback_modes", ())) != FALLBACK_MODES:
        raise ValueError("fallback axis does not match the registered replication")
    construction = config.get("construction")
    if not isinstance(construction, Mapping):
        raise ValueError("construction is required")
    if tuple(tuple(edge) for edge in construction.get("intervention_edges", ())) != INTERVENTION_EDGES:
        raise ValueError("intervention graph does not match the registered replication")
    if tuple(tuple(edge) for edge in construction.get("stability_edges", ())) != STABILITY_EDGES:
        raise ValueError("stability graph does not match the registered replication")
    if tuple(config.get("predictors", {}).get("spectral_features", ())) != FEATURE_NAMES:
        raise ValueError("spectral feature list does not match the registered replication")
    return actual


def generate_replication_genomes(
    registration: Mapping[str, object],
) -> list[ControllerGenome]:
    grid = registration["genome_grid"]
    return [
        ControllerGenome(str(evidence), float(threshold), str(fallback))
        for evidence in grid["evidence_modes"]
        for threshold in grid["confidence_thresholds"]
        for fallback in grid["fallback_modes"]
    ]


def split_replication_genomes(
    genomes: Sequence[ControllerGenome],
    registration: Mapping[str, object],
) -> tuple[list[ControllerGenome], list[ControllerGenome]]:
    split = registration["genome_split"]
    discovery_families = set(split["discovery_fallback_families"])
    heldout_families = set(split["heldout_fallback_families"])
    if discovery_families & heldout_families:
        raise ValueError("discovery and heldout fallback families overlap")
    if discovery_families | heldout_families != set(FALLBACK_MODES):
        raise ValueError("fallback family split is incomplete")
    discovery = sorted(
        [genome for genome in genomes if genome.fallback_mode in discovery_families],
        key=lambda genome: genome.genome_id,
    )
    heldout = sorted(
        [genome for genome in genomes if genome.fallback_mode in heldout_families],
        key=lambda genome: genome.genome_id,
    )
    return discovery, heldout


def paired_bootstrap_increment(
    spectral_predictions: Sequence[float],
    categorical_predictions: Sequence[float],
    outcomes: Sequence[float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    if not (
        len(spectral_predictions) == len(categorical_predictions) == len(outcomes)
    ):
        raise ValueError("paired bootstrap vectors must have equal length")
    if len(outcomes) < 4:
        raise ValueError("paired bootstrap needs at least four heldout genomes")
    observed_spectral = spearman_correlation(spectral_predictions, outcomes)
    observed_categorical = spearman_correlation(categorical_predictions, outcomes)
    observed_delta = observed_spectral - observed_categorical
    rng = random.Random(seed)
    deltas = []
    for _ in range(iterations):
        indices = [rng.randrange(len(outcomes)) for _ in outcomes]
        spectral = [spectral_predictions[index] for index in indices]
        categorical = [categorical_predictions[index] for index in indices]
        values = [outcomes[index] for index in indices]
        deltas.append(
            spearman_correlation(spectral, values)
            - spearman_correlation(categorical, values)
        )
    ordered = sorted(deltas)
    lower_index = max(0, int(math.floor(0.025 * iterations)))
    upper_index = min(iterations - 1, int(math.ceil(0.975 * iterations)) - 1)
    return {
        "iterations": iterations,
        "seed": seed,
        "spectral_rho": observed_spectral,
        "categorical_rho": observed_categorical,
        "rho_delta": observed_delta,
        "lower_95": ordered[lower_index],
        "upper_95": ordered[upper_index],
        "bootstrap_positive_share": mean(value > 0.0 for value in deltas),
    }


def _categorical_feature_names(genomes: Sequence[ControllerGenome]) -> tuple[str, ...]:
    return tuple(_architecture_features(genomes[0]))


def run_replication(registration: Mapping[str, object]) -> dict[str, object]:
    config_sha256 = validate_frozen_config(registration)
    benchmark = registration["benchmark"]
    tasks = generate_control_tasks(
        n_train=int(benchmark["calibration_per_application"]),
        n_eval=int(benchmark["evaluation_per_application"]),
        seed=int(benchmark["seed"]),
    )
    calibration_tasks = [task for task in tasks if task.split == "train"]
    evaluation_tasks = [task for task in tasks if task.split == "eval"]
    correction = fit_correction_model(
        calibration_tasks,
        candidate_alphas=registration["correction_infused"]["candidate_alphas"],
    )
    genomes = generate_replication_genomes(registration)
    discovery, heldout = split_replication_genomes(genomes, registration)
    expected = registration["genome_split"]
    if len(genomes) != int(registration["genome_grid"]["expected_genomes"]):
        raise ValueError("registered genome count mismatch")
    if len(discovery) != int(expected["expected_discovery"]):
        raise ValueError("registered discovery count mismatch")
    if len(heldout) != int(expected["expected_heldout"]):
        raise ValueError("registered heldout count mismatch")

    construction = registration["construction"]
    null_config = registration["matched_null"]
    geometry: dict[str, dict[str, object]] = {}
    null_features: dict[str, list[dict[str, float]]] = {}
    for genome in genomes:
        stable_seed = int(sha256(genome.genome_id.encode("utf-8")).hexdigest()[:8], 16)
        observed, null = analyze_genome_geometry(
            calibration_tasks,
            genome,
            correction,
            construction=construction,
            null_replicates=int(null_config["replicates"]),
            null_seed=int(null_config["base_seed"]) + stable_seed,
        )
        geometry[genome.genome_id] = observed
        null_features[genome.genome_id] = null
    task_corpus_sha256 = _task_corpus_sha256(tasks)
    geometry_receipt = canonical_sha256(
        {
            "config_sha256": config_sha256,
            "task_corpus_sha256": task_corpus_sha256,
            "features": {
                genome_id: value["feature_sha256"]
                for genome_id, value in sorted(geometry.items())
            },
        }
    )

    discovery_outcomes = {
        genome.genome_id: evaluate_genome_outcomes(
            evaluation_tasks, genome, correction
        )
        for genome in discovery
    }
    predictor_config = registration["predictors"]
    ridge = float(predictor_config["ridge"])
    spectral_predictor = fit_ridge_predictor(
        [geometry[genome.genome_id]["features"] for genome in discovery],
        [
            float(discovery_outcomes[genome.genome_id]["selection_objective"])
            for genome in discovery
        ],
        feature_names=FEATURE_NAMES,
        ridge=ridge,
    )
    architecture_rows = {
        genome.genome_id: _architecture_features(genome) for genome in genomes
    }
    categorical_feature_names = _categorical_feature_names(genomes)
    categorical_predictor = fit_ridge_predictor(
        [architecture_rows[genome.genome_id] for genome in discovery],
        [
            float(discovery_outcomes[genome.genome_id]["selection_objective"])
            for genome in discovery
        ],
        feature_names=categorical_feature_names,
        ridge=ridge,
    )
    predictions = [
        {
            **genome.to_jsonable(),
            "feature_sha256": geometry[genome.genome_id]["feature_sha256"],
            "spectral_prediction": spectral_predictor.predict(
                geometry[genome.genome_id]["features"]
            ),
            "categorical_prediction": categorical_predictor.predict(
                architecture_rows[genome.genome_id]
            ),
        }
        for genome in heldout
    ]
    prediction_material = {
        "config_sha256": config_sha256,
        "geometry_receipt_sha256": geometry_receipt,
        "spectral_predictor": spectral_predictor.to_jsonable(),
        "categorical_predictor": categorical_predictor.to_jsonable(),
        "predictions": predictions,
    }
    prediction_receipt = canonical_sha256(prediction_material)

    # Heldout outcomes are evaluated only after both predictors are sealed.
    heldout_outcomes = {
        genome.genome_id: evaluate_genome_outcomes(
            evaluation_tasks, genome, correction
        )
        for genome in heldout
    }
    spectral_values = [float(row["spectral_prediction"]) for row in predictions]
    categorical_values = [float(row["categorical_prediction"]) for row in predictions]
    observed_values = [
        float(heldout_outcomes[str(row["genome_id"])]["selection_objective"])
        for row in predictions
    ]
    bootstrap_config = registration["paired_bootstrap"]
    paired = paired_bootstrap_increment(
        spectral_values,
        categorical_values,
        observed_values,
        iterations=int(bootstrap_config["iterations"]),
        seed=int(bootstrap_config["seed"]),
    )
    top_k = int(registration["incremental_gate"]["top_k"])
    spectral_rows = [
        {
            "genome_id": row["genome_id"],
            "predicted_selection_objective": row["spectral_prediction"],
        }
        for row in predictions
    ]
    categorical_rows = [
        {
            "genome_id": row["genome_id"],
            "predicted_selection_objective": row["categorical_prediction"],
        }
        for row in predictions
    ]
    spectral_top = _top_k_metrics(
        spectral_rows, heldout_outcomes, top_k=top_k
    )
    categorical_top = _top_k_metrics(
        categorical_rows, heldout_outcomes, top_k=top_k
    )
    top_uplift_delta = float(spectral_top["uplift_vs_heldout_mean"]) - float(
        categorical_top["uplift_vs_heldout_mean"]
    )

    null_rhos = []
    for replicate in range(int(null_config["replicates"])):
        null_predictor = fit_ridge_predictor(
            [null_features[genome.genome_id][replicate] for genome in discovery],
            [
                float(discovery_outcomes[genome.genome_id]["selection_objective"])
                for genome in discovery
            ],
            feature_names=FEATURE_NAMES,
            ridge=ridge,
        )
        null_predictions = [
            null_predictor.predict(null_features[genome.genome_id][replicate])
            for genome in heldout
        ]
        null_rhos.append(spearman_correlation(null_predictions, observed_values))
    null_p = (1 + sum(value >= float(paired["spectral_rho"]) for value in null_rhos)) / (
        len(null_rhos) + 1
    )

    gate = registration["incremental_gate"]
    checks = {
        "spectral_rho": float(paired["spectral_rho"])
        >= float(gate["minimum_spectral_rho"]),
        "rho_delta": float(paired["rho_delta"]) >= float(gate["minimum_rho_delta"]),
        "bootstrap_lower": float(paired["lower_95"])
        > float(gate["minimum_bootstrap_lower_delta"]),
        "matched_null": null_p <= float(gate["maximum_matched_null_p"]),
        "top_k_uplift_delta": top_uplift_delta
        >= float(gate["minimum_top_k_uplift_delta"]),
    }

    family_summary = {}
    for fallback in registration["genome_split"]["heldout_fallback_families"]:
        ids = [
            genome.genome_id for genome in heldout if genome.fallback_mode == fallback
        ]
        family_summary[fallback] = {
            "genomes": len(ids),
            "mean_objective": mean(
                float(heldout_outcomes[genome_id]["selection_objective"])
                for genome_id in ids
            ),
            "mean_utility": mean(
                float(heldout_outcomes[genome_id]["mean_utility"])
                for genome_id in ids
            ),
            "mean_unsafe_rate": mean(
                float(heldout_outcomes[genome_id]["unsafe_rate"])
                for genome_id in ids
            ),
        }

    result: dict[str, object] = {
        "schema": "controller_mesh_sheaf_forward_replication_v2",
        "study_id": registration["study_id"],
        "config_sha256": config_sha256,
        "task_corpus": {
            "sha256": task_corpus_sha256,
            "calibration_tasks": len(calibration_tasks),
            "evaluation_tasks": len(evaluation_tasks),
            "applications": sorted({task.application for task in tasks}),
        },
        "correction_model": correction.to_jsonable(),
        "genome_counts": {
            "total": len(genomes),
            "discovery": len(discovery),
            "heldout": len(heldout),
        },
        "genome_split": {
            "discovery": [genome.genome_id for genome in discovery],
            "heldout": [genome.genome_id for genome in heldout],
            "discovery_fallback_families": registration["genome_split"][
                "discovery_fallback_families"
            ],
            "heldout_fallback_families": registration["genome_split"][
                "heldout_fallback_families"
            ],
        },
        "stage_receipts": {
            "calibration_geometry_sha256": geometry_receipt,
            "discovery_outcomes_sha256": canonical_sha256(discovery_outcomes),
            "dual_prediction_sha256": prediction_receipt,
            "heldout_outcomes_sha256": canonical_sha256(heldout_outcomes),
            "ordering": [
                "calibration_geometry_sealed",
                "discovery_outcomes_revealed",
                "spectral_and_categorical_predictions_sealed",
                "whole_family_heldout_outcomes_revealed",
            ],
        },
        "spectral_predictor": spectral_predictor.to_jsonable(),
        "categorical_predictor": categorical_predictor.to_jsonable(),
        "predictions": predictions,
        "discovery_outcomes": discovery_outcomes,
        "heldout_outcomes": heldout_outcomes,
        "geometry": geometry,
        "replication_result": {
            "paired_bootstrap": paired,
            "matched_null_mean_rho": mean(null_rhos),
            "matched_null_one_sided_p": null_p,
            "matched_null_replicates": len(null_rhos),
            "spectral_top_k": spectral_top,
            "categorical_top_k": categorical_top,
            "top_k_uplift_delta": top_uplift_delta,
            "gate_checks": checks,
            "gate_passed": all(checks.values()),
            "gate": gate,
        },
        "heldout_family_summary": family_summary,
        "matched_null": registration["matched_null"],
        "claim_boundary": registration["claim_boundary"],
    }
    result["analysis_receipt_sha256"] = canonical_sha256(result)
    return result


def markdown_report(result: Mapping[str, object]) -> str:
    replication = result["replication_result"]
    paired = replication["paired_bootstrap"]
    spectral_top = replication["spectral_top_k"]
    categorical_top = replication["categorical_top_k"]
    lines = [
        "# Independent Whole-Family Spectral Replication",
        "",
        f"Protocol: `{result['study_id']}`.",
        "",
        "## Design",
        "",
        f"- genomes: `{result['genome_counts']['total']}`",
        f"- discovery / heldout: `{result['genome_counts']['discovery']} / {result['genome_counts']['heldout']}`",
        f"- discovery fallback families: `{result['genome_split']['discovery_fallback_families']}`",
        f"- heldout fallback families: `{result['genome_split']['heldout_fallback_families']}`",
        "- spectral and categorical predictions were sealed together before heldout outcomes",
        "",
        "## Co-Primary Result",
        "",
        f"- spectral rho: `{paired['spectral_rho']:+.3f}`",
        f"- categorical rho: `{paired['categorical_rho']:+.3f}`",
        f"- paired rho delta: `{paired['rho_delta']:+.3f}`",
        f"- paired bootstrap 95% interval: `[{paired['lower_95']:+.3f}, {paired['upper_95']:+.3f}]`",
        f"- matched N0 p: `{replication['matched_null_one_sided_p']:.4f}`",
        f"- spectral top-{spectral_top['top_k']} uplift: `{spectral_top['uplift_vs_heldout_mean']:+.4f}`",
        f"- categorical top-{categorical_top['top_k']} uplift: `{categorical_top['uplift_vs_heldout_mean']:+.4f}`",
        f"- top-k uplift delta: `{replication['top_k_uplift_delta']:+.4f}`",
        f"- registered incremental gate passed: `{replication['gate_passed']}`",
        "",
        "## Gate Checks",
        "",
        "| Check | Passed |",
        "|---|---:|",
    ]
    for name, passed in replication["gate_checks"].items():
        lines.append(f"| `{name}` | `{passed}` |")
    lines.extend(
        [
            "",
            "## Held-Out Families",
            "",
            "| Fallback family | Genomes | Objective | Utility | Unsafe |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for family, row in result["heldout_family_summary"].items():
        lines.append(
            f"| `{family}` | {row['genomes']} | {row['mean_objective']:.4f} | "
            f"{row['mean_utility']:.4f} | {row['mean_unsafe_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            str(result["claim_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)
