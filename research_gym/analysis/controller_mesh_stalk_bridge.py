"""Forward controller prediction with a measured Qwen external-product stalk."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import math
from pathlib import Path
import random
from statistics import mean
from typing import Mapping, Sequence

import torch

from research_gym.analysis.controller_mesh_forward import (
    EVIDENCE_MODES,
    FALLBACK_MODES,
    FEATURE_NAMES,
    INTERVENTION_EDGES,
    STABILITY_EDGES,
    ControllerGenome,
    _architecture_features,
    _feature_vector,
    _spectrum,
    _task_corpus_sha256,
    _top_k_metrics,
    calibration_message_matrices,
    evaluate_genome_outcomes,
    fit_correction_model,
    fit_ridge_predictor,
)
from research_gym.analysis.controller_mesh_forward_replication import (
    generate_replication_genomes,
    paired_bootstrap_increment,
    split_replication_genomes,
)
from research_gym.analysis.controller_mesh_sheaf import (
    MESH_NODES,
    canonical_sha256,
    message_basis,
    spearman_correlation,
)
from research_gym.analysis.measured_stalk_bridge import (
    sha256_file,
    validate_measured_stalk,
)
from research_gym.envs.control_tasks import generate_control_tasks


BRIDGE_FEATURE_NAMES = (
    "bridge_spectral_gap",
    "bridge_low_band_fraction",
    "bridge_slow_mode_fraction",
    "bridge_heat_trace_t1",
    "bridge_heat_trace_t4",
    "bridge_heat_trace_t16",
)


def frozen_config_sha256(config: Mapping[str, object]) -> str:
    material = {
        key: value for key, value in config.items() if key != "frozen_config_sha256"
    }
    return canonical_sha256(material)


def validate_bridge_config(config: Mapping[str, object]) -> str:
    expected = str(config.get("frozen_config_sha256") or "")
    actual = frozen_config_sha256(config)
    if not expected or expected != actual:
        raise ValueError(
            "measured-stalk bridge config hash mismatch: "
            f"expected={expected or '<missing>'} actual={actual}"
        )
    grid = config.get("genome_grid")
    if not isinstance(grid, Mapping):
        raise ValueError("bridge genome grid is required")
    if tuple(grid.get("evidence_modes", ())) != EVIDENCE_MODES:
        raise ValueError("bridge evidence axis differs from the registered grid")
    if tuple(grid.get("fallback_modes", ())) != FALLBACK_MODES:
        raise ValueError("bridge fallback axis differs from the registered grid")
    construction = config.get("controller_construction")
    if not isinstance(construction, Mapping):
        raise ValueError("controller construction is required")
    if tuple(tuple(edge) for edge in construction["intervention_edges"]) != INTERVENTION_EDGES:
        raise ValueError("bridge intervention graph differs from registration")
    if tuple(tuple(edge) for edge in construction["stability_edges"]) != STABILITY_EDGES:
        raise ValueError("bridge stability graph differs from registration")
    bridge = config.get("bridge_construction")
    if not isinstance(bridge, Mapping):
        raise ValueError("bridge construction is required")
    if tuple(bridge.get("features", ())) != BRIDGE_FEATURE_NAMES:
        raise ValueError("bridge feature family differs from registration")
    predictors = config["predictors"]
    if tuple(predictors["controller_features"]) != FEATURE_NAMES:
        raise ValueError("controller feature family differs from registration")
    if tuple(predictors["combined_features"]) != FEATURE_NAMES + BRIDGE_FEATURE_NAMES:
        raise ValueError("combined feature family differs from registration")
    return actual


def load_registered_stalk(
    config: Mapping[str, object], *, root: str | Path
) -> dict[str, object]:
    registration = config["measured_stalk"]
    path = (Path(root) / str(registration["path"])).resolve()
    if not path.is_file() or sha256_file(path) != registration["file_sha256"]:
        raise ValueError("registered measured-stalk file is missing or changed")
    import json

    stalk = json.loads(path.read_text(encoding="utf-8"))
    receipt = validate_measured_stalk(stalk)
    if receipt != registration["source_receipt_sha256"]:
        raise ValueError("measured-stalk receipt differs from bridge registration")
    if stalk["source_config_sha256"] != registration["source_config_sha256"]:
        raise ValueError("measured-stalk source config differs from bridge registration")
    if stalk["site"] != registration["required_site"]:
        raise ValueError("measured-stalk site differs from bridge registration")
    if int(stalk["rank"]) != int(registration["required_rank"]):
        raise ValueError("measured-stalk rank differs from bridge registration")
    return stalk


def measured_graph_eigenvalues(
    stalk: Mapping[str, object],
    *,
    restriction_edges: Sequence[Mapping[str, object]] | None = None,
) -> tuple[float, ...]:
    edges = list(restriction_edges or stalk["restriction_edges"])
    nodes = sorted(
        {
            str(row[key])
            for row in edges
            for key in ("source_node", "target_node")
        }
    )
    index = {node: position for position, node in enumerate(nodes)}
    delta = torch.zeros((len(edges), len(nodes)), dtype=torch.float64)
    for edge_index, row in enumerate(edges):
        weight = float(row["restriction_weight"])
        sign = int(row["restriction_sign"])
        if not 0.0 < weight <= 1.0 or sign not in {-1, 1}:
            raise ValueError("invalid measured restriction")
        scale = math.sqrt(weight)
        delta[edge_index, index[str(row["source_node"])]] = scale
        delta[edge_index, index[str(row["target_node"])]] = -sign * scale
    values = torch.linalg.eigvalsh(delta.T @ delta).clamp_min(0.0)
    maximum = float(values[-1].item())
    if maximum > 0.0:
        values = values / maximum
    return tuple(float(value) for value in values.tolist())


def shuffled_measured_edges(
    stalk: Mapping[str, object], *, seed: int
) -> list[dict[str, object]]:
    edges = [dict(row) for row in stalk["restriction_edges"]]
    rng = random.Random(seed)
    for edge_type in sorted({str(row["edge_type"]) for row in edges}):
        indices = [
            index
            for index, row in enumerate(edges)
            if row["edge_type"] == edge_type
        ]
        weights = [float(edges[index]["restriction_weight"]) for index in indices]
        rng.shuffle(weights)
        for index, weight in zip(indices, weights):
            edges[index]["restriction_weight"] = weight
    return edges


def external_product_eigenvalues(
    controller_eigenvalues: Sequence[float],
    measured_eigenvalues: Sequence[float],
    *,
    coupling: float,
) -> tuple[float, ...]:
    values = [
        float(controller) + coupling * float(measured)
        for controller in controller_eigenvalues
        for measured in measured_eigenvalues
    ]
    maximum = max(values, default=0.0)
    if maximum > 0.0:
        values = [value / maximum for value in values]
    return tuple(sorted(values))


def bridge_feature_vector(
    eigenvalues: Sequence[float], construction: Mapping[str, object]
) -> dict[str, float]:
    zero = float(construction["zero_tolerance"])
    low = float(construction["low_band_max"])
    slow = float(construction["slow_band_max"])
    spectrum_values = tuple(map(float, eigenvalues))
    positive = [value for value in spectrum_values if value > zero]
    heat_times = tuple(map(float, construction["heat_times"]))
    if heat_times != (1.0, 4.0, 16.0):
        raise ValueError("bridge heat times differ from registration")
    return {
        "bridge_spectral_gap": min(positive) if positive else 0.0,
        "bridge_low_band_fraction": mean(
            value <= low for value in spectrum_values
        ),
        "bridge_slow_mode_fraction": mean(
            zero < value <= slow for value in spectrum_values
        ),
        "bridge_heat_trace_t1": mean(
            math.exp(-heat_times[0] * value) for value in spectrum_values
        ),
        "bridge_heat_trace_t4": mean(
            math.exp(-heat_times[1] * value) for value in spectrum_values
        ),
        "bridge_heat_trace_t16": mean(
            math.exp(-heat_times[2] * value) for value in spectrum_values
        ),
    }


def _controller_geometry(
    calibration_tasks,
    genome: ControllerGenome,
    correction,
    construction: Mapping[str, object],
):
    matrices = calibration_message_matrices(calibration_tasks, genome, correction)
    bases = {
        node: message_basis(
            matrices[node],
            energy_fraction=float(construction["centered_energy_fraction"]),
            max_rank=int(construction["max_centered_rank_per_stalk"]),
            singular_tolerance=float(construction["singular_tolerance"]),
        )
        for node in MESH_NODES
    }
    intervention = _spectrum(bases, INTERVENTION_EDGES, construction)
    stability = _spectrum(bases, STABILITY_EDGES, construction)
    return intervention, stability


def _fit_predictor(
    feature_rows,
    genomes,
    outcomes,
    *,
    feature_names,
    ridge,
):
    return fit_ridge_predictor(
        [feature_rows[genome.genome_id] for genome in genomes],
        [float(outcomes[genome.genome_id]["selection_objective"]) for genome in genomes],
        feature_names=feature_names,
        ridge=ridge,
    )


def run_bridge_study(
    registration: Mapping[str, object],
    stalk: Mapping[str, object],
) -> dict[str, object]:
    config_sha256 = validate_bridge_config(registration)
    source_receipt = validate_measured_stalk(stalk)
    if source_receipt != registration["measured_stalk"]["source_receipt_sha256"]:
        raise ValueError("bridge source receipt differs from registration")
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
    if len(genomes) != int(registration["genome_grid"]["expected_genomes"]):
        raise ValueError("bridge genome count differs from registration")
    if len(discovery) != int(registration["genome_split"]["expected_discovery"]):
        raise ValueError("bridge discovery count differs from registration")
    if len(heldout) != int(registration["genome_split"]["expected_heldout"]):
        raise ValueError("bridge heldout count differs from registration")

    controller_construction = registration["controller_construction"]
    bridge_construction = {
        **controller_construction,
        **registration["bridge_construction"],
    }
    measured_values = measured_graph_eigenvalues(stalk)
    null_config = registration["matched_null"]
    null_measured_values = [
        measured_graph_eigenvalues(
            stalk,
            restriction_edges=shuffled_measured_edges(
                stalk,
                seed=int(null_config["base_seed"]) + replicate,
            ),
        )
        for replicate in range(int(null_config["replicates"]))
    ]
    controller_rows = {}
    bridge_rows = {}
    combined_rows = {}
    null_combined_rows = [dict() for _ in null_measured_values]
    geometry = {}
    coupling = float(registration["bridge_construction"]["coupling"])
    for genome in genomes:
        intervention, stability = _controller_geometry(
            calibration_tasks, genome, correction, controller_construction
        )
        controller_features = _feature_vector(intervention, stability)
        bridge_values = external_product_eigenvalues(
            stability.eigenvalues, measured_values, coupling=coupling
        )
        bridge_features = bridge_feature_vector(bridge_values, bridge_construction)
        combined = {**controller_features, **bridge_features}
        controller_rows[genome.genome_id] = controller_features
        bridge_rows[genome.genome_id] = bridge_features
        combined_rows[genome.genome_id] = combined
        for replicate, null_values in enumerate(null_measured_values):
            null_bridge = bridge_feature_vector(
                external_product_eigenvalues(
                    stability.eigenvalues, null_values, coupling=coupling
                ),
                bridge_construction,
            )
            null_combined_rows[replicate][genome.genome_id] = {
                **controller_features,
                **null_bridge,
            }
        geometry[genome.genome_id] = {
            "controller_features": controller_features,
            "bridge_features": bridge_features,
            "combined_feature_sha256": canonical_sha256(combined),
            "controller_stability_eigenvalues": list(stability.eigenvalues),
            "bridge_eigenvalues": list(bridge_values),
            "stalk_ranks": {
                "intervention": len(intervention.eigenvalues),
                "stability": len(stability.eigenvalues),
                "external_product": len(bridge_values),
            },
        }

    task_corpus_sha256 = _task_corpus_sha256(tasks)
    measured_graph_receipt = canonical_sha256(
        {
            "source_receipt_sha256": source_receipt,
            "eigenvalues": measured_values,
            "null_eigenvalues": null_measured_values,
        }
    )
    geometry_receipt = canonical_sha256(
        {
            "config_sha256": config_sha256,
            "source_receipt_sha256": source_receipt,
            "measured_graph_receipt_sha256": measured_graph_receipt,
            "task_corpus_sha256": task_corpus_sha256,
            "features": {
                genome_id: row["combined_feature_sha256"]
                for genome_id, row in sorted(geometry.items())
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
    combined_predictor = _fit_predictor(
        combined_rows,
        discovery,
        discovery_outcomes,
        feature_names=FEATURE_NAMES + BRIDGE_FEATURE_NAMES,
        ridge=ridge,
    )
    controller_predictor = _fit_predictor(
        controller_rows,
        discovery,
        discovery_outcomes,
        feature_names=FEATURE_NAMES,
        ridge=ridge,
    )
    categorical_rows = {
        genome.genome_id: _architecture_features(genome) for genome in genomes
    }
    categorical_names = tuple(categorical_rows[genomes[0].genome_id])
    categorical_predictor = _fit_predictor(
        categorical_rows,
        discovery,
        discovery_outcomes,
        feature_names=categorical_names,
        ridge=ridge,
    )
    predictions = [
        {
            **genome.to_jsonable(),
            "combined_feature_sha256": geometry[genome.genome_id][
                "combined_feature_sha256"
            ],
            "combined_prediction": combined_predictor.predict(
                combined_rows[genome.genome_id]
            ),
            "controller_prediction": controller_predictor.predict(
                controller_rows[genome.genome_id]
            ),
            "categorical_prediction": categorical_predictor.predict(
                categorical_rows[genome.genome_id]
            ),
        }
        for genome in heldout
    ]
    prediction_material = {
        "config_sha256": config_sha256,
        "source_receipt_sha256": source_receipt,
        "geometry_receipt_sha256": geometry_receipt,
        "combined_predictor": asdict(combined_predictor),
        "controller_predictor": asdict(controller_predictor),
        "categorical_predictor": asdict(categorical_predictor),
        "predictions": predictions,
    }
    prediction_receipt = canonical_sha256(prediction_material)

    heldout_outcomes = {
        genome.genome_id: evaluate_genome_outcomes(
            evaluation_tasks, genome, correction
        )
        for genome in heldout
    }
    outcomes = [
        float(heldout_outcomes[str(row["genome_id"])]["selection_objective"])
        for row in predictions
    ]
    combined_values = [float(row["combined_prediction"]) for row in predictions]
    controller_values = [float(row["controller_prediction"]) for row in predictions]
    categorical_values = [float(row["categorical_prediction"]) for row in predictions]
    bootstrap = registration["paired_bootstrap"]
    versus_categorical = paired_bootstrap_increment(
        combined_values,
        categorical_values,
        outcomes,
        iterations=int(bootstrap["iterations"]),
        seed=int(bootstrap["categorical_seed"]),
    )
    versus_controller = paired_bootstrap_increment(
        combined_values,
        controller_values,
        outcomes,
        iterations=int(bootstrap["iterations"]),
        seed=int(bootstrap["controller_seed"]),
    )

    top_k = int(registration["incremental_gate"]["top_k"])
    top_metrics = {}
    for name, field in (
        ("combined", "combined_prediction"),
        ("controller", "controller_prediction"),
        ("categorical", "categorical_prediction"),
    ):
        top_metrics[name] = _top_k_metrics(
            [
                {
                    "genome_id": row["genome_id"],
                    "predicted_selection_objective": row[field],
                }
                for row in predictions
            ],
            heldout_outcomes,
            top_k=top_k,
        )
    best_baseline_uplift = max(
        float(top_metrics[name]["uplift_vs_heldout_mean"])
        for name in ("controller", "categorical")
    )
    top_delta = (
        float(top_metrics["combined"]["uplift_vs_heldout_mean"])
        - best_baseline_uplift
    )

    null_rhos = []
    for rows in null_combined_rows:
        predictor = _fit_predictor(
            rows,
            discovery,
            discovery_outcomes,
            feature_names=FEATURE_NAMES + BRIDGE_FEATURE_NAMES,
            ridge=ridge,
        )
        null_predictions = [
            predictor.predict(rows[genome.genome_id]) for genome in heldout
        ]
        null_rhos.append(spearman_correlation(null_predictions, outcomes))
    combined_rho = float(versus_categorical["spectral_rho"])
    null_p = (1 + sum(value >= combined_rho for value in null_rhos)) / (
        len(null_rhos) + 1
    )
    gate = registration["incremental_gate"]
    checks = {
        "combined_rho": combined_rho >= float(gate["minimum_combined_rho"]),
        "rho_delta_vs_categorical": float(versus_categorical["rho_delta"])
        >= float(gate["minimum_rho_delta_vs_each_baseline"]),
        "rho_delta_vs_controller": float(versus_controller["rho_delta"])
        >= float(gate["minimum_rho_delta_vs_each_baseline"]),
        "bootstrap_lower_vs_categorical": float(versus_categorical["lower_95"])
        > float(gate["minimum_bootstrap_lower_delta"]),
        "bootstrap_lower_vs_controller": float(versus_controller["lower_95"])
        > float(gate["minimum_bootstrap_lower_delta"]),
        "matched_null": null_p <= float(gate["maximum_matched_null_p"]),
        "top_k_uplift_delta": top_delta
        >= float(gate["minimum_top_k_uplift_delta_vs_best_baseline"]),
    }

    family_summary = {}
    for fallback in registration["genome_split"]["heldout_fallback_families"]:
        ids = [
            genome.genome_id
            for genome in heldout
            if genome.fallback_mode == fallback
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
        "schema": "controller_mesh_measured_stalk_bridge_v1",
        "study_id": registration["study_id"],
        "config_sha256": config_sha256,
        "measured_stalk": {
            "source_receipt_sha256": source_receipt,
            "model_id": stalk["model_id"],
            "site": stalk["site"],
            "rank": stalk["rank"],
            "activation_chunk_manifest_sha256": stalk["activation_chunks"][
                "manifest_sha256"
            ],
            "restriction_edge_count": len(stalk["restriction_edges"]),
            "graph_eigenvalues": list(measured_values),
            "null_graph_eigenvalues": [
                list(values) for values in null_measured_values
            ],
            "graph_receipt_sha256": measured_graph_receipt,
        },
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
            "source_stalk_sha256": source_receipt,
            "measured_graph_sha256": measured_graph_receipt,
            "calibration_geometry_sha256": geometry_receipt,
            "discovery_outcomes_sha256": canonical_sha256(discovery_outcomes),
            "triple_prediction_sha256": prediction_receipt,
            "heldout_outcomes_sha256": canonical_sha256(heldout_outcomes),
            "ordering": [
                "outcome_free_measured_stalk_sealed",
                "fresh_calibration_geometry_sealed",
                "discovery_outcomes_revealed",
                "combined_controller_and_categorical_predictions_sealed",
                "whole_family_heldout_outcomes_revealed",
            ],
        },
        "combined_predictor": asdict(combined_predictor),
        "controller_predictor": asdict(controller_predictor),
        "categorical_predictor": asdict(categorical_predictor),
        "predictions": predictions,
        "discovery_outcomes": discovery_outcomes,
        "heldout_outcomes": heldout_outcomes,
        "geometry": geometry,
        "bridge_result": {
            "versus_categorical": versus_categorical,
            "versus_controller": versus_controller,
            "combined_rho": combined_rho,
            "controller_rho": spearman_correlation(controller_values, outcomes),
            "categorical_rho": spearman_correlation(categorical_values, outcomes),
            "matched_null_mean_rho": mean(null_rhos),
            "matched_null_one_sided_p": null_p,
            "matched_null_replicates": len(null_rhos),
            "matched_null_rhos": null_rhos,
            "top_k": top_metrics,
            "top_k_uplift_delta_vs_best_baseline": top_delta,
            "gate_checks": checks,
            "gate_passed": all(checks.values()),
            "gate": gate,
        },
        "heldout_family_summary": family_summary,
        "claim_boundary": registration["claim_boundary"],
    }
    result["analysis_receipt_sha256"] = canonical_sha256(result)
    return result


def markdown_report(result: Mapping[str, object]) -> str:
    bridge = result["bridge_result"]
    categorical = bridge["versus_categorical"]
    controller = bridge["versus_controller"]
    lines = [
        "# Measured-Stalk Controller Bridge",
        "",
        f"Protocol: `{result['study_id']}`.",
        "",
        "## Source",
        "",
        f"- model / site: `{result['measured_stalk']['model_id']}` / `{result['measured_stalk']['site']}`",
        f"- measured rank: `{result['measured_stalk']['rank']}`",
        f"- restriction edges: `{result['measured_stalk']['restriction_edge_count']}`",
        f"- source receipt: `{result['measured_stalk']['source_receipt_sha256']}`",
        "",
        "## Result",
        "",
        f"- combined / controller / categorical rho: `{bridge['combined_rho']:+.4f} / {bridge['controller_rho']:+.4f} / {bridge['categorical_rho']:+.4f}`",
        f"- combined minus categorical rho: `{categorical['rho_delta']:+.4f}` with 95% interval `[{categorical['lower_95']:+.4f}, {categorical['upper_95']:+.4f}]`",
        f"- combined minus controller rho: `{controller['rho_delta']:+.4f}` with 95% interval `[{controller['lower_95']:+.4f}, {controller['upper_95']:+.4f}]`",
        f"- matched N0 p: `{bridge['matched_null_one_sided_p']:.4f}`",
        f"- top-k uplift delta versus best baseline: `{bridge['top_k_uplift_delta_vs_best_baseline']:+.4f}`",
        f"- registered bridge gate passed: `{bridge['gate_passed']}`",
        "",
        "## Gate Checks",
        "",
        "| Check | Passed |",
        "|---|---:|",
    ]
    for name, passed in bridge["gate_checks"].items():
        lines.append(f"| `{name}` | `{passed}` |")
    lines.extend(["", "## Boundary", "", str(result["claim_boundary"]), ""])
    return "\n".join(lines)
