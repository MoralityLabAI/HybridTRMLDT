"""Forward spectral prediction over typed controller genomes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from statistics import mean
from typing import Mapping, Sequence

try:
    import torch
    from torch import Tensor
except ImportError as exc:  # pragma: no cover - exercised without neural extra
    raise ImportError(
        "controller_mesh_forward requires the optional 'neural' extra"
    ) from exc

from research_gym.analysis.controller_mesh_sheaf import (
    MESH_NODES,
    canonical_sha256,
    laplacian_spectrum,
    message_basis,
    permuted_bases,
    sheaf_laplacian,
    spearman_correlation,
)
from research_gym.envs.control_tasks import ControlTask, generate_control_tasks


EVIDENCE_MODES = ("none", "soft", "exact", "dual")
CONFIDENCE_THRESHOLDS = (0.0, 0.1, 0.35, 0.75)
FALLBACK_MODES = ("identical", "ldt", "safe_ldt", "correction_infused")

INTERVENTION_EDGES = (
    ("evidence", "gate"),
    ("gate", "fallback"),
    ("fallback", "executor"),
)
STABILITY_EDGES = (
    ("proposer", "evidence"),
    ("evidence", "gate"),
    ("gate", "executor"),
    ("proposer", "executor"),
)
FEATURE_NAMES = (
    "intervention_spectral_gap",
    "intervention_low_band_rank",
    "intervention_slow_mode_rank",
    "stability_spectral_gap",
    "stability_low_band_rank",
    "stability_slow_mode_rank",
)


def frozen_config_sha256(config: Mapping[str, object]) -> str:
    material = {key: value for key, value in config.items() if key != "frozen_config_sha256"}
    return canonical_sha256(material)


def validate_frozen_config(config: Mapping[str, object]) -> str:
    expected = str(config.get("frozen_config_sha256") or "")
    actual = frozen_config_sha256(config)
    if not expected or expected != actual:
        raise ValueError(
            "controller-mesh forward config hash mismatch: "
            f"expected={expected or '<missing>'} actual={actual}"
        )
    genome = config.get("genome_grid")
    if not isinstance(genome, Mapping):
        raise ValueError("genome_grid is required")
    if tuple(genome.get("evidence_modes", ())) != EVIDENCE_MODES:
        raise ValueError("evidence genome axis does not match the registered grid")
    if tuple(float(value) for value in genome.get("confidence_thresholds", ())) != CONFIDENCE_THRESHOLDS:
        raise ValueError("confidence genome axis does not match the registered grid")
    if tuple(genome.get("fallback_modes", ())) != FALLBACK_MODES:
        raise ValueError("fallback genome axis does not match the registered grid")
    construction = config.get("construction")
    if not isinstance(construction, Mapping):
        raise ValueError("construction is required")
    if tuple(tuple(edge) for edge in construction.get("intervention_edges", ())) != INTERVENTION_EDGES:
        raise ValueError("intervention topology does not match the registered graph")
    if tuple(tuple(edge) for edge in construction.get("stability_edges", ())) != STABILITY_EDGES:
        raise ValueError("stability topology does not match the registered graph")
    if tuple(config.get("predictor", {}).get("features", ())) != FEATURE_NAMES:
        raise ValueError("predictor features do not match the registered spectra")
    return actual


@dataclass(frozen=True)
class ControllerGenome:
    evidence_mode: str
    confidence_threshold: float
    fallback_mode: str

    @property
    def genome_id(self) -> str:
        threshold = int(round(100 * self.confidence_threshold))
        return (
            f"e-{self.evidence_mode}__g-{threshold:03d}__f-{self.fallback_mode}"
        )

    def to_jsonable(self) -> dict[str, object]:
        return {"genome_id": self.genome_id, **asdict(self)}


@dataclass(frozen=True)
class CorrectionModel:
    alpha_by_skill: Mapping[str, float]
    calibration_count_by_skill: Mapping[str, int]
    candidate_alphas: tuple[float, ...]

    def to_jsonable(self) -> dict[str, object]:
        return {
            "alpha_by_skill": dict(sorted(self.alpha_by_skill.items())),
            "calibration_count_by_skill": dict(
                sorted(self.calibration_count_by_skill.items())
            ),
            "candidate_alphas": list(self.candidate_alphas),
        }


@dataclass(frozen=True)
class GenomeTrace:
    task_id: str
    proposed_action: str
    fallback_action: str
    selected_action: str
    accepted: bool
    evidence_passed: bool
    confidence_passed: bool
    margin: float
    cost: float


@dataclass(frozen=True)
class RidgePredictor:
    feature_names: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    ridge: float

    def predict(self, features: Mapping[str, float]) -> float:
        standardized = [
            (float(features[name]) - center) / scale
            for name, center, scale in zip(
                self.feature_names, self.feature_means, self.feature_scales
            )
        ]
        return self.intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, standardized)
        )

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


def generate_genomes() -> list[ControllerGenome]:
    return [
        ControllerGenome(evidence, threshold, fallback)
        for evidence in EVIDENCE_MODES
        for threshold in CONFIDENCE_THRESHOLDS
        for fallback in FALLBACK_MODES
    ]


def split_genomes(
    genomes: Sequence[ControllerGenome],
) -> tuple[list[ControllerGenome], list[ControllerGenome]]:
    """Hold out one hash-selected threshold in every evidence/fallback cell."""

    cells: dict[tuple[str, str], list[ControllerGenome]] = defaultdict(list)
    for genome in genomes:
        cells[(genome.evidence_mode, genome.fallback_mode)].append(genome)
    discovery = []
    heldout = []
    for values in cells.values():
        ordered = sorted(
            values,
            key=lambda genome: sha256(genome.genome_id.encode("utf-8")).hexdigest(),
        )
        heldout.append(ordered[0])
        discovery.extend(ordered[1:])
    return (
        sorted(discovery, key=lambda genome: genome.genome_id),
        sorted(heldout, key=lambda genome: genome.genome_id),
    )


def _rank(scores: Mapping[str, float], actions: Sequence[str]) -> list[str]:
    order = {action: index for index, action in enumerate(actions)}
    return sorted(actions, key=lambda action: (-float(scores[action]), order[action]))


def _margin(scores: Mapping[str, float], actions: Sequence[str]) -> float:
    ranked = _rank(scores, actions)
    return float(scores[ranked[0]] - scores[ranked[1]])


def _blended_scores(task: ControlTask, alpha: float) -> dict[str, float]:
    return {
        action: alpha * float(task.trm_scores[action])
        + (1.0 - alpha) * float(task.ldt_scores[action])
        for action in task.actions
    }


def fit_correction_model(
    calibration_tasks: Sequence[ControlTask],
    *,
    candidate_alphas: Sequence[float],
) -> CorrectionModel:
    """Fit a bounded safe correction blend on independently scored hard cases."""

    by_skill: dict[str, list[ControlTask]] = defaultdict(list)
    for task in calibration_tasks:
        proposal = _rank(task.trm_scores, task.actions)[0]
        if proposal != task.optimal_action or proposal not in task.environment_allowed:
            by_skill[task.skill].append(task)
    all_by_skill: dict[str, list[ControlTask]] = defaultdict(list)
    for task in calibration_tasks:
        all_by_skill[task.skill].append(task)
    selected: dict[str, float] = {}
    counts: dict[str, int] = {}
    candidates = tuple(float(value) for value in candidate_alphas)
    for skill, all_tasks in sorted(all_by_skill.items()):
        tasks = by_skill[skill] or all_tasks
        counts[skill] = len(tasks)

        def score(alpha: float) -> tuple[float, float, float]:
            chosen = [
                _rank(_blended_scores(task, alpha), task.environment_allowed)[0]
                for task in tasks
            ]
            utility = mean(
                float(task.utilities[action]) for task, action in zip(tasks, chosen)
            )
            accuracy = mean(
                action == task.optimal_action for task, action in zip(tasks, chosen)
            )
            return utility, accuracy, -alpha

        selected[skill] = max(candidates, key=score)
    return CorrectionModel(selected, counts, candidates)


def route_genome(
    task: ControlTask,
    genome: ControllerGenome,
    correction: CorrectionModel,
) -> GenomeTrace:
    proposed = _rank(task.trm_scores, task.actions)[0]
    margin = _margin(task.trm_scores, task.actions)
    if genome.evidence_mode == "none":
        evidence_passed = True
    elif genome.evidence_mode == "soft":
        evidence_passed = proposed in task.soft_candidates
    elif genome.evidence_mode == "exact":
        evidence_passed = proposed in task.environment_allowed
    elif genome.evidence_mode == "dual":
        evidence_passed = (
            proposed in task.environment_allowed and proposed in task.soft_candidates
        )
    else:
        raise ValueError(f"unknown evidence mode: {genome.evidence_mode}")
    confidence_passed = margin >= genome.confidence_threshold
    accepted = evidence_passed and confidence_passed

    if genome.fallback_mode == "identical":
        fallback = proposed
        fallback_cost = 0.0
    elif genome.fallback_mode == "ldt":
        fallback = _rank(task.ldt_scores, task.actions)[0]
        fallback_cost = 1.0
    elif genome.fallback_mode == "safe_ldt":
        fallback = _rank(task.ldt_scores, task.environment_allowed)[0]
        fallback_cost = 1.2
    elif genome.fallback_mode == "correction_infused":
        alpha = correction.alpha_by_skill[task.skill]
        fallback = _rank(
            _blended_scores(task, alpha), task.environment_allowed
        )[0]
        fallback_cost = 1.4
    else:
        raise ValueError(f"unknown fallback mode: {genome.fallback_mode}")

    evidence_cost = {"none": 0.0, "soft": 0.1, "exact": 0.2, "dual": 0.3}[
        genome.evidence_mode
    ]
    selected = proposed if accepted else fallback
    return GenomeTrace(
        task_id=task.task_id,
        proposed_action=proposed,
        fallback_action=fallback,
        selected_action=selected,
        accepted=accepted,
        evidence_passed=evidence_passed,
        confidence_passed=confidence_passed,
        margin=margin,
        cost=1.0 + evidence_cost + (0.0 if accepted else fallback_cost),
    )


def _one_hot(index: int, width: int = 4) -> list[float]:
    values = [0.0] * width
    values[index] = 1.0
    return values


def _action_index(task: ControlTask, action: str) -> int:
    return task.actions.index(action)


def calibration_message_matrices(
    calibration_tasks: Sequence[ControlTask],
    genome: ControllerGenome,
    correction: CorrectionModel,
) -> dict[str, Tensor]:
    """Build typed messages without reading utility or optimal-action fields."""

    rows: dict[str, list[list[float]]] = {node: [] for node in MESH_NODES}
    soundness_vocab = ("env_sound_dead", "model_sound_dead", "experience_sound_dead", "unknown")
    for task in sorted(calibration_tasks, key=lambda value: value.task_id):
        trace = route_genome(task, genome, correction)
        action_order = list(task.actions)
        proposed_index = _action_index(task, trace.proposed_action)
        fallback_index = _action_index(task, trace.fallback_action)
        selected_index = _action_index(task, trace.selected_action)
        trm_scores = [float(task.trm_scores[action]) for action in action_order]
        ldt_scores = [float(task.ldt_scores[action]) for action in action_order]
        exact_mask = [float(action in task.environment_allowed) for action in action_order]
        soft_mask = [float(action in task.soft_candidates) for action in action_order]
        if genome.evidence_mode == "none":
            active_mask = [1.0] * len(action_order)
        elif genome.evidence_mode == "soft":
            active_mask = soft_mask
        elif genome.evidence_mode == "exact":
            active_mask = exact_mask
        else:
            active_mask = [left * right for left, right in zip(exact_mask, soft_mask)]
        soundness = _one_hot(soundness_vocab.index(task.soft_soundness.value))
        alpha = correction.alpha_by_skill[task.skill]
        blended = _blended_scores(task, alpha)
        if genome.fallback_mode == "identical":
            fallback_scores = trm_scores
        elif genome.fallback_mode in {"ldt", "safe_ldt"}:
            fallback_scores = ldt_scores
        else:
            fallback_scores = [float(blended[action]) for action in action_order]

        rows["proposer"].append(trm_scores + _one_hot(proposed_index))
        rows["evidence"].append(
            exact_mask
            + soft_mask
            + active_mask
            + soundness
            + [float(trace.evidence_passed)]
        )
        rows["gate"].append(
            [
                trace.margin,
                float(trace.confidence_passed),
                float(trace.evidence_passed),
                float(trace.accepted),
            ]
        )
        rows["fallback"].append(
            fallback_scores
            + _one_hot(fallback_index)
            + [float(not trace.accepted)]
        )
        rows["executor"].append(
            _one_hot(selected_index) + [float(trace.accepted), float(not trace.accepted)]
        )
    return {
        node: torch.tensor(values, dtype=torch.float64) for node, values in rows.items()
    }


def _spectrum(
    bases: Mapping[str, Tensor],
    edges: Sequence[tuple[str, str]],
    construction: Mapping[str, object],
):
    return laplacian_spectrum(
        sheaf_laplacian(bases, edges=edges),
        zero_tolerance=float(construction["zero_tolerance"]),
        low_band_max=float(construction["low_band_max"]),
        slow_band_max=float(construction["slow_band_max"]),
    )


def _feature_vector(intervention, stability) -> dict[str, float]:
    return {
        "intervention_spectral_gap": intervention.spectral_gap,
        "intervention_low_band_rank": float(intervention.low_band_rank),
        "intervention_slow_mode_rank": float(intervention.slow_mode_rank),
        "stability_spectral_gap": stability.spectral_gap,
        "stability_low_band_rank": float(stability.low_band_rank),
        "stability_slow_mode_rank": float(stability.slow_mode_rank),
    }


def _phase_boundary_distance(eigenvalues: Sequence[float], boundary: float) -> float:
    return min(abs(float(value) - boundary) for value in eigenvalues)


def analyze_genome_geometry(
    calibration_tasks: Sequence[ControlTask],
    genome: ControllerGenome,
    correction: CorrectionModel,
    *,
    construction: Mapping[str, object],
    null_replicates: int,
    null_seed: int,
) -> tuple[dict[str, object], list[dict[str, float]]]:
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
    features = _feature_vector(intervention, stability)
    null_features = []
    for replicate in range(null_replicates):
        shuffled = permuted_bases(
            bases, seed=null_seed + 1000003 * replicate
        )
        null_features.append(
            _feature_vector(
                _spectrum(shuffled, INTERVENTION_EDGES, construction),
                _spectrum(shuffled, STABILITY_EDGES, construction),
            )
        )
    low_band = float(construction["low_band_max"])
    return (
        {
            "features": features,
            "feature_sha256": canonical_sha256(features),
            "stalk_ranks": {node: int(bases[node].shape[1]) for node in MESH_NODES},
            "intervention_spectrum": intervention.to_jsonable(),
            "stability_spectrum": stability.to_jsonable(),
            "intervention_phase_boundary_distance": _phase_boundary_distance(
                intervention.eigenvalues, low_band
            ),
            "stability_phase_boundary_distance": _phase_boundary_distance(
                stability.eigenvalues, low_band
            ),
        },
        null_features,
    )


def evaluate_genome_outcomes(
    evaluation_tasks: Sequence[ControlTask],
    genome: ControllerGenome,
    correction: CorrectionModel,
) -> dict[str, float | int]:
    traces = [route_genome(task, genome, correction) for task in evaluation_tasks]
    selected_utility = [
        float(task.utilities[trace.selected_action])
        for task, trace in zip(evaluation_tasks, traces)
    ]
    safe = [
        trace.selected_action in task.environment_allowed
        for task, trace in zip(evaluation_tasks, traces)
    ]
    accuracy = [
        is_safe and trace.selected_action == task.optimal_action
        for task, trace, is_safe in zip(evaluation_tasks, traces, safe)
    ]
    utility = mean(selected_utility)
    unsafe_rate = mean(not value for value in safe)
    accuracy_rate = mean(accuracy)
    cost = mean(trace.cost for trace in traces)
    return {
        "episodes": len(traces),
        "mean_utility": utility,
        "accuracy": accuracy_rate,
        "unsafe_rate": unsafe_rate,
        "mean_cost": cost,
        "selection_objective": utility
        + 0.25 * accuracy_rate
        - 2.0 * unsafe_rate
        - 0.025 * cost,
        "acceptance_rate": mean(trace.accepted for trace in traces),
        "action_change_rate": mean(
            trace.selected_action != trace.proposed_action for trace in traces
        ),
    }


def fit_ridge_predictor(
    rows: Sequence[Mapping[str, float]],
    targets: Sequence[float],
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
    ridge: float,
) -> RidgePredictor:
    if len(rows) != len(targets) or len(rows) < 2:
        raise ValueError("predictor fit needs equal feature and target rows")
    matrix = torch.tensor(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=torch.float64,
    )
    y = torch.tensor(targets, dtype=torch.float64)
    centers = matrix.mean(dim=0)
    scales = matrix.std(dim=0, unbiased=False)
    scales = torch.where(scales > 1e-12, scales, torch.ones_like(scales))
    standardized = (matrix - centers) / scales
    design = torch.cat(
        (torch.ones((len(rows), 1), dtype=torch.float64), standardized), dim=1
    )
    penalty = torch.eye(design.shape[1], dtype=torch.float64) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = torch.linalg.solve(
        design.T @ design + penalty, design.T @ y
    )
    return RidgePredictor(
        feature_names=tuple(feature_names),
        feature_means=tuple(float(value) for value in centers.tolist()),
        feature_scales=tuple(float(value) for value in scales.tolist()),
        intercept=float(coefficients[0].item()),
        coefficients=tuple(float(value) for value in coefficients[1:].tolist()),
        ridge=float(ridge),
    )


def _task_corpus_sha256(tasks: Sequence[ControlTask]) -> str:
    return canonical_sha256([task.to_jsonable() for task in tasks])


def _top_k_metrics(
    predictions: Sequence[Mapping[str, object]],
    outcomes: Mapping[str, Mapping[str, float | int]],
    *,
    top_k: int,
) -> dict[str, object]:
    predicted = sorted(
        predictions,
        key=lambda row: (-float(row["predicted_selection_objective"]), str(row["genome_id"])),
    )
    actual = sorted(
        predictions,
        key=lambda row: (
            -float(outcomes[str(row["genome_id"])]["selection_objective"]),
            str(row["genome_id"]),
        ),
    )
    selected_ids = [str(row["genome_id"]) for row in predicted[:top_k]]
    actual_ids = [str(row["genome_id"]) for row in actual[:top_k]]
    all_values = [
        float(outcomes[str(row["genome_id"])]["selection_objective"])
        for row in predictions
    ]
    selected_values = [
        float(outcomes[genome_id]["selection_objective"])
        for genome_id in selected_ids
    ]
    return {
        "top_k": top_k,
        "predicted_top_genomes": selected_ids,
        "actual_top_genomes": actual_ids,
        "overlap": len(set(selected_ids) & set(actual_ids)),
        "precision": len(set(selected_ids) & set(actual_ids)) / top_k,
        "mean_selected_objective": mean(selected_values),
        "heldout_mean_objective": mean(all_values),
        "uplift_vs_heldout_mean": mean(selected_values) - mean(all_values),
    }


def _architecture_features(genome: ControllerGenome) -> dict[str, float]:
    features = {
        f"evidence_{value}": float(genome.evidence_mode == value)
        for value in EVIDENCE_MODES
    }
    features.update(
        {
            f"fallback_{value}": float(genome.fallback_mode == value)
            for value in FALLBACK_MODES
        }
    )
    features["confidence_threshold"] = genome.confidence_threshold
    features["confidence_threshold_squared"] = genome.confidence_threshold**2
    return features


def _posthoc_predictor_diagnostic(
    discovery: Sequence[ControllerGenome],
    heldout: Sequence[ControllerGenome],
    discovery_outcomes: Mapping[str, Mapping[str, float | int]],
    heldout_outcomes: Mapping[str, Mapping[str, float | int]],
    feature_rows: Mapping[str, Mapping[str, float]],
    *,
    feature_names: Sequence[str],
    ridge: float,
    top_k: int,
) -> dict[str, object]:
    predictor = fit_ridge_predictor(
        [feature_rows[genome.genome_id] for genome in discovery],
        [
            float(discovery_outcomes[genome.genome_id]["selection_objective"])
            for genome in discovery
        ],
        feature_names=feature_names,
        ridge=ridge,
    )
    predictions = [
        {
            "genome_id": genome.genome_id,
            "predicted_selection_objective": predictor.predict(
                feature_rows[genome.genome_id]
            ),
        }
        for genome in heldout
    ]
    observed = [
        float(heldout_outcomes[genome.genome_id]["selection_objective"])
        for genome in heldout
    ]
    predicted = [float(row["predicted_selection_objective"]) for row in predictions]
    return {
        "status": "post_hoc_after_registered_forward_gate",
        "feature_names": list(feature_names),
        "heldout_spearman_rho": spearman_correlation(predicted, observed),
        "top_k": _top_k_metrics(predictions, heldout_outcomes, top_k=top_k),
        "predictor": predictor.to_jsonable(),
    }


def run_forward_study(registration: Mapping[str, object]) -> dict[str, object]:
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
    genomes = generate_genomes()
    discovery, heldout = split_genomes(genomes)
    if len(genomes) != int(registration["genome_grid"]["expected_genomes"]):
        raise ValueError("registered genome count mismatch")
    if len(discovery) != int(registration["genome_split"]["expected_discovery"]):
        raise ValueError("registered discovery genome count mismatch")
    if len(heldout) != int(registration["genome_split"]["expected_heldout"]):
        raise ValueError("registered heldout genome count mismatch")

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
    geometry_receipt = canonical_sha256(
        {
            "config_sha256": config_sha256,
            "task_corpus_sha256": _task_corpus_sha256(tasks),
            "features": {
                genome_id: value["feature_sha256"]
                for genome_id, value in sorted(geometry.items())
            },
        }
    )

    # Stage 1 outcome reveal is restricted to discovery genomes.
    discovery_outcomes = {
        genome.genome_id: evaluate_genome_outcomes(
            evaluation_tasks, genome, correction
        )
        for genome in discovery
    }
    predictor = fit_ridge_predictor(
        [geometry[genome.genome_id]["features"] for genome in discovery],
        [
            float(discovery_outcomes[genome.genome_id]["selection_objective"])
            for genome in discovery
        ],
        ridge=float(registration["predictor"]["ridge"]),
    )
    predictions = [
        {
            **genome.to_jsonable(),
            "feature_sha256": geometry[genome.genome_id]["feature_sha256"],
            "predicted_selection_objective": predictor.predict(
                geometry[genome.genome_id]["features"]
            ),
        }
        for genome in heldout
    ]
    prediction_material = {
        "config_sha256": config_sha256,
        "geometry_receipt_sha256": geometry_receipt,
        "predictor": predictor.to_jsonable(),
        "predictions": predictions,
    }
    prediction_receipt = canonical_sha256(prediction_material)

    # Stage 2 reveal occurs only after heldout predictions are sealed above.
    heldout_outcomes = {
        genome.genome_id: evaluate_genome_outcomes(
            evaluation_tasks, genome, correction
        )
        for genome in heldout
    }
    predicted_values = [
        float(row["predicted_selection_objective"]) for row in predictions
    ]
    observed_values = [
        float(heldout_outcomes[str(row["genome_id"])]["selection_objective"])
        for row in predictions
    ]
    observed_rho = spearman_correlation(predicted_values, observed_values)

    null_rhos = []
    for replicate in range(int(null_config["replicates"])):
        null_predictor = fit_ridge_predictor(
            [null_features[genome.genome_id][replicate] for genome in discovery],
            [
                float(discovery_outcomes[genome.genome_id]["selection_objective"])
                for genome in discovery
            ],
            ridge=float(registration["predictor"]["ridge"]),
        )
        null_predictions = [
            null_predictor.predict(null_features[genome.genome_id][replicate])
            for genome in heldout
        ]
        null_rhos.append(spearman_correlation(null_predictions, observed_values))
    matched_p = (1 + sum(value >= observed_rho for value in null_rhos)) / (
        len(null_rhos) + 1
    )
    top_k = _top_k_metrics(
        predictions,
        heldout_outcomes,
        top_k=int(registration["forward_gate"]["top_k"]),
    )
    gate = registration["forward_gate"]
    gate_passed = (
        observed_rho >= float(gate["minimum_spearman_rho"])
        and matched_p <= float(gate["maximum_matched_null_p"])
        and float(top_k["uplift_vs_heldout_mean"]) >= float(gate["minimum_top_k_uplift"])
    )

    architecture_rows = {
        genome.genome_id: _architecture_features(genome) for genome in genomes
    }
    architecture_feature_names = tuple(next(iter(architecture_rows.values())))
    intervention_features = FEATURE_NAMES[:3]
    stability_features = FEATURE_NAMES[3:]
    geometry_feature_rows = {
        genome.genome_id: geometry[genome.genome_id]["features"] for genome in genomes
    }
    posthoc = {
        "categorical_architecture_baseline": _posthoc_predictor_diagnostic(
            discovery,
            heldout,
            discovery_outcomes,
            heldout_outcomes,
            architecture_rows,
            feature_names=architecture_feature_names,
            ridge=float(registration["predictor"]["ridge"]),
            top_k=int(gate["top_k"]),
        ),
        "intervention_only_ablation": _posthoc_predictor_diagnostic(
            discovery,
            heldout,
            discovery_outcomes,
            heldout_outcomes,
            geometry_feature_rows,
            feature_names=intervention_features,
            ridge=float(registration["predictor"]["ridge"]),
            top_k=int(gate["top_k"]),
        ),
        "stability_only_ablation": _posthoc_predictor_diagnostic(
            discovery,
            heldout,
            discovery_outcomes,
            heldout_outcomes,
            geometry_feature_rows,
            feature_names=stability_features,
            ridge=float(registration["predictor"]["ridge"]),
            top_k=int(gate["top_k"]),
        ),
    }

    boundary = sorted(
        heldout,
        key=lambda genome: min(
            float(geometry[genome.genome_id]["intervention_phase_boundary_distance"]),
            float(geometry[genome.genome_id]["stability_phase_boundary_distance"]),
        ),
    )[: int(registration["forward_gate"]["top_k"])]
    result: dict[str, object] = {
        "schema": "controller_mesh_sheaf_forward_v1",
        "study_id": registration["study_id"],
        "config_sha256": config_sha256,
        "task_corpus": {
            "sha256": _task_corpus_sha256(tasks),
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
        },
        "stage_receipts": {
            "calibration_geometry_sha256": geometry_receipt,
            "discovery_outcomes_sha256": canonical_sha256(discovery_outcomes),
            "heldout_prediction_sha256": prediction_receipt,
            "heldout_outcomes_sha256": canonical_sha256(heldout_outcomes),
            "ordering": [
                "calibration_geometry_sealed",
                "discovery_outcomes_revealed",
                "heldout_predictions_sealed",
                "heldout_outcomes_revealed",
            ],
        },
        "predictor": predictor.to_jsonable(),
        "predictions": predictions,
        "heldout_outcomes": heldout_outcomes,
        "discovery_outcomes": discovery_outcomes,
        "geometry": geometry,
        "forward_result": {
            "heldout_spearman_rho": observed_rho,
            "matched_null_mean_rho": mean(null_rhos),
            "matched_null_one_sided_p": matched_p,
            "matched_null_replicates": len(null_rhos),
            "top_k": top_k,
            "gate_passed": gate_passed,
            "gate": gate,
        },
        "posthoc_diagnostics": posthoc,
        "phase_boundary_panel": [
            {
                "genome_id": genome.genome_id,
                "minimum_boundary_distance": min(
                    float(geometry[genome.genome_id]["intervention_phase_boundary_distance"]),
                    float(geometry[genome.genome_id]["stability_phase_boundary_distance"]),
                ),
                "selection_objective": heldout_outcomes[genome.genome_id][
                    "selection_objective"
                ],
            }
            for genome in boundary
        ],
        "matched_null": registration["matched_null"],
        "claim_boundary": registration["claim_boundary"],
    }
    result["analysis_receipt_sha256"] = canonical_sha256(result)
    return result


def markdown_report(result: Mapping[str, object]) -> str:
    forward = result["forward_result"]
    top = forward["top_k"]
    lines = [
        "# Forward Spectral Prediction of Controller Genomes",
        "",
        f"Protocol: `{result['study_id']}`.",
        "",
        "## Staging",
        "",
        (
            "Calibration geometry for all genomes was sealed before discovery outcomes were "
            "revealed. Held-out predictions were then sealed before held-out outcomes were "
            "computed."
        ),
        "",
        f"- genomes: `{result['genome_counts']['total']}`",
        f"- discovery / heldout: `{result['genome_counts']['discovery']} / {result['genome_counts']['heldout']}`",
        f"- calibration / evaluation tasks: `{result['task_corpus']['calibration_tasks']} / {result['task_corpus']['evaluation_tasks']}`",
        "",
        "## Forward Result",
        "",
        f"- held-out Spearman rho: `{forward['heldout_spearman_rho']:+.3f}`",
        f"- matched N0 mean rho: `{forward['matched_null_mean_rho']:+.3f}`",
        f"- matched N0 one-sided p: `{forward['matched_null_one_sided_p']:.4f}`",
        f"- predicted top-{top['top_k']} uplift: `{top['uplift_vs_heldout_mean']:+.4f}`",
        f"- predicted/actual top-{top['top_k']} overlap: `{top['overlap']}/{top['top_k']}`",
        f"- registered forward gate passed: `{forward['gate_passed']}`",
        "",
        "## Held-Out Predictions",
        "",
        "| Genome | Predicted objective | Actual objective | Utility | Unsafe |",
        "|---|---:|---:|---:|---:|",
    ]
    outcomes = result["heldout_outcomes"]
    for row in sorted(
        result["predictions"],
        key=lambda value: -float(value["predicted_selection_objective"]),
    ):
        actual = outcomes[row["genome_id"]]
        lines.append(
            f"| `{row['genome_id']}` | {row['predicted_selection_objective']:+.4f} | "
            f"{actual['selection_objective']:+.4f} | {actual['mean_utility']:.4f} | "
            f"{actual['unsafe_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Post-Hoc Diagnostics",
            "",
            "These diagnostics were not part of the registered gate.",
            "",
            "| Predictor | Held-out rho | Top-k uplift |",
            "|---|---:|---:|",
        ]
    )
    for label, key in (
        ("Categorical genome baseline", "categorical_architecture_baseline"),
        ("Intervention spectrum only", "intervention_only_ablation"),
        ("Stability spectrum only", "stability_only_ablation"),
    ):
        diagnostic = result["posthoc_diagnostics"][key]
        lines.append(
            f"| {label} | {diagnostic['heldout_spearman_rho']:+.3f} | "
            f"{diagnostic['top_k']['uplift_vs_heldout_mean']:+.4f} |"
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
