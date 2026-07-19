"""Serializable contracts shared by planning, execution, and receipts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


TRAINABILITY_ONLY = "trainability-boundary and schedule-cost measurement only"


@dataclass(frozen=True)
class PredictiveDistribution:
    mean: float
    std: float
    lower: float
    upper: float
    units: str


@dataclass(frozen=True)
class TheoryPrediction:
    unique_parameters: int
    applied_parameters_per_token: int
    gradient_applied_parameters_per_token: int
    expanded_visits: int
    gradient_visible_visits: int
    retained_state_edges: int
    gamma_assumption: float
    p_star: float
    stability_functional: float


@dataclass(frozen=True)
class EmpiricalPrediction:
    finite_run_probability: float
    maximum_gradient_norm: PredictiveDistribution
    loss_ratio: PredictiveDistribution
    gamma: PredictiveDistribution
    p_star: PredictiveDistribution
    scale_transfer_probability: float
    boundary_observations: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ResourceForecast:
    scale_rung: str
    unique_parameters: int
    parameter_bytes: int
    optimizer_bytes: int
    estimated_peak_memory_bytes: int
    estimated_flops: int
    estimated_step_seconds: float
    locally_executable: bool
    exclusion_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionPrediction:
    information_gain: float
    falsification_value: float
    diversity: float
    estimated_compute: float
    acquisition_score: float
    decision_kind: str
    rankable: bool
    exclusion_reasons: tuple[str, ...]


@dataclass(frozen=True)
class TrainingProposal:
    proposal_id: str
    proposal_hash: str
    algebra_hash: str
    model_hash: str
    run_hash: str
    name: str
    mutation_family: str
    mutation: Mapping[str, Any]
    scale_rung: str
    model: Mapping[str, Any]
    run: Mapping[str, Any]
    theory: TheoryPrediction
    empirical: EmpiricalPrediction
    resources: tuple[ResourceForecast, ...]
    decision: DecisionPrediction
    promotion_rules: tuple[str, ...]
    stopping_rules: tuple[str, ...]
    claim_scope: str = TRAINABILITY_ONLY

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceProfile:
    ram_bytes: int
    vram_bytes: int
    cpu_pct: int
    io_bytes_per_second: int
    timeout_seconds: int
    max_stage_a_proposals: int


@dataclass(frozen=True)
class PriorEvidence:
    tied_gamma: float
    tied_gamma_r_squared: float
    untied_gamma: float
    untied_gamma_r_squared: float
    p3_confirmed_all_seeds: bool
    boundary_observations: tuple[Mapping[str, Any], ...]
    stable_runs: int
    total_runs: int
    source_receipt_sha256: str
