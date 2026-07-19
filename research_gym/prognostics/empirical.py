"""Receipt-calibrated small-data predictive distributions."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .censoring import CensoredObservation, ObservationKind
from .features import proposal_features
from .schemas import (
    EmpiricalPrediction,
    PredictiveDistribution,
    PriorEvidence,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ingest_lsa_receipt(path: Path) -> PriorEvidence:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    boundary = tuple(
        {
            "rounds": row["rounds"],
            "observation": CensoredObservation(
                ObservationKind.LEFT_CENSORED,
                upper=float(row["machine_boundary"]),
            ).to_dict(),
            "predicted_boundary": row["predicted_boundary"],
        }
        for row in receipt["boundary_phase"]["outcomes"]
    )
    return PriorEvidence(
        tied_gamma=receipt["gamma_phase"]["fits"]["tied"]["gamma"],
        tied_gamma_r_squared=receipt["gamma_phase"]["fits"]["tied"]["r_squared"],
        untied_gamma=receipt["gamma_phase"]["fits"]["untied"]["gamma"],
        untied_gamma_r_squared=receipt["gamma_phase"]["fits"]["untied"]["r_squared"],
        p3_confirmed_all_seeds=receipt["gamma_phase"]["p3"]["confirmed_all_seeds"],
        boundary_observations=boundary,
        stable_runs=receipt["boundary_phase"]["record_count"],
        total_runs=receipt["boundary_phase"]["record_count"],
        source_receipt_sha256=_sha256(path),
    )


class EmpiricalCalibrator:
    """Transparent hierarchical additive approximation for sparse receipts."""

    def __init__(self, prior: PriorEvidence) -> None:
        self.prior = prior

    def predict(
        self,
        candidate: Mapping[str, Any],
        *,
        unique_parameters: int,
    ) -> EmpiricalPrediction:
        features = proposal_features(candidate)
        tied = features["tied_fraction"]
        gamma_mean = (
            tied * self.prior.tied_gamma
            + (1.0 - tied) * self.prior.untied_gamma
        )
        fit_quality = (
            tied * self.prior.tied_gamma_r_squared
            + (1.0 - tied) * self.prior.untied_gamma_r_squared
        )
        gamma_std = 0.06 + 0.25 * (1.0 - fit_quality)
        scale_log = math.log(unique_parameters / 5_000_000)
        graph_fraction = features["retained_state_edge_fraction"]
        visit_pressure = max(0.0, features["expanded_visit_count"] - 4.0)
        finite_alpha = self.prior.stable_runs + 1
        finite_beta = self.prior.total_runs - self.prior.stable_runs + 1
        finite_probability = finite_alpha / (finite_alpha + finite_beta)
        finite_probability -= 0.015 * visit_pressure + 0.01 * max(0.0, scale_log)
        finite_probability = min(0.995, max(0.05, finite_probability))
        gradient_mean = 4.0 + 2.5 * visit_pressure + 2.0 * graph_fraction + scale_log
        gradient_std = 2.0 + 0.5 * visit_pressure + 0.3 * max(0.0, scale_log)
        loss_mean = 0.75 + 0.05 * visit_pressure + 0.03 * max(0.0, scale_log)
        loss_std = 0.20 + 0.04 * visit_pressure
        p_mean = (1.0 + gamma_mean) / 4.0
        p_std = gamma_std / 4.0 + 0.05
        transfer = max(
            0.05,
            min(
                0.9,
                0.45
                + 0.15 * fit_quality
                - 0.2 * bool(self.prior.boundary_observations)
                - 0.04 * max(0.0, scale_log),
            ),
        )
        return EmpiricalPrediction(
            finite_run_probability=finite_probability,
            maximum_gradient_norm=PredictiveDistribution(
                gradient_mean,
                gradient_std,
                max(0.0, gradient_mean - 1.96 * gradient_std),
                gradient_mean + 1.96 * gradient_std,
                "l2_norm",
            ),
            loss_ratio=PredictiveDistribution(
                loss_mean,
                loss_std,
                max(0.0, loss_mean - 1.96 * loss_std),
                loss_mean + 1.96 * loss_std,
                "final_over_initial",
            ),
            gamma=PredictiveDistribution(
                gamma_mean,
                gamma_std,
                gamma_mean - 1.96 * gamma_std,
                gamma_mean + 1.96 * gamma_std,
                "exponent",
            ),
            p_star=PredictiveDistribution(
                p_mean,
                p_std,
                p_mean - 1.96 * p_std,
                p_mean + 1.96 * p_std,
                "residual_exponent",
            ),
            scale_transfer_probability=transfer,
            boundary_observations=self.prior.boundary_observations,
        )
