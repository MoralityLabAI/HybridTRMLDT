"""Deterministic multi-fidelity proposal planner."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from lsa.canonical import digest
from lsa.scale import solve_decoder_width

from .acquisition import acquisition_decision
from .empirical import EmpiricalCalibrator, ingest_lsa_receipt
from .schemas import (
    ResourceForecast,
    ResourceProfile,
    TRAINABILITY_ONLY,
    TrainingProposal,
)
from .theory import theory_prediction


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LoopSchedulePlanner:
    def __init__(
        self,
        *,
        base_receipt: Path,
        scale_ladder: Mapping[str, Any],
        mutation_space: Mapping[str, Any],
        resource_profile: ResourceProfile,
        promotion_policy: Mapping[str, Any],
        code_commit: str,
    ) -> None:
        self.prior = ingest_lsa_receipt(base_receipt)
        self.base_receipt = base_receipt
        self.scale_ladder = scale_ladder
        self.mutation_space = mutation_space
        self.resource_profile = resource_profile
        self.promotion_policy = promotion_policy
        self.code_commit = code_commit
        self.calibrator = EmpiricalCalibrator(self.prior)

    def _resource_forecasts(
        self, candidate: Mapping[str, Any]
    ) -> tuple[ResourceForecast, ...]:
        forecasts = []
        visits = int(candidate["expanded_visits"])
        sequence = int(candidate["sequence_length"])
        for rung in self.scale_ladder["rungs"]:
            shape = solve_decoder_width(
                int(rung["target_unique_parameters"]),
                physical_blocks=int(candidate["physical_modules"]),
                vocab_size=self.scale_ladder["shape_policy"]["vocab_size"],
                head_dim=self.scale_ladder["shape_policy"]["head_dim"],
                tolerance=self.scale_ladder["shape_policy"]["target_tolerance"],
            )
            parameter_bytes = shape.unique_parameters * 2
            optimizer_bytes = shape.unique_parameters * 8
            activation_bytes = (
                sequence * shape.hidden_size * 2 * max(4, visits * 3)
            )
            peak = parameter_bytes + optimizer_bytes + activation_bytes
            flops = int(
                visits
                * sequence
                * (24 * shape.hidden_size**2 + 4 * sequence * shape.hidden_size)
            )
            step_seconds = flops / self.mutation_space["calibration"]["local_flops_per_second"]
            exclusions = []
            if peak > self.resource_profile.vram_bytes:
                exclusions.append("estimated_vram_exceeds_profile")
            if rung["name"] != "S0":
                exclusions.append("stage_a_executes_s0_only")
            forecasts.append(
                ResourceForecast(
                    scale_rung=rung["name"],
                    unique_parameters=shape.unique_parameters,
                    parameter_bytes=parameter_bytes,
                    optimizer_bytes=optimizer_bytes,
                    estimated_peak_memory_bytes=peak,
                    estimated_flops=flops,
                    estimated_step_seconds=step_seconds,
                    locally_executable=not exclusions,
                    exclusion_reasons=tuple(exclusions),
                )
            )
        return tuple(forecasts)

    def _candidate_records(self) -> list[dict[str, Any]]:
        return [dict(candidate) for candidate in self.mutation_space["candidates"]]

    def propose(self, max_proposals: int) -> tuple[TrainingProposal, ...]:
        candidates = self._candidate_records()
        proposals = []
        seen_algebra = set()
        for index, candidate in enumerate(candidates):
            algebra_identity = digest(
                {
                    key: candidate[key]
                    for key in sorted(candidate)
                    if key not in {"name", "information_gain", "falsification_value"}
                }
            )
            if algebra_identity in seen_algebra:
                continue
            seen_algebra.add(algebra_identity)
            forecasts = self._resource_forecasts(candidate)
            s0 = forecasts[0]
            shape = solve_decoder_width(
                self.scale_ladder["rungs"][0]["target_unique_parameters"],
                physical_blocks=int(candidate["physical_modules"]),
                vocab_size=self.scale_ladder["shape_policy"]["vocab_size"],
                head_dim=self.scale_ladder["shape_policy"]["head_dim"],
                tolerance=self.scale_ladder["shape_policy"]["target_tolerance"],
            )
            tied_fraction = {"fully_tied": 1.0, "grouped": 0.5, "alternating": 0.5, "untied": 0.0}[candidate["tying"]]
            gamma = tied_fraction * self.prior.tied_gamma + (1 - tied_fraction) * self.prior.untied_gamma
            module_parameters = shape.core_parameters // max(1, int(candidate["physical_modules"]))
            theory = theory_prediction(
                candidate,
                unique_parameters=shape.unique_parameters,
                module_parameters=module_parameters,
                gamma_assumption=gamma,
            )
            empirical = self.calibrator.predict(
                candidate, unique_parameters=shape.unique_parameters
            )
            decision_kind = candidate.get("decision_kind", "novel_mutation")
            decision = acquisition_decision(
                information_gain=float(candidate["information_gain"]),
                falsification_value=float(candidate["falsification_value"]),
                diversity=float(candidate["diversity"]),
                estimated_compute=max(s0.estimated_step_seconds, 1e-9),
                decision_kind=decision_kind,
                resource=s0,
                known_invalid=bool(candidate.get("known_invalid", False)),
            )
            model_context = {
                "family": self.scale_ladder["family"],
                "rung": "S0",
                "hidden_size": shape.hidden_size,
                "num_heads": shape.num_heads,
                "physical_blocks": shape.physical_blocks,
                "vocab_size": shape.vocab_size,
                "unique_parameters": shape.unique_parameters,
                "embedding_parameters": shape.embedding_parameters,
                "core_parameters": shape.core_parameters,
                "candidate": candidate,
            }
            model_identity = digest(
                {"algebra_hash": algebra_identity, "model": model_context}
            )
            run_context = {
                "seed": self.promotion_policy["stage_a"]["seed"],
                "exposure_budget": self.promotion_policy["stage_a"]["exposure_budget"],
                "checkpoints_pct": self.promotion_policy["checkpoints_pct"],
                "code_commit": self.code_commit,
            }
            run_identity = digest(
                {"model_hash": model_identity, "run": run_context}
            )
            proposal_hash = digest(
                {
                    "algebra_hash": algebra_identity,
                    "model_hash": model_identity,
                    "run_hash": run_identity,
                    "theory": asdict(theory),
                    "empirical": asdict(empirical),
                    "resources": [asdict(value) for value in forecasts],
                    "decision": asdict(decision),
                    "claim_scope": TRAINABILITY_ONLY,
                }
            )
            proposals.append(
                TrainingProposal(
                    proposal_id=f"LSPG-S0-{index + 1:02d}",
                    proposal_hash=proposal_hash,
                    algebra_hash=algebra_identity,
                    model_hash=model_identity,
                    run_hash=run_identity,
                    name=candidate["name"],
                    mutation_family=candidate["mutation_family"],
                    mutation=candidate,
                    scale_rung="S0",
                    model=model_context,
                    run=run_context,
                    theory=theory,
                    empirical=empirical,
                    resources=forecasts,
                    decision=decision,
                    promotion_rules=tuple(self.promotion_policy["promotion_requires"]),
                    stopping_rules=tuple(self.promotion_policy["stopping_rules"]),
                )
            )
        rankable = [proposal for proposal in proposals if proposal.decision.rankable]
        rankable.sort(
            key=lambda proposal: (
                proposal.decision.decision_kind != "grid_extension",
                -proposal.decision.acquisition_score,
                proposal.proposal_hash,
            )
        )
        return tuple(rankable[:max_proposals])
