"""Registered promotion and winner gates for LSPG architecture discovery."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from lsa.topology import ScheduleTopology, standardized_descriptor_vectors

from .planner import ArchitectureProposal


@dataclass(frozen=True)
class PromotionDecision:
    proposal_id: str
    passed: bool
    mean_macro_delta: float
    positive_seed_count: int
    maximum_task_regression: float
    mean_step_time_ratio: float | None
    reasons: tuple[str, ...]


def extension_required(*, passing_candidates: int, reserve_already_run: bool) -> bool:
    return passing_candidates < 2 and not reserve_already_run


def provisional_gate(
    proposal: ArchitectureProposal,
    control: ArchitectureProposal,
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int] = (401, 409, 419),
    max_task_regression: float = 0.03,
    max_step_time_ratio: float = 1.10,
) -> PromotionDecision:
    candidate_rows = {int(row["seed"]): row for row in rows if row["proposal_id"] == proposal.proposal_id}
    control_rows = {int(row["seed"]): row for row in rows if row["proposal_id"] == control.proposal_id}
    reasons: list[str] = []
    missing = [seed for seed in seeds if seed not in candidate_rows or seed not in control_rows]
    if missing:
        reasons.append("missing_paired_seed_receipts")
    macro_deltas: list[float] = []
    family_deltas: dict[str, list[float]] = {}
    time_ratios: list[float] = []
    for seed in seeds:
        if seed not in candidate_rows or seed not in control_rows:
            continue
        candidate = candidate_rows[seed]
        baseline = control_rows[seed]
        for row in (candidate, baseline):
            if row.get("status") != "completed":
                reasons.append("incomplete_cell")
            if not row.get("integrity_passed", False):
                reasons.append("integrity_failure")
            if not row.get("cleanup_passed", False):
                reasons.append("cleanup_failure")
            if float(row.get("max_gradient_norm", math.inf)) > 100.0:
                reasons.append("gradient_stop")
            ratio = row.get("final_over_initial_loss")
            if ratio is None or float(ratio) > 10.0:
                reasons.append("loss_ratio_stop")
        macro_deltas.append(float(candidate["macro_exact"]) - float(baseline["macro_exact"]))
        for family, value in candidate["by_family"].items():
            family_deltas.setdefault(family, []).append(
                float(value) - float(baseline["by_family"][family])
            )
        if candidate.get("mean_step_seconds") is not None and baseline.get("mean_step_seconds"):
            time_ratios.append(float(candidate["mean_step_seconds"]) / float(baseline["mean_step_seconds"]))
        if candidate.get("unique_parameters") != baseline.get("unique_parameters"):
            reasons.append("parameter_mismatch")
        if candidate.get("estimated_flops") != baseline.get("estimated_flops"):
            reasons.append("flops_mismatch")
    positive = sum(delta > 0.0 for delta in macro_deltas)
    mean_delta = sum(macro_deltas) / len(macro_deltas) if macro_deltas else float("-inf")
    worst = min(
        (sum(values) / len(values) for values in family_deltas.values()), default=float("-inf")
    )
    mean_time = sum(time_ratios) / len(time_ratios) if time_ratios else None
    if positive < 2:
        reasons.append("positive_in_fewer_than_two_seeds")
    if worst < -max_task_regression:
        reasons.append("task_regression_above_three_points")
    if mean_time is None or mean_time > max_step_time_ratio:
        reasons.append("step_time_not_matched")
    return PromotionDecision(
        proposal_id=proposal.proposal_id,
        passed=not reasons,
        mean_macro_delta=mean_delta,
        positive_seed_count=positive,
        maximum_task_regression=worst,
        mean_step_time_ratio=mean_time,
        reasons=tuple(sorted(set(reasons))),
    )


def select_distinct_winners(
    decisions: Sequence[PromotionDecision],
    proposals: Mapping[str, ArchitectureProposal],
    *,
    maximum: int = 2,
    minimum_distance: float = 0.35,
) -> tuple[str, ...]:
    passing = sorted(
        (decision for decision in decisions if decision.passed),
        key=lambda value: (-value.mean_macro_delta, value.proposal_id),
    )
    if not passing:
        return ()
    topologies = {
        proposal_id: ScheduleTopology.from_mapping(proposals[proposal_id].topology)
        for proposal_id in proposals
        if proposals[proposal_id].role == "candidate"
    }
    vectors = standardized_descriptor_vectors(tuple(topologies.values()))
    selected: list[str] = []
    for decision in passing:
        vector = vectors[topologies[decision.proposal_id].topology_hash]
        distances = []
        for existing in selected:
            other = vectors[topologies[existing].topology_hash]
            distances.append(
                math.sqrt(sum((left - right) ** 2 for left, right in zip(vector, other)))
                / math.sqrt(len(vector))
            )
        if not distances or min(distances) >= minimum_distance:
            selected.append(decision.proposal_id)
        if len(selected) == maximum:
            break
    return tuple(selected)


def final_gate(
    provisional: PromotionDecision,
    *,
    bootstrap_lower: float,
    holm_rejected: bool,
) -> PromotionDecision:
    reasons = list(provisional.reasons)
    if bootstrap_lower <= 0.0:
        reasons.append("paired_bootstrap_interval_not_positive")
    if not holm_rejected:
        reasons.append("holm_adjusted_test_not_significant")
    return PromotionDecision(
        proposal_id=provisional.proposal_id,
        passed=not reasons,
        mean_macro_delta=provisional.mean_macro_delta,
        positive_seed_count=provisional.positive_seed_count,
        maximum_task_regression=provisional.maximum_task_regression,
        mean_step_time_ratio=provisional.mean_step_time_ratio,
        reasons=tuple(sorted(set(reasons))),
    )
