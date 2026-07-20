"""Deterministic stage selection and paired final inference."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from lsa.canonical import digest

from .metrics import hierarchical_paired_bootstrap
from .planner import ArchitectureProposal
from .promotion import PromotionDecision, provisional_gate, select_distinct_winners


def _integrity_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if row.get("status") != "completed":
        reasons.append("incomplete_cell")
    if not row.get("integrity_passed", False):
        reasons.append("integrity_failure")
    if not row.get("cleanup_passed", False):
        reasons.append("cleanup_failure")
    if row.get("macro_exact") is None:
        reasons.append("missing_task_metric")
    if float(row.get("max_gradient_norm", math.inf)) > 100.0:
        reasons.append("gradient_stop")
    ratio = row.get("final_over_initial_loss")
    if ratio is None or float(ratio) > 10.0:
        reasons.append("loss_ratio_stop")
    return reasons


def a1_shortlist(
    candidate_ids: Sequence[str],
    proposals: Mapping[str, ArchitectureProposal],
    rows: Sequence[Mapping[str, Any]],
    *,
    maximum: int = 6,
    maximum_task_regression: float = 0.03,
    maximum_step_time_ratio: float = 1.10,
) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...]]:
    by_proposal = {str(row["proposal_id"]): row for row in rows}
    decisions: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        proposal = proposals[candidate_id]
        control_id = proposal.matched_control_id
        reasons: list[str] = []
        if candidate_id not in by_proposal or control_id not in by_proposal:
            reasons.append("missing_paired_receipt")
            decisions.append(
                {"proposal_id": candidate_id, "eligible": False, "macro_delta": None, "reasons": reasons}
            )
            continue
        candidate = by_proposal[candidate_id]
        control = by_proposal[str(control_id)]
        reasons.extend(_integrity_reasons(candidate))
        reasons.extend(_integrity_reasons(control))
        delta = float(candidate["macro_exact"]) - float(control["macro_exact"])
        family_deltas = [
            float(value) - float(control["by_family"][family])
            for family, value in candidate["by_family"].items()
        ]
        worst = min(family_deltas, default=float("-inf"))
        if worst < -maximum_task_regression:
            reasons.append("task_regression_above_three_points")
        if candidate.get("unique_parameters") != control.get("unique_parameters"):
            reasons.append("parameter_mismatch")
        if candidate.get("estimated_flops") != control.get("estimated_flops"):
            reasons.append("flops_mismatch")
        baseline_time = control.get("mean_step_seconds")
        ratio = (
            float(candidate["mean_step_seconds"]) / float(baseline_time)
            if candidate.get("mean_step_seconds") is not None and baseline_time
            else None
        )
        if ratio is None or ratio > maximum_step_time_ratio:
            reasons.append("step_time_not_matched")
        decisions.append(
            {
                "proposal_id": candidate_id,
                "eligible": not reasons,
                "macro_delta": delta,
                "maximum_task_regression": worst,
                "step_time_ratio": ratio,
                "reasons": sorted(set(reasons)),
            }
        )
    ranked = sorted(
        (value for value in decisions if value["eligible"]),
        key=lambda value: (-float(value["macro_delta"]), value["proposal_id"]),
    )
    return tuple(value["proposal_id"] for value in ranked[:maximum]), tuple(decisions)


def provisional_selection(
    candidate_ids: Sequence[str],
    proposals: Mapping[str, ArchitectureProposal],
    rows: Sequence[Mapping[str, Any]],
    *,
    maximum: int,
    structurally_distinct: bool = False,
) -> tuple[tuple[str, ...], tuple[PromotionDecision, ...]]:
    decisions = tuple(
        provisional_gate(
            proposals[candidate_id],
            proposals[str(proposals[candidate_id].matched_control_id)],
            rows,
        )
        for candidate_id in candidate_ids
    )
    if structurally_distinct:
        selected = select_distinct_winners(decisions, proposals, maximum=maximum)
    else:
        selected = tuple(
            value.proposal_id
            for value in sorted(
                (decision for decision in decisions if decision.passed),
                key=lambda value: (-value.mean_macro_delta, value.proposal_id),
            )[:maximum]
        )
    return selected, decisions


def load_prediction_rows(result: Mapping[str, Any], *, root: Path) -> list[dict[str, Any]]:
    train_depth = str(int(result["train_visits"]))
    if train_depth not in result["prediction_artifacts"]:
        raise ValueError("training-depth prediction artifact is missing")
    artifact = result["prediction_artifacts"][train_depth]
    path = Path(artifact["path"])
    if not path.is_absolute():
        path = root / path
    if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
        raise ValueError("prediction artifact hash mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != int(artifact["rows"]):
        raise ValueError("prediction artifact row-count mismatch")
    return rows


def paired_randomization_pvalue(
    deltas: Sequence[float], *, resamples: int = 10_000, seed: int = 977
) -> float:
    if not deltas:
        raise ValueError("paired randomization requires deltas")
    observed = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    exceedances = 0
    for _ in range(resamples):
        permuted = sum(value if rng.random() < 0.5 else -value for value in deltas) / len(deltas)
        exceedances += int(permuted >= observed)
    return (exceedances + 1) / (resamples + 1)


def holm_rejections(pvalues: Mapping[str, float], *, alpha: float = 0.05) -> dict[str, bool]:
    ordered = sorted(pvalues.items(), key=lambda value: (value[1], value[0]))
    rejected = {name: False for name in pvalues}
    for index, (name, pvalue) in enumerate(ordered):
        if pvalue > alpha / (len(ordered) - index):
            break
        rejected[name] = True
    return rejected


def final_inference(
    candidate_ids: Sequence[str],
    proposals: Mapping[str, ArchitectureProposal],
    result_rows: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    bootstrap_resamples: int = 10_000,
) -> dict[str, Any]:
    indexed = {
        (str(row["proposal_id"]), int(row["seed"])): row for row in result_rows
    }
    statistics: dict[str, Any] = {}
    pvalues: dict[str, float] = {}
    for candidate_id in candidate_ids:
        control_id = str(proposals[candidate_id].matched_control_id)
        paired_by_seed: dict[int, list[float]] = {}
        for seed in (401, 409, 419):
            candidate = indexed[(candidate_id, seed)]
            control = indexed[(control_id, seed)]
            candidate_predictions = {
                row["example_id"]: int(row["exact"])
                for row in load_prediction_rows(candidate, root=root)
            }
            control_predictions = {
                row["example_id"]: int(row["exact"])
                for row in load_prediction_rows(control, root=root)
            }
            if candidate_predictions.keys() != control_predictions.keys():
                raise ValueError("paired prediction IDs do not match")
            paired_by_seed[seed] = [
                candidate_predictions[key] - control_predictions[key]
                for key in sorted(candidate_predictions)
            ]
        mean, lower, upper = hierarchical_paired_bootstrap(
            paired_by_seed, resamples=bootstrap_resamples
        )
        flattened = [value for values in paired_by_seed.values() for value in values]
        pvalue = paired_randomization_pvalue(flattened, resamples=bootstrap_resamples)
        pvalues[candidate_id] = pvalue
        statistics[candidate_id] = {
            "paired_examples": len(flattened),
            "bootstrap_mean": mean,
            "bootstrap_lower_95": lower,
            "bootstrap_upper_95": upper,
            "paired_randomization_p_one_sided": pvalue,
        }
    rejected = holm_rejections(pvalues)
    winners = []
    for candidate_id, value in statistics.items():
        value["holm_rejected"] = rejected[candidate_id]
        value["passed"] = value["bootstrap_lower_95"] > 0.0 and rejected[candidate_id]
        if value["passed"]:
            winners.append(candidate_id)
    payload = {
        "schema_version": 1,
        "candidate_statistics": statistics,
        "qualifying_architectures": sorted(winners),
        "qualifying_count": len(winners),
        "claim_scope": "held-out matched-control transfer within the registered campaign only",
    }
    payload["analysis_hash"] = digest(payload)
    return payload
