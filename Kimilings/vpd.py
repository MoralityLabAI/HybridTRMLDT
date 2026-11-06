"""Scientific reporting over measured expert-local VPD and causal ablations."""
from __future__ import annotations

from collections import defaultdict
from math import isfinite

MEASURES = ("parameter_relative_error", "all_on_mse", "importance_mse",
            "adversarial_mse", "all_off_mse")


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def summarize_vpd(rows: list[dict]) -> dict:
    """Compute reconstruction/sparsity surfaces; never infer task scores."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    if not rows:
        raise ValueError("VPD records required")
    for r in rows:
        fields = ("target_id", "task_family", "activation_source", "seed",
                  "step", "mask_budget", "method", "active_count",
                  "total_components", *MEASURES)
        if any(k not in r for k in fields):
            raise ValueError("Incomplete VPD record")
        if r["activation_source"] not in ("synthetic", "routed_kimi"):
            raise ValueError("Unknown activation provenance")
        if r["activation_source"] == "routed_kimi" and not all(
            r.get(k) for k in ("activation_receipt", "source_model_revision",
                               "reference_forward_receipt", "router_trace_receipt")
        ):
            raise ValueError("Routed Kimi claim requires capture, forward and router receipts")
        if any(type(r[k]) is not int or r[k] < 0 for k in
               ("seed", "step", "mask_budget", "active_count", "total_components")):
            raise ValueError("Counts and steps must be nonnegative integers")
        if r["total_components"] == 0 or r["active_count"] > r["total_components"]:
            raise ValueError("Invalid VPD active-component count")
        if any(type(r[k]) not in (int, float) or not isfinite(r[k])
               or r[k] < 0 for k in MEASURES):
            raise ValueError("VPD errors must be finite nonnegative numbers")
        grouped[(r["target_id"], r["activation_source"], r["task_family"],
                 r["method"], r["mask_budget"])].append(r)
    results = []
    for (target, source, family, method, budget), samples in sorted(grouped.items()):
        metrics = {name: _mean([s[name] for s in samples]) for name in MEASURES}
        l0 = _mean([s["active_count"] for s in samples])
        total = _mean([s["total_components"] for s in samples])
        # 0 means no improvement on all-off; 1 means importance-only
        # reconstruction matched all-on. Preserve values outside [0,1].
        denom = metrics["all_off_mse"] - metrics["all_on_mse"]
        rescue = (metrics["all_off_mse"] - metrics["importance_mse"]) / denom if abs(denom) > 1e-12 else None
        results.append({
            "target": target, "activation_source": source, "family": family,
            "method": method, "mask_budget": budget, "n": len(samples),
            "seeds": sorted({s["seed"] for s in samples}),
            "min_step": min(s["step"] for s in samples),
            "max_step": max(s["step"] for s in samples),
            "active_l0": l0, "total_components": total,
            "active_fraction": l0 / total,
            "importance_rescue_ratio": rescue,
            **metrics,
            "status": "instrument_only" if source == "synthetic"
                      else "routed_expert_local_measurement",
        })
    return {
        "protocol": "KIMILINGS-01", "analysis": "vpd_reconstruction_sparsity",
        "groups": results,
        "limitations": ("No convergence, mechanism identity or whole-model task "
                        "effect can be inferred from summary rows alone."),
    }


def causal_transfer(rows: list[dict]) -> dict:
    """Paired, equal-budget intervention deltas against matched controls."""
    if not rows:
        raise ValueError("Transfer intervention rows required")
    grouped = defaultdict(list)
    for r in rows:
        req = ("component_family", "test_family", "task_id", "seed", "mask_budget",
               "loss_unmasked", "loss_ablated", "loss_control", "control_method",
               "activation_source", "matched_budget")
        if any(k not in r for k in req):
            raise ValueError("Missing causal transfer field")
        if r["activation_source"] != "routed_kimi" or not all(
            r.get(k) for k in ("activation_receipt", "reference_forward_receipt")
        ):
            raise ValueError("Reasoning-conditioned transfer needs real routed activations")
        if r["matched_budget"] is not True:
            raise ValueError("Matched intervention budgets are mandatory")
        if r["control_method"] not in (
            "random_rank_matched", "magnitude", "architectural_unit", "shuffled_importance"
        ):
            raise ValueError("Unknown matched control")
        for field in ("loss_unmasked", "loss_ablated", "loss_control"):
            if type(r[field]) not in (float, int) or not isfinite(r[field]) or r[field] < 0:
                raise ValueError("Losses must be finite and nonnegative")
        if type(r["mask_budget"]) is not int or r["mask_budget"] < 0:
            raise ValueError("Bad intervention budget")
        grouped[(r["component_family"], r["test_family"],
                 r["mask_budget"], r["control_method"])].append(r)
    output = []
    for (component_family, test_family, budget, control), samples in sorted(grouped.items()):
        paired_effect = [s["loss_ablated"] - s["loss_unmasked"] for s in samples]
        baseline_effect = [s["loss_control"] - s["loss_unmasked"] for s in samples]
        output.append({
            "component_family": component_family,
            "test_family": test_family, "mask_budget": budget,
            "control": control, "n": len(samples),
            "ablated_delta": _mean(paired_effect),
            "control_delta": _mean(baseline_effect),
            "differential_delta": _mean(
                [a - b for a, b in zip(paired_effect, baseline_effect)]
            ),
            "seeds": sorted({s["seed"] for s in samples}),
            "status": "expert_local_causal_effect_only",
        })
    return {"protocol": "KIMILINGS-01", "analysis": "paired_transfer", "cells": output}
