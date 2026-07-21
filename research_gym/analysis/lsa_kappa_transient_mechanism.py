"""Mechanism diagnostics for visit-alignment transients."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def interference_ratio(sum_norm: float, component_norms: Sequence[float]) -> float:
    """Return squared summed norm over component self-energy.

    Values below one imply a negative aggregate pairwise cross-term because
    ``||sum v||^2 = sum ||v||^2 + 2 sum_{r<s} <v_r, v_s>``.
    """

    if not component_norms:
        raise ValueError("interference ratio requires at least one component")
    values = tuple(float(value) for value in component_norms)
    if sum_norm < 0.0 or not math.isfinite(sum_norm):
        raise ValueError("sum norm must be finite and nonnegative")
    if any(value < 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("component norms must be finite and nonnegative")
    self_energy = sum(value * value for value in values)
    if self_energy == 0.0:
        raise ValueError("interference ratio is undefined at zero self-energy")
    return float(sum_norm) ** 2 / self_energy


def classify_interference(ratio: float, *, tolerance: float = 1e-9) -> str:
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")
    if ratio < 0.0 or not math.isfinite(ratio):
        raise ValueError("interference ratio must be finite and nonnegative")
    if ratio < 1.0 - tolerance:
        return "net_destructive"
    if ratio > 1.0 + tolerance:
        return "net_constructive"
    return "aggregate_orthogonal"


def decompose_estimate(estimate: Mapping[str, Any]) -> dict[str, Any]:
    rounds = int(estimate["visible_rounds"])
    u_norms = tuple(float(value) for value in estimate["u_norms"])
    g_norms = tuple(float(value) for value in estimate["g_norms"])
    if rounds <= 0 or len(u_norms) != rounds or len(g_norms) != rounds:
        raise ValueError("estimate norm vectors must match visible_rounds")
    if max(u_norms) == 0.0 or max(g_norms) == 0.0:
        raise ValueError("kappa decomposition requires nonzero component norms")

    u_sum_norm = float(estimate["u_sum_norm"])
    g_sum_norm = float(estimate["g_sum_norm"])
    scale = math.sqrt(rounds)
    u_max_normalized_sum = u_sum_norm / (scale * max(u_norms))
    g_max_normalized_sum = g_sum_norm / (scale * max(g_norms))
    reconstructed = u_max_normalized_sum * g_max_normalized_sum
    observed = float(estimate["kappa"])
    if not math.isclose(reconstructed, observed, rel_tol=2e-6, abs_tol=1e-9):
        raise ValueError("stored estimate does not reconstruct kappa")

    u_ratio = interference_ratio(u_sum_norm, u_norms)
    g_ratio = interference_ratio(g_sum_norm, g_norms)
    return {
        "sensitivity_interference_ratio": u_ratio,
        "sensitivity_cross_term_fraction": u_ratio - 1.0,
        "sensitivity_interference_class": classify_interference(u_ratio),
        "gradient_interference_ratio": g_ratio,
        "gradient_cross_term_fraction": g_ratio - 1.0,
        "gradient_interference_class": classify_interference(g_ratio),
        "sensitivity_max_normalized_sum": u_max_normalized_sum,
        "gradient_max_normalized_sum": g_max_normalized_sum,
        "reconstructed_kappa": reconstructed,
    }


def training_location(exposure: int, *, batch_size: int, rounds: int) -> dict[str, int]:
    quantum = batch_size * rounds
    if exposure <= 0 or quantum <= 0 or exposure % quantum:
        raise ValueError("exposure must be a positive, reachable training checkpoint")
    step = exposure // quantum
    return {"optimizer_step": step, "training_stream": step - 1}
