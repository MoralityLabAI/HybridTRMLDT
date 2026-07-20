"""Deterministic summaries for the registered LSA v0.1 extension."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Any, Iterable, Sequence

from lsa.kappa_probe import fit_power_law, geometric_mean, spread_ratio


def quantile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("quantile requires values and a probability in [0,1]")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def fit_records(
    records: Iterable[dict[str, Any]],
    *,
    rounds: Sequence[int],
    regime: str,
    exposure: int,
) -> dict[str, Any]:
    by_round: dict[int, list[float]] = defaultdict(list)
    for row in records:
        if row["regime"] == regime and row["exposure"] == exposure:
            by_round[int(row["rounds"])].append(float(row["kappa"]))
    missing = [round_count for round_count in rounds if not by_round[round_count]]
    if missing:
        raise ValueError(f"missing kappa cells for rounds {missing}")
    means = [geometric_mean(by_round[round_count]) for round_count in rounds]
    fit = fit_power_law(rounds, means)
    return {
        **fit.to_dict(),
        "exposure": exposure,
        "regime": regime,
        "geometric_mean_kappa": dict(zip(map(str, rounds), means)),
        "spread_ratio": {
            str(round_count): spread_ratio(by_round[round_count]) for round_count in rounds
        },
    }


def bootstrap_gamma(
    records: Iterable[dict[str, Any]],
    *,
    rounds: Sequence[int],
    regime: str,
    exposure: int,
    seeds: Sequence[int],
    samples: int,
    bootstrap_seed: int,
    interval_mass: float,
) -> dict[str, float | int]:
    if samples <= 0 or not 0.0 < interval_mass < 1.0:
        raise ValueError("invalid bootstrap configuration")
    lookup = {
        (int(row["rounds"]), int(row["seed"])): float(row["kappa"])
        for row in records
        if row["regime"] == regime and row["exposure"] == exposure
    }
    missing = [key for key in ((r, s) for r in rounds for s in seeds) if key not in lookup]
    if missing:
        raise ValueError(f"missing seed cells: {missing}")
    rng = random.Random(bootstrap_seed + exposure)
    draws: list[float] = []
    for _ in range(samples):
        selected = [rng.choice(tuple(seeds)) for _ in seeds]
        kappas = [geometric_mean([lookup[(r, seed)] for seed in selected]) for r in rounds]
        draws.append(fit_power_law(rounds, kappas).gamma)
    tail = (1.0 - interval_mass) / 2.0
    return {
        "samples": samples,
        "lower": quantile(draws, tail),
        "median": quantile(draws, 0.5),
        "upper": quantile(draws, 1.0 - tail),
    }


def classify_gamma_trajectory(config: dict[str, Any], fits: Sequence[dict[str, Any]]) -> str:
    trajectory = config["gamma_trajectory"]
    ordered = sorted(fits, key=lambda row: int(row["exposure"]))
    gamma = [float(row["gamma"]) for row in ordered]
    if len(gamma) < 2:
        raise ValueError("trajectory classification requires at least two checkpoints")
    nondecreasing = sum(later >= earlier for earlier, later in zip(gamma, gamma[1:]))
    if (
        gamma[-1] - gamma[0] >= trajectory["growth_minimum_delta"]
        and nondecreasing >= trajectory["growth_minimum_nondecreasing_transitions"]
    ):
        return "learned_growth"
    if max(gamma) - min(gamma) <= trajectory["invariant_maximum_range"]:
        return "invariant"
    return "mixed"


def score_r16_holdout(config: dict[str, Any], measured_kappa: float) -> dict[str, Any]:
    holdout = config["r16_holdout"]
    prediction = float(holdout["predicted_kappa_r16"])
    ratio = measured_kappa / prediction
    lower, upper = holdout["confirmation_ratio_interval"]
    return {
        "predicted_kappa": prediction,
        "measured_kappa": measured_kappa,
        "ratio": ratio,
        "log_residual": math.log(measured_kappa) - math.log(prediction),
        "confirmed": lower <= ratio <= upper,
    }


def stable_training(training: dict[str, Any]) -> bool:
    initial = training["initial_loss"]
    final = training["final_loss"]
    return bool(
        not training["nonfinite"]
        and initial is not None
        and math.isfinite(final)
        and training["max_gradient_norm"] <= 100.0
        and final / initial <= 10.0
    )


def empirical_boundary(
    rows: Iterable[dict[str, Any]], p_grid: Sequence[float]
) -> dict[str, Any]:
    by_p: dict[float, list[bool]] = defaultdict(list)
    for row in rows:
        by_p[float(row["p"])].append(bool(row["stable"]))
    stable = {float(p): sum(by_p[float(p)]) >= 2 for p in p_grid}
    observed: float | None = None
    for index, p in enumerate(p_grid):
        if all(stable[float(later)] for later in p_grid[index:]):
            observed = float(p)
            break
    if observed is None:
        censoring = "right_censored"
        scientific = f">{max(p_grid):g}"
    elif observed == min(p_grid) and all(stable.values()):
        censoring = "left_censored"
        scientific = f"<={min(p_grid):g}"
    else:
        censoring = "exact_grid"
        scientific = f"{observed:g}"
    return {
        "machine_boundary": observed,
        "censoring": censoring,
        "scientific_boundary": scientific,
        "stable_by_p": {f"{p:g}": stable[float(p)] for p in p_grid},
    }
