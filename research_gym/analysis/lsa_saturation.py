"""Registered functional-form analysis for the LSA saturation addendum."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import defaultdict
import math
import random
from typing import Any, Iterable, Mapping, Sequence

from lsa.kappa_probe import fit_power_law, geometric_mean
from research_gym.analysis.lsa_v0_1 import quantile


@dataclass(frozen=True)
class CurveFit:
    model: str
    parameters: Mapping[str, float]
    fitted: tuple[float, ...]
    sse: float
    r_squared: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_xy(rounds: Sequence[int], kappas: Sequence[float]) -> None:
    if len(rounds) != len(kappas) or len(rounds) < 3:
        raise ValueError("curve fitting requires at least three paired observations")
    if any(rounds[index] >= rounds[index + 1] for index in range(len(rounds) - 1)):
        raise ValueError("round counts must be strictly increasing")
    if any(value <= 0 or not math.isfinite(value) for value in (*rounds, *kappas)):
        raise ValueError("rounds and kappas must be finite and positive")


def _fit_quality(actual: Sequence[float], fitted: Sequence[float]) -> tuple[float, float]:
    mean = sum(actual) / len(actual)
    sse = sum((left - right) ** 2 for left, right in zip(actual, fitted))
    total = sum((value - mean) ** 2 for value in actual)
    r_squared = 1.0 if total == 0.0 and sse == 0.0 else 1.0 - sse / total
    return sse, r_squared


def fit_logarithmic(rounds: Sequence[int], kappas: Sequence[float]) -> CurveFit:
    """Fit ``a + b log(R)`` with the registered nonnegative-slope constraint."""

    _validate_xy(rounds, kappas)
    x = [math.log(value) for value in rounds]
    x_mean = sum(x) / len(x)
    y_mean = sum(kappas) / len(kappas)
    denominator = sum((value - x_mean) ** 2 for value in x)
    slope = sum((left - x_mean) * (right - y_mean) for left, right in zip(x, kappas))
    slope /= denominator
    if slope < 0.0:
        slope = 0.0
    intercept = y_mean - slope * x_mean
    fitted = tuple(intercept + slope * value for value in x)
    sse, r_squared = _fit_quality(kappas, fitted)
    return CurveFit(
        model="logarithmic",
        parameters={"a": intercept, "b": slope},
        fitted=fitted,
        sse=sse,
        r_squared=r_squared,
    )


def fit_saturating_exponential(
    rounds: Sequence[int],
    kappas: Sequence[float],
    *,
    tau_minimum: float,
    tau_maximum: float,
    tau_points: int,
) -> CurveFit:
    """Fit ``K - A exp(-R/tau)`` over the frozen deterministic tau grid."""

    _validate_xy(rounds, kappas)
    if tau_minimum <= 0 or tau_maximum <= tau_minimum or tau_points < 2:
        raise ValueError("invalid tau grid")
    maximum_kappa = max(kappas)
    log_ratio = math.log(tau_maximum / tau_minimum)
    candidates: list[tuple[tuple[float, float, float, float], CurveFit]] = []
    for index in range(tau_points):
        tau = tau_minimum * math.exp(log_ratio * index / (tau_points - 1))
        x = [-math.exp(-round_count / tau) for round_count in rounds]
        x_mean = sum(x) / len(x)
        y_mean = sum(kappas) / len(kappas)
        denominator = sum((value - x_mean) ** 2 for value in x)
        if denominator <= 1e-24:
            continue
        amplitude = sum(
            (left - x_mean) * (right - y_mean) for left, right in zip(x, kappas)
        ) / denominator
        ceiling = y_mean - amplitude * x_mean
        if amplitude < 0.0 or ceiling < maximum_kappa:
            continue
        fitted = tuple(ceiling + amplitude * value for value in x)
        sse, r_squared = _fit_quality(kappas, fitted)
        fit = CurveFit(
            model="saturating_exponential",
            parameters={"K": ceiling, "A": amplitude, "tau": tau},
            fitted=fitted,
            sse=sse,
            r_squared=r_squared,
        )
        candidates.append(((sse, tau, ceiling, amplitude), fit))
    if not candidates:
        raise ValueError("no saturating exponential satisfies the registered constraints")
    return min(candidates, key=lambda item: item[0])[1]


def predict_curve(fit: CurveFit, rounds: float) -> float:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if fit.model == "logarithmic":
        return fit.parameters["a"] + fit.parameters["b"] * math.log(rounds)
    if fit.model == "saturating_exponential":
        return fit.parameters["K"] - fit.parameters["A"] * math.exp(
            -rounds / fit.parameters["tau"]
        )
    raise ValueError(f"unsupported curve model: {fit.model}")


def fit_registered_models(
    rounds: Sequence[int], kappas: Sequence[float], config: Mapping[str, Any]
) -> dict[str, CurveFit]:
    tau = config["candidate_models"]["saturating_exponential"]["tau_grid"]
    return {
        "saturating_exponential": fit_saturating_exponential(
            rounds,
            kappas,
            tau_minimum=float(tau["minimum"]),
            tau_maximum=float(tau["maximum"]),
            tau_points=int(tau["points"]),
        ),
        "logarithmic": fit_logarithmic(rounds, kappas),
    }


def geometric_means_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    rounds: Sequence[int],
    regime: str,
    exposure: int,
) -> tuple[list[float], dict[int, list[float]]]:
    values: dict[int, list[float]] = defaultdict(list)
    for row in records:
        if row["regime"] == regime and int(row["exposure"]) == exposure:
            values[int(row["rounds"])].append(float(row["kappa"]))
    missing = [round_count for round_count in rounds if not values[round_count]]
    if missing:
        raise ValueError(f"missing registered round cells: {missing}")
    return [geometric_mean(values[round_count]) for round_count in rounds], values


def bootstrap_predictions(
    values: Mapping[int, Sequence[float]],
    *,
    rounds: Sequence[int],
    prediction_round: int,
    config: Mapping[str, Any],
) -> dict[str, dict[str, float | int]]:
    specification = config["fit_uncertainty"]
    samples = int(specification["samples"])
    interval_mass = float(specification["interval_mass"])
    if samples <= 0 or not 0.0 < interval_mass < 1.0:
        raise ValueError("invalid bootstrap specification")
    rng = random.Random(int(specification["seed"]))
    predictions: dict[str, list[float]] = {
        "saturating_exponential": [],
        "logarithmic": [],
    }
    for _ in range(samples):
        means = []
        for round_count in rounds:
            source = tuple(float(value) for value in values[round_count])
            draw = [rng.choice(source) for _ in source]
            means.append(geometric_mean(draw))
        fits = fit_registered_models(rounds, means, config)
        for name, fit in fits.items():
            predictions[name].append(predict_curve(fit, prediction_round))
    tail = (1.0 - interval_mass) / 2.0
    return {
        name: {
            "samples": samples,
            "lower": quantile(draws, tail),
            "median": quantile(draws, 0.5),
            "upper": quantile(draws, 1.0 - tail),
        }
        for name, draws in predictions.items()
    }


def local_effective_gammas(
    rounds: Sequence[int], kappas: Sequence[float]
) -> list[dict[str, float | int]]:
    _validate_xy(rounds, kappas)
    values = []
    for first_r, second_r, first_k, second_k in zip(
        rounds, rounds[1:], kappas, kappas[1:]
    ):
        values.append(
            {
                "from_rounds": first_r,
                "to_rounds": second_r,
                "gamma": math.log(second_k / first_k) / math.log(second_r / first_r),
            }
        )
    return values


def choose_holdout_model(
    measured_kappa: float,
    predictions: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    errors = {name: abs(measured_kappa - value) for name, value in predictions.items()}
    discriminator = config["discriminator"]
    absolute = float(discriminator["minimum_absolute_error_advantage"])
    ratio_limit = float(discriminator["maximum_winner_to_loser_error_ratio"])
    sat_error = errors["saturating_exponential"]
    log_error = errors["logarithmic"]

    def wins(winner: float, loser: float) -> bool:
        ratio = winner / loser if loser > 0.0 else math.inf
        return winner + absolute <= loser and ratio <= ratio_limit

    if wins(sat_error, log_error):
        winner = "saturating_exponential"
    elif wins(log_error, sat_error):
        winner = "logarithmic"
    else:
        winner = "unresolved"
    return {"winner": winner, "errors": errors}


def score_saturation_support(
    *,
    winner: str,
    saturating_fit: CurveFit,
    kappas: Mapping[int, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    rule = config["saturation_support_rule"]
    denominator = kappas[8] - kappas[2]
    tail_growth = kappas[64] - kappas[16]
    tail_fraction = tail_growth / denominator if denominator > 0.0 else math.inf
    checks = {
        "discriminator": winner == rule["required_discriminator"],
        "fit_r_squared": saturating_fit.r_squared >= float(rule["minimum_fit_r_squared"]),
        "r64_ceiling": kappas[64] <= float(rule["maximum_r64_kappa"]),
        "tail_growth_fraction": tail_fraction <= float(rule["maximum_tail_growth_fraction"]),
        "tail_growth_floor": tail_growth >= float(rule["minimum_tail_growth"]),
    }
    return {
        "supported": all(checks.values()),
        "checks": checks,
        "tail_growth": tail_growth,
        "tail_growth_fraction": tail_fraction,
    }


def legacy_power_law(rounds: Sequence[int], kappas: Sequence[float]) -> dict[str, Any]:
    return fit_power_law(rounds, kappas).to_dict()
