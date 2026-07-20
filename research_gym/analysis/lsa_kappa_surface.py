"""Registered analysis for the loop-schedule kappa-by-training surface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import defaultdict
import math
import random
from typing import Any, Iterable, Mapping, Sequence

from lsa.kappa_probe import fit_power_law, geometric_mean
from research_gym.analysis.lsa_v0_1 import quantile


MODEL_NAMES = ("separable", "smooth_interaction", "r64_change_point")


@dataclass(frozen=True)
class SurfaceFit:
    model: str
    parameters: tuple[float, ...]
    parameter_names: tuple[str, ...]
    fitted: tuple[float, ...]
    sse: float
    r_squared: float
    aicc: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def surface_coordinates(rounds: int, exposure: int) -> tuple[float, float, float]:
    if rounds <= 0 or exposure < 0:
        raise ValueError("surface coordinates require positive rounds and nonnegative exposure")
    return (
        math.log2(rounds / 16.0),
        math.log2(1.0 + exposure / 512.0),
        1.0 if rounds == 64 else 0.0,
    )


def _design_row(model: str, rounds: int, exposure: int) -> tuple[float, ...]:
    x, u, i64 = surface_coordinates(rounds, exposure)
    if model == "separable":
        return (1.0, x, u)
    if model == "smooth_interaction":
        return (1.0, x, u, x * u)
    if model == "r64_change_point":
        return (1.0, x, u, x * u, i64, i64 * u)
    raise ValueError(f"unsupported surface model: {model}")


def _parameter_names(model: str) -> tuple[str, ...]:
    names = {
        "separable": ("a", "b", "c"),
        "smooth_interaction": ("a", "b", "c", "d"),
        "r64_change_point": ("a", "b", "c", "d", "q", "z"),
    }
    try:
        return names[model]
    except KeyError as error:
        raise ValueError(f"unsupported surface model: {model}") from error


def _solve(matrix: Sequence[Sequence[float]], target: Sequence[float]) -> list[float]:
    size = len(target)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("linear solve requires a square matrix")
    augmented = [list(map(float, row)) + [float(value)] for row, value in zip(matrix, target)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-12:
            raise ValueError("surface design is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def fit_surface(model: str, cells: Sequence[Mapping[str, float | int]]) -> SurfaceFit:
    names = _parameter_names(model)
    if len(cells) <= len(names) + 1:
        raise ValueError("AICc requires more cells than parameters plus one")
    design = [
        _design_row(model, int(cell["rounds"]), int(cell["exposure"]))
        for cell in cells
    ]
    response = [math.log(float(cell["kappa"])) for cell in cells]
    if any(not math.isfinite(value) for value in response):
        raise ValueError("surface response must be finite and positive")
    count = len(names)
    gram = [
        [sum(row[left] * row[right] for row in design) for right in range(count)]
        for left in range(count)
    ]
    cross = [sum(row[index] * value for row, value in zip(design, response)) for index in range(count)]
    parameters = tuple(_solve(gram, cross))
    fitted = tuple(sum(left * right for left, right in zip(row, parameters)) for row in design)
    mean = sum(response) / len(response)
    sse = sum((actual - predicted) ** 2 for actual, predicted in zip(response, fitted))
    total = sum((actual - mean) ** 2 for actual in response)
    r_squared = 1.0 if total == 0.0 and sse == 0.0 else 1.0 - sse / total
    n = len(response)
    k = len(parameters)
    variance = max(sse / n, 1e-300)
    aic = n * math.log(variance) + 2.0 * k
    aicc = aic + 2.0 * k * (k + 1.0) / (n - k - 1.0)
    return SurfaceFit(model, parameters, names, fitted, sse, r_squared, aicc)


def predict_surface(fit: SurfaceFit, rounds: int, exposure: int) -> float:
    row = _design_row(fit.model, rounds, exposure)
    return math.exp(sum(left * right for left, right in zip(row, fit.parameters)))


def blocked_cv_rmse(model: str, cells: Sequence[Mapping[str, float | int]]) -> float:
    exposures = sorted({int(cell["exposure"]) for cell in cells})
    squared_errors: list[float] = []
    for exposure in exposures:
        training = [cell for cell in cells if int(cell["exposure"]) != exposure]
        held_out = [cell for cell in cells if int(cell["exposure"]) == exposure]
        fit = fit_surface(model, training)
        for cell in held_out:
            predicted = math.log(predict_surface(fit, int(cell["rounds"]), exposure))
            actual = math.log(float(cell["kappa"]))
            squared_errors.append((actual - predicted) ** 2)
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def compare_surface_models(
    cells: Sequence[Mapping[str, float | int]], config: Mapping[str, Any]
) -> dict[str, Any]:
    fits = {name: fit_surface(name, cells) for name in MODEL_NAMES}
    cv = {name: blocked_cv_rmse(name, cells) for name in MODEL_NAMES}
    ranked = sorted(MODEL_NAMES, key=lambda name: (fits[name].aicc, name))
    best, runner_up = ranked[:2]
    model_specification = config.get("candidate_models", config.get("surface_models"))
    if model_specification is None:
        raise ValueError("surface model specification is missing")
    comparison = model_specification["comparison"]
    aicc_advantage = fits[runner_up].aicc - fits[best].aicc
    cv_ratio = cv[best] / cv[runner_up] if cv[runner_up] > 0.0 else math.inf
    winner = (
        best
        if aicc_advantage >= 4.0 and cv_ratio <= 0.9
        else "unresolved"
    )
    return {
        "winner": winner,
        "aicc_rank": ranked,
        "aicc_advantage": aicc_advantage,
        "blocked_cv_rmse_ratio": cv_ratio,
        "fits": {name: fit.to_dict() for name, fit in fits.items()},
        "blocked_cv_rmse": cv,
        "registered_rule": comparison["winner_rule"],
    }


def geometric_surface(
    records: Iterable[Mapping[str, Any]], *, regime: str
) -> tuple[list[dict[str, float | int]], dict[tuple[int, int], dict[int, float]]]:
    values: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
    for row in records:
        if row["regime"] != regime:
            continue
        key = (int(row["rounds"]), int(row["exposure"]))
        seed = int(row["seed"])
        if seed in values[key]:
            raise ValueError(f"duplicate surface cell seed: {key}, {seed}")
        values[key][seed] = float(row["kappa"])
    cells = [
        {"rounds": rounds, "exposure": exposure, "kappa": geometric_mean(seed_values.values())}
        for (rounds, exposure), seed_values in sorted(values.items())
    ]
    return cells, values


def curvature_intervals(
    values: Mapping[tuple[int, int], Mapping[int, float]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    specification = config["depth_curvature"]
    frozen = config["frozen_training"]
    exposures = [
        int(value)
        for value in frozen.get("measurement_exposures", frozen.get("surface_exposures", ()))
    ]
    seeds = [int(value) for value in config["frozen_training"]["seeds"]]
    rng = random.Random(int(specification["bootstrap_seed"]))
    samples = int(specification["bootstrap_samples"])
    mass = float(specification["interval_mass"])
    tail = (1.0 - mass) / 2.0
    output = []
    for exposure in exposures:
        for rounds in (16, 32, 64):
            missing = set(seeds) - set(values[(rounds, exposure)])
            if missing:
                raise ValueError(f"missing curvature seeds at R={rounds}, E={exposure}: {sorted(missing)}")
        means = {
            rounds: geometric_mean(values[(rounds, exposure)].values())
            for rounds in (16, 32, 64)
        }
        point = math.log(means[64]) - 2.0 * math.log(means[32]) + math.log(means[16])
        draws = []
        for _ in range(samples):
            sampled_seeds = [rng.choice(seeds) for _ in seeds]
            sampled_means = {
                rounds: geometric_mean(
                    [values[(rounds, exposure)][seed] for seed in sampled_seeds]
                )
                for rounds in (16, 32, 64)
            }
            draws.append(
                math.log(sampled_means[64])
                - 2.0 * math.log(sampled_means[32])
                + math.log(sampled_means[16])
            )
        output.append(
            {
                "exposure": exposure,
                "curvature": point,
                "lower": quantile(draws, tail),
                "median": quantile(draws, 0.5),
                "upper": quantile(draws, 1.0 - tail),
                "samples": samples,
            }
        )
    return output


def curvature_onset(intervals: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> int | None:
    specification = config["depth_curvature"]
    threshold = float(specification["transition_threshold"])
    equivalence_upper = float(specification["practical_equivalence_region"][0])
    eligible = {
        int(value)
        for value in specification.get("onset_candidates", (512, 1024, 2048))
    }
    for row in sorted(intervals, key=lambda value: int(value["exposure"])):
        exposure = int(row["exposure"])
        if (
            exposure in eligible
            and float(row["curvature"]) <= threshold
            and float(row["upper"]) < equivalence_upper
        ):
            return exposure
    return None


def depth_gammas(cells: Sequence[Mapping[str, float | int]]) -> list[dict[str, Any]]:
    by_exposure: dict[int, dict[int, float]] = defaultdict(dict)
    for cell in cells:
        by_exposure[int(cell["exposure"])][int(cell["rounds"])] = float(cell["kappa"])
    output = []
    for exposure, values in sorted(by_exposure.items()):
        if set(values) != {16, 32, 64}:
            raise ValueError(f"incomplete depth fit at exposure {exposure}")
        fit = fit_power_law([16, 32, 64], [values[16], values[32], values[64]])
        output.append({"exposure": exposure, **fit.to_dict()})
    return output


def score_untied_control(cells: Sequence[Mapping[str, float | int]], config: Mapping[str, Any]) -> dict[str, Any]:
    fits = depth_gammas(cells)
    equivalent = [abs(float(row["gamma"])) <= 0.1 for row in fits]
    terminal = next(row for row in fits if int(row["exposure"]) == 4096)
    passed = sum(equivalent) >= 4 and abs(float(terminal["gamma"])) <= 0.1
    return {"passed": passed, "equivalent_checkpoints": sum(equivalent), "fits": fits}


def score_terminal_replication(
    records: Iterable[Mapping[str, Any]],
    prior: Mapping[tuple[str, int, int], float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    tolerance = float(config["terminal_replication"]["maximum_absolute_log_ratio"])
    comparisons = []
    for row in records:
        if int(row["exposure"]) != 4096:
            continue
        key = (str(row["regime"]), int(row["rounds"]), int(row["seed"]))
        if key not in prior:
            raise ValueError(f"missing prior terminal cell: {key}")
        difference = abs(math.log(float(row["kappa"]) / float(prior[key])))
        comparisons.append(
            {
                "regime": key[0],
                "rounds": key[1],
                "seed": key[2],
                "prior_kappa": prior[key],
                "new_kappa": float(row["kappa"]),
                "absolute_log_ratio": difference,
                "passed": difference <= tolerance,
            }
        )
    return {
        "passed": len(comparisons) == 18 and all(row["passed"] for row in comparisons),
        "maximum_absolute_log_ratio": max(row["absolute_log_ratio"] for row in comparisons),
        "comparisons": comparisons,
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman correlation requires paired values")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left_ranks, right_ranks)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left_ranks)
        * sum((value - right_mean) ** 2 for value in right_ranks)
    )
    return numerator / denominator if denominator > 0.0 else math.nan


def score_gradient_coupling(
    records: Sequence[Mapping[str, Any]], onset: int | None, config: Mapping[str, Any]
) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    kappa: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in records:
        exposure = int(row["exposure"])
        if exposure == 0:
            continue
        key = (str(row["regime"]), int(row["rounds"]), exposure)
        grouped[key].append(float(row["interval_gradient"]["maximum"]))
        kappa[key].append(float(row["kappa"]))
    ratios: dict[tuple[int, int], float] = {}
    for rounds in (16, 32, 64):
        for exposure in (512, 1024, 2048, 4096):
            tied = geometric_mean(grouped[("tied", rounds, exposure)])
            untied = geometric_mean(grouped[("untied", rounds, exposure)])
            ratios[(rounds, exposure)] = tied / untied
    co_localized = False
    onset_ratios = None
    if onset is not None:
        onset_ratios = {str(rounds): ratios[(rounds, onset)] for rounds in (16, 32, 64)}
        threshold = float(config["gradient_coupling"]["support_threshold"])
        co_localized = (
            onset_ratios["64"] >= threshold
            and onset_ratios["64"] > onset_ratios["16"]
            and onset_ratios["64"] > onset_ratios["32"]
        )
    negative_delta: list[float] = []
    log_stress: list[float] = []
    for rounds in (16, 32, 64):
        previous = geometric_mean(
            [
                float(row["kappa"])
                for row in records
                if row["regime"] == "tied"
                and int(row["rounds"]) == rounds
                and int(row["exposure"]) == 0
            ]
        )
        for exposure in (512, 1024, 2048, 4096):
            current = geometric_mean(kappa[("tied", rounds, exposure)])
            negative_delta.append(-math.log(current / previous))
            log_stress.append(math.log(ratios[(rounds, exposure)]))
            previous = current
    return {
        "co_localized": co_localized,
        "onset_ratios": onset_ratios,
        "stress_ratios": {
            f"R{rounds}_E{exposure}": value
            for (rounds, exposure), value in sorted(ratios.items())
        },
        "spearman_negative_delta_log_kappa_vs_log_stress": spearman(negative_delta, log_stress),
    }


def classify_surface(
    *,
    winner: str,
    onset: int | None,
    gradient_supported: bool,
    replication_passed: bool,
    untied_passed: bool,
) -> str:
    if not replication_passed:
        return "instrument_drift"
    if not untied_passed:
        return "control_failure"
    if winner == "r64_change_point" and onset is not None:
        return (
            "depth_training_transition"
            if gradient_supported
            else "geometric_transition_without_gradient_support"
        )
    if winner in {"separable", "smooth_interaction"} and onset is None:
        return "smooth_learning"
    return "form_unresolved"
