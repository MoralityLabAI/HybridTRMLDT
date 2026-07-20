from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from research_gym.analysis.lsa_saturation import (
    bootstrap_predictions,
    choose_holdout_model,
    fit_logarithmic,
    fit_registered_models,
    fit_saturating_exponential,
    local_effective_gammas,
    predict_curve,
    score_saturation_support,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads(
        (ROOT / "configs" / "loop_schedule_algebra_saturation_addendum_v1.json").read_text()
    )


def test_saturation_addendum_config_rehashes_registration() -> None:
    path = ROOT / "configs" / "loop_schedule_algebra_saturation_addendum_v1.json"
    registration = json.loads(
        (
            ROOT
            / "configs"
            / "loop_schedule_algebra_saturation_addendum_v1_registration.json"
        ).read_text()
    )

    assert hashlib.sha256(path.read_bytes()).hexdigest() == registration["config_sha256"]
    assert not registration["outcomes_observed"]
    assert registration["untouched_holdout_round"] == 64


def test_registered_curve_fits_recover_synthetic_forms() -> None:
    rounds = [2, 4, 8, 16, 32]
    sat_kappa = [2.8 - 2.0 * math.exp(-value / 6.0) for value in rounds]
    log_kappa = [0.7 + 0.5 * math.log(value) for value in rounds]

    sat_fit = fit_saturating_exponential(
        rounds,
        sat_kappa,
        tau_minimum=0.25,
        tau_maximum=256.0,
        tau_points=4097,
    )
    log_fit = fit_logarithmic(rounds, log_kappa)

    assert sat_fit.r_squared > 0.999999
    assert sat_fit.parameters["K"] == pytest.approx(2.8, rel=0.002)
    assert predict_curve(sat_fit, 64) == pytest.approx(2.8, rel=0.002)
    assert log_fit.r_squared == pytest.approx(1.0)
    assert predict_curve(log_fit, 64) == pytest.approx(0.7 + 0.5 * math.log(64))


def test_holdout_discriminator_requires_both_registered_margins() -> None:
    config = _config()

    clear = choose_holdout_model(
        2.6,
        {"saturating_exponential": 2.61, "logarithmic": 2.9},
        config,
    )
    too_close = choose_holdout_model(
        2.6,
        {"saturating_exponential": 2.61, "logarithmic": 2.64},
        config,
    )

    assert clear["winner"] == "saturating_exponential"
    assert too_close["winner"] == "unresolved"


def test_saturation_support_requires_holdout_and_tail_checks() -> None:
    config = _config()
    rounds = [2, 4, 8, 16, 32]
    kappas = [2.8 - 2.0 * math.exp(-value / 6.0) for value in rounds]
    fit = fit_registered_models(rounds, kappas, config)["saturating_exponential"]
    measured = dict(zip(rounds, kappas))
    measured[64] = 2.799

    supported = score_saturation_support(
        winner="saturating_exponential",
        saturating_fit=fit,
        kappas=measured,
        config=config,
    )
    rejected = score_saturation_support(
        winner="unresolved",
        saturating_fit=fit,
        kappas=measured,
        config=config,
    )

    assert supported["supported"]
    assert not rejected["supported"]


def test_bootstrap_predictions_are_deterministic() -> None:
    config = _config()
    config["fit_uncertainty"]["samples"] = 20
    rounds = [2, 4, 8, 16, 32]
    values = {
        round_count: [
            (2.8 - 2.0 * math.exp(-round_count / 6.0)) * multiplier
            for multiplier in (0.99, 1.0, 1.01)
        ]
        for round_count in rounds
    }

    first = bootstrap_predictions(
        values, rounds=rounds, prediction_round=64, config=config
    )
    second = bootstrap_predictions(
        values, rounds=rounds, prediction_round=64, config=config
    )

    assert first == second
    assert first["saturating_exponential"]["lower"] > 0


def test_local_effective_gamma_exposes_a_plateau() -> None:
    values = local_effective_gammas([2, 4, 8, 16], [1.0, 1.5, 2.0, 2.05])

    assert values[-1]["gamma"] < values[0]["gamma"]
    assert values[-1]["from_rounds"] == 8


def test_sealed_r64_holdout_preserves_the_registered_negative_result() -> None:
    result_path = (
        ROOT
        / "experiments"
        / "loop_schedule_algebra_saturation_addendum_v1"
        / "holdout_r64_result.json"
    )
    if not result_path.exists():
        pytest.skip("R64 holdout has not been materialized")
    result = json.loads(result_path.read_text())
    summary = result["summary"]

    assert summary["prediction_seal"]["commit"] == (
        "d91b745007c25bc01403aa12b67de73acb969c77"
    )
    assert summary["prediction_seal"]["sha256"] == (
        "910f4d994be6d24afb8bb370b9b6e42a2b907df7b4b82245516286295119ed42"
    )
    assert summary["classification"] == "form_unresolved"
    assert summary["holdout_decision"]["winner"] == "unresolved"
    assert not summary["saturation_support"]["supported"]
    assert summary["tied_geometric_mean_kappa"]["64"] == pytest.approx(
        1.5098975282233111
    )
    assert not summary["untied_flatness_control"]["passed"]
