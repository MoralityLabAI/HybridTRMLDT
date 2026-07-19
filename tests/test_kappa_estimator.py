from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from lsa.kappa_probe import (  # noqa: E402
    ToyLoop,
    alignment_coefficient,
    estimate_kappa,
    fit_power_law,
)


def test_alignment_coefficient_separates_orthogonal_and_aligned_visits() -> None:
    rounds = 4
    orthogonal = tuple(torch.eye(rounds)[index] for index in range(rounds))
    aligned = tuple(torch.tensor([1.0, 0.0, 0.0, 0.0]) for _ in range(rounds))

    assert alignment_coefficient(orthogonal, orthogonal) == pytest.approx(1.0)
    assert alignment_coefficient(aligned, aligned) == pytest.approx(rounds)


def test_single_visit_kappa_is_one_independent_of_residual_scale() -> None:
    torch.manual_seed(3)
    model = ToyLoop(hidden_size=8, rounds=1, tied=True, beta=0.25)
    inputs = torch.randn(2, 8)
    estimate = estimate_kappa(model, inputs, torch.zeros_like(inputs), power_iterations=2)

    assert estimate.kappa == pytest.approx(1.0)


def test_kappa_probe_is_deterministic_and_visit_bounded() -> None:
    torch.manual_seed(7)
    model = ToyLoop(hidden_size=8, rounds=3, tied=True, beta=0.5)
    inputs = torch.randn(2, 8)
    targets = torch.tanh(inputs)

    first = estimate_kappa(model, inputs, targets, power_iterations=2, seed=19)
    second = estimate_kappa(model, inputs, targets, power_iterations=2, seed=19)

    assert first == second
    assert first.visible_rounds == 3
    assert math.isfinite(first.kappa) and first.kappa > 0
    assert all(value > 0 for value in first.u_norms)
    assert all(value > 0 for value in first.g_norms)


def test_masked_probe_only_recomputes_selected_visits() -> None:
    torch.manual_seed(11)
    model = ToyLoop(hidden_size=8, rounds=4, tied=False, beta=0.75)
    inputs = torch.randn(2, 8)
    targets = torch.zeros_like(inputs)

    estimate = estimate_kappa(
        model,
        inputs,
        targets,
        visible_visits=(2, 3),
        power_iterations=2,
        seed=5,
    )

    assert estimate.visible_rounds == 2
    assert len(estimate.u_norms) == len(estimate.g_norms) == 2


def test_power_law_fit_recovers_known_exponent() -> None:
    rounds = (2.0, 4.0, 8.0)
    kappas = tuple(1.5 * value**0.8 for value in rounds)

    fit = fit_power_law(rounds, kappas)

    assert fit.gamma == pytest.approx(0.8)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.predict(16) == pytest.approx(1.5 * 16**0.8)
