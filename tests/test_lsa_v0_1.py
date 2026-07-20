from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

torch = pytest.importorskip("torch")

from lsa.kappa_probe import estimate_kappa  # noqa: E402
from lsa.mini_transformer_probe import MiniTransformerLoop  # noqa: E402
from research_gym.analysis.lsa_v0_1 import (  # noqa: E402
    bootstrap_gamma,
    classify_gamma_trajectory,
    empirical_boundary,
    score_r16_holdout,
)
from research_gym.scripts.bench_loop_schedule_algebra_v0_1 import (  # noqa: E402
    measurement_seed,
    prefix_context_batch,
    steps_for_exposure_budget,
)
from research_gym.scripts.report_lsa_v0_1 import generate  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads((ROOT / "configs" / "loop_schedule_algebra_v0_1.json").read_text())


def test_v0_1_config_rehashes_registration() -> None:
    path = ROOT / "configs" / "loop_schedule_algebra_v0_1.json"
    registration = json.loads(
        (ROOT / "configs" / "loop_schedule_algebra_v0_1_registration.json").read_text()
    )

    assert hashlib.sha256(path.read_bytes()).hexdigest() == registration["config_sha256"]
    assert not registration["outcomes_observed"]


def test_r16_prediction_is_frozen_from_v0_fit() -> None:
    holdout = _config()["r16_holdout"]
    predicted = pytest.approx(
        torch.exp(torch.tensor(holdout["sealed_tied_intercept"])).item()
        * 16 ** holdout["sealed_tied_gamma"],
        rel=1e-6,
    )

    assert holdout["predicted_kappa_r16"] == predicted
    assert score_r16_holdout(_config(), holdout["predicted_kappa_r16"])["confirmed"]
    assert not score_r16_holdout(_config(), holdout["predicted_kappa_r16"] * 1.3)["confirmed"]


def test_mini_transformer_probe_is_deterministic_and_visit_bounded() -> None:
    torch.manual_seed(13)
    model = MiniTransformerLoop(
        hidden_size=8,
        rounds=3,
        tied=True,
        heads=2,
        mlp_ratio=2,
        beta=0.25,
    )
    inputs = torch.randn(2, 4, 8)
    targets = torch.tanh(inputs.cumsum(dim=1))

    first = estimate_kappa(model, inputs, targets, power_iterations=2, seed=31)
    second = estimate_kappa(model, inputs, targets, power_iterations=2, seed=31)

    assert first == second
    assert 0.0 <= first.kappa <= 3.0


def test_untied_mini_transformer_counts_physical_parameters_per_visit() -> None:
    tied = MiniTransformerLoop(8, 4, tied=True, heads=2, mlp_ratio=2)
    untied = MiniTransformerLoop(8, 4, tied=False, heads=2, mlp_ratio=2)

    tied_parameters = sum(parameter.numel() for parameter in tied.parameters())
    untied_parameters = sum(parameter.numel() for parameter in untied.parameters())

    assert untied_parameters == 4 * tied_parameters


def test_prefix_context_task_is_deterministic_and_contextual() -> None:
    first_inputs, first_targets = prefix_context_batch(
        7, 2, 8, 4, torch.device("cpu"), stream=19
    )
    second_inputs, second_targets = prefix_context_batch(
        7, 2, 8, 4, torch.device("cpu"), stream=19
    )

    assert torch.equal(first_inputs, second_inputs)
    assert torch.equal(first_targets, second_targets)
    assert not torch.equal(first_targets[:, 1], torch.tanh(first_inputs[:, 1]))


def test_measurement_seed_is_fixed_across_checkpoint_exposures() -> None:
    assert measurement_seed(101, 0) == 50_101
    assert measurement_seed(101, 0) == measurement_seed(101, 4096)


def test_exposure_budget_uses_v0_ceiling_semantics() -> None:
    assert steps_for_exposure_budget(4096, 8, 6) == 86
    assert steps_for_exposure_budget(4096, 8, 12) == 43
    assert steps_for_exposure_budget(4096, 8, 8) == 64


def _synthetic_records(exposure: int) -> list[dict]:
    records = []
    for rounds in (2, 4, 8, 16):
        for seed, multiplier in zip((101, 103, 107), (0.98, 1.0, 1.02)):
            records.append(
                {
                    "regime": "tied",
                    "exposure": exposure,
                    "rounds": rounds,
                    "seed": seed,
                    "kappa": multiplier * rounds**0.4,
                }
            )
    return records


def test_seed_bootstrap_is_deterministic() -> None:
    records = _synthetic_records(4096)
    arguments = {
        "rounds": (2, 4, 8, 16),
        "regime": "tied",
        "exposure": 4096,
        "seeds": (101, 103, 107),
        "samples": 100,
        "bootstrap_seed": 86017,
        "interval_mass": 0.95,
    }

    first = bootstrap_gamma(records, **arguments)
    second = bootstrap_gamma(records, **arguments)

    assert first == second
    assert first["lower"] == pytest.approx(0.4)
    assert first["upper"] == pytest.approx(0.4)


def test_trajectory_labels_are_predeclared_and_exclusive() -> None:
    config = _config()
    exposures = config["shared_training"]["checkpoint_exposures"]

    growth = [{"exposure": exposure, "gamma": 0.2 + index * 0.04} for index, exposure in enumerate(exposures)]
    invariant = [{"exposure": exposure, "gamma": 0.4 + index * 0.01} for index, exposure in enumerate(exposures)]
    mixed = [{"exposure": exposure, "gamma": value} for exposure, value in zip(exposures, (0.2, 0.5, 0.3, 0.6, 0.31))]

    assert classify_gamma_trajectory(config, growth) == "learned_growth"
    assert classify_gamma_trajectory(config, invariant) == "invariant"
    assert classify_gamma_trajectory(config, mixed) == "mixed"


def test_boundary_preserves_censoring_instead_of_promoting_grid_edge() -> None:
    grid = [0.0, 0.05, 0.1, 0.15]
    all_stable = [
        {"p": p, "stable": True} for p in grid for _ in range(3)
    ]
    unstable_low = [
        {"p": p, "stable": p >= 0.1} for p in grid for _ in range(3)
    ]

    assert empirical_boundary(all_stable, grid)["censoring"] == "left_censored"
    assert empirical_boundary(all_stable, grid)["scientific_boundary"] == "<=0"
    assert empirical_boundary(unstable_low, grid)["machine_boundary"] == 0.1
    assert empirical_boundary(unstable_low, grid)["censoring"] == "exact_grid"


def test_claim_scope_excludes_task_performance() -> None:
    scope = _config()["claim_scope"].lower()

    assert "no task-performance" in scope
    assert "sample-efficiency" in scope


def test_report_figures_are_deterministic_valid_svg(tmp_path: Path) -> None:
    experiment = ROOT / "experiments" / "loop_schedule_algebra_v0_1"
    first = generate(experiment, tmp_path / "first")
    second = generate(experiment, tmp_path / "second")

    assert len(first) == len(second) == 3
    for first_path, second_path in zip(first, second):
        assert first_path.read_bytes() == second_path.read_bytes()
        assert ET.parse(first_path).getroot().tag.endswith("svg")


def test_final_v0_1_receipt_rehashes_all_valid_phases() -> None:
    receipt_path = ROOT / "data" / "benchmarks" / "lsa_v0_1_receipt.json"
    receipt = json.loads(receipt_path.read_text())

    assert receipt["status"] == "complete"
    assert receipt["primary_findings"]["trajectory_classification"] == "learned_growth"
    assert not receipt["primary_findings"]["r16_holdout"]["confirmed"]
    assert not receipt["external_validity"]["replication_supported"]
    assert receipt["learning_rate_ladder"]["all_cells_stable"]
    for phase in receipt["phases"].values():
        records = ROOT / phase["records_path"]
        result = ROOT / phase["result_path"]
        assert hashlib.sha256(records.read_bytes()).hexdigest() == phase["records_sha256"]
        assert hashlib.sha256(result.read_bytes()).hexdigest() == phase["result_sha256"]
    assert receipt_path.read_bytes() == (
        ROOT / "experiments" / "loop_schedule_algebra_v0_1" / "result_receipt.json"
    ).read_bytes()
