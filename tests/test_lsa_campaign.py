from __future__ import annotations

import json
from pathlib import Path

from research_gym.integrity import verify_file_sha256
from research_gym.scripts.bench_loop_schedule_algebra import (
    _empirical_boundary,
    _gamma_summary,
    _is_stable,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads((ROOT / "configs" / "loop_schedule_algebra_v0.json").read_text())


def test_registered_cells_and_stop_rules_are_explicit() -> None:
    config = _config()

    assert config["kappa_probe"]["rounds"] == [2, 4, 8]
    assert len(config["kappa_probe"]["seeds"]) == 3
    assert config["kappa_probe"]["power_iterations"] == 5
    assert config["stop_branches"]["max_across_seed_kappa_spread"] == 10.0
    assert config["stop_branches"]["minimum_power_law_r_squared"] == 0.8


def test_instability_stops_gamma_fit() -> None:
    config = _config()
    records = []
    for regime in ("tied", "untied"):
        for rounds in (2, 4, 8):
            values = (1.0, 1.1, 1.2)
            if regime == "tied" and rounds == 4:
                values = (1.0, 2.0, 11.0)
            for seed, value in zip((101, 103, 107), values):
                records.append(
                    {
                        "record_id": f"{regime}-{rounds}-{seed}",
                        "kind": "gamma",
                        "regime": regime,
                        "rounds": rounds,
                        "seed": seed,
                        "kappa": value,
                    }
                )

    summary = _gamma_summary(config, records)

    assert summary["status"] == "estimator_unstable"
    assert summary["fits"] == {}


def test_boundary_requires_stable_tail_and_seed_majority() -> None:
    config = _config()
    rows = []
    for p in config["p2"]["p_grid"]:
        for seed in config["kappa_probe"]["seeds"]:
            rows.append({"p": p, "stable": p >= 0.4 and seed != 107})

    assert _empirical_boundary(config, rows) == 0.4


def test_stability_rule_rejects_gradient_and_loss_explosions() -> None:
    config = _config()
    base = {
        "nonfinite": False,
        "initial_loss": 1.0,
        "final_loss": 0.5,
        "max_gradient_norm": 2.0,
    }

    assert _is_stable(config, base)
    assert not _is_stable(config, {**base, "max_gradient_norm": 101.0})
    assert not _is_stable(config, {**base, "final_loss": 11.0})


def test_sealed_receipt_rehashes_records_and_prediction() -> None:
    receipt = json.loads((ROOT / "data" / "benchmarks" / "lsa_v0_receipt.json").read_text())

    for section, key in (
        ("gamma_phase", "records_path"),
        ("boundary_phase", "records_path"),
        ("prediction", "path"),
    ):
        path = ROOT / receipt[section][key]
        expected = receipt[section][
            "sha256" if section == "prediction" else "records_sha256"
        ]
        assert verify_file_sha256(path, expected)

    boundary_rows = [
        json.loads(line)
        for line in (ROOT / receipt["boundary_phase"]["records_path"]).read_text().splitlines()
    ]
    assert len(boundary_rows) == receipt["boundary_phase"]["record_count"] == 66
    assert all(row["stable"] for row in boundary_rows)
    assert not receipt["boundary_phase"]["p2_confirmed"]
