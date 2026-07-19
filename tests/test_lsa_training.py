from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from research_gym.neural.training import screen_proposal  # noqa: E402


def _tiny_proposal() -> dict:
    mutation = {
        "expanded_visits": 2,
        "gradient_visible_visits": 2,
        "retained_state_edges": 1,
        "supervision_points": 1,
        "physical_modules": 1,
        "tying": "fully_tied",
        "alpha": 1.0,
        "beta": 0.25,
        "normalization": "pre",
        "carry": "reset",
        "sequence_length": 8,
    }
    return {
        "proposal_id": "tiny",
        "proposal_hash": "abc",
        "mutation": mutation,
        "model": {"vocab_size": 32, "hidden_size": 16, "num_heads": 2},
    }


def test_screening_emits_all_registered_checkpoint_percentages(tmp_path: Path) -> None:
    result = screen_proposal(
        _tiny_proposal(),
        output_dir=tmp_path,
        exposure_budget=64,
        checkpoints_pct=(1, 3, 10, 30, 100),
        seed=7,
    )

    assert result.status == "completed"
    assert {row["percentage"] for row in result.checkpoints} == {1, 3, 10, 30, 100}
    assert (tmp_path / "checkpoints" / "tiny" / "latest.pt").exists()
