from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_gym.prognostics.censoring import CensoredObservation, ObservationKind
from research_gym.prognostics.empirical import ingest_lsa_receipt


ROOT = Path(__file__).resolve().parents[1]


def test_p2_round_trips_as_left_censored_not_exact_endpoint() -> None:
    evidence = ingest_lsa_receipt(ROOT / "data" / "benchmarks" / "lsa_v0_receipt.json")

    assert len(evidence.boundary_observations) == 2
    for row in evidence.boundary_observations:
        observation = row["observation"]
        assert observation["kind"] == ObservationKind.LEFT_CENSORED.value
        assert observation["upper"] == 0.15
        assert observation["value"] is None


def test_censored_likelihood_uses_cdf_mass() -> None:
    left = CensoredObservation(ObservationKind.LEFT_CENSORED, upper=0.0)
    right = CensoredObservation(ObservationKind.RIGHT_CENSORED, lower=0.0)
    interval = CensoredObservation(
        ObservationKind.INTERVAL_CENSORED, lower=-1.0, upper=1.0
    )

    assert left.probability(mean=0.0, std=1.0) == pytest.approx(0.5)
    assert right.probability(mean=0.0, std=1.0) == pytest.approx(0.5)
    assert interval.probability(mean=0.0, std=1.0) == pytest.approx(0.682689, rel=1e-5)
