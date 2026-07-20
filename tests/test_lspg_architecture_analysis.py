from __future__ import annotations

from research_gym.architecture_discovery.analysis import (
    holm_rejections,
    paired_randomization_pvalue,
)


def test_paired_randomization_separates_consistent_gain_from_null() -> None:
    assert paired_randomization_pvalue([1.0] * 20, resamples=2_000) < 0.01
    assert paired_randomization_pvalue([0.0] * 20, resamples=2_000) == 1.0


def test_holm_rejections_stop_after_first_failure() -> None:
    assert holm_rejections({"a": 0.01, "b": 0.02, "c": 0.2}) == {
        "a": True,
        "b": True,
        "c": False,
    }
    assert holm_rejections({"a": 0.02, "b": 0.021, "c": 0.022}) == {
        "a": False,
        "b": False,
        "c": False,
    }
