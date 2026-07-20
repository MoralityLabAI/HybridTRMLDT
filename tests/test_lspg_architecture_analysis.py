from __future__ import annotations

import json
from pathlib import Path

from lsa.canonical import digest
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


def test_execution_addendum_is_self_attested() -> None:
    root = Path(__file__).resolve().parents[1]
    value = json.loads(
        (root / "configs/lsa/architecture_execution_addendum_v1.json").read_text(
            encoding="utf-8"
        )
    )
    claimed = value.pop("frozen_config_sha256")
    assert value["status"] == "frozen_before_task_outcomes"
    assert digest(value) == claimed


def test_gradient_execution_amendment_is_self_attested_and_pre_outcome() -> None:
    root = Path(__file__).resolve().parents[1]
    value = json.loads(
        (root / "configs/lsa/architecture_execution_amendment_v1_1.json").read_text(
            encoding="utf-8"
        )
    )
    claimed = value.pop("frozen_config_sha256")
    assert value["status"] == "frozen_after_resource_failure_before_task_outcomes"
    assert not value["task_outcomes_observed"]
    assert digest(value) == claimed


def test_precision_execution_amendment_is_self_attested_and_pre_outcome() -> None:
    root = Path(__file__).resolve().parents[1]
    value = json.loads(
        (root / "configs/lsa/architecture_execution_amendment_v1_2.json").read_text(
            encoding="utf-8"
        )
    )
    claimed = value.pop("frozen_config_sha256")
    assert value["status"] == "frozen_after_precision_diagnostics_before_task_outcomes"
    assert value["replacement_rule"]["training_precision"] == "fp32"
    assert not value["task_outcomes_observed"]
    assert digest(value) == claimed


def test_lsad_v1_closeout_is_self_attested_zero_winner() -> None:
    root = Path(__file__).resolve().parents[1]
    value = json.loads(
        (
            root
            / "experiments/loop_schedule_architecture_discovery_v1/campaign/closeout_v1.json"
        ).read_text(encoding="utf-8")
    )
    claimed = value.pop("closeout_hash")

    assert value["status"] == "completed_zero_winner"
    assert value["qualifying_architectures"] == []
    assert value["primary_metric"]["effective_denominator"] == 768
    assert value["reserve"]["status"] == "sealed_unopened"
    assert digest(value) == claimed
