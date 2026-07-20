from __future__ import annotations

import json

import torch

from research_gym.integrity import canonical_file_sha256
from research_gym.scripts.bench_loop_schedule_kappa_transient_v0_2_recovery import (
    DEFAULT_CONFIG,
    REGISTRATION,
    _admitted_parent_records,
    _exact_equal,
    load_registered_config,
)


def test_recovery_registration_hashes_config_before_outcomes() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    assert canonical_file_sha256(DEFAULT_CONFIG) == registration["config_sha256"]
    assert registration["new_recovery_outcomes_observed"] is False
    assert registration["registration_status"] == "frozen_before_recovery_outcomes"


def test_recovery_admits_only_registered_parent_measurements() -> None:
    config, _, _ = load_registered_config(DEFAULT_CONFIG)
    records = _admitted_parent_records(config)
    identities = {(row["seed"], row["exposure"]) for row in records}

    assert len(records) == 10
    assert {(107, 1024), (107, 2048)} <= identities
    assert (107, 4096) not in identities
    assert all(row["regime"] == "tied" for row in records)


def test_tensor_exact_gate_rejects_changed_state() -> None:
    _exact_equal({"value": torch.tensor([1.0])}, {"value": torch.tensor([1.0])})
    try:
        _exact_equal({"value": torch.tensor([1.0])}, {"value": torch.tensor([2.0])})
    except RuntimeError as error:
        assert "tensor-exact replay failed" in str(error)
    else:
        raise AssertionError("changed tensor state passed the replay gate")
