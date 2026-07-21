from __future__ import annotations

import json

from research_gym.integrity import canonical_file_sha256
from research_gym.scripts.bench_loop_schedule_kappa_data_order_v0_2_1_recovery import (
    DEFAULT_CONFIG,
    REGISTRATION,
    admitted_records,
    load_registered_config,
)


def test_recovery_registration_hashes_config_before_aggregate_branch() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    assert canonical_file_sha256(DEFAULT_CONFIG) == registration["config_sha256"]
    assert registration["aggregate_branch_computed"] is False
    assert registration["new_training_cells"] == 0


def test_recovery_admits_exact_complete_seed_decoupled_grid() -> None:
    config, _ = load_registered_config(DEFAULT_CONFIG)
    records = admitted_records(config)
    assert len(records) == len({row["record_id"] for row in records}) == 20
    assert {row["seed_channels"]["data_order_seed"] for row in records} == {
        103,
        211,
        223,
        227,
    }
    assert all(row["seed_channels"]["model_seed"] == 103 for row in records)
