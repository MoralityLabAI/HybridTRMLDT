from __future__ import annotations

import json

from research_gym.integrity import canonical_file_sha256
from research_gym.scripts.bench_loop_schedule_kappa_splice_v0_2_2 import (
    DEFAULT_CONFIG,
    REGISTRATION,
    classify_splice,
    continuation_streams,
    load_registered_config,
    validate,
)


def test_registration_hashes_config_before_splice_outcomes(tmp_path) -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    assert canonical_file_sha256(DEFAULT_CONFIG) == registration["config_sha256"]
    assert registration["new_splice_outcomes_observed"] is False
    assert registration["new_prefix_training"] == 0
    config, config_hash, _ = load_registered_config(DEFAULT_CONFIG)
    assert validate(config, config_hash, tmp_path)["new_continuation_arms"] == 5


def test_continuation_streams_start_after_checkpoint() -> None:
    assert continuation_streams(2048, 4096, quantum=512) == [4, 5, 6, 7]
    assert continuation_streams(4096, 8192, quantum=512) == list(range(8, 16))


def test_splice_classifier_preserves_registered_asymmetry() -> None:
    assert classify_splice(True, True, False) == "suffix_controlled_exit"
    assert classify_splice(False, False, True) == "prehistory_controlled_exit"
    assert classify_splice(True, False, False) == "mixed_state_suffix_control"
