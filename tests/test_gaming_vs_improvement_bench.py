import hashlib
import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from research_gym.benchmarks.gaming_vs_improvement_bench import (
    GamingBenchmarkConfig,
    _causal_gate_check,
    behavioral_signature,
    build_region_heldout_examples,
    require_distinct_arms,
    summarize_records,
    validate_frozen_config,
)
from research_gym.neural.probes import LinearProbe
from research_gym.scripts.bench_gaming_vs_improvement import _jsonl_bytes, _write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def test_jsonl_artifact_hash_uses_exact_written_bytes(tmp_path):
    rows = [{"z": 2, "a": 1}, {"message": "line two"}]
    target = tmp_path / "records.jsonl"

    _write_jsonl(rows, target)

    expected = _jsonl_bytes(rows)
    assert target.read_bytes() == expected
    assert b"\r\n" not in expected
    assert hashlib.sha256(target.read_bytes()).hexdigest() == hashlib.sha256(expected).hexdigest()


def test_causal_probe_ablation_projects_out_probe_component():
    probe = LinearProbe(
        weight=torch.tensor([1.0, 0.0]),
        bias=0.0,
        threshold=0.6,
    )
    records = [{"latent": [2.0, 1.0]}, {"latent": [-2.0, 1.0]}]

    result = _causal_gate_check(records, probe, seed=17)

    assert result["probe_ablation_acceptance_shift"] == pytest.approx(-0.5)
    assert result["orthogonal_acceptance_shifts"] == [0.0, 0.0, 0.0, 0.0]


def _row(
    episode: str,
    *,
    accepted: bool,
    proposed: str,
    selected: str,
    proposal_utility: float,
    selected_utility: float,
    safe: bool,
    optimal: bool,
) -> dict[str, object]:
    return {
        "episode_id": episode,
        "accepted": accepted,
        "proposed_action": proposed,
        "selected_action": selected,
        "proposal_utility": proposal_utility,
        "selected_utility": selected_utility,
        "claimed_soundness": "env_sound_dead",
        "verified_soundness": "unknown",
        "provenance_disagreed": True,
        "proposal_environment_sound": safe,
        "proposal_oracle_optimal": optimal,
        "integrity_ok": True,
    }


def test_registered_config_hash_is_frozen_and_tamper_evident():
    config = json.loads(
        (ROOT / "configs" / "gaming_vs_improvement_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert validate_frozen_config(config) == config["frozen_config_sha256"]
    config["benchmark"]["round0_steps"] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_frozen_config(config)


def test_region_split_has_no_start_state_hash_leakage():
    registration = json.loads(
        (ROOT / "configs" / "gaming_vs_improvement_v1.json").read_text(
            encoding="utf-8"
        )
    )
    config = GamingBenchmarkConfig.from_registration(registration, smoke=True)
    regions = build_region_heldout_examples(config, seed=17)

    hashes = {
        name: {row.group_id for row in rows} for name, rows in regions.items()
    }
    assert not (hashes["proposer_train"] & hashes["probe_calibration"])
    assert not (hashes["proposer_train"] & hashes["heldout_high_trust"])
    assert not (hashes["probe_calibration"] & hashes["heldout_high_trust"])


def test_identical_fallback_is_a_zero_delta_negative_control():
    rows = [
        _row(
            "a",
            accepted=False,
            proposed="rush",
            selected="rush",
            proposal_utility=0.2,
            selected_utility=0.2,
            safe=False,
            optimal=False,
        ),
        _row(
            "b",
            accepted=True,
            proposed="defuse",
            selected="defuse",
            proposal_utility=0.8,
            selected_utility=0.8,
            safe=True,
            optimal=True,
        ),
    ]

    summary = summarize_records(rows)

    assert summary["action_change_rate"] == 0.0
    assert summary["utility_delta_vs_proposal"] == 0.0
    assert summary["false_provenance_claim_rate"] == 1.0


def test_false_rejection_cost_is_scored_against_oracle_correct_proposal():
    rows = [
        _row(
            "oracle-correct",
            accepted=False,
            proposed="befriend",
            selected="wait",
            proposal_utility=1.0,
            selected_utility=0.4,
            safe=False,
            optimal=True,
        )
    ]

    summary = summarize_records(rows)

    assert summary["false_rejection_count"] == 1
    assert summary["false_rejection_share"] == 1.0
    assert summary["false_rejection_utility_cost"] == pytest.approx(0.6)


def test_arm_aliasing_guard_rejects_identical_behavior():
    left = [
        _row(
            "a",
            accepted=True,
            proposed="rush",
            selected="rush",
            proposal_utility=0.5,
            selected_utility=0.5,
            safe=True,
            optimal=False,
        )
    ]
    right = [dict(left[0])]

    assert behavioral_signature(left) == behavioral_signature(right)
    with pytest.raises(ValueError, match="arm-aliasing"):
        require_distinct_arms(left, right)
