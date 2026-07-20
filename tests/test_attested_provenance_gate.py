from __future__ import annotations

import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch", exc_type=ImportError)

from research_gym.attestation.provenance_gate import (
    CLAIM_BOUNDARY,
    RSIAttestationBackend,
    canonical_sha256,
    run_rsi_conformance,
    validate_registration,
)
from research_gym.benchmarks.attested_provenance_gate_bench import (
    _tamper_controls,
    verify_local_sources,
)
from research_gym.core.typed_soundness import SoundnessType
from research_gym.envs.coupled_storyworld import CoupledStoryworldEnv, StoryState
from research_gym.integrity import verify_file_sha256
from research_gym.neural.rollout import StoryExample, action_environment_sound


ROOT = Path(__file__).resolve().parents[1]
RSI_ROOT = Path(r"C:\projects\RSITopology")
CONFIG_PATH = ROOT / "configs" / "attested_provenance_gate_v1.json"


def _registration():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _backend():
    if not RSI_ROOT.exists():
        pytest.skip("hash-pinned RSITopology checkout is not available")
    registration = _registration()
    return RSIAttestationBackend(RSI_ROOT, registration["source_integrity"])


def _examples():
    env = CoupledStoryworldEnv()
    for state in env.all_states():
        if env.terminal(state) or env.target(state):
            continue
        example = StoryExample(
            episode_id="fixture-0",
            scenario="secret_ending",
            state=state,
            horizon=6,
            split="test",
            region="fixture",
        )
        sound_count = sum(
            action_environment_sound(env, example, action)
            for action in env.self_actions
        )
        if 2 <= sound_count < len(env.self_actions):
            return [example]
    raise AssertionError("fixture search found no mixed-soundness state")


def test_frozen_attestation_registration_and_local_sources():
    registration = _registration()

    assert validate_registration(registration) == registration["frozen_config_sha256"]
    assert registration["claim_boundary"] == CLAIM_BOUNDARY
    assert registration["benchmark"]["seeds"] == [17, 29, 43]
    assert registration["benchmark"]["expert_iteration_rounds"] == 5
    verify_local_sources(ROOT, registration)


def test_attested_envelope_authorizes_true_claim_and_denies_unregistered_false_claim():
    registration = _registration()
    backend = _backend()
    mechanics = backend.mechanics_identity(registration)
    env = CoupledStoryworldEnv()
    examples = _examples()
    registry = backend.build_registry(
        examples,
        env.self_actions,
        mechanics,
        lambda example, action: action_environment_sound(env, example, action),
    )
    true_pair = next(
        (example, action)
        for example in examples
        for action in env.self_actions
        if action_environment_sound(env, example, action)
    )
    false_pair = next(
        (example, action)
        for example in examples
        for action in env.self_actions
        if not action_environment_sound(env, example, action)
    )

    true_envelope = backend.claim_envelope(
        registry, *true_pair, SoundnessType.ENV_SOUND_DEAD.value, mechanics
    )
    true_decision = backend.verify_envelope(
        registry, true_envelope, *true_pair, SoundnessType.ENV_SOUND_DEAD.value, mechanics
    )
    false_envelope = backend.claim_envelope(
        registry, *false_pair, SoundnessType.ENV_SOUND_DEAD.value, mechanics
    )
    false_decision = backend.verify_envelope(
        registry, false_envelope, *false_pair, SoundnessType.ENV_SOUND_DEAD.value, mechanics
    )

    assert true_decision["authorized"]
    assert true_decision["certification_level"] == "holonomy_clean"
    assert not false_decision["authorized"]
    assert "envelope:not_issued" in false_decision["failures"]
    assert "anchor:not_registered" in false_decision["failures"]


def test_registered_tamper_classes_are_detected():
    registration = _registration()
    backend = _backend()
    mechanics = backend.mechanics_identity(registration)
    env = CoupledStoryworldEnv()
    examples = _examples()
    registry = backend.build_registry(
        examples,
        env.self_actions,
        mechanics,
        lambda example, action: action_environment_sound(env, example, action),
    )

    controls = _tamper_controls(
        backend, registry, examples, env.self_actions, mechanics
    )

    assert [row["tamper_class"] for row in controls] == [
        "spectrum_preserving_anchor_substitution",
        "band_registry_failover",
        "reanchor_preserved_payload_hash",
    ]
    assert all(row["status"] == "DETECTED" for row in controls)


def test_rsi_golden_vectors_and_registry_fixture_conform():
    registration = _registration()
    receipt = run_rsi_conformance(_backend(), registration["source_integrity"])

    assert receipt["golden_vector_count"] >= 200
    assert receipt["golden_vector_failure_count"] == 0
    assert receipt["fixture_authorized"]
    assert receipt["passed"]


def test_sealed_result_receipt_when_present():
    receipt_path = ROOT / "data" / "benchmarks" / "attested_provenance_gate_v1_receipt.json"
    if not receipt_path.exists():
        pytest.skip("full attested benchmark has not been sealed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    result_path = ROOT / receipt["results_path"]
    records_path = ROOT / receipt["records_path"]
    report_path = ROOT / receipt["report_path"]

    assert verify_file_sha256(result_path, receipt["results_sha256"])
    assert verify_file_sha256(records_path, receipt["records_sha256"])
    assert verify_file_sha256(report_path, receipt["report_sha256"])
    assert receipt["records_reverified"]
    assert receipt["split_overlap"] == 0
    assert receipt["decision_receipt_failure_count"] == 0
