from research_gym.core.hybrid import (
    CandidateState,
    HybridMode,
    HybridStepResult,
    LatticeProposal,
    MembranePolicy,
    certify_and_apply,
    exact_mechanics_verifier,
)
from research_gym.core.typed_soundness import SoundnessType


def state(*values: str) -> CandidateState:
    return CandidateState({"slot": frozenset(values)})


def proposal(values: tuple[str, ...], soundness: SoundnessType) -> LatticeProposal:
    return LatticeProposal(proposed_state=state(*values), soundness=soundness)


def test_default_verifier_path_preserves_historical_json_shape():
    result = certify_and_apply(
        state("a", "b"),
        proposal(("a",), SoundnessType.ENV_SOUND_DEAD),
        policy=MembranePolicy(),
    )

    assert set(result.to_jsonable()) == {
        "accepted",
        "before",
        "after",
        "proposal",
        "reason",
        "soft_store",
    }
    assert result.claimed_soundness is None
    assert result.verified_soundness is None
    assert result.provenance_disagreed is None


def test_verifier_can_reject_claim_and_records_disagreement():
    policy = MembranePolicy(
        provenance_verifier=lambda proposal, context: SoundnessType.UNKNOWN
    )
    result = certify_and_apply(
        state("a", "b"),
        proposal(("a",), SoundnessType.ENV_SOUND_DEAD),
        policy=policy,
    )

    assert not result.accepted
    assert result.claimed_soundness == SoundnessType.ENV_SOUND_DEAD
    assert result.verified_soundness == SoundnessType.UNKNOWN
    assert result.provenance_disagreed is True
    assert result.to_jsonable()["verified_soundness"] == "unknown"
    assert HybridStepResult.from_jsonable(result.to_jsonable()) == result


def test_verifier_can_promote_grounded_unknown_claim():
    policy = MembranePolicy(
        provenance_verifier=lambda proposal, context: SoundnessType.ENV_SOUND_DEAD
    )
    result = certify_and_apply(
        state("a", "b"),
        proposal(("a",), SoundnessType.UNKNOWN),
        policy=policy,
    )

    assert result.accepted
    assert result.after == state("a")
    assert result.provenance_disagreed is True


def test_exact_mechanics_verifier_adapts_boolean_checker():
    candidate = proposal(("a",), SoundnessType.UNKNOWN)

    assert exact_mechanics_verifier(candidate, lambda _: True) == SoundnessType.ENV_SOUND_DEAD
    assert exact_mechanics_verifier(candidate, lambda _: False) == SoundnessType.UNKNOWN
    assert exact_mechanics_verifier(candidate, {}) is None


def test_environment_sound_accepted():
    result = certify_and_apply(state("a", "b"), proposal(("a",), SoundnessType.ENV_SOUND_DEAD))

    assert result.accepted
    assert result.after == state("a")
    assert result.soft_store is None


def test_model_sound_rejected_and_soft_stored_by_default():
    candidate = proposal(("a",), SoundnessType.MODEL_SOUND_DEAD)
    result = certify_and_apply(state("a", "b"), candidate)

    assert not result.accepted
    assert result.after == state("a", "b")
    assert result.soft_store == candidate
    assert "not hard-applicable" in result.reason


def test_experience_sound_rejected_and_soft_stored_by_default():
    candidate = proposal(("a",), SoundnessType.EXPERIENCE_SOUND_DEAD)
    result = certify_and_apply(state("a", "b"), candidate)

    assert not result.accepted
    assert result.after == state("a", "b")
    assert result.soft_store == candidate
    assert "not hard-applicable" in result.reason


def test_unknown_rejected_without_soft_store():
    candidate = proposal(("a",), SoundnessType.UNKNOWN)
    result = certify_and_apply(state("a", "b"), candidate)

    assert not result.accepted
    assert result.after == state("a", "b")
    assert result.soft_store is None


def test_model_sound_accepted_only_with_explicit_policy_flag():
    candidate = proposal(("a",), SoundnessType.MODEL_SOUND_DEAD)

    default_result = certify_and_apply(state("a", "b"), candidate)
    allowed_result = certify_and_apply(
        state("a", "b"),
        candidate,
        policy=MembranePolicy(allow_model_sound=True),
    )

    assert not default_result.accepted
    assert allowed_result.accepted
    assert allowed_result.after == state("a")


def test_experience_sound_accepted_only_with_explicit_policy_flag():
    candidate = proposal(("a",), SoundnessType.EXPERIENCE_SOUND_DEAD)

    default_result = certify_and_apply(state("a", "b"), candidate)
    allowed_result = certify_and_apply(
        state("a", "b"),
        candidate,
        policy=MembranePolicy(allow_experience_sound=True),
    )

    assert not default_result.accepted
    assert allowed_result.accepted
    assert allowed_result.after == state("a")


def test_non_monotone_proposal_rejected():
    candidate = proposal(("a", "b", "c"), SoundnessType.ENV_SOUND_DEAD)
    result = certify_and_apply(state("a", "b"), candidate)

    assert not result.accepted
    assert result.after == state("a", "b")
    assert "non-monotone" in result.reason


def test_bottom_proposal_requires_abstain_mode():
    current = state("a")
    proposed = LatticeProposal(
        proposed_state=state(),
        soundness=SoundnessType.ENV_SOUND_DEAD,
        mode=HybridMode.DEDUCE,
    )
    abstain = LatticeProposal(
        proposed_state=state(),
        soundness=SoundnessType.ENV_SOUND_DEAD,
        mode=HybridMode.ABSTAIN,
    )

    rejected = certify_and_apply(current, proposed)
    accepted = certify_and_apply(current, abstain)

    assert not rejected.accepted
    assert accepted.accepted
    assert accepted.after.is_bottom


def test_hybrid_decision_roundtrip_serialization():
    result = certify_and_apply(state("a", "b"), proposal(("a",), SoundnessType.ENV_SOUND_DEAD))
    restored = HybridStepResult.from_jsonable(result.to_jsonable())

    assert restored == result
