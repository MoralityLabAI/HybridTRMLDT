from research_gym.core.hybrid import (
    CandidateState, HybridMode, LatticeProposal,
    certify_and_apply, certify_and_apply_verified,
)
from research_gym.core.typed_soundness import SoundnessType


def state(*values):
    return CandidateState({"action": frozenset(values)})


def p(values, claimed=SoundnessType.ENV_SOUND_DEAD, mode=HybridMode.DEDUCE):
    return LatticeProposal(state(*values), soundness=claimed, mode=mode)


def test_unverified_env_sound_claim_cannot_authorize():
    proposal = p(("a",))
    result = certify_and_apply_verified(
        state("a", "b"), proposal, provenance_verifier=None)
    assert not result.accepted
    assert result.after == state("a", "b")
    assert result.verified_soundness == SoundnessType.UNKNOWN


def test_none_or_unknown_verification_fails_closed():
    for verdict in (None, SoundnessType.UNKNOWN, SoundnessType.MODEL_SOUND_DEAD):
        result = certify_and_apply_verified(
            state("a", "b"), p(("a",)),
            provenance_verifier=lambda _p, _c: verdict)
        assert not result.accepted
        assert result.after == state("a", "b")


def test_independent_verifier_can_authorize_claimed_unknown_once():
    calls = []
    def verifier(proposal, context):
        calls.append((proposal, context))
        return SoundnessType.ENV_SOUND_DEAD
    proposal = p(("a",), claimed=SoundnessType.UNKNOWN)
    result = certify_and_apply_verified(
        state("a", "b"), proposal,
        provenance_verifier=verifier,
        verifier_context={"test": True})
    assert result.accepted
    assert result.after == state("a")
    assert len(calls) == 1
    assert result.provenance_disagreed is True


def test_strict_verified_refinement_requires_progress():
    result = certify_and_apply_verified(
        state("a", "b"), p(("a", "b")),
        provenance_verifier=lambda *_: SoundnessType.ENV_SOUND_DEAD)
    assert not result.accepted
    assert "progress" in result.reason


def test_verified_refinement_rejects_widening_and_unmarked_bottom():
    verified = lambda *_: SoundnessType.ENV_SOUND_DEAD
    assert not certify_and_apply_verified(
        state("a"), p(("a", "b")), provenance_verifier=verified).accepted
    assert not certify_and_apply_verified(
        state("a"), p(()), provenance_verifier=verified).accepted
    abstain = certify_and_apply_verified(
        state("a"), p((), mode=HybridMode.ABSTAIN), provenance_verifier=verified)
    assert abstain.accepted and abstain.after.is_bottom


def test_legacy_experiment_path_is_unchanged():
    legacy = certify_and_apply(state("a", "b"), p(("a",)))
    assert legacy.accepted
