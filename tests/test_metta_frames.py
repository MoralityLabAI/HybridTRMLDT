from pathlib import Path

from research_gym.core.hybrid import CandidateState, HybridMode, LatticeProposal, certify_and_apply
from research_gym.core.metta_frames import DeductionFrame, MettaFrameKind, frame_from_jsonable
from research_gym.envs.metta_synthetic import default_frames, frames_from_directives
from research_gym.core.typed_soundness import SoundnessType


def test_default_metta_frames_include_all_kinds():
    frames = default_frames()
    kinds = {frame.kind for frame in frames}
    assert MettaFrameKind.EXECUTION in kinds
    assert MettaFrameKind.DEDUCTION in kinds
    assert MettaFrameKind.REPAIR in kinds
    assert MettaFrameKind.ROUTING in kinds


def test_frame_roundtrip_for_deduction_soundness():
    frame = [f for f in default_frames() if isinstance(f, DeductionFrame)][0]
    restored = frame_from_jsonable(frame.to_jsonable())
    assert isinstance(restored, DeductionFrame)
    assert restored.soundness == frame.soundness
    assert restored.kind == MettaFrameKind.DEDUCTION


def test_parser_handles_quoted_repair():
    frames = frames_from_directives('!repair diagnostic=bad buggy="(x y)" repaired="(x z)" passed=true')
    assert frames[0].to_jsonable()["buggy_code"] == "(x y)"


def test_hybrid_membrane_rejects_soft_by_default():
    current = CandidateState({"slot": frozenset({"a", "b"})})
    proposed = CandidateState({"slot": frozenset({"a"})})
    proposal = LatticeProposal(proposed_state=proposed, soundness=SoundnessType.EXPERIENCE_SOUND_DEAD)
    result = certify_and_apply(current, proposal)
    assert not result.accepted
    assert result.after == current


def test_hybrid_membrane_accepts_env_sound_refinement():
    current = CandidateState({"slot": frozenset({"a", "b"})})
    proposed = CandidateState({"slot": frozenset({"a"})})
    proposal = LatticeProposal(proposed_state=proposed, soundness=SoundnessType.ENV_SOUND_DEAD, mode=HybridMode.DEDUCE)
    result = certify_and_apply(current, proposal)
    assert result.accepted
    assert result.after.domains["slot"] == frozenset({"a"})
