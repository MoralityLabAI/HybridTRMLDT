"""Map Kimi expert-bank decisions through the repository's certified lattice."""
from __future__ import annotations

from collections.abc import Callable

from research_gym.core.hybrid import (
    CandidateState, LatticeProposal, certify_and_apply_verified,
)
from research_gym.core.typed_soundness import SoundnessType


def certify_bank_subset(
    candidate_indices: list[int],
    allowed_indices: list[int],
    environment_verifier: Callable[[], bool],
) -> tuple[bool, set[int], str]:
    """Require independent environment authority for monotone hard narrowing.

    The callback is provided by the trusted inference/environment harness and
    called once through the real HybridTRMLDT certified membrane.
    """
    current = CandidateState({"experts": frozenset(str(i) for i in candidate_indices)})
    proposed = CandidateState({"experts": frozenset(str(i) for i in allowed_indices)})
    proposal = LatticeProposal(
        proposed_state=proposed, soundness=SoundnessType.ENV_SOUND_DEAD,
    )
    def checker(_proposal: LatticeProposal, _context: object) -> SoundnessType:
        return (SoundnessType.ENV_SOUND_DEAD if environment_verifier() is True
                else SoundnessType.UNKNOWN)
    decision = certify_and_apply_verified(
        current, proposal, provenance_verifier=checker, require_progress=False
    )
    kept = {int(i) for i in decision.after.domains["experts"]}
    return decision.accepted, kept, decision.reason
