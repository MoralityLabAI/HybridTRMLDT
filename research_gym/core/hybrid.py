from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, Mapping

from .typed_soundness import SoundnessType, is_hard


class HybridMode(str, Enum):
    """High-level mode selected by a hybrid LDT/TRM controller."""

    DEDUCE = "deduce"
    BRANCH = "branch"
    EXPLORE = "explore"
    ABSTAIN = "abstain"
    EXPAND_ABSTRACTION = "expand_abstraction"


@dataclass(frozen=True)
class CandidateState:
    """Pointwise powerset lattice over symbolic candidates.

    This is the generic lattice needed for LDT-style candidate-survival heads.
    It is intentionally small and independent of torch.
    """

    domains: Mapping[str, FrozenSet[str]]

    @property
    def is_bottom(self) -> bool:
        return any(len(values) == 0 for values in self.domains.values())

    def meet(self, other: "CandidateState") -> "CandidateState":
        keys = set(self.domains) | set(other.domains)
        merged: Dict[str, FrozenSet[str]] = {}
        for key in keys:
            if key not in self.domains or key not in other.domains:
                raise KeyError(f"CandidateState key mismatch: {key}")
            merged[key] = frozenset(self.domains[key] & other.domains[key])
        return CandidateState(merged)

    def join(self, other: "CandidateState") -> "CandidateState":
        keys = set(self.domains) | set(other.domains)
        merged: Dict[str, FrozenSet[str]] = {}
        for key in keys:
            if key not in self.domains or key not in other.domains:
                raise KeyError(f"CandidateState key mismatch: {key}")
            merged[key] = frozenset(self.domains[key] | other.domains[key])
        return CandidateState(merged)

    def is_refinement_of(self, other: "CandidateState") -> bool:
        """Return true when this state is pointwise no wider than `other`."""
        if set(self.domains) != set(other.domains):
            return False
        return all(self.domains[key].issubset(other.domains[key]) for key in self.domains)

    def eliminate(self, slot: str, removed: set[str] | frozenset[str]) -> "CandidateState":
        if slot not in self.domains:
            raise KeyError(slot)
        domains = dict(self.domains)
        domains[slot] = frozenset(v for v in self.domains[slot] if v not in removed)
        return CandidateState(domains)

    def to_jsonable(self) -> Dict[str, list[str]]:
        return {k: sorted(v) for k, v in self.domains.items()}

    @classmethod
    def from_jsonable(cls, data: Mapping[str, list[str]]) -> "CandidateState":
        return cls({k: frozenset(v) for k, v in data.items()})


@dataclass(frozen=True)
class LatticeProposal:
    """A TRM-side proposal expressed as a lattice refinement request."""

    proposed_state: CandidateState
    soundness: SoundnessType
    mode: HybridMode = HybridMode.DEDUCE
    conflict_score: float = 0.0
    soft_store: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "proposed_state": self.proposed_state.to_jsonable(),
            "soundness": self.soundness.value,
            "mode": self.mode.value,
            "conflict_score": self.conflict_score,
            "soft_store": self.soft_store,
            "metadata": self.metadata,
        }

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> "LatticeProposal":
        return cls(
            proposed_state=CandidateState.from_jsonable(data["proposed_state"]),
            soundness=SoundnessType(data["soundness"]),
            mode=HybridMode(data.get("mode", HybridMode.DEDUCE.value)),
            conflict_score=float(data.get("conflict_score", 0.0)),
            soft_store=bool(data.get("soft_store", True)),
            metadata=dict(data.get("metadata", {})),
        )


HybridProposal = LatticeProposal

ProvenanceVerifier = Callable[[LatticeProposal, Any], SoundnessType | None]


@dataclass(frozen=True)
class MembranePolicy:
    """Policy controlling which provenance types may hard-apply."""

    allow_model_sound: bool = False
    allow_experience_sound: bool = False
    store_soft_proposals: bool = True
    provenance_verifier: ProvenanceVerifier | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def can_hard_apply(self, soundness: SoundnessType) -> bool:
        if is_hard(soundness):
            return True
        if soundness == SoundnessType.MODEL_SOUND_DEAD:
            return self.allow_model_sound
        if soundness == SoundnessType.EXPERIENCE_SOUND_DEAD:
            return self.allow_experience_sound
        return False

    def to_jsonable(self) -> Dict[str, bool]:
        return {
            "allow_model_sound": self.allow_model_sound,
            "allow_experience_sound": self.allow_experience_sound,
            "store_soft_proposals": self.store_soft_proposals,
        }

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> "MembranePolicy":
        return cls(
            allow_model_sound=bool(data.get("allow_model_sound", False)),
            allow_experience_sound=bool(data.get("allow_experience_sound", False)),
            store_soft_proposals=bool(data.get("store_soft_proposals", True)),
        )


@dataclass(frozen=True)
class HybridStepResult:
    """Result of passing a proposal through the typed lattice membrane."""

    accepted: bool
    before: CandidateState
    after: CandidateState
    proposal: LatticeProposal
    reason: str
    soft_store: LatticeProposal | None = None
    claimed_soundness: SoundnessType | None = None
    verified_soundness: SoundnessType | None = None
    provenance_disagreed: bool | None = None

    def to_jsonable(self) -> Dict[str, Any]:
        data = {
            "accepted": self.accepted,
            "before": self.before.to_jsonable(),
            "after": self.after.to_jsonable(),
            "proposal": self.proposal.to_jsonable(),
            "reason": self.reason,
            "soft_store": self.soft_store.to_jsonable() if self.soft_store else None,
        }
        # Keep historical traces byte-compatible when no verifier supplied a verdict.
        if self.verified_soundness is not None:
            data.update(
                {
                    "claimed_soundness": self.claimed_soundness.value
                    if self.claimed_soundness
                    else None,
                    "verified_soundness": self.verified_soundness.value,
                    "provenance_disagreed": bool(self.provenance_disagreed),
                }
            )
        return data

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> "HybridStepResult":
        soft_store = data.get("soft_store")
        return cls(
            accepted=bool(data["accepted"]),
            before=CandidateState.from_jsonable(data["before"]),
            after=CandidateState.from_jsonable(data["after"]),
            proposal=LatticeProposal.from_jsonable(data["proposal"]),
            reason=str(data["reason"]),
            soft_store=LatticeProposal.from_jsonable(soft_store) if soft_store else None,
            claimed_soundness=(
                SoundnessType(data["claimed_soundness"])
                if data.get("claimed_soundness")
                else None
            ),
            verified_soundness=(
                SoundnessType(data["verified_soundness"])
                if data.get("verified_soundness")
                else None
            ),
            provenance_disagreed=(
                bool(data["provenance_disagreed"])
                if "provenance_disagreed" in data
                else None
            ),
        )


HybridDecision = HybridStepResult


def _soft_store_for(
    policy: MembranePolicy,
    proposal: LatticeProposal,
    effective_soundness: SoundnessType | None = None,
) -> LatticeProposal | None:
    soft_soundness = {SoundnessType.MODEL_SOUND_DEAD, SoundnessType.EXPERIENCE_SOUND_DEAD}
    soundness = effective_soundness or proposal.soundness
    if policy.store_soft_proposals and proposal.soft_store and soundness in soft_soundness:
        return proposal
    return None


def exact_mechanics_verifier(
    proposal: LatticeProposal,
    context: Any,
) -> SoundnessType | None:
    """Resolve provenance through an environment-owned mechanics callback.

    The core remains environment-agnostic. A context can be a callable, expose a
    ``verify_provenance`` method, or provide ``mechanics_checker`` in a mapping.
    Checkers may return a SoundnessType, bool, or None. A false boolean is typed
    UNKNOWN rather than promoted to a model-derived claim.
    """

    checker: Any = None
    if callable(context):
        checker = context
    elif isinstance(context, Mapping):
        checker = context.get("mechanics_checker")
    elif context is not None:
        checker = getattr(context, "verify_provenance", None)
    if not callable(checker):
        return None
    verdict = checker(proposal)
    if verdict is None or isinstance(verdict, SoundnessType):
        return verdict
    if isinstance(verdict, bool):
        return SoundnessType.ENV_SOUND_DEAD if verdict else SoundnessType.UNKNOWN
    raise TypeError("mechanics checker must return SoundnessType, bool, or None")


def certify_and_apply(
    current: CandidateState,
    proposal: LatticeProposal,
    *,
    allow_soft: bool = False,
    policy: MembranePolicy | None = None,
    verifier_context: Any = None,
) -> HybridStepResult:
    """Apply a proposal only if its provenance permits the requested update.

    Hard updates require environment-sound provenance. Soft model/experience
    updates can be allowed for experiments, but are rejected by default.
    """

    if policy is None:
        policy = MembranePolicy(allow_model_sound=allow_soft, allow_experience_sound=allow_soft)

    verified_soundness = (
        policy.provenance_verifier(proposal, verifier_context)
        if policy.provenance_verifier is not None
        else None
    )
    effective_soundness = verified_soundness or proposal.soundness
    verification_fields = (
        {
            "claimed_soundness": proposal.soundness,
            "verified_soundness": verified_soundness,
            "provenance_disagreed": verified_soundness != proposal.soundness,
        }
        if verified_soundness is not None
        else {}
    )

    if proposal.mode in {HybridMode.EXPLORE, HybridMode.EXPAND_ABSTRACTION}:
        return HybridStepResult(
            False,
            current,
            current,
            proposal,
            "rejected: exploratory proposal is not a certified deduction",
            _soft_store_for(policy, proposal, effective_soundness),
            **verification_fields,
        )

    if not proposal.proposed_state.is_refinement_of(current):
        return HybridStepResult(
            False,
            current,
            current,
            proposal,
            "rejected: non-monotone proposal widens the lattice state",
            _soft_store_for(policy, proposal, effective_soundness),
            **verification_fields,
        )

    if not policy.can_hard_apply(effective_soundness):
        return HybridStepResult(
            False,
            current,
            current,
            proposal,
            f"rejected: {effective_soundness.value} provenance is not hard-applicable by policy",
            _soft_store_for(policy, proposal, effective_soundness),
            **verification_fields,
        )

    refined = current.meet(proposal.proposed_state)
    if refined == current:
        return HybridStepResult(
            True,
            current,
            refined,
            proposal,
            "accepted: no-op monotone refinement",
            **verification_fields,
        )
    if refined.is_bottom and proposal.mode != HybridMode.ABSTAIN:
        return HybridStepResult(
            False,
            current,
            current,
            proposal,
            "rejected: proposal collapses lattice to bottom without abstain mode",
            _soft_store_for(policy, proposal, effective_soundness),
            **verification_fields,
        )
    return HybridStepResult(
        True,
        current,
        refined,
        proposal,
        "accepted: certified monotone refinement",
        **verification_fields,
    )
