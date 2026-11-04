"""CCA-05: proof-carrying execution and bounded certified repair.

The proposer only nominates actions. Execution requires a certificate that a
host-owned checker accepts for the *current* context, and every repair
candidate passes the same checker. Admissibility is never read from proposer
scores: scores may order certified actions, not add or remove them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, FrozenSet, Mapping, Sequence

from .hybrid import (
    CandidateState,
    HybridMode,
    LatticeProposal,
    ProvenanceVerifier,
    certify_and_apply_verified,
)
from .typed_soundness import SoundnessType


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def bind(context: Mapping[str, Any]) -> str:
    """Canonical binding of everything a verdict depends on."""
    return canonical_json(dict(context))


@dataclass(frozen=True)
class ActionCertificate:
    """Issuer's claim that `action` is admissible in the bound context.

    `successor` is the deployed transition the issuer evaluated and `witness`
    is a checkable continuation. The certificate is evidence for a checker; it
    authorizes nothing by itself.
    """

    binding: str
    action: str
    successor: tuple
    witness: tuple[str, ...]
    issuer: str

    def to_jsonable(self) -> dict[str, object]:
        return {
            "binding": self.binding,
            "action": self.action,
            "successor": list(self.successor),
            "witness": list(self.witness),
            "issuer": self.issuer,
        }

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> "ActionCertificate":
        return cls(
            binding=str(data["binding"]),
            action=str(data["action"]),
            successor=tuple(data["successor"]),
            witness=tuple(str(a) for a in data["witness"]),
            issuer=str(data["issuer"]),
        )

    def digest(self) -> str:
        return canonical_digest(self.to_jsonable())


CertificateChecker = Callable[[Mapping[str, Any], str, ActionCertificate], bool]


def execute_certified(
    context: Mapping[str, Any],
    action: str,
    certificate: ActionCertificate | None,
    checker: CertificateChecker,
    *,
    domain: CandidateState | None = None,
    slot: str = "action",
) -> str | None:
    """Return `action` only if its certificate checks for this exact context.

    A missing certificate, a certificate for another action or context (a stale
    replay), an action outside the current lattice domain, or a checker verdict
    other than True all return None (abstain).
    """
    if certificate is None or certificate.action != action:
        return None
    if certificate.binding != bind(context):
        return None
    if domain is not None and action not in domain.domains.get(slot, frozenset()):
        return None
    return action if checker(context, action, certificate) is True else None


def rank_certified(
    certified: FrozenSet[str],
    scores: Mapping[str, float],
    order: Sequence[str],
) -> tuple[str, ...]:
    """Order certified actions by proposer score, ties by `order`.

    The result is a permutation of `certified`: ranking cannot change the set.
    Non-finite or missing scores raise instead of being silently ordered.
    """
    position = {a: i for i, a in enumerate(order)}
    for a in certified:
        if a not in position:
            raise ValueError(f"action outside the order: {a}")
        if a not in scores or not isfinite(scores[a]):
            raise ValueError(f"missing or non-finite score for {a}")
    return tuple(sorted(certified, key=lambda a: (-scores[a], position[a])))


def lattice_rank(state: CandidateState) -> int:
    """rho(C) = sum_j |C_j| (CCA-01 finite lattice termination)."""
    return sum(len(v) for v in state.domains.values())


@dataclass(frozen=True)
class RepairOutcome:
    action: str | None
    abstained: bool
    proposals: tuple[str, ...]
    ranks: tuple[int, ...]
    admissibility_checks: int
    elimination_checks: int
    reason: str


def bounded_recurrent_repair(
    initial: CandidateState,
    propose: Callable[[CandidateState], str],
    admissible: Callable[[str], bool],
    elimination_verifier: ProvenanceVerifier,
    verifier_context: Any,
    *,
    max_iterations: int,
    slot: str = "action",
    claimed: SoundnessType = SoundnessType.ENV_SOUND_DEAD,
) -> RepairOutcome:
    """Propose, check, and on rejection commit a verified elimination.

    The proposer's recurrence on its latent workspace is invisible here. Only
    eliminations accepted by certify_and_apply_verified change the lattice, and
    each strictly lowers the integer rank, so at most
    min(max_iterations, rank(initial)) refinements are committed. The
    elimination verifier receives {"base": verifier_context, "current": C} so it
    can recompute exactly which actions the proposal removes from C.
    A nomination outside the current domain fails closed.
    """
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    current = initial
    ranks = [lattice_rank(current)]
    proposals: list[str] = []
    checks = eliminations = 0

    def outcome(action: str | None, reason: str) -> RepairOutcome:
        return RepairOutcome(action, action is None, tuple(proposals), tuple(ranks),
                             checks, eliminations, reason)

    for _ in range(max_iterations):
        domain = current.domains[slot]
        if not domain:
            return outcome(None, "abstain: empty domain")
        nominee = propose(current)
        proposals.append(nominee)
        if nominee not in domain:
            return outcome(None, "abstain: nomination outside current domain")
        checks += 1
        if admissible(nominee):
            return outcome(nominee, "executed: admissible nomination")
        target = current.eliminate(slot, {nominee})
        mode = HybridMode.ABSTAIN if not target.domains[slot] else HybridMode.DEDUCE
        proposal = LatticeProposal(target, soundness=claimed, mode=mode,
                                   metadata={"eliminated": nominee})
        eliminations += 1
        result = certify_and_apply_verified(
            current, proposal,
            provenance_verifier=elimination_verifier,
            verifier_context={"base": verifier_context, "current": current},
        )
        if not result.accepted:
            return outcome(None, "abstain: elimination not verified")
        new_rank = lattice_rank(result.after)
        if new_rank >= ranks[-1]:
            raise AssertionError("committed refinement did not lower the rank")
        current = result.after
        ranks.append(new_rank)
    if not current.domains[slot]:
        return outcome(None, "abstain: empty domain")
    return outcome(None, "abstain: iteration bound reached")
