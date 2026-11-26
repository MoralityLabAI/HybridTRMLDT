"""Host-owned frame/epoch membrane and exact scalar incremental cache.

The cache is a mathematical fixture, not neural K/V reuse.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from research_gym.core.hybrid import CandidateState, LatticeProposal, certify_and_apply_verified
from research_gym.core.typed_soundness import SoundnessType

from .prefix_memory import validate


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def freeze(state: CandidateState) -> CandidateState:
    return CandidateState(MappingProxyType({k: frozenset(v) for k, v in state.domains.items()}))


@dataclass(frozen=True)
class Premise:
    source: str
    version: int
    slot: str
    excluded: str


@dataclass(frozen=True)
class Snapshot:
    frame: int
    epoch: int
    key: str
    state: CandidateState


@dataclass(frozen=True)
class Receipt:
    accepted: bool
    reason: str
    before_key: str
    after_key: str
    frame: int
    epoch: int


class FrameMachine:
    """Serializes host-owned premise edits and verified monotone refinements."""

    def __init__(self, universe: CandidateState, premises: tuple[Premise, ...], root: str):
        if not root or universe.is_bottom or not universe.domains:
            raise ValueError("invalid root or universe")
        self._universe = freeze(universe)
        self._root = root
        self._facts = self._validate_facts(premises)
        self._state = self._universe
        self._frame = 0
        self._epoch = 0
        self._history: list[Snapshot] = [self.snapshot()]

    def _validate_facts(self, facts: tuple[Premise, ...]) -> tuple[Premise, ...]:
        if len({f.source for f in facts}) != len(facts):
            raise ValueError("duplicate source")
        for f in facts:
            if (not f.source or type(f.version) is not int or f.version < 1
                    or f.slot not in self._universe.domains
                    or f.excluded not in self._universe.domains[f.slot]):
                raise ValueError("invalid premise")
        return tuple(sorted(facts, key=lambda f: f.source))

    @property
    def history(self) -> tuple[Snapshot, ...]:
        return tuple(self._history)

    def snapshot(self) -> Snapshot:
        state = freeze(self._state)
        key = digest([self._root, self._frame, self._epoch, state.to_jsonable(),
                      [(f.source, f.version, f.slot, f.excluded) for f in self._facts]])
        return Snapshot(self._frame, self._epoch, key, state)

    def _exact_state(self) -> CandidateState:
        domains = {k: set(v) for k, v in self._universe.domains.items()}
        for f in self._facts:
            domains[f.slot].discard(f.excluded)
        return freeze(CandidateState(domains))

    def apply(self, snapshot_key: str, proposal: LatticeProposal) -> Receipt:
        before = self.snapshot()
        if snapshot_key != before.key:
            return Receipt(False, "stale_snapshot", before.key, before.key, before.frame, before.epoch)
        exact = self._exact_state()

        def checker(p: LatticeProposal, _context: Any) -> SoundnessType:
            # Every genuinely possible candidate must survive; provenance claims
            # cannot establish that an elimination is warranted.
            if (set(p.proposed_state.domains) == set(exact.domains)
                    and exact.is_refinement_of(p.proposed_state)):
                return SoundnessType.ENV_SOUND_DEAD
            return SoundnessType.UNKNOWN

        result = certify_and_apply_verified(before.state, proposal, provenance_verifier=checker)
        if result.accepted:
            self._state = freeze(result.after)
            self._epoch += 1
            self._history.append(self.snapshot())
        after = self.snapshot()
        return Receipt(result.accepted, result.reason, before.key, after.key, after.frame, after.epoch)

    def replace_premise(self, new: Premise) -> Snapshot:
        """Trusted environment operation; reset public state from current premises."""
        old = next((f for f in self._facts if f.source == new.source), None)
        if old is None or new.version <= old.version:
            raise ValueError("unknown or non-advancing premise version")
        facts = tuple(new if f.source == new.source else f for f in self._facts)
        self._facts = self._validate_facts(facts)
        self._frame += 1
        self._epoch = 0
        self._state = self._exact_state()
        self._history.append(self.snapshot())
        return self.snapshot()


@dataclass(frozen=True)
class CacheResult:
    values: tuple[int, ...]
    recomputed: tuple[int, ...]
    reused: tuple[int, ...]


class ScalarBank:
    """Exact fixed-point cache with dependency-local refresh.

    Context is an external content binding. A change conservatively invalidates all
    entries. Within one context, graph and input edits invalidate their union closure.
    """

    def __init__(self, modulus: int = 17):
        if type(modulus) is not int or modulus <= 1:
            raise ValueError("invalid modulus")
        self._modulus = modulus
        self._state: tuple[int, ...] = ()
        self._inputs: tuple[int, ...] = ()
        self._parents: tuple[tuple[int, ...], ...] = ()
        self._context: str | None = None

    def evaluate(self, inputs: tuple[int, ...], parents: tuple[tuple[int, ...], ...],
                 context: str) -> CacheResult:
        inputs = tuple(inputs)
        parents = tuple(tuple(sorted(row)) for row in parents)
        validate(parents)
        if not context or len(inputs) != len(parents) or any(type(x) is not int for x in inputs):
            raise ValueError("invalid inputs or context")
        n = len(inputs)
        if self._context != context or len(self._state) != n:
            affected = set(range(n))
        else:
            affected = {i for i in range(n)
                        if inputs[i] != self._inputs[i] or parents[i] != self._parents[i]}
            # Both graphs use the same strict prefix order. A single forward
            # sweep computes the descendant closure in their union.
            for i in range(n):
                if affected.intersection(parents[i]) or affected.intersection(self._parents[i]):
                    affected.add(i)
        values: list[int] = []
        for i in range(n):
            if i in affected:
                values.append((inputs[i] + sum(values[j] for j in parents[i])) % self._modulus)
            else:
                values.append(self._state[i])
        self._state = tuple(values)
        self._inputs = inputs
        self._parents = parents
        self._context = context
        return CacheResult(self._state, tuple(sorted(affected)), tuple(i for i in range(n) if i not in affected))
