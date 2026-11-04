"""Dual-LDT-05: universal-branch certificates for finite typed action programs.

A certificate binds the program to an environment, initial state, and all
possible intermediate states. Checker recomputes every branch using host
semantics; a model-authored certificate cannot grant permissions by itself.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from typing import Callable, Hashable, Iterable, TypeVar

S = TypeVar("S", bound=Hashable)


@dataclass(frozen=True)
class ProgramProof:
    binding: str
    program_id: str
    steps: tuple[str, ...]
    levels: tuple[tuple[object, ...], ...]
    checker_version: str = "dual05-branch-certificate-v1"


def sound_layers(
    initial: S,
    steps: tuple[str, ...],
    *,
    authorized: Callable[[S, str], bool],
    successor: Callable[[S, str], frozenset[S]],
    invariant: Callable[[S], bool],
) -> tuple[tuple[S, ...], ...] | None:
    """All-branches safety, with *fresh* authorization at every step.

    A plan is refused when any represented branch is illegal, unknown, empty
    or exits the invariant. No branch is discarded to make the plan pass.
    """
    if not steps or not invariant(initial):
        return None
    frontier = frozenset({initial})
    levels: list[tuple[S, ...]] = [tuple(sorted(frontier, key=repr))]
    for primitive in steps:
        next_frontier: set[S] = set()
        for state in frontier:
            if authorized(state, primitive) is not True:
                return None
            options = successor(state, primitive)
            if not options or any(not invariant(s) for s in options):
                return None
            next_frontier.update(options)
        if not next_frontier:
            return None
        frontier = frozenset(next_frontier)
        levels.append(tuple(sorted(frontier, key=repr)))
    return tuple(levels)


def bind_program(world_digest: str, initial: object, program_id: str,
                 steps: tuple[str, ...]) -> str:
    if not world_digest or not program_id or not steps:
        raise ValueError("missing binding content")
    if not hasattr(initial, "__dataclass_fields__"):
        raise TypeError("program binding requires an explicit dataclass state")
    payload = {"world_digest": world_digest,
               "initial": asdict(initial),
               "id": program_id, "steps": list(steps)}
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"))
                  .encode("utf-8")).hexdigest()


def issue_program(
    world_digest: str, initial: S, program_id: str, steps: tuple[str, ...],
    *,
    authorized: Callable[[S, str], bool],
    successor: Callable[[S, str], frozenset[S]],
    invariant: Callable[[S], bool],
) -> ProgramProof | None:
    layers = sound_layers(initial, steps, authorized=authorized,
                          successor=successor, invariant=invariant)
    if layers is None:
        return None
    return ProgramProof(bind_program(world_digest, initial, program_id, steps),
                        program_id, steps, layers)


def check_program(
    certificate: ProgramProof | None, world_digest: str, initial: S,
    program_id: str, steps: tuple[str, ...],
    *,
    authorized: Callable[[S, str], bool],
    successor: Callable[[S, str], frozenset[S]],
    invariant: Callable[[S], bool],
) -> bool:
    """Verifier replays host semantics, including all policy-epoch changes."""
    if not isinstance(certificate, ProgramProof):
        return False
    if (certificate.checker_version != "dual05-branch-certificate-v1"
            or certificate.program_id != program_id
            or certificate.steps != steps):
        return False
    try:
        if certificate.binding != bind_program(world_digest, initial, program_id, steps):
            return False
        expected = sound_layers(initial, steps, authorized=authorized,
                                successor=successor, invariant=invariant)
        return expected is not None and expected == certificate.levels
    except (ValueError, KeyError, TypeError):
        return False
