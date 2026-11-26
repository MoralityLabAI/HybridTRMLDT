"""Exact strict-prefix recurrence fixture, independent of neural implementations."""
from __future__ import annotations

from collections.abc import Sequence


def validate(parents: Sequence[Sequence[int]]) -> None:
    if not parents:
        raise ValueError("empty graph")
    for i, row in enumerate(parents):
        if len(set(row)) != len(row) or any(type(j) is not int or not 0 <= j < i for j in row):
            raise ValueError("duplicate or non-prefix dependency")


def bound(parents: Sequence[Sequence[int]]) -> int:
    validate(parents)
    depths: list[int] = []
    for row in parents:
        depths.append(1 + max((depths[j] for j in row), default=0))
    return max(depths)


def _check(x: Sequence[int], parents: Sequence[Sequence[int]], modulus: int) -> None:
    validate(parents)
    if len(x) != len(parents) or modulus <= 1:
        raise ValueError("shape or modulus mismatch")


def synchronous(x: Sequence[int], parents: Sequence[Sequence[int]],
                initial: Sequence[int], loops: int, modulus: int = 17) -> tuple[int, ...]:
    _check(x, parents, modulus)
    if len(initial) != len(x) or loops < 0:
        raise ValueError("shape or loop mismatch")
    state = tuple(initial)
    for _ in range(loops):
        state = tuple((v + sum(state[j] for j in parents[i])) % modulus for i, v in enumerate(x))
    return state


def topological(x: Sequence[int], parents: Sequence[Sequence[int]],
                modulus: int = 17) -> tuple[int, ...]:
    _check(x, parents, modulus)
    state: list[int] = []
    for i, v in enumerate(x):
        state.append((v + sum(state[j] for j in parents[i])) % modulus)
    return tuple(state)


def invalidate(parents: Sequence[Sequence[int]], changed: set[int]) -> set[int]:
    """Structural descendants, including changed nodes; never silently widen a hard state."""
    validate(parents)
    if not changed <= set(range(len(parents))):
        raise ValueError("unknown changed node")
    affected = set(changed)
    for i, row in enumerate(parents):
        if any(j in affected for j in row):
            affected.add(i)
    return affected
