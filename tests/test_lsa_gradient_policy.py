from __future__ import annotations

from lsa.gradient_policy import (
    ExplicitStateEdges,
    FullStateEdges,
    GradientPolicy,
    LastKStateEdges,
    NoStateEdges,
)
from lsa.schedule import FullGradient, LastKGradient, VisitSymbol, parse_word


def _visits():
    symbol = VisitSymbol("B", "block", frozenset({"stream"}), "stream")
    return parse_word("B^4", {"B": symbol}).expand()


def test_parameter_and_state_gradient_masks_are_independent() -> None:
    visits = _visits()
    parameter_only = GradientPolicy(FullGradient(), NoStateEdges())
    state_tail = GradientPolicy(LastKGradient(1), FullStateEdges())

    assert parameter_only.parameter_positions(visits) == frozenset({0, 1, 2, 3})
    assert parameter_only.state_edge_positions(visits) == frozenset()
    assert state_tail.parameter_positions(visits) == frozenset({3})
    assert state_tail.state_edge_positions(visits) == frozenset({0, 1, 2})


def test_edge_masks_index_state_transitions_not_visits() -> None:
    visits = _visits()

    assert LastKStateEdges(2).select(visits) == frozenset({2})
    assert ExplicitStateEdges(frozenset({0, 2})).select(visits) == frozenset({0, 2})


def test_legacy_mask_preserves_visible_tail_semantics() -> None:
    visits = _visits()
    policy = GradientPolicy.from_legacy(LastKGradient(2))

    assert policy.parameter_positions(visits) == frozenset({2, 3})
    assert policy.state_edge_positions(visits) == frozenset({2})
