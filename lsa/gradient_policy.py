"""Separate parameter-gradient visibility from state-graph retention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .schedule import ExpandedVisit, GradientMask


class EdgeMask(Protocol):
    name: str

    def select(self, visits: tuple[ExpandedVisit, ...]) -> frozenset[int]:
        """Select edges by source visit position; edge i connects visit i to i+1."""


@dataclass(frozen=True)
class FullStateEdges:
    name: str = "full-state-edges"

    def select(self, visits: tuple[ExpandedVisit, ...]) -> frozenset[int]:
        return frozenset(range(max(0, len(visits) - 1)))


@dataclass(frozen=True)
class NoStateEdges:
    name: str = "no-state-edges"

    def select(self, visits: tuple[ExpandedVisit, ...]) -> frozenset[int]:
        return frozenset()


@dataclass(frozen=True)
class LastKStateEdges:
    k: int
    name: str = "last-k-state-edges"

    def select(self, visits: tuple[ExpandedVisit, ...]) -> frozenset[int]:
        if self.k < 0:
            raise ValueError("k must be non-negative")
        first_target = max(1, len(visits) - self.k + 1)
        return frozenset(range(first_target - 1, max(0, len(visits) - 1)))


@dataclass(frozen=True)
class ExplicitStateEdges:
    positions: frozenset[int]
    name: str = "explicit-state-edges"

    def select(self, visits: tuple[ExpandedVisit, ...]) -> frozenset[int]:
        valid = frozenset(range(max(0, len(visits) - 1)))
        if not self.positions.issubset(valid):
            raise ValueError("state-edge mask contains an out-of-range edge")
        return self.positions


@dataclass(frozen=True)
class GradientDerivedStateEdges:
    """Compatibility mapping for v0 masks.

    An edge remains live when both adjacent applications are parameter-visible.
    This makes a last-k visit mask retain only the graph inside its visible tail.
    """

    visit_mask: GradientMask
    name: str = "legacy-mask-derived-state-edges"

    def select(self, visits: tuple[ExpandedVisit, ...]) -> frozenset[int]:
        selected = self.visit_mask.select(visits)
        return frozenset(
            source
            for source in range(max(0, len(visits) - 1))
            if source in selected and source + 1 in selected
        )


@dataclass(frozen=True)
class GradientPolicy:
    parameter_visits: GradientMask
    retained_state_edges: EdgeMask
    detach_carry_between_segments: bool = False

    @classmethod
    def from_legacy(cls, mask: GradientMask) -> "GradientPolicy":
        return cls(mask, GradientDerivedStateEdges(mask))

    def parameter_positions(
        self, visits: tuple[ExpandedVisit, ...]
    ) -> frozenset[int]:
        return self.parameter_visits.select(visits)

    def state_edge_positions(
        self, visits: tuple[ExpandedVisit, ...]
    ) -> frozenset[int]:
        return self.retained_state_edges.select(visits)
