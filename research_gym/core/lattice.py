from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping


@dataclass(frozen=True)
class Interval:
    """Closed integer interval with a bottom value represented by low > high."""

    low: int
    high: int

    @classmethod
    def top(cls, low: int, high: int) -> "Interval":
        return cls(low, high)

    @classmethod
    def bottom(cls) -> "Interval":
        return cls(1, 0)

    @property
    def is_bottom(self) -> bool:
        return self.low > self.high

    def meet(self, other: "Interval") -> "Interval":
        return Interval(max(self.low, other.low), min(self.high, other.high))

    def join(self, other: "Interval") -> "Interval":
        if self.is_bottom:
            return other
        if other.is_bottom:
            return self
        return Interval(min(self.low, other.low), max(self.high, other.high))

    def clamp(self, domain: "Interval") -> "Interval":
        return self.meet(domain)

    def contains(self, value: int) -> bool:
        return self.low <= value <= self.high

    def values(self) -> Iterable[int]:
        if self.is_bottom:
            return []
        return range(self.low, self.high + 1)


@dataclass(frozen=True)
class Box:
    """Pointwise product lattice over named integer intervals."""

    intervals: Mapping[str, Interval]

    @property
    def is_bottom(self) -> bool:
        return any(v.is_bottom for v in self.intervals.values())

    def meet(self, other: "Box") -> "Box":
        keys = set(self.intervals) | set(other.intervals)
        merged: Dict[str, Interval] = {}
        for key in keys:
            if key not in self.intervals or key not in other.intervals:
                raise KeyError(f"Box key mismatch: {key}")
            merged[key] = self.intervals[key].meet(other.intervals[key])
        return Box(merged)

    def join(self, other: "Box") -> "Box":
        keys = set(self.intervals) | set(other.intervals)
        merged: Dict[str, Interval] = {}
        for key in keys:
            if key not in self.intervals or key not in other.intervals:
                raise KeyError(f"Box key mismatch: {key}")
            merged[key] = self.intervals[key].join(other.intervals[key])
        return Box(merged)

    def contains_point(self, point: Mapping[str, int]) -> bool:
        return all(self.intervals[k].contains(v) for k, v in point.items())

    @classmethod
    def from_points(cls, points: Iterable[Mapping[str, int]], keys: Iterable[str]) -> "Box":
        points = list(points)
        keys = list(keys)
        if not points:
            return cls({k: Interval.bottom() for k in keys})
        return cls({k: Interval(min(p[k] for p in points), max(p[k] for p in points)) for k in keys})
