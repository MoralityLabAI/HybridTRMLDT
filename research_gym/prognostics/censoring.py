"""Native exact and censored observations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class ObservationKind(str, Enum):
    EXACT = "exact"
    LEFT_CENSORED = "left_censored"
    RIGHT_CENSORED = "right_censored"
    INTERVAL_CENSORED = "interval_censored"


@dataclass(frozen=True)
class CensoredObservation:
    kind: ObservationKind
    value: float | None = None
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if self.kind is ObservationKind.EXACT and self.value is None:
            raise ValueError("exact observations require value")
        if self.kind is ObservationKind.LEFT_CENSORED and self.upper is None:
            raise ValueError("left-censored observations require upper")
        if self.kind is ObservationKind.RIGHT_CENSORED and self.lower is None:
            raise ValueError("right-censored observations require lower")
        if self.kind is ObservationKind.INTERVAL_CENSORED:
            if self.lower is None or self.upper is None or self.lower >= self.upper:
                raise ValueError("interval-censored observations require lower < upper")

    def probability(self, *, mean: float, std: float) -> float:
        if std <= 0:
            raise ValueError("predictive standard deviation must be positive")

        def cdf(value: float) -> float:
            return 0.5 * (1.0 + math.erf((value - mean) / (std * math.sqrt(2.0))))

        if self.kind is ObservationKind.EXACT:
            assert self.value is not None
            z = (self.value - mean) / std
            return math.exp(-0.5 * z * z) / (std * math.sqrt(2.0 * math.pi))
        if self.kind is ObservationKind.LEFT_CENSORED:
            assert self.upper is not None
            return cdf(self.upper)
        if self.kind is ObservationKind.RIGHT_CENSORED:
            assert self.lower is not None
            return 1.0 - cdf(self.lower)
        assert self.lower is not None and self.upper is not None
        return cdf(self.upper) - cdf(self.lower)

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "kind": self.kind.value,
            "value": self.value,
            "lower": self.lower,
            "upper": self.upper,
        }
