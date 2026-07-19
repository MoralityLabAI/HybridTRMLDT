"""Normalization topology and stability-normalization validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NormPlacement(str, Enum):
    PRE = "pre"
    POST = "post"
    SANDWICH = "sandwich"
    NONE = "none"


@dataclass(frozen=True)
class NormSpec:
    placement: NormPlacement
    applications: int
    affine: bool = True

    def __post_init__(self) -> None:
        if self.applications < 0:
            raise ValueError("normalization applications must be non-negative")
        if self.placement is NormPlacement.NONE and self.applications != 0:
            raise ValueError("placement=none requires zero applications")
        if self.placement is not NormPlacement.NONE and self.applications == 0:
            raise ValueError("a normalization placement requires at least one application")


class ControlKind(str, Enum):
    VIABLE = "viable"
    KNOWN_INVALID = "known_invalid"


@dataclass(frozen=True)
class StabilityNormalizationSpec:
    """Record where residual scale is normalized in the stability analysis."""

    kappa_applies_residual_scale: bool = False
    stability_functional_applies_residual_scale: bool = True
    control_kind: ControlKind = ControlKind.VIABLE

    def validate(self) -> None:
        double = (
            self.kappa_applies_residual_scale
            and self.stability_functional_applies_residual_scale
        )
        if double and self.control_kind is not ControlKind.KNOWN_INVALID:
            raise ValueError(
                "residual double normalization is rejected for viable proposals"
            )

    @property
    def rankable(self) -> bool:
        self.validate()
        return self.control_kind is ControlKind.VIABLE
