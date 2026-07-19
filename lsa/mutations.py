"""Bounded, code-free mutation descriptions for schedule proposals."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .canonical import digest
from .normalization import StabilityNormalizationSpec


class MutationFamily(str, Enum):
    TYING = "tying"
    SCHEDULE = "schedule"
    GRADIENT = "gradient"
    RESIDUAL = "residual"
    SUPERVISION = "supervision"
    CARRY = "carry"
    NORMALIZATION = "normalization"


@dataclass(frozen=True)
class AlgebraMutation:
    family: MutationFamily
    name: str
    parameters: Mapping[str, Any]
    normalization: StabilityNormalizationSpec = StabilityNormalizationSpec()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("mutation name must be non-empty")
        self.normalization.validate()

    @property
    def mutation_hash(self) -> str:
        return digest(
            {
                "family": self.family.value,
                "name": self.name,
                "parameters": dict(self.parameters),
                "normalization": {
                    "kappa_applies_residual_scale": self.normalization.kappa_applies_residual_scale,
                    "stability_functional_applies_residual_scale": self.normalization.stability_functional_applies_residual_scale,
                    "control_kind": self.normalization.control_kind.value,
                },
            }
        )

    @property
    def rankable(self) -> bool:
        return self.normalization.rankable
