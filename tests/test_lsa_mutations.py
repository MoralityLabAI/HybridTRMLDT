from __future__ import annotations

import pytest

from lsa.mutations import AlgebraMutation, MutationFamily
from lsa.normalization import ControlKind, StabilityNormalizationSpec


def test_double_residual_normalization_is_rejected_for_viable_proposal() -> None:
    with pytest.raises(ValueError, match="double normalization"):
        AlgebraMutation(
            MutationFamily.NORMALIZATION,
            "invalid-double-scale",
            {},
            StabilityNormalizationSpec(True, True, ControlKind.VIABLE),
        )


def test_known_invalid_double_normalization_is_retained_but_not_rankable() -> None:
    control = AlgebraMutation(
        MutationFamily.NORMALIZATION,
        "known-invalid-double-scale",
        {},
        StabilityNormalizationSpec(True, True, ControlKind.KNOWN_INVALID),
    )

    assert not control.rankable


def test_mutation_hash_is_deterministic() -> None:
    left = AlgebraMutation(MutationFamily.SCHEDULE, "loop-4", {"rounds": 4})
    right = AlgebraMutation(MutationFamily.SCHEDULE, "loop-4", {"rounds": 4})

    assert left.mutation_hash == right.mutation_hash
