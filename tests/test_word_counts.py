from __future__ import annotations

import pytest

from lsa.instances import (
    ArchitectureInstance,
    CarrySpec,
    ModuleSpec,
    ResidualSpec,
    SupervisionPoint,
    extraction_registry,
)
from lsa.invariants import (
    flops_per_token,
    gradient_residual_visit_count,
    gradient_rounds,
    gradient_visible_labels,
    parameter_count,
    predicted_minimal_exponent,
    residual_visit_count,
    rounds,
    stability_functional,
)
from lsa.schedule import (
    ExplicitGradient,
    FullGradient,
    HRMTextWarmupGradient,
    LastKGradient,
    LastOuterCycleGradient,
    VisitSymbol,
    parse_word,
)


def _hrm_text_instance(bp_steps: int = 5) -> ArchitectureInstance:
    symbols = {
        "L": VisitSymbol("L", "low", frozenset({"z_L", "z_H"}), "z_L"),
        "H": VisitSymbol("H", "high", frozenset({"z_H", "z_L"}), "z_H"),
    }
    word = parse_word("(L^3.H)^2", symbols)
    return ArchitectureInstance(
        name="hrm-text-hand-count",
        states=frozenset({"z_H", "z_L"}),
        modules={
            "low": ModuleSpec("low", 2, 100, 20, ResidualSpec(1.0, 1.0)),
            "high": ModuleSpec("high", 4, 200, 40, ResidualSpec(1.0, 0.5)),
        },
        word=word,
        gradient_mask=HRMTextWarmupGradient(bp_steps, h_cycles=2, l_cycles=3),
        supervision=(SupervisionPoint(len(word.expand()) - 1),),
        carry=CarrySpec(),
    )


def test_hrm_text_bp_steps_five_reproduces_corrected_visible_suffix() -> None:
    instance = _hrm_text_instance()

    assert gradient_visible_labels(instance) == ("H1", "L4", "L5", "L6", "H2")
    assert set(gradient_visible_labels(instance)) == {"L4", "L5", "L6", "H1", "H2"}
    assert gradient_rounds(instance)["high"] == 2


def test_word_invariants_use_physical_module_identity() -> None:
    instance = _hrm_text_instance()

    assert rounds(instance) == {"low": 6, "high": 2}
    assert residual_visit_count(instance) == 20
    assert gradient_residual_visit_count(instance) == 14
    assert parameter_count(instance) == 300
    assert flops_per_token(instance) == 200
    assert stability_functional(instance, {"low": 2.0, "high": 1.0}) == pytest.approx(14.0)


def test_parser_and_named_masks() -> None:
    x = VisitSymbol("X", "shared", frozenset({"state"}), "state")
    word = parse_word("(X^2.X)^2", {"X": x})
    visits = word.expand()

    assert tuple(visit.label for visit in visits) == tuple(f"X{i}" for i in range(1, 7))
    assert FullGradient().select(visits) == frozenset(range(6))
    assert LastKGradient(2).select(visits) == frozenset({4, 5})
    assert LastOuterCycleGradient(3).select(visits) == frozenset({3, 4, 5})
    assert ExplicitGradient(frozenset({0, 5})).select(visits) == frozenset({0, 5})


def test_registry_never_marks_unresolved_hybrid_training_fields_verified() -> None:
    rows = {row.name: row for row in extraction_registry()}

    assert rows["HRM-Text bp_warmup"].complete
    assert not rows["Gym Hybrid TRM/LDT"].complete
    assert rows["Gym Hybrid TRM/LDT"].cells["g"].provenance.status.value == "FILL"


def test_predicted_exponent_uses_binding_module() -> None:
    assert predicted_minimal_exponent({"low": 0.4, "high": 0.8}) == pytest.approx(0.45)
