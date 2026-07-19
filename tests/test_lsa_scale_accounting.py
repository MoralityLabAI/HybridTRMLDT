from __future__ import annotations

from lsa.cost import CostSpec, RunContext, estimate_decoder_cost
from lsa.instances import ArchitectureInstance, ModuleSpec, ResidualSpec, SupervisionPoint
from lsa.invariants import (
    applied_parameter_count,
    gradient_applied_parameter_count,
    parameter_count,
)
from lsa.scale import DEFAULT_SCALE_LADDER, solve_decoder_width
from lsa.schedule import FullGradient, LastKGradient, VisitSymbol, parse_word


def _loop(*, tied: bool, rounds: int, mask=None) -> ArchitectureInstance:
    symbols = {}
    modules = {}
    names = []
    for index in range(rounds):
        module = "shared" if tied else f"block_{index}"
        name = f"V{index}"
        modules.setdefault(
            module, ModuleSpec(module, 2, 100, 200, ResidualSpec(1.0, 1.0))
        )
        symbols[name] = VisitSymbol(name, module, frozenset({"stream"}), "stream")
        names.append(name)
    word = parse_word(".".join(names), symbols)
    return ArchitectureInstance(
        name="loop",
        states=frozenset({"stream"}),
        modules=modules,
        word=word,
        gradient_mask=mask or FullGradient(),
        supervision=(SupervisionPoint(rounds - 1),),
    )


def test_tied_loop_count_changes_applied_not_unique_parameters() -> None:
    two = _loop(tied=True, rounds=2)
    four = _loop(tied=True, rounds=4)

    assert parameter_count(two) == parameter_count(four) == 100
    assert applied_parameter_count(two) == 200
    assert applied_parameter_count(four) == 400


def test_untying_visits_increases_unique_parameters_only() -> None:
    tied = _loop(tied=True, rounds=4)
    untied = _loop(tied=False, rounds=4)

    assert parameter_count(tied) == 100
    assert parameter_count(untied) == 400
    assert applied_parameter_count(tied) == applied_parameter_count(untied) == 400


def test_gradient_applied_parameters_follow_parameter_mask() -> None:
    instance = _loop(tied=True, rounds=4, mask=LastKGradient(2))

    assert gradient_applied_parameter_count(instance) == 200


def test_every_scale_rung_lands_within_declared_tolerance() -> None:
    for rung in DEFAULT_SCALE_LADDER:
        shape = solve_decoder_width(rung.target_unique_parameters, tolerance=0.03)
        assert shape.relative_error <= 0.03


def test_transformer_cost_depends_on_sequence_and_gradient_visibility() -> None:
    shape = solve_decoder_width(5_000_000)
    spec = CostSpec(
        unique_parameters=shape.unique_parameters,
        parameter_bytes=shape.unique_parameters * 2,
        attention_cost_model="quadratic",
        activation_cost_model="decoder_v0",
        embedding_parameters=shape.embedding_parameters,
        core_parameters=shape.core_parameters,
    )
    short = estimate_decoder_cost(
        cost=spec,
        context=RunContext(64, 1, 2, True, "eager"),
        module_parameters=shape.core_parameters // 2,
        expanded_visits=4,
        gradient_visible_visits=1,
        hidden_size=shape.hidden_size,
    )
    long = estimate_decoder_cost(
        cost=spec,
        context=RunContext(128, 1, 2, True, "eager"),
        module_parameters=shape.core_parameters // 2,
        expanded_visits=4,
        gradient_visible_visits=4,
        hidden_size=shape.hidden_size,
    )

    assert long.estimated_flops > 2 * short.estimated_flops
    assert long.estimated_activation_bytes > short.estimated_activation_bytes
