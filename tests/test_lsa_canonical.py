from __future__ import annotations

from lsa.canonical import algebra_hash, canonical_algebra
from lsa.instances import (
    ArchitectureInstance,
    ModuleSpec,
    ResidualSpec,
    SupervisionPoint,
)
from lsa.schedule import FullGradient, VisitSymbol, parse_word


def _instance(module_name: str, symbol_name: str) -> ArchitectureInstance:
    symbol = VisitSymbol(
        symbol_name, module_name, frozenset({"stream", "input"}), "stream"
    )
    word = parse_word(f"{symbol_name}^3", {symbol_name: symbol})
    return ArchitectureInstance(
        name="cosmetic-name",
        states=frozenset({"stream"}),
        modules={
            module_name: ModuleSpec(
                module_name, 2, 100, 200, ResidualSpec(1.0, 0.5)
            )
        },
        word=word,
        gradient_mask=FullGradient(),
        supervision=(SupervisionPoint(2),),
    )


def test_cosmetic_module_and_symbol_names_do_not_change_algebra_hash() -> None:
    left = _instance("physical_block", "B")
    right = _instance("renamed_tie_class", "LOOP")

    assert canonical_algebra(left) == canonical_algebra(right)
    assert algebra_hash(left) == algebra_hash(right)


def test_schedule_change_changes_algebra_hash() -> None:
    left = _instance("block", "B")
    symbol = VisitSymbol("B", "block", frozenset({"stream", "input"}), "stream")
    word = parse_word("B^4", {"B": symbol})
    right = ArchitectureInstance(
        name="longer",
        states=frozenset({"stream"}),
        modules={"block": left.modules["block"]},
        word=word,
        gradient_mask=FullGradient(),
        supervision=(SupervisionPoint(3),),
    )

    assert algebra_hash(left) != algebra_hash(right)
