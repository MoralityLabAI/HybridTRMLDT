from __future__ import annotations

from lsa.topology import (
    canonical_module_word,
    enumerate_schedule_grammar,
    is_known_schedule_pattern,
    minimal_period,
    periodic_control,
    random_schedule_null,
    schedule_descriptors,
    select_discovery_batches,
)


def test_module_relabeling_has_one_canonical_identity() -> None:
    assert canonical_module_word((7, 3, 7, 9, 3)) == (0, 1, 0, 2, 1)
    assert canonical_module_word((4, 8, 4, 2, 8)) == (0, 1, 0, 2, 1)


def test_grammar_contains_only_balanced_primitive_unknown_words() -> None:
    pool = enumerate_schedule_grammar()

    assert pool
    for topology in pool:
        word = topology.macro_word
        counts = [word.count(module) for module in range(topology.physical_modules)]
        assert max(counts) - min(counts) <= 1
        assert minimal_period(word) == len(word)
        assert not is_known_schedule_pattern(word)


def test_discovery_and_reserve_batches_are_deterministic_and_disjoint() -> None:
    pool = enumerate_schedule_grammar()
    first, reserve = select_discovery_batches(pool)
    repeated_first, repeated_reserve = select_discovery_batches(pool)

    assert len(first) == 12
    assert len(reserve) == 12
    assert first == repeated_first
    assert reserve == repeated_reserve
    assert {item.topology_hash for item in first}.isdisjoint(
        item.topology_hash for item in reserve
    )
    assert {
        (item.physical_modules, item.train_visits) for item in first
    } == {(modules, length) for modules in (2, 3, 4) for length in (6, 8)}


def test_controls_are_stable_and_not_candidates() -> None:
    pool = enumerate_schedule_grammar()
    control = periodic_control(3, 8)
    null = random_schedule_null(pool, modules=3, length=8)

    assert control.topology_kind == "fixed_periodic_control"
    assert null.topology_kind == "balanced_schedule_null"
    assert control.topology_hash != null.topology_hash
    assert schedule_descriptors(null.macro_word).module_balance == 1.0 - (1.0 / 3.0)
