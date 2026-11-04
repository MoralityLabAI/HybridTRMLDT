from itertools import product

import pytest

from research_gym.core.cca_abstraction_refiner import PredicateBank, refine_to_authorize
from research_gym.core.cca_compositional import maximal_sound_actions


def test_refinement_excludes_unsafe_concrete_witnesses():
    states = frozenset(range(8))
    bank = PredicateBank(states, {
        f"bit{i}": {s: bool(s & (1 << i)) for s in states} for i in range(3)
    })
    perms = {s: frozenset({"read"} | ({"write"} if (s & 3) == 3 else set()))
             for s in states}
    receipt = refine_to_authorize(bank, perms, 7, "write")
    assert receipt.authorized
    assert len(receipt.steps) == 2
    assert len(receipt.possible_states) == 2
    assert "write" in receipt.robust_allowed
    for step in receipt.steps:
        assert step.after < step.before
        assert step.removed_counterexamples
        assert all("write" not in perms[s] for s in step.removed_counterexamples)
        assert step.witness in step.removed_counterexamples
        assert step.after == bank.class_of(7, receipt.selected_predicates[
            :receipt.selected_predicates.index(step.predicate) + 1])


def test_all_four_state_permission_functions_never_authorize_unsafe():
    states = frozenset(range(4))
    bank = PredicateBank(states, {
        f"eq{i}": {s: s == i for s in states} for i in states
    })
    for bits in range(16):
        perms = {s: frozenset({"a"} if (bits & (1 << s)) else set()) for s in states}
        for target in states:
            receipt = refine_to_authorize(bank, perms, target, "a")
            if "a" in perms[target]:
                assert receipt.authorized
                assert "a" in receipt.robust_allowed
                assert all("a" in perms[s] for s in receipt.possible_states)
            else:
                assert receipt.status == "denied_at_true_state"
                assert not receipt.authorized
            assert len(receipt.steps) <= len(states) - 1
            for step in receipt.steps:
                before = maximal_sound_actions(step.before, perms)
                after = maximal_sound_actions(step.after, perms)
                assert before <= after


def test_unsplittable_cell_stays_unresolved():
    bank = PredicateBank(frozenset({0, 1}), {})
    perms = {0: frozenset({"a"}), 1: frozenset()}
    receipt = refine_to_authorize(bank, perms, 0, "a")
    assert receipt.status == "unresolved_predicates"
    assert not receipt.authorized
    assert len(receipt.steps) == 0
    assert receipt.possible_states == frozenset({0, 1})


def test_incomplete_or_mutable_predicate_sources_fail():
    with pytest.raises(ValueError):
        PredicateBank(frozenset({0, 1}), {"p": {0: True}})
    with pytest.raises(ValueError):
        PredicateBank(frozenset({0}), {"p": {0: 1}})
    m = {0: False, 1: True}
    bank = PredicateBank(frozenset({0, 1}), {"p": m})
    m[0] = True
    assert bank.values["p"][0] is False
    with pytest.raises(TypeError):
        bank.values["p"][0] = True
    with pytest.raises(ValueError):
        refine_to_authorize(bank, {0: frozenset()}, 0, "a")
    with pytest.raises(ValueError):
        refine_to_authorize(bank, {0: frozenset(), 1: frozenset()}, 0, "a",
                            initial_predicates=("unknown",))


def test_truth_table_split_is_equivalent_to_full_concrete_reasoning():
    states = frozenset(range(4))
    bank = PredicateBank(states, {f"eq{k}": {s: s == k for s in states}
                                  for k in states})
    for flags in product((False, True), repeat=4):
        perms = {s: frozenset({"a", "b"} if flags[s] else {"b"}) for s in states}
        for target in states:
            r = refine_to_authorize(bank, perms, target, "a")
            assert r.authorized == flags[target]
            assert r.robust_allowed <= perms[target]
