from dataclasses import replace
from fractions import Fraction
from itertools import product
import math
import pytest

from experiments.dual_ldt_06.environment import (
    World, State, PERSONAS, PROGRAMS, IDLE, DENOMINATOR, evaluate,
    count_upper_numerator, world, world_ids, input_features, outcomes, initial,
)
from experiments.dual_ldt_06.search import independent_value, verify, ordered_indices, search


def toy():
    return World("toy", (False, False, False), (True, True, True), (1, 1, 1), (4, 4, 4),
                 ((1, 0, 0, 0), (0, 2, 0, 0), (0, 0, 0, 0)),
                 ((0, 0, 0, 0), (0, 4, 0, 0), (0, 0, 0, 0)), (-1, 0, -1), 2, 4)


def test_order_has_a_real_outcome_effect():
    w, theta = toy(), (1, 1, 1, 1)
    forward, reverse = evaluate(w, (3, 4, IDLE)), evaluate(w, (4, 3, IDLE))
    assert forward.feasible and reverse.feasible
    assert forward.numerator(theta) == 5*DENOMINATOR
    assert reverse.numerator(theta) == DENOMINATOR
    assert forward.numerator(theta) - reverse.numerator(theta) == 4*DENOMINATOR


def test_preparation_enables_a_delayed_payoff_and_keeps_host_recovery():
    w = replace(toy(), hazards=(True, False, False), features=((10, 0, 0, 0), (0, 2, 0, 0), (0, 0, 0, 0)))
    assert not evaluate(w, (3, IDLE, IDLE)).feasible
    plan = evaluate(w, (0, 3, IDLE))
    assert plan.feasible and plan.numerator((1, 1, 1, 1)) == 8*DENOMINATOR
    assert all(s.mode in ("active", "maintenance") for layer in plan.levels for s in layer)


def test_quarter_probabilities_and_completion_once():
    w = replace(toy(), probability_quarters=(2, 4, 4))
    # Two chances at project 0 complete with probability 3/4, payoff 1, cost 2.
    r = evaluate(w, (3, 3, IDLE))
    assert Fraction(r.numerator((1, 1, 1, 1)), DENOMINATOR) == Fraction(-5, 4)
    full = evaluate(toy(), (3, 3, IDLE))
    assert full.numerator((1, 1, 1, 1)) == -DENOMINATOR


def test_current_authority_rechecked_at_third_step():
    w = replace(toy(), forbidden_project=0)
    assert evaluate(w, (3, IDLE, IDLE)).feasible
    assert not evaluate(w, (IDLE, IDLE, 3)).feasible
    assert not evaluate(replace(w, initial_credits=2), (0, 1, 2)).feasible


def test_all_branch_check_denies_a_rare_isolated_outcome():
    w = replace(toy(), hazards=(True, False, False))
    branches = outcomes(w, initial(w), 3)
    assert any(s.mode == "isolated" and p > 0 for s, p, _ in branches)
    assert not evaluate(w, (3, IDLE, IDLE)).feasible


def test_exact_forward_and_recursive_values_and_bounds_dev_only():
    for wid in world_ids("dev")[:8]:
        w = world(wid)
        for plan in PROGRAMS:
            result = evaluate(w, plan)
            for theta in PERSONAS.values():
                assert verify(w, plan, theta, result)
                if result.feasible:
                    assert result.numerator(theta) <= count_upper_numerator(w, plan, theta)


def test_anytime_certificates_under_good_and_adversarial_rankings():
    w, theta = toy(), (1, 1, 1, 1)
    evaluations = [evaluate(w, p) for p in PROGRAMS]
    oracle = max(e.numerator(theta) for e in evaluations if e.feasible)
    upper = tuple(count_upper_numerator(w, p, theta) for p in PROGRAMS)
    for scores in (list(upper), [-u for u in upper], [0.]*len(PROGRAMS)):
        rows = search(w, theta, ordered_indices(scores), upper, oracle)
        assert rows[-1]["regret_numerator"] == 0
        assert all(r["regret_numerator"] <= r["bound_gap_numerator"] for r in rows)
        assert [r["return_numerator"] for r in rows] == sorted(r["return_numerator"] for r in rows)
        assert all(evaluate(w, PROGRAMS[r["selected"]]).feasible for r in rows)


def test_forged_utility_record_is_rejected():
    w, plan = toy(), (3, 4, IDLE)
    result = evaluate(w, plan)
    forged = replace(result, outcome_numerators=(9999, 9999, 9999, 9999))
    assert not verify(w, plan, (1, 1, 1, 1), forged)


def test_frozen_ids_and_no_outcome_leakage_in_inputs():
    assert not set(world_ids("train")) & set(world_ids("report"))
    w = world("dev000")
    assert w == world("dev000")
    assert len(input_features(w, PERSONAS["trickster"], PROGRAMS[0])) == 77
    with pytest.raises(ValueError):
        world("report064")
    with pytest.raises(ValueError):
        ordered_indices([float("nan")]*343)
    with pytest.raises(ValueError):
        evaluate(w, (9, 0, 0))
