"""Proof-checked anytime search; predictive scores are advisory only."""
from __future__ import annotations
from dataclasses import asdict
from fractions import Fraction
from math import isfinite
from time import perf_counter

from experiments.dual_ldt_06.environment import (
    DENOMINATOR, HORIZON, IDLE, PROGRAMS, Evaluation, World, initial,
    invariant, outcomes, evaluate, count_upper_numerator,
)

BUDGETS = (1, 4, 16, 64, 343)
IDLE_INDEX = PROGRAMS.index((IDLE,) * HORIZON)


def independent_value(w: World, plan: tuple[int, ...], theta: tuple[int, ...], counter: list[int] | None = None):
    """Recursive rational replay; fails on ANY positive-probability unsafe branch.

    This is independent of evaluate's forward mass propagation, though both
    deliberately trust the same host transition model.
    """
    def visit(state, t):
        if not invariant(state):
            return None
        if t == len(plan):
            return Fraction(0)
        if counter is not None:
            counter[0] += 1
        branches = outcomes(w, state, plan[t])
        if not branches:
            return None
        if sum(p for _, p, _ in branches) != 4:
            raise AssertionError("invalid probability table")
        value = Fraction(-(plan[t] != IDLE))
        for successor, p, feature in branches:
            continuation = visit(successor, t + 1)
            if continuation is None:
                return None
            value += Fraction(p, 4) * (sum(t * f for t, f in zip(theta, feature)) + continuation)
        return value
    return visit(initial(w), 0)


def verify(w: World, plan: tuple[int, ...], theta: tuple[int, ...], result: Evaluation, counter: list[int] | None = None) -> bool:
    actual = independent_value(w, plan, theta, counter)
    if actual is None:
        return not result.feasible
    return (result.feasible and actual == Fraction(result.numerator(theta), DENOMINATOR)
            and len(result.levels) == HORIZON + 1
            and result.levels[0] == (initial(w),)
            and all(invariant(s) for layer in result.levels for s in layer))


def ordered_indices(scores: list[float]) -> tuple[int, ...]:
    if len(scores) != len(PROGRAMS) or any(not isfinite(float(s)) for s in scores):
        raise ValueError("finite score for every registered program required")
    # Keep the current fallback identical across every method and count its check.
    return (IDLE_INDEX,) + tuple(sorted((i for i in range(len(PROGRAMS)) if i != IDLE_INDEX),
                                       key=lambda i: (-float(scores[i]), i)))


def search(w: World, theta: tuple[int, ...], ranking: tuple[int, ...],
           uppers: tuple[int, ...], oracle: int, ordering_seconds: float = 0.,
           bound_seconds: float = 0.) -> list[dict]:
    """Cold candidate checks; incumbent and upper bound are exact integers / 64.

    Complete all budgets even after an early certificate to keep fixed-budget
    comparisons paired; separately record the first sound stopping certificate.
    """
    if (len(ranking) != len(PROGRAMS) or set(ranking) != set(range(len(PROGRAMS)))
            or ranking[0] != IDLE_INDEX or len(uppers) != len(PROGRAMS)):
        raise ValueError("incomplete search or bound domain")
    incumbent, value, expansions, infeasible = IDLE_INDEX, None, 0, 0
    seen, rows = set(), []
    first_certificate = None
    checker_calls = [0]
    t0 = perf_counter()
    previous_value, previous_upper = None, None
    for k, idx in enumerate(ranking, 1):
        result = evaluate(w, PROGRAMS[idx])
        expansions += result.expansions
        if not verify(w, PROGRAMS[idx], theta, result, checker_calls):
            raise AssertionError("independent trajectory replay disagreed")
        seen.add(idx)
        if result.feasible:
            candidate = result.numerator(theta)
            if candidate > uppers[idx]:
                raise AssertionError("count upper certificate unsound")
            if value is None or candidate > value or (candidate == value and idx < incumbent):
                incumbent, value = idx, candidate
        else:
            infeasible += 1
        if value is None:
            raise AssertionError("idle fallback was not feasible")
        upper = max([value, *(b for j, b in enumerate(uppers) if j not in seen)])
        if previous_value is not None and (value < previous_value or upper > previous_upper):
            raise AssertionError("anytime monotonicity broken")
        if not value <= oracle <= upper:
            raise AssertionError("oracle outside certified regret interval")
        previous_value, previous_upper = value, upper
        if upper == value and first_certificate is None:
            first_certificate = k
        if k in BUDGETS:
            rows.append({
                "budget": k, "selected": incumbent, "return_numerator": value,
                "oracle_numerator": oracle, "regret_numerator": oracle-value,
                "upper_numerator": upper, "bound_gap_numerator": upper-value,
                "exact_optimum_attained": value == oracle,
                "exact_optimum_certified": upper == value,
                "infeasible_proposals": infeasible, "issuer_expansions": expansions,
                "checker_expansions": checker_calls[0],
                "primitive_expansions": expansions + checker_calls[0],
                "ordering_seconds": ordering_seconds, "bound_seconds": bound_seconds,
                "search_seconds": perf_counter()-t0,
                "end_to_end_seconds": perf_counter()-t0 + ordering_seconds + bound_seconds,
                "first_exact_certificate": first_certificate,
            })
    if rows[-1]["regret_numerator"] != 0 or rows[-1]["bound_gap_numerator"] != 0:
        raise AssertionError("exhaustive search failed to establish the true optimum")
    return rows
