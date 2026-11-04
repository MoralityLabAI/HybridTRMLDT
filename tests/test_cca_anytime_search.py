"""Finite exhaustive checks of certified anytime search."""
from itertools import permutations, product
from math import isclose
import pytest
from research_gym.core.cca_anytime_search import (
    UpperCertificate, ActionVerdict, certified_anytime_search,
)

def run(values, safe, order, budget, upper=None, eps=0):
    actions = tuple(values)
    upper = upper or {a: 1. for a in actions}
    bounds = {a: UpperCertificate(a, upper[a], "frozen", "host") for a in actions}
    checks = []
    def check_bound(ctx, a, cert):
        checks.append(("bound", a))
        return ctx["binding"] == cert.binding and cert.evidence == "host"
    def evaluate(ctx, a):
        return ActionVerdict(a, a in safe, values[a] if a in safe else None, "host")
    def check_verdict(ctx, a, verdict):
        checks.append(("verdict", a))
        return (verdict.evidence == "host" and verdict.feasible == (a in safe)
                and verdict.utility == (values[a] if a in safe else None))
    result = certified_anytime_search(
        {"binding": "frozen"}, actions, order, bounds,
        check_bound=check_bound, evaluate=evaluate, check_verdict=check_verdict,
        max_evaluations=budget, epsilon=eps,
    )
    return result, checks

def test_exhaustive_safe_incumbent_and_sound_regret_bounds():
    actions = ("a", "b", "c")
    for utility_values in product((0., 0.3, 1.), repeat=3):
        values = dict(zip(actions, utility_values))
        for flags in product((False, True), repeat=3):
            safe = frozenset(a for a, f in zip(actions, flags) if f)
            for ranking in permutations(actions):
                for budget in range(4):
                    r, checks = run(values, safe, ranking, budget)
                    assert set(r.checked) <= set(actions)
                    if r.action is not None:
                        assert r.action in safe
                        assert r.lower == values[r.action]
                        best = max(values[a] for a in safe)
                        assert best - r.lower <= r.gap + 1e-12
                        assert r.upper + 1e-12 >= best
                    if r.status == "epsilon_optimal":
                        assert r.action is not None and r.gap <= 1e-12
                    if r.status == "proven_empty":
                        assert not safe and len(r.checked) == 3
                    lower = [s.lower for s in r.steps if s.lower is not None]
                    upper = [s.upper for s in r.steps if s.upper is not None]
                    assert lower == sorted(lower)
                    assert upper == sorted(upper, reverse=True)
                    assert len([c for c in checks if c[0] == "bound"]) == 3

def test_verified_bound_certificates_allow_early_optimality():
    values = {"a": .2, "b": .8, "c": .3}
    r, _ = run(values, set(values), ("b", "a", "c"), 3, values)
    assert r.action == "b" and r.status == "epsilon_optimal"
    assert r.checked == ("b",) and r.gap == 0

def test_no_false_optimality_with_only_global_upper():
    values = {"a": .4, "b": .9}
    r, _ = run(values, set(values), ("a", "b"), 1)
    assert r.status == "budget_exhausted"
    assert r.action == "a" and isclose(r.gap, .6)

def test_proven_empty_requires_all_negative_verdicts():
    values = {"a": .2, "b": .5}
    r, _ = run(values, set(), ("b", "a"), 2)
    assert r.status == "proven_empty" and r.action is None
    r, _ = run(values, set(), ("b", "a"), 1)
    assert r.status == "budget_exhausted" and r.action is None

def test_deduplicated_untrusted_proposer_fallback_is_complete():
    values = {"a": .4, "b": 1., "c": .6}
    r, _ = run(values, set(values), ("a", "a"), 3)
    assert r.checked == ("a", "b")
    assert r.skipped_duplicates == 1 and r.status == "epsilon_optimal"
    assert r.action == "b"

def test_unverified_bound_or_negative_verdict_fails_closed():
    proof = {"a": UpperCertificate("a", 1., "somewhere", "host")}
    kw = dict(context={"binding": "frozen"}, actions=("a",),
              proposal_order=("a",), upper_bounds=proof, max_evaluations=1,
              check_bound=lambda ctx, a, cert: False,
              evaluate=lambda ctx, a: ActionVerdict("a", False, None, "bad"),
              check_verdict=lambda ctx, a, verdict: False)
    with pytest.raises(ValueError, match="upper bound"):
        certified_anytime_search(**kw)
    kw["check_bound"] = lambda *_: True
    with pytest.raises(ValueError, match="verdict"):
        certified_anytime_search(**kw)

def test_evaluation_utility_over_verified_upper_bound_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        certified_anytime_search(
            {}, ("a",), ("a",), {"a": UpperCertificate("a", .2, "x", "p")},
            check_bound=lambda *_: True,
            evaluate=lambda *_: ActionVerdict("a", True, .3, "p"),
            check_verdict=lambda *_: True, max_evaluations=1)

def test_outside_domain_rejected_and_invalid_budget_rejected():
    bound = {"a": UpperCertificate("a", 1., "x", "p")}
    kw = dict(context={}, actions=("a",), proposal_order=("invalid",),
              upper_bounds=bound, check_bound=lambda *_: True,
              evaluate=lambda *_: ActionVerdict("a", True, .3, "p"),
              check_verdict=lambda *_: True, max_evaluations=1)
    with pytest.raises(ValueError, match="outside domain"):
        certified_anytime_search(**kw)
    kw["proposal_order"] = ("a",)
    kw["max_evaluations"] = -1
    with pytest.raises(ValueError, match="budget"):
        certified_anytime_search(**kw)
