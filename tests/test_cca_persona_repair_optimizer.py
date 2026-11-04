import math
from itertools import combinations
import pytest
from research_gym.core.cca_persona_repair_optimizer import (
    evaluate_retractions, optimal_repair, pareto_repair_frontier,
)

def test_fisher_vs_distance_pareto_tradeoff():
    domain = frozenset("abc")
    safe = frozenset("ab")
    p = {a: 1.0 for a in domain}
    persona = {"a": 0.0, "b": 1.0, "c": 2.0}
    task = {"a": 0.0, "b": 1.0, "c": 0.5}
    dist = lambda a, b: 0.0 if a == b else (1.0 if b == "a" else 3.0)
    scores = evaluate_retractions(domain, safe, p, persona, task, dist)
    assert len(scores) == 2
    by_target = {s.repair("c"): s for s in scores}
    assert math.isclose(by_target["a"].intervention_cost, 1/3)
    assert math.isclose(by_target["b"].intervention_cost, 1.0)
    assert math.isclose(by_target["a"].fisher_loss, 2/3)
    assert math.isclose(by_target["b"].fisher_loss, 1/6)
    assert len(pareto_repair_frontier(scores)) == 2
    assert optimal_repair(scores).repair("c") == "a"
    assert optimal_repair(scores, fisher_weight=4).repair("c") == "b"
    for s in scores:
        assert all(s.repair(a) == a for a in safe)
        assert all(s.repair(a) in safe for a in domain)

def test_exact_enumeration_count_and_certification():
    universe = frozenset(range(5))
    q = {a: 1 for a in universe}
    u = {a: float(a) for a in universe}
    for k in (1, 2, 3):
        for group in combinations(universe, k):
            safe = frozenset(group)
            scores = evaluate_retractions(universe, safe, q, u, u,
                                          lambda a,b: abs(a-b), max_retractions=200)
            assert len(scores) == k ** (5-k)
            assert all(all(s.repair(a) == a for a in safe) for s in scores)
            assert all(s.intervention_cost >= 0 and s.fisher_loss >= -1e-12 for s in scores)

def test_joint_repair_removes_sequential_cycle():
    universe = frozenset({0, 1, 2})
    scores = evaluate_retractions(universe, frozenset({2}),
                                 {a: 1.0 for a in universe},
                                 {a: float(a) for a in universe},
                                 {a: float(a) for a in universe},
                                 lambda a,b: abs(a-b))
    assert len(scores) == 1
    assert all(scores[0].repair(a) == 2 for a in universe)

def test_oversized_search_and_bad_inputs_are_explicit():
    universe = frozenset(range(8))
    safe = frozenset(range(4))
    q = {a: 1 for a in universe}
    u = {a: float(a) for a in universe}
    with pytest.raises(ValueError, match="enumeration cap exceeded"):
        evaluate_retractions(universe, safe, q, u, u,
                             lambda a,b: 1.0, max_retractions=50)
    with pytest.raises(ValueError):
        evaluate_retractions(universe, frozenset(), q, u, u, lambda a,b: 0)
    with pytest.raises(ValueError):
        evaluate_retractions(universe, safe, q, u, u,
                             lambda a,b: float("nan"))
