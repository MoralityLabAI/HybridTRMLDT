"""Finite-property and analytic-regression tests of CCA-06."""
from itertools import combinations, product
from math import exp, isclose
import random

import pytest

from research_gym.core.cca_compositional import (
    FiniteRetraction, maximal_sound_actions, scalar_persona_metrics,
    vector_persona_fisher,
)


def all_retractions(universe):
    for size in range(1, len(universe) + 1):
        for subset in combinations(universe, size):
            outsiders = [a for a in universe if a not in subset]
            for destinations in product(subset, repeat=len(outsiders)):
                m = dict(zip(outsiders, destinations))
                m.update({a: a for a in subset})
                yield FiniteRetraction(frozenset(universe), frozenset(subset), m)


def test_all_commuting_retractions_compose_on_four_actions():
    rs = list(all_retractions(tuple(range(4))))
    assert len(rs) == 41
    commuting = 0
    for r in rs:
        for q in rs:
            if r.commutes_with(q):
                commuting += 1
                joint = r.compose(q)
                assert joint.permitted == r.permitted & q.permitted
                assert all(joint(joint(a)) == joint(a) for a in joint.universe)
                assert all(joint(a) in joint.permitted for a in joint.universe)
            else:
                with pytest.raises(ValueError, match="do not commute"):
                    r.compose(q)
    assert commuting == 601


def test_noncommuting_repair_cycle_is_real():
    u = frozenset({0, 1, 2})
    r = FiniteRetraction(u, frozenset({0, 2}), {0: 0, 1: 0, 2: 2})
    q = FiniteRetraction(u, frozenset({1, 2}), {0: 1, 1: 1, 2: 2})
    assert not r.commutes_with(q)
    assert q(0) == 1 and r(1) == 0
    assert r.permitted & q.permitted == frozenset({2})


def test_retraction_rejects_invalid_mapping():
    with pytest.raises(ValueError):
        FiniteRetraction(frozenset({0, 1}), frozenset({0}), {0: 0})
    with pytest.raises(ValueError):
        FiniteRetraction(frozenset({0, 1}), frozenset({0}), {0: 0, 1: 1})
    with pytest.raises(ValueError):
        FiniteRetraction(frozenset({0, 1}), frozenset({0}), {0: 1, 1: 0})


def test_maximal_authorization_soundness_and_refinement():
    u = frozenset(range(4))
    subsets = [frozenset(a for a in u if mask & (1 << a)) for mask in range(16)]
    for safe0, safe1, safe2 in product(subsets, repeat=3):
        permissions = {0: safe0, 1: safe1, 2: safe2}
        coarse = maximal_sound_actions(frozenset({0, 1, 2}), permissions)
        refined = maximal_sound_actions(frozenset({0, 1}), permissions)
        assert coarse <= refined <= safe0
        for a in coarse:
            assert all(a in permissions[s] for s in (0, 1, 2))
    assert maximal_sound_actions(frozenset(), {0: u}) == frozenset()
    assert maximal_sound_actions(frozenset({0, 1}), {0: u}) == frozenset()


def test_persona_repair_reverses_monotone_preference():
    u = frozenset("abc")
    r = FiniteRetraction(u, frozenset("ab"), {"a": "a", "b": "b", "c": "a"})
    q = dict.fromkeys(u, 1.0)
    utility = {"a": 0.0, "b": 1.0, "c": 2.0}
    values = [scalar_persona_metrics(q, utility, r, t) for t in (0.0, 0.5, 1.0, 2.0)]
    assert all(v.post_derivative < 0 for v in values[1:])
    assert all(values[i].mean_post > values[i + 1].mean_post for i in range(3))
    assert isclose(values[0].mean_post, 1.0 / 3.0)
    assert isclose(values[-1].mean_post, exp(2) / (1 + exp(2) + exp(4)))
    assert all(isclose(v.fisher_pre, v.fisher_post + v.fisher_lost, abs_tol=1e-12)
               for v in values)


def test_repaired_utility_derivative_is_covariance():
    random.seed(7)
    u = frozenset(range(5))
    r = FiniteRetraction(u, frozenset({0, 1, 2}), {0: 0, 1: 1, 2: 2, 3: 1, 4: 0})
    q = {a: random.random() + .2 for a in u}
    utility = {a: random.uniform(-3, 3) for a in u}
    for strength in (-.7, 0., .1, 1., 2.5):
        eps = 1e-5
        left = scalar_persona_metrics(q, utility, r, strength - eps).mean_post
        right = scalar_persona_metrics(q, utility, r, strength + eps).mean_post
        observed = (right - left) / (2 * eps)
        exact = scalar_persona_metrics(q, utility, r, strength).post_derivative
        assert isclose(observed, exact, rel_tol=2e-8, abs_tol=2e-8)


def test_vector_fisher_data_processing_identity_and_psd():
    u = frozenset(range(5))
    r = FiniteRetraction(u, frozenset({0, 1, 2}), {0: 0, 1: 1, 2: 2, 3: 1, 4: 0})
    features = {0: (0., 0.), 1: (1., -2.), 2: (2., 1.),
                3: (-1., 3.), 4: (3., -1.)}
    x = vector_persona_fisher({a: 1.0 for a in u}, features, (.3, -.8), r)
    for i in range(2):
        for j in range(2):
            assert isclose(x.before[i][j], x.after[i][j] + x.lost[i][j], abs_tol=1e-12)
    for v in product((-3., -1., 0., 1., 3.), repeat=2):
        qform = sum(v[i] * x.lost[i][j] * v[j] for i in range(2) for j in range(2))
        assert qform >= -1e-12


def test_repairs_preserving_features_lose_no_fisher_information():
    u = frozenset({"x", "y", "z"})
    r = FiniteRetraction(u, frozenset({"x", "y"}), {"x": "x", "y": "y", "z": "x"})
    metrics = vector_persona_fisher(
        {a: 1. for a in u},
        {"x": (1., 3.), "y": (2., 1.), "z": (1., 3.)},
        (.7, -.1), r)
    assert all(abs(x) < 1e-12 for row in metrics.lost for x in row)


def test_statistics_validate_domains():
    r = FiniteRetraction(frozenset({"a"}), frozenset({"a"}), {"a": "a"})
    with pytest.raises(ValueError):
        scalar_persona_metrics({"a": 1.}, {"other": 2.}, r, 0)
    with pytest.raises(ValueError):
        vector_persona_fisher({"a": 1.}, {"a": (1., 2.)}, (0.,), r)


def test_retraction_captures_immutable_mapping():
    m = {0: 0, 1: 0}
    r = FiniteRetraction(frozenset({0, 1}), frozenset({0}), m)
    m[1] = 1
    assert r(1) == 0
    with pytest.raises(TypeError):
        r.outputs[1] = 1
