from fractions import Fraction
from itertools import product
import math
import random

import pytest

from research_gym.core.constitutional_control import (
    Constitution, ConstitutionOp, LaneHead, minimal_repair,
    persona_distribution, robust_tag_certificate,
)


def test_all_constitutions_preserve_floor():
    c = Constitution(frozenset("abcd"), frozenset("abc"),
                     frozenset("ab"), frozenset("bc"))
    expected = {"beast_only": frozenset("bc"),
                "jinn_only": frozenset("ab"),
                "meet": frozenset("b"),
                "parallax": frozenset("abc")}
    for op in ConstitutionOp:
        assert c.allowed(op) == expected[op.value]
        assert c.allowed(op) <= c.hard_safe


def test_every_small_constitution_obeys_boundary():
    universe = frozenset(range(4))
    sets = [frozenset(i for i in universe if bits & (1 << i))
            for bits in range(1 << len(universe))]
    for floor in sets:
        for jinn in sets:
            for beast in sets:
                c = Constitution(universe, floor, jinn, beast)
                for op in ConstitutionOp:
                    assert c.allowed(op) <= floor


def test_reject_margin_matches_exhaustive_binary_domain():
    random.seed(11)
    for _ in range(120):
        n = 5
        weight = lambda: tuple(Fraction(random.randrange(-4, 5)) for _ in range(n))
        h = LaneHead(weight(), weight(), weight(),
                     tuple(Fraction(random.randrange(-4, 5)) for _ in range(3)))
        for j in range(n):
            for lane, w, b in (("train", h.train, h.biases[0]),
                               ("hold", h.hold, h.biases[1])):
                actual = max(b - h.biases[2]
                             + sum((w[i] - h.reject[i]) * bits[i] for i in range(n))
                             for bits in product((0, 1), repeat=n) if bits[j] == 1)
                assert h.max_competing_margin(lane, j) == actual
            safe = all(h.winner(bits) == "reject" for bits in product((0, 1), repeat=n)
                       if bits[j] == 1)
            assert h.certifies_critical_rejection([j]) == safe


def test_ties_are_unsafe_and_empty_critical_list_rejected():
    h = LaneHead((Fraction(0),), (Fraction(0),), (Fraction(0),),
                 (Fraction(0), Fraction(0), Fraction(0)))
    assert not h.certifies_critical_rejection([0])
    with pytest.raises(ValueError):
        h.certifies_critical_rejection([])


def test_unknown_or_critical_possible_tags_blocks():
    ban = frozenset({"critical"})
    assert robust_tag_certificate("text", lambda _: [frozenset({"safe"})], ban)
    assert not robust_tag_certificate("text", lambda _: [], ban)
    assert not robust_tag_certificate("text", lambda _: [frozenset({"safe"}),
                                                           frozenset({"critical"})], ban)


def test_persona_expected_utility_monotone():
    baseline = {"a": 0.7, "b": 0.2, "c": 0.1}
    utility = {"a": -1.0, "b": 0.1, "c": 2.0}
    allowed = frozenset(baseline)
    means = []
    for strength in (0, .2, .7, 2, 8):
        dist = persona_distribution(baseline, utility, allowed, strength)
        assert math.isclose(sum(dist.values()), 1.0, rel_tol=1e-12)
        means.append(sum(dist[a] * utility[a] for a in allowed))
    assert means == sorted(means)
    with pytest.raises(ValueError):
        persona_distribution(baseline, utility, frozenset(), 0)


def test_repair_safe_idempotent_and_abstains():
    allowed = frozenset({"b", "c"})
    d = lambda x, y: abs(ord(x) - ord(y))
    r = minimal_repair("a", allowed, d)
    assert r.action == "b" and not r.abstained
    assert minimal_repair(r.action, allowed, d).action == r.action
    assert minimal_repair("c", allowed, d).action == "c"
    assert minimal_repair("a", frozenset(), d).abstained
    assert minimal_repair("a", frozenset(), d).action is None


def test_viability_kernel_catches_multi_step_trap():
    from research_gym.core.constitutional_control import viability_kernel, shield_actions
    safe = frozenset({"start", "bait", "safe"})
    actions = {"start": frozenset({"tempt", "safe_route"}),
               "bait": frozenset({"continue"}),
               "safe": frozenset({"wait"})}
    succ = {("start", "tempt"): frozenset({"bait"}),
            ("start", "safe_route"): frozenset({"safe"}),
            ("bait", "continue"): frozenset({"bad"}),
            ("safe", "wait"): frozenset({"safe"})}
    win = viability_kernel(safe, actions, succ)
    assert win == frozenset({"start", "safe"})
    assert shield_actions("start", win, actions, succ) == frozenset({"safe_route"})
    assert shield_actions("bait", win, actions, succ) == frozenset()
    for state in win:
        for act in shield_actions(state, win, actions, succ):
            assert succ[(state, act)] <= win


def test_missing_or_empty_transition_fails_closed():
    from research_gym.core.constitutional_control import viability_kernel, shield_actions
    safe = frozenset({"start"})
    actions = {"start": frozenset({"undocumented", "empty"})}
    successors = {("start", "empty"): frozenset()}
    assert viability_kernel(safe, actions, successors) == frozenset()
    assert shield_actions("start", safe, actions, successors) == frozenset()


def test_counterexample_iff_reject_dominance_fails():
    from research_gym.core.constitutional_control import critical_counterexample
    random.seed(17)
    for _ in range(90):
        n = 5
        ws = [tuple(Fraction(random.randrange(-3, 4)) for _ in range(n))
              for _ in range(3)]
        bs = tuple(Fraction(random.randrange(-3, 4)) for _ in range(3))
        h = LaneHead(*ws, bs)
        witness = critical_counterexample(h, [2])
        assert (witness is None) == h.certifies_critical_rejection([2])
        if witness is not None:
            assert witness[2] == 1 and h.winner(witness) != "reject"


def test_structural_veto_preserves_noncritical_choices():
    from research_gym.core.constitutional_control import StructuralVeto
    h = LaneHead((Fraction(3), Fraction(2)), (Fraction(0), Fraction(2)),
                 (Fraction(1), Fraction(-2)), (Fraction(0),) * 3)
    v = StructuralVeto(h, frozenset({1}))
    for bits in product((0, 1), repeat=2):
        assert v.winner(bits) == ("reject" if bits[1] else h.winner(bits))


def test_checkpoint_audit_requires_host_critical_tags():
    from research_gym.core.constitutional_control import audit_saved_head
    record = {
        "feature_names": ["public", "critical"],
        "classes": ["train", "hold", "reject"],
        "weights": [[3, 2], [0, 0], [0, 0]],
        "bias": [0, 0, 0],
        "critical_veto_tags": ["critical"],
    }
    outcome = audit_saved_head(record, frozenset({"critical"}))
    assert not outcome["underlying_linear_reject_certificate"]
    assert not outcome["effective_reject_certificate"]
    verified = audit_saved_head(record, frozenset({"critical"}), wrapper_verified=True)
    assert verified["effective_reject_certificate"]
    record["critical_veto_tags"] = ["public"]
    outcome = audit_saved_head(record, frozenset({"critical"}), wrapper_verified=True)
    assert not outcome["effective_reject_certificate"]


def test_one_pass_reject_projection_matches_structural_veto_everywhere():
    from research_gym.core.constitutional_control import (
        project_reject_weights, StructuralVeto,
    )
    random.seed(63)
    for _ in range(150):
        n = 6
        ws = [tuple(Fraction(random.randrange(-5, 6)) for _ in range(n))
              for _ in range(3)]
        bs = tuple(Fraction(random.randrange(-5, 6)) for _ in range(3))
        head = LaneHead(*ws, bs)
        critical = frozenset({1, 4})
        projected = project_reject_weights(head, critical)
        veto = StructuralVeto(head, critical)
        assert projected.certifies_critical_rejection(critical)
        for bits in product((0, 1), repeat=n):
            assert projected.winner(bits) == veto.winner(bits)
            if not any(bits[j] for j in critical):
                assert projected.winner(bits) == head.winner(bits)


def test_projection_requires_positive_rational_margin():
    from research_gym.core.constitutional_control import project_reject_weights
    h = LaneHead((Fraction(0),), (Fraction(0),), (Fraction(0),),
                 (Fraction(0),) * 3)
    with pytest.raises(ValueError):
        project_reject_weights(h, [0], margin=Fraction(0))
    with pytest.raises(ValueError):
        project_reject_weights(h, [0], margin=0.2)
