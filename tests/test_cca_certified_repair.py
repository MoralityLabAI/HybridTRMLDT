import math
import random

import pytest

from research_gym.core.certified_repair import (
    ActionCertificate,
    bind,
    bounded_recurrent_repair,
    execute_certified,
    lattice_rank,
    rank_certified,
)
from research_gym.core.hybrid import CandidateState
from research_gym.core.typed_soundness import SoundnessType

ACTIONS = ("a", "b", "c", "d", "e")


def dom(*values):
    return CandidateState({"action": frozenset(values)})


def recompute_verifier(safe):
    """Elimination authority recomputed from `safe`; the proposal label is ignored."""
    def check(proposal, ctx):
        before = ctx["current"].domains["action"]
        after = proposal.proposed_state.domains["action"]
        removed = before - after
        if not after <= before or not removed:
            return SoundnessType.UNKNOWN
        return SoundnessType.ENV_SOUND_DEAD if not (removed & safe) else SoundnessType.UNKNOWN
    return check


def test_rank_is_a_permutation_with_vocabulary_ties():
    certified = frozenset({"b", "d", "e"})
    ranked = rank_certified(certified, {"a": 9.0, "b": 1.0, "d": 1.0, "e": 2.0}, ACTIONS)
    assert ranked == ("e", "b", "d")
    assert set(ranked) == certified


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_rank_refuses_nonfinite_scores(bad):
    with pytest.raises(ValueError):
        rank_certified(frozenset({"a", "b"}), {"a": bad, "b": 0.0}, ACTIONS)


def test_rank_refuses_missing_score_and_unknown_action():
    with pytest.raises(ValueError):
        rank_certified(frozenset({"a"}), {}, ACTIONS)
    with pytest.raises(ValueError):
        rank_certified(frozenset({"z"}), {"z": 0.0}, ACTIONS)


def test_execute_requires_matching_binding_action_domain_and_checker():
    ctx = {"state": [0, 1], "horizon": 3}
    cert = ActionCertificate(bind(ctx), "a", (1,), ("b",), "test")
    ok = lambda _c, _a, _cert: True
    assert execute_certified(ctx, "a", cert, ok) == "a"
    assert execute_certified({"state": [0, 2], "horizon": 3}, "a", cert, ok) is None
    assert execute_certified(dict(ctx, horizon=2), "a", cert, ok) is None
    assert execute_certified(ctx, "b", cert, ok) is None
    assert execute_certified(ctx, "a", None, ok) is None
    assert execute_certified(ctx, "a", cert, ok, domain=dom("b")) is None
    assert execute_certified(ctx, "a", cert, lambda *_: False) is None
    assert execute_certified(ctx, "a", cert, lambda *_: 1) is None  # truthy is not True


def test_certificate_json_round_trip_and_digest():
    cert = ActionCertificate(bind({"x": 1}), "a", (1, 2), ("b", "c"), "test")
    again = ActionCertificate.from_jsonable(cert.to_jsonable())
    assert again == cert and again.digest() == cert.digest()


def test_bounded_repair_terminates_with_strict_rank_decrease_on_random_instances():
    rng = random.Random(5)
    for _ in range(2000):
        safe = frozenset(a for a in ACTIONS if rng.random() < 0.4)
        prefs = list(ACTIONS)
        rng.shuffle(prefs)
        propose = lambda c: next(a for a in prefs if a in c.domains["action"])
        out = bounded_recurrent_repair(dom(*ACTIONS), propose, lambda a: a in safe,
                                       recompute_verifier(safe), None, max_iterations=5)
        assert all(x > y for x, y in zip(out.ranks, out.ranks[1:]))
        assert len(out.ranks) - 1 <= lattice_rank(dom(*ACTIONS))
        if safe:
            assert out.action == next(a for a in prefs if a in safe)
        else:
            assert out.abstained
        assert out.action is None or out.action in safe


def test_bounded_repair_identity_and_idempotence():
    safe = frozenset({"c", "e"})
    for first in ACTIONS:
        order = [first] + [a for a in ACTIONS if a != first]
        propose = lambda c, o=order: next(a for a in o if a in c.domains["action"])
        out = bounded_recurrent_repair(dom(*ACTIONS), propose, lambda a: a in safe,
                                       recompute_verifier(safe), None, max_iterations=5)
        if first in safe:
            assert out.action == first and out.ranks == (5,)
        again = bounded_recurrent_repair(
            dom(*ACTIONS), lambda c, a=out.action: a, lambda a: a in safe,
            recompute_verifier(safe), None, max_iterations=5)
        assert again.action == out.action


def test_bounded_repair_iteration_bound_and_fail_closed_cases():
    safe = frozenset({"e"})
    propose = lambda c: sorted(c.domains["action"])[0]
    short = bounded_recurrent_repair(dom(*ACTIONS), propose, lambda a: a in safe,
                                     recompute_verifier(safe), None, max_iterations=2)
    assert short.abstained and "bound" in short.reason and len(short.proposals) == 2
    outside = bounded_recurrent_repair(dom("a", "b"), lambda c: "z", lambda a: True,
                                       recompute_verifier(safe), None, max_iterations=5)
    assert outside.abstained and outside.admissibility_checks == 0
    label_only = bounded_recurrent_repair(dom("a", "e"), lambda c: "e" if "a" not in c.domains["action"] else "a",
                                          lambda a: False, lambda *_: SoundnessType.UNKNOWN, None,
                                          max_iterations=5)
    assert label_only.abstained and "not verified" in label_only.reason
    empty = bounded_recurrent_repair(dom(), lambda c: "a", lambda a: True,
                                     recompute_verifier(safe), None, max_iterations=5)
    assert empty.abstained and empty.proposals == ()
