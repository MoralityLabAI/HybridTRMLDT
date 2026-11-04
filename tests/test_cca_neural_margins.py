from itertools import combinations, product
from math import inf, isclose
import random

import pytest

from research_gym.core.cca_neural_margins import (
    adversarial_logit_witness, anchored_hidden_l2_radius,
    certified_ranked_action, certify_logit_infinity,
    guaranteed_safe_logit_noise, winner,
)

ACTIONS = ("a", "b", "c")


def test_exhaustive_three_action_gap_certificate_and_witness():
    epsilons = (0., 0.01, 0.1, 0.5, 1., 2.)
    for z in product((-2., -1., 0., 1., 2.), repeat=3):
        for mask in product((False, True), repeat=3):
            allowed = frozenset(a for a, keep in zip(ACTIONS, mask) if keep)
            cert = certify_logit_infinity(z, ACTIONS, allowed)
            assert cert.selected == winner(z, ACTIONS)
            assert cert.viable == bool(allowed)
            for eps in epsilons:
                perturbed = adversarial_logit_witness(z, ACTIONS, allowed, eps)
                actual_safe = winner(perturbed, ACTIONS) in allowed
                if guaranteed_safe_logit_noise(cert, eps):
                    assert actual_safe
                if 0 < len(allowed) < len(ACTIONS) and eps > cert.radius:
                    assert not actual_safe
                choice = certified_ranked_action(perturbed, ACTIONS, allowed)
                assert (choice in allowed) if allowed else choice is None


def test_tie_priority_and_certification_at_boundary():
    assert winner((0, 0, 0), ACTIONS) == "a"
    cert = certify_logit_infinity((1., 0., 0.), ACTIONS, {"a"})
    assert cert.radius == 0.5
    assert guaranteed_safe_logit_noise(cert, 0.499)
    assert not guaranteed_safe_logit_noise(cert, 0.5)
    assert winner(adversarial_logit_witness((1., 0., 0.), ACTIONS, {"a"}, .51),
                  ACTIONS) != "a"


def test_anchored_hidden_radius_and_adversarial_vectors():
    rows = ((1., 0.), (0., 1.), (-1., 0.))
    logits = (2., 1., 0.)
    cert = anchored_hidden_l2_radius(logits, rows, ACTIONS, {"a"})
    assert cert == pytest.approx(1 / (2**0.5))
    rng = random.Random(19)
    for _ in range(250):
        x, y = rng.uniform(-1, 1), rng.uniform(-1, 1)
        scale = (x*x+y*y)**0.5 or 1.
        perturb = tuple(q * 0.99 * cert / scale for q in (x, y))
        altered = tuple(z+sum(a*b for a,b in zip(row, perturb))
                        for z,row in zip(logits, rows))
        assert winner(altered, ACTIONS) == "a"
    # This is a conservative anchor certificate: another safe competitor
    # may remain a winner after the original anchor loses to an unsafe action.
    assert anchored_hidden_l2_radius(logits, rows, ACTIONS, set(ACTIONS)) == inf
    assert anchored_hidden_l2_radius(logits, rows, ACTIONS, set()) is None
    assert anchored_hidden_l2_radius(logits, rows, ACTIONS, {"b"}) == 0


def test_cannot_self_attest_action_set_or_unknown_logits():
    with pytest.raises(ValueError):
        certify_logit_infinity((0., 1., 2.), ACTIONS, {"invented"})
    with pytest.raises(ValueError):
        winner((1., float("nan"), 3.), ACTIONS)
    with pytest.raises(ValueError):
        winner((1., float("inf"), 3.), ACTIONS)
    with pytest.raises(ValueError):
        guaranteed_safe_logit_noise(certify_logit_infinity((1., 2., 3.), ACTIONS, {"c"}), -1)
    with pytest.raises(ValueError):
        anchored_hidden_l2_radius((1., 2., 3.), ((0.,), (1.,)), ACTIONS, {"c"})


def test_logit_certificate_does_not_claim_upstream_state_safety():
    cert = certify_logit_infinity((2., 1., 0.), ACTIONS, {"a"})
    assert cert.radius == .5
    assert guaranteed_safe_logit_noise(cert, .2)
    # Re-labeling the environment's permitted set invalidates this particular
    # certificate. The permission vocabulary is never learned from logits.
    wrong_context = certify_logit_infinity((2., 1., 0.), ACTIONS, {"c"})
    assert not guaranteed_safe_logit_noise(wrong_context, .2)


def test_preference_winner_set_is_robust_to_all_endpoint_noises():
    from research_gym.core.cca_neural_margins import (
        potentially_selected_certified, shielded_identity_radius
    )
    for logits in product((-1., 0., 1.), repeat=3):
        for mask in product((False, True), repeat=3):
            allowed = frozenset(a for a,keep in zip(ACTIONS,mask) if keep)
            for eps in (0., 0.25, 0.75, 2.):
                possible = potentially_selected_certified(logits, ACTIONS, allowed, eps)
                assert bool(possible) == bool(allowed)
                for corners in product((-eps, eps), repeat=3):
                    noisy = [z+e for z,e in zip(logits,corners)]
                    choice = certified_ranked_action(noisy, ACTIONS, allowed)
                    if choice is not None:
                        assert choice in possible
                radius = shielded_identity_radius(logits, ACTIONS, allowed)
                if allowed and radius is not None and eps < radius:
                    nominal = certified_ranked_action(logits, ACTIONS, allowed)
                    assert possible == (nominal,)


def test_utility_may_degrade_inside_certified_set():
    from research_gym.core.cca_neural_margins import (
        potentially_selected_certified, shielded_identity_radius
    )
    # CCA-12A's uniform safe-down, unsafe-up witness does not change the
    # relative rankings of certified actions; individual logit noise can.
    actions = ("useful", "bad_but_permitted", "forbidden")
    z = (1.0, 0.95, -2.0)
    allowed = {"useful", "bad_but_permitted"}
    eps = 0.03
    assert shielded_identity_radius(z, actions, allowed) == pytest.approx(0.025)
    assert "bad_but_permitted" in potentially_selected_certified(z, actions, allowed, eps)
    assert certified_ranked_action((1.0-eps, 0.95+eps, -2.0),
                                   actions, allowed) == "bad_but_permitted"
    # Outer shield remains within permission even if preference is reversed.
    assert certified_ranked_action((1.0-eps,0.95+eps,-2.0),actions,allowed) in allowed
