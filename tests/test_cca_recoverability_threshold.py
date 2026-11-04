"""Exact finite checks for CCA-O/P/Q and failure-mode regressions."""
from fractions import Fraction as F
from itertools import product

import pytest

from research_gym.core.cca_recoverability_threshold import (
    RecoveryEnvelope, rational, fixed_action_regret_bound,
    stopped_barrier_exit_bound, union_risk_bound,
)


def test_boundary_and_sharpness():
    env = RecoveryEnvelope(F(1, 2), 1, F(1, 10), 1)
    assert env.critical_disturbance() == F(2, 5)
    assert env.equilibrium_bound(F(2, 5)) == 1
    assert env.uniform_viability_certificate(1, F(2, 5))
    assert set(env.envelope_path(1, [F(2, 5)] * 20)) == {F(1)}
    assert not env.uniform_viability_certificate(1, F(41, 100))
    assert env.first_possible_exit(env.envelope_path(1, [F(41, 100)])) == 1
    assert env.first_possible_exit(env.envelope_path(1, [F(2, 5)] * 4)) is None


def test_burst_can_be_absorbed_beyond_uniform_threshold():
    env = RecoveryEnvelope(F(1, 2), 1, 0, 1)
    assert env.critical_disturbance() == F(1, 2)
    assert not env.uniform_viability_certificate(0, F(3, 4))
    path = env.envelope_path(0, [F(3, 4), 0, 0, 0])
    assert path == (0, F(3, 4), F(3, 8), F(3, 16), F(3, 32))
    assert env.first_possible_exit(path) is None


def test_infeasible_even_without_disturbance():
    env = RecoveryEnvelope(F(3, 4), 1, F(3, 10), 1)
    assert env.critical_disturbance() is None
    assert not env.uniform_viability_certificate(0, 0)
    assert env.first_possible_exit(env.envelope_path(1, [0])) == 1


def test_exact_envelope_maximizes_each_step():
    for numerator, denominator, residual, margin, gain in product(
        range(4), (4,), (F(0), F(1, 8), F(1, 2)), (F(1), F(2)), (F(1), F(2))
    ):
        env = RecoveryEnvelope(F(numerator, denominator), gain, residual, margin)
        critical = env.critical_disturbance()
        if critical is None:
            assert not env.uniform_viability_certificate(margin, 0)
            continue
        assert env.uniform_viability_certificate(margin, critical)
        for e in env.envelope_path(margin, [critical] * 8):
            assert e <= margin
        beyond = critical + F(1, 128)
        assert not env.uniform_viability_certificate(margin, beyond)
        assert env.first_possible_exit(env.envelope_path(margin, [beyond])) == 1


def test_nonconstant_disturbance_is_convolution():
    env = RecoveryEnvelope(F(2, 3), 3, F(1, 7), 2)
    noises = [F(1, 10), F(3, 10), 0, F(1, 5)]
    path = env.envelope_path(F(1, 3), noises)
    for t in range(1, len(path)):
        expected = env.rho ** t * path[0]
        expected += sum(
            env.rho ** (t-1-j) * (env.gain * noises[j] + env.residual)
            for j in range(t)
        )
        assert path[t] == expected


def test_fixed_action_regret_bound_is_sharp_for_two_actions():
    e, L = F(3, 8), F(2)
    regret = (L * e) - (-L * e)
    assert regret == fixed_action_regret_bound(e, L)
    with pytest.raises(ValueError):
        fixed_action_regret_bound(-1, 1)


def test_stopped_barrier_and_monitor_risk():
    assert stopped_barrier_exit_bound(F(1, 10), F(1, 100), 1, 20) == F(3, 10)
    assert stopped_barrier_exit_bound(0, 0, 1, 1000) == 0
    assert stopped_barrier_exit_bound(0, F(1, 10), 1, 20) == 1
    assert union_risk_bound(F(3, 10), F(1, 100), F(1, 100)) == F(8, 25)
    assert union_risk_bound(F(3, 4), F(3, 4)) == 1


def test_expected_drift_does_not_imply_pathwise_invariance():
    spike_p, spike = F(1, 100), F(10)
    assert spike_p * spike <= F(1, 2)
    assert spike > 1
    assert stopped_barrier_exit_bound(0, F(1, 2), 1, 1) >= spike_p


@pytest.mark.parametrize("invalid", [0.1, True, object()])
def test_float_and_non_rationals_rejected(invalid):
    with pytest.raises(TypeError):
        rational(invalid)


def test_bad_arguments_rejected():
    for args in ((1, 1, 0, 1), (0, 0, 0, 1), (0, 1, -1, 1), (0, 1, 0, 0)):
        with pytest.raises(ValueError):
            RecoveryEnvelope(*args)
    env = RecoveryEnvelope(0, 1, 0, 1)
    with pytest.raises(ValueError):
        env.envelope_path(0, [-1])
    with pytest.raises(ValueError):
        env.envelope_path(-1, [])
    with pytest.raises(ValueError):
        env.equilibrium_bound(-1)
    with pytest.raises(ValueError):
        env.uniform_viability_certificate(-1, 0)
    with pytest.raises(ValueError):
        stopped_barrier_exit_bound(0, 0, 0, 3)
    with pytest.raises(ValueError):
        stopped_barrier_exit_bound(0, 0, 1, -1)
    with pytest.raises(ValueError):
        union_risk_bound(F(11, 10))
