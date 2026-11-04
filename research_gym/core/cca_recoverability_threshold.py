"""CCA-11: exact-arithmetic robust recoverability certificates.

These are abstract control envelopes, not neural-model verification. A trusted
plant/repair analysis must establish the stepwise envelope and viability radius.
Outer action safety belongs to an independently enforced execution shield.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable


def rational(value: Fraction | int | str) -> Fraction:
    """Reject floats so certificates do not silently round their assumptions."""
    if type(value) is bool or not isinstance(value, (Fraction, int, str)):
        raise TypeError("Use Fraction, int, or exact decimal/fraction string; floats are not certified")
    return Fraction(value)


@dataclass(frozen=True)
class RecoveryEnvelope:
    """e[t+1] <= rho*e[t] + gain*eta[t] + residual; viable if e[t] <= margin.

    All quantities are nonnegative exact rational bounds. The viability claim
    must come from an independent environment-owned checker or proof.
    """

    rho: Fraction
    gain: Fraction
    residual: Fraction
    margin: Fraction

    def __post_init__(self) -> None:
        for field in ("rho", "gain", "residual", "margin"):
            object.__setattr__(self, field, rational(getattr(self, field)))
        if not 0 <= self.rho < 1:
            raise ValueError("rho must lie in [0, 1)")
        if self.gain <= 0 or self.margin <= 0 or self.residual < 0:
            raise ValueError("gain and margin positive; residual nonnegative")

    def critical_disturbance(self) -> Fraction | None:
        """Sharp uniform robust-invariance threshold; None means infeasible at eta=0."""
        capacity = (1 - self.rho) * self.margin - self.residual
        return capacity / self.gain if capacity >= 0 else None

    def equilibrium_bound(self, disturbance: Fraction | int | str) -> Fraction:
        eta = rational(disturbance)
        if eta < 0:
            raise ValueError("disturbance must be nonnegative")
        return (self.gain * eta + self.residual) / (1 - self.rho)

    def uniform_viability_certificate(
        self, initial: Fraction | int | str, disturbance: Fraction | int | str
    ) -> bool:
        """Every-future-step certificate for every bounded disturbance sequence.

        False means this argument does not certify, not that a trajectory fails.
        """
        e0, eta = rational(initial), rational(disturbance)
        if e0 < 0 or eta < 0:
            raise ValueError("error and disturbance must be nonnegative")
        eta_crit = self.critical_disturbance()
        return e0 <= self.margin and eta_crit is not None and eta <= eta_crit

    def next_step_disturbance_budget(
        self,
        current_error: Fraction | int | str,
        successor_radius: Fraction | int | str,
    ) -> Fraction | None:
        """CCA-S: next-step noise compatible with a local viability radius.

        For a certified successor of radius m_next, require
        rho*e_current + gain*eta + residual <= m_next.
        None means even eta=0 is uncertified by this envelope.
        """
        e, m_next = rational(current_error), rational(successor_radius)
        if e < 0 or m_next < 0:
            raise ValueError("error and successor radius must be nonnegative")
        capacity = m_next - self.rho * e - self.residual
        return capacity / self.gain if capacity >= 0 else None

    def envelope_path(
        self, initial: Fraction | int | str, disturbances: Iterable[Fraction | int | str]
    ) -> tuple[Fraction, ...]:
        """Exact worst-case upper envelope for a finite disturbance schedule."""
        e = rational(initial)
        if e < 0:
            raise ValueError("initial error must be nonnegative")
        path = [e]
        for raw in disturbances:
            eta = rational(raw)
            if eta < 0:
                raise ValueError("disturbance must be nonnegative")
            e = self.rho * e + self.gain * eta + self.residual
            path.append(e)
        return tuple(path)

    def first_possible_exit(self, path: Iterable[Fraction]) -> int | None:
        """First envelope index above the viable radius, not an observed failure."""
        return next((i for i, e in enumerate(path) if e > self.margin), None)


def fixed_action_regret_bound(
    error: Fraction | int | str, lipschitz: Fraction | int | str
) -> Fraction:
    """2*L*error when both states share the same certified feasible-action set.

    Each action's utility is L-Lipschitz; the nominal action is optimal there.
    """
    e, L = rational(error), rational(lipschitz)
    if e < 0 or L < 0:
        raise ValueError("nonnegative error and Lipschitz constant required")
    return 2 * e * L


def stopped_barrier_exit_bound(
    initial: Fraction | int | str,
    drift: Fraction | int | str,
    barrier: Fraction | int | str,
    horizon: int,
) -> Fraction:
    """Upper bound on P(hit B >= barrier by time T).

    ASSUMPTIONS: B_t >= 0, B_0 <= initial, and, up to the first hit,
    E[B_(t+1) | F_t] <= B_t + drift. Stopping plus Markov gives
    P(hit before T) <= (initial + T*drift)/barrier.
    Arithmetic only; this code cannot validate the stochastic drift premise.
    """
    b0, d, m = rational(initial), rational(drift), rational(barrier)
    if b0 < 0 or d < 0 or m <= 0 or type(horizon) is not int or horizon < 0:
        raise ValueError("invalid stopped barrier arguments")
    return min(Fraction(1), (b0 + horizon * d) / m)


def union_risk_bound(*probabilities: Fraction | int | str) -> Fraction:
    """Union bound over failure modes; no independence assumption."""
    values = tuple(rational(p) for p in probabilities)
    if any(p < 0 or p > 1 for p in values):
        raise ValueError("each probability must lie in [0,1]")
    return min(Fraction(1), sum(values, Fraction(0)))
