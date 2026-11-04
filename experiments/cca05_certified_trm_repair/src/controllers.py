"""CCA-05 controller arms over a TRM proposer interface (no torch).

A proposer exposes final_logits(scenario, state, mask) (pass-4 action logits in
vocabulary order, computed on the candidate mask) and pass_logits(scenario,
state) (full mask, passes 1 to 8). Every certified arm executes only through
execute_certified with the environment's independent certificate checker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Protocol

from research_gym.core.certified_repair import (
    bounded_recurrent_repair,
    execute_certified,
    rank_certified,
)
from research_gym.core.constitutional_control import minimal_repair
from research_gym.core.hybrid import CandidateState
from research_gym.core.typed_soundness import SoundnessType
from research_gym.envs.coupled_storyworld import StoryState

import semantics as M

FULL = frozenset(M.ACTIONS)
RANGES = (6.0, 4.0, 4.0, float(M.ENV.max_scene))
PASSES_PER_QUERY = 4
A4_MAX_ITERATIONS = len(M.ACTIONS)


class Proposer(Protocol):
    def final_logits(self, scenario: str, state: StoryState, mask: frozenset[str]) -> tuple[float, ...]: ...

    def pass_logits(self, scenario: str, state: StoryState) -> tuple[tuple[float, ...], ...]: ...


def masked_argmax(logits: tuple[float, ...], mask: frozenset[str]) -> str:
    """TRMProposer.propose: logits outside the mask are -inf, first index wins ties."""
    best = None
    for i, a in enumerate(M.ACTIONS):
        if a in mask and (best is None or logits[i] > logits[best]):
            best = i
    if best is None:
        raise ValueError("empty candidate mask")
    return M.ACTIONS[best]


def trm_choice(proposer: Proposer, scenario: str, state: StoryState, mask: frozenset[str]) -> str:
    return masked_argmax(proposer.final_logits(scenario, state, mask), mask)


@dataclass
class Decision:
    arm: str
    proposal: str | None
    executed: str | None
    reason: str
    distinct_verifications: int = 0
    verifier_invocations: int = 0
    elimination_checks: int = 0
    proposer_queries: int = 0
    utility_evaluations: int = 0
    certificate_checks: int = 0
    iterations: int = 0
    ranks: tuple[int, ...] = ()
    certificate: str | None = None
    proposals: tuple[str, ...] = field(default_factory=tuple)

    @property
    def abstained(self) -> bool:
        return self.executed is None

    @property
    def recurrent_passes(self) -> int:
        return PASSES_PER_QUERY * self.proposer_queries


def _finalize(d: Decision, verify: M.CountingVerifier, action: str | None,
              domain: frozenset[str] = FULL) -> Decision:
    d.distinct_verifications = verify.distinct
    d.verifier_invocations = verify.invocations
    if action is None:
        d.executed = None
        return d
    ctx = M.context(verify.scenario, verify.state, verify.horizon)
    cert = M.issue(verify.scenario, verify.state, verify.horizon, action)
    d.certificate_checks += 1
    executed = execute_certified(ctx, action, cert, M.check_certificate,
                                 domain=CandidateState({"action": domain}))
    if executed is None:
        d.reason = "abstain: certificate refused"
    d.executed = executed
    d.certificate = cert.digest() if cert is not None and executed is not None else None
    return d


def identity_distance(a: str, b: str) -> float:
    return 0.0 if a == b else 1.0


def transition_distance(state: StoryState) -> Callable[[str, str], float]:
    """L1 distance between the deployed successors, each coordinate over its range."""
    def d(a: str, b: str) -> float:
        sa, sb = M.st(M.successor(state, a)), M.st(M.successor(state, b))
        return sum(abs(x - y) / r for x, y, r in zip(sa, sb, RANGES))
    return d


def elimination_verifier(verify: M.CountingVerifier):
    """Provenance check for a committed elimination: recompute every removed
    action's admissibility in the bound state. The proposal's label is ignored."""
    def check(proposal, ctx) -> SoundnessType:
        base, current = ctx["base"], ctx["current"]
        if base != M.context(verify.scenario, verify.state, verify.horizon):
            return SoundnessType.UNKNOWN
        before = current.domains["action"]
        after = proposal.proposed_state.domains.get("action", frozenset())
        removed = before - after
        if not after <= before or not removed:
            return SoundnessType.UNKNOWN
        return (SoundnessType.ENV_SOUND_DEAD if all(not verify(a) for a in removed)
                else SoundnessType.UNKNOWN)
    return check


def decide(arm: str, proposer: Proposer | None, scenario: str, state: StoryState,
           horizon: int, *, a4_max_iterations: int = A4_MAX_ITERATIONS) -> Decision:
    verify = M.CountingVerifier(scenario, state, horizon)
    raw = None
    if arm not in ("L0", "O"):
        raw = trm_choice(proposer, scenario, state, FULL)
    d = Decision(arm, raw, None, "", proposer_queries=0 if raw is None else 1)

    if arm == "A0":
        d.executed, d.reason = raw, "raw TRM"
        return d
    if arm == "L0":
        d.reason = "symbolic LDT fallback"
        return _finalize(d, verify, M.ldt_fallback(verify))
    if arm == "O":
        allowed = [a for a in M.ACTIONS if verify(a)]
        d.utility_evaluations = len(allowed)
        d.reason = "certified oracle"
        return _finalize(d, verify, M.oracle_action(scenario, state, horizon, allowed))
    if arm == "A4":
        out = bounded_recurrent_repair(
            CandidateState({"action": FULL}),
            lambda c: trm_choice(proposer, scenario, state, c.domains["action"]),
            verify, elimination_verifier(verify),
            M.context(scenario, state, horizon),
            max_iterations=a4_max_iterations,
        )
        d.proposer_queries = len(out.proposals)
        d.proposals, d.ranks, d.iterations = out.proposals, out.ranks, len(out.proposals)
        d.elimination_checks = out.elimination_checks
        d.reason = out.reason
        return _finalize(d, verify, out.action)
    if arm == "A5":
        allowed = frozenset(a for a in M.ACTIONS if verify(a))
        if not allowed:
            d.reason = "abstain: empty certified set"
            return _finalize(d, verify, None)
        logits = proposer.final_logits(scenario, state, FULL)
        ranked = rank_certified(allowed, dict(zip(M.ACTIONS, logits)), M.ACTIONS)
        d.reason = "TRM-ranked certified action"
        return _finalize(d, verify, ranked[0])

    if verify(raw):
        d.reason = "certified proposal"
        return _finalize(d, verify, raw)
    if arm == "A1":
        d.reason = "abstain: proposal rejected"
        return _finalize(d, verify, None)
    if arm in ("A2id", "A2tr"):
        allowed = frozenset(a for a in M.ACTIONS if verify(a))
        metric = identity_distance if arm == "A2id" else transition_distance(state)
        rep = minimal_repair(raw, allowed, metric)
        d.reason = "abstain: empty certified set" if rep.abstained else "nearest certified action"
        return _finalize(d, verify, rep.action)
    if arm == "A3":
        fb = M.ldt_fallback(verify)
        d.reason = "certified LDT fallback" if fb else "abstain: no certified fallback"
        return _finalize(d, verify, fb)
    raise ValueError(f"unknown arm: {arm}")


ARMS = ("A0", "A1", "A2id", "A2tr", "A3", "A4", "A5", "L0", "O")
TRM_ARMS = ("A0", "A1", "A2id", "A2tr", "A3", "A4", "A5")


# --------------------------------------------------------------------------- #
# Trajectories under the environment's transitions and rival model
# --------------------------------------------------------------------------- #


@dataclass
class Trajectory:
    arm: str
    actions: list[str]
    abstained_at: int | None
    unsafe_steps: int
    final_state: tuple[int, int, int, int]
    value: float
    goal_met: bool
    distinct_verifications: int
    proposer_queries: int
    certificate_checks: int


def run_trajectory(arm: str, proposer: Proposer | None, scenario: str, start: StoryState,
                   horizon: int = M.HORIZON,
                   chooser: Callable[[str, StoryState, int], str | None] | None = None) -> Trajectory:
    """Act until the horizon or a terminal state. `chooser` replaces decide() for
    uncertified negative controls."""
    s, k = start, horizon
    actions: list[str] = []
    abstained_at = None
    unsafe = verifications = queries = checks = 0
    while k > 0 and not M.ENV.terminal(s) and not M.ENV.target(s):
        if chooser is not None:
            a = chooser(scenario, s, k)
        else:
            d = decide(arm, proposer, scenario, s, k)
            a = d.executed
            verifications += d.distinct_verifications
            queries += d.proposer_queries
            checks += d.certificate_checks
        if a is None:
            abstained_at = len(actions)
            break
        unsafe += not M.admissible(scenario, s, k, a)
        actions.append(a)
        s, k = M.successor(s, a), k - 1
    value = M.terminal_value(s, scenario)
    return Trajectory(arm, actions, abstained_at, unsafe, M.st(s), value,
                      abstained_at is None and M.goal(scenario, s),
                      verifications, queries, checks)


@lru_cache(maxsize=None)
def uniform_certified_value(scenario: str, state: StoryState, k: int) -> float | None:
    """Exact expected trajectory value of the uniform-random certified policy."""
    if k <= 0 or M.ENV.terminal(state) or M.ENV.target(state):
        return M.terminal_value(state, scenario)
    allowed = [a for a in M.ACTIONS if M.admissible(scenario, state, k, a)]
    if not allowed:
        return None
    vals = [uniform_certified_value(scenario, M.successor(state, a), k - 1) for a in allowed]
    if any(v is None for v in vals):
        raise AssertionError("certified successor with an empty certified set")
    return sum(vals) / len(vals)


def myopic_admissible(scenario: str, state: StoryState, action: str) -> bool:
    """Negative control: one-step check on the successor's own state.

    Secret ending: the successor is the target, or is nonterminal with trust >= 1
    and heat <= 2 (the target's non-progress conditions). Moral: the successor's
    moral value already meets TAU. Each accepted action looks acceptable alone.
    """
    nxt = M.successor(state, action)
    if scenario == "secret_ending":
        return M.ENV.target(nxt) or (not M.ENV.terminal(nxt) and nxt.trust >= 1 and nxt.heat <= 2)
    return M.terminal_value(nxt, scenario) >= M.TAU


def myopic_chooser(proposer: Proposer):
    def choose(scenario: str, state: StoryState, k: int) -> str | None:
        allowed = frozenset(a for a in M.ACTIONS if myopic_admissible(scenario, state, a))
        if not allowed:
            return None
        logits = proposer.final_logits(scenario, state, FULL)
        return rank_certified(allowed, dict(zip(M.ACTIONS, logits)), M.ACTIONS)[0]
    return choose
