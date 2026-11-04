"""CCA-05 invariants on the whole benchmark domain with synthetic proposers.

Safety, identity, idempotence, bounded termination, persona independence and
provenance integrity hold for any proposer, so they are checked here without
the private checkpoints; test_replay.py covers the real tapes.
"""

import inspect

import pytest

import controllers as C
import negative_controls as NC
import semantics as M
from research_gym.core.hybrid import CandidateState, LatticeProposal, certify_and_apply_verified
from research_gym.core.typed_soundness import SoundnessType

DOM = M.bench_domain()
PROPOSERS = {
    "adversarial": NC.FnProposer(NC.adversarial),
    "random": NC.FnProposer(NC.seeded(3)),
    "utility": NC.FnProposer(lambda sc, s, m: [M.utility(sc, s, M.HORIZON, a) for a in M.ACTIONS]),
}
CERTIFIED_ARMS = ("A1", "A2id", "A2tr", "A3", "A4", "A5", "L0", "O")


def forcing(action):
    return NC.FnProposer(lambda sc, s, m: [1.0 if a == action else 0.0 for a in M.ACTIONS])


@pytest.mark.parametrize("name", sorted(PROPOSERS))
def test_safety_identity_and_abstention(name):
    p = PROPOSERS[name]
    for sc, s in DOM:
        S = M.certified_set(sc, s, M.HORIZON)
        for arm in CERTIFIED_ARMS:
            d = C.decide(arm, p, sc, s, M.HORIZON)
            assert d.executed is None or d.executed in S
            if arm != "A1":
                assert (d.executed is None) == (not S)
            if d.proposal is not None and d.proposal in S:
                assert d.executed == d.proposal, (arm, sc, s)


def test_repair_idempotence():
    p = PROPOSERS["adversarial"]
    for sc, s in DOM[::7]:
        for arm in ("A2id", "A2tr", "A3", "A4", "A5"):
            first = C.decide(arm, p, sc, s, M.HORIZON).executed
            if first is None:
                continue
            assert C.decide(arm, forcing(first), sc, s, M.HORIZON).executed == first


def test_a4_bounded_with_strict_rank_decrease():
    p = PROPOSERS["adversarial"]
    for sc, s in DOM:
        d = C.decide("A4", p, sc, s, M.HORIZON)
        assert d.iterations <= C.A4_MAX_ITERATIONS
        assert all(x > y for x, y in zip(d.ranks, d.ranks[1:]))
        assert len(d.ranks) - 1 <= len(M.ACTIONS)


def test_certified_set_takes_no_proposer_and_ranking_cannot_change_it():
    assert list(inspect.signature(M.certified_set).parameters) == ["scenario", "state", "horizon"]
    for sc, s in DOM[::5]:
        S = M.certified_set(sc, s, M.HORIZON)
        executed = {C.decide("A5", p, sc, s, M.HORIZON).executed for p in PROPOSERS.values()}
        assert executed <= (S | {None})


def test_false_env_sound_label_cannot_authorize_hard_refinement():
    for sc, s in DOM[::3]:
        full = CandidateState({"action": C.FULL})
        v = M.CountingVerifier(sc, s, M.HORIZON)
        for a in M.certified_set(sc, s, M.HORIZON):
            prop = LatticeProposal(full.eliminate("action", {a}), soundness=SoundnessType.ENV_SOUND_DEAD)
            res = certify_and_apply_verified(
                full, prop, provenance_verifier=C.elimination_verifier(v),
                verifier_context={"base": M.context(sc, s, M.HORIZON), "current": full})
            assert not res.accepted


def test_stale_context_cannot_reuse_an_elimination_verifier():
    sc, s = next((sc, s) for sc, s in DOM if 0 < len(M.certified_set(sc, s, M.HORIZON)) < 5)
    bad = next(a for a in M.ACTIONS if a not in M.certified_set(sc, s, M.HORIZON))
    full = CandidateState({"action": C.FULL})
    v = M.CountingVerifier(sc, s, M.HORIZON)
    prop = LatticeProposal(full.eliminate("action", {bad}), soundness=SoundnessType.UNKNOWN)
    other = M.context(sc, M.successor(s, "rush"), M.HORIZON)
    res = certify_and_apply_verified(full, prop, provenance_verifier=C.elimination_verifier(v),
                                     verifier_context={"base": other, "current": full})
    assert not res.accepted


def test_secret_ending_utility_is_constant_on_the_certified_set():
    for sc, s in DOM:
        if sc != "secret_ending":
            continue
        S = M.certified_set(sc, s, M.HORIZON)
        for a in M.ACTIONS:
            u = M.utility(sc, s, M.HORIZON, a)
            assert (u == 1.0) if a in S else (u <= 0.05)


def test_certified_trajectories_reach_the_goal():
    p = PROPOSERS["random"]
    for sc, s in DOM[::4]:
        if not M.certified_set(sc, s, M.HORIZON):
            continue
        for arm in ("A4", "A5", "L0"):
            t = C.run_trajectory(arm, p, sc, s)
            assert t.unsafe_steps == 0 and t.abstained_at is None and t.goal_met


def test_viability_kernel_equals_verifier_and_certificates_check():
    for sc in M.SCENARIOS:
        for (s, k), allowed in M.kernel_shield(sc, "model").items():
            assert allowed == M.certified_set(sc, s, k)
    for sc, s in DOM[::11]:
        for a in M.ACTIONS:
            cert = M.issue(sc, s, M.HORIZON, a)
            assert (cert is not None) == M.admissible(sc, s, M.HORIZON, a)
            if cert is not None:
                assert M.check_certificate(M.context(sc, s, M.HORIZON), a, cert)
