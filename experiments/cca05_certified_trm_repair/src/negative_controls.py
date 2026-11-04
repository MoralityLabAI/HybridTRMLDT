"""CCA-05 Phase 6 negative controls (no torch). Counts are over D_bench."""

from __future__ import annotations

import itertools
import random
from collections import Counter

from research_gym.core.certified_repair import (
    ActionCertificate,
    bounded_recurrent_repair,
    execute_certified,
)
from research_gym.core.hybrid import (
    CandidateState,
    HybridMode,
    LatticeProposal,
    certify_and_apply,
    certify_and_apply_verified,
)
from research_gym.core.typed_soundness import SoundnessType

import controllers as C
import semantics as M

ENV_SOUND = SoundnessType.ENV_SOUND_DEAD


class FnProposer:
    """Synthetic proposer: logits from a function of (scenario, state, mask)."""

    def __init__(self, fn):
        self.fn = fn

    def final_logits(self, scenario, state, mask):
        return tuple(self.fn(scenario, state, mask))

    def pass_logits(self, scenario, state):
        return (tuple(self.fn(scenario, state, C.FULL)),) * 8


def adversarial(sc, s, mask):
    """Prefers the lowest-utility action: proposes inadmissible actions first."""
    return [-M.utility(sc, s, M.HORIZON, a) for a in M.ACTIONS]


def seeded(seed):
    def fn(sc, s, mask):
        rng = random.Random(f"{seed}|{sc}|{M.st(s)}|{sorted(mask)}")
        return [rng.random() for _ in M.ACTIONS]
    return fn


def _strict(current, proposal, verify):
    return certify_and_apply_verified(
        current, proposal, provenance_verifier=C.elimination_verifier(verify),
        verifier_context={"base": M.context(verify.scenario, verify.state, verify.horizon),
                          "current": current})


def claim_as_provenance(rows, decisions, claims, threshold) -> dict:
    """The TRM's own provenance claim used to commit {proposal} as a hard refinement.

    Historical certify_and_apply hard-applies whatever the label says; the strict
    path recomputes whether every removed action is inadmissible.
    """
    full = CandidateState({"action": C.FULL})
    c = Counter()
    for r, d in zip(rows, decisions):
        claimed = ENV_SOUND if claims[r["key"]] >= threshold else SoundnessType.UNKNOWN
        prop = LatticeProposal(CandidateState({"action": frozenset({d.proposal})}), soundness=claimed,
                               mode=HybridMode.DEDUCE, metadata={"selected_action": d.proposal})
        sound = all(a not in r["S"] for a in C.FULL - {d.proposal})
        hist = certify_and_apply(full, prop).accepted
        strict = _strict(full, prop, M.CountingVerifier(r["scenario"], r["state"], M.HORIZON)).accepted
        c["historical_accepted"] += hist
        c["historical_accepted_unsound"] += hist and not sound
        c["strict_accepted"] += strict
        c["strict_accepted_unsound"] += strict and not sound
    return {"commit_proposal_as_singleton": dict(c)}


def run_all() -> dict:
    dom = M.bench_domain()
    h = M.HORIZON
    out = {}

    # 1-3. Proposer-authored or missing provenance.
    c = Counter()
    for sc, s in dom:
        S = M.certified_set(sc, s, h)
        full = CandidateState({"action": C.FULL})
        for a in M.ACTIONS:
            prop = LatticeProposal(full.eliminate("action", {a}), soundness=ENV_SOUND)
            v = M.CountingVerifier(sc, s, h)
            strict = _strict(full, prop, v).accepted
            if a in S:
                c["false_env_sound_dead_eliminations"] += 1
                c["strict_accepted_false"] += strict
                c["historical_accepted_false"] += certify_and_apply(full, prop).accepted
            else:
                c["true_eliminations"] += 1
                c["strict_accepted_true"] += strict
                unk = LatticeProposal(full.eliminate("action", {a}), soundness=SoundnessType.UNKNOWN)
                c["strict_accepted_true_labelled_unknown"] += _strict(full, unk, v).accepted
                c["missing_verifier_accepted_true"] += certify_and_apply_verified(
                    full, prop, provenance_verifier=None).accepted
                for verdict in (None, SoundnessType.UNKNOWN, SoundnessType.MODEL_SOUND_DEAD):
                    c["inconclusive_verdict_accepted"] += certify_and_apply_verified(
                        full, prop, provenance_verifier=lambda _p, _c, x=verdict: x).accepted
    out["provenance"] = dict(c)

    # 4, 9. Contradictory masks, stale and forged certificates.
    c = Counter()
    for sc, s in dom:
        for a in M.certified_set(sc, s, h):
            cert = M.issue(sc, s, h, a)
            ctx = M.context(sc, s, h)
            c["issued"] += 1
            c["executes_when_valid"] += execute_certified(ctx, a, cert, M.check_certificate) == a
            c["executes_outside_mask"] += execute_certified(
                ctx, a, cert, M.check_certificate,
                domain=CandidateState({"action": C.FULL - {a}})) is not None
            nxt = M.successor(s, a)
            if not M.ENV.terminal(nxt):
                c["stale_state_replays"] += 1
                c["stale_state_executed"] += execute_certified(
                    M.context(sc, nxt, h - 1), a, cert, M.check_certificate) is not None
            c["stale_horizon_executed"] += execute_certified(
                M.context(sc, s, h - 1), a, cert, M.check_certificate) is not None
            c["other_action_executed"] += sum(
                execute_certified(ctx, b, cert, M.check_certificate) is not None for b in M.ACTIONS if b != a)
            other_env = dict(ctx, env="0" * 64)
            forged_binding = ActionCertificate(M.bind(other_env), a, cert.successor, cert.witness, cert.issuer)
            c["foreign_env_executed"] += execute_certified(other_env, a, forged_binding, M.check_certificate) is not None
            bad_succ = ActionCertificate(cert.binding, a, (9, 9, 9, 9), cert.witness, cert.issuer)
            c["forged_successor_executed"] += execute_certified(ctx, a, bad_succ, M.check_certificate) is not None
        for a in M.ACTIONS:
            if a in M.certified_set(sc, s, h):
                continue
            ctx = M.context(sc, s, h)
            nxt = M.successor(s, a)
            for w in ((), tuple(M.ACTIONS[:1]) * (h - 1)):
                forged = ActionCertificate(M.bind(ctx), a, M.st(nxt), w, "forger")
                c["forged_inadmissible_tried"] += 1
                c["forged_inadmissible_executed"] += execute_certified(ctx, a, forged, M.check_certificate) is not None
            c["missing_certificate_executed"] += execute_certified(ctx, a, None, M.check_certificate) is not None
    out["certificates"] = dict(c)

    # Exhaustive witness soundness for horizons 1-3 (every witness of length <= h-1).
    c = Counter()
    for sc in M.SCENARIOS:
        for s in M.ENV.all_states():
            if M.ENV.terminal(s):
                continue
            for k in (1, 2, 3):
                for a in M.ACTIONS:
                    adm = M.admissible(sc, s, k, a)
                    nxt = M.successor(s, a)
                    for n in range(k):
                        for w in itertools.product(M.ACTIONS, repeat=n):
                            ok = M.witness_ends_in_goal(sc, nxt, k - 1, w)
                            c["witnesses_checked"] += 1
                            c["accepted_witness_for_inadmissible"] += ok and not adm
    out["witness_soundness_h1_to_h3"] = dict(c)

    # 5. Nonmonotone refinements.
    c = Counter()
    for sc, s in dom:
        S = M.certified_set(sc, s, h)
        for a in M.ACTIONS:
            current = CandidateState({"action": C.FULL - {a}})
            widen = LatticeProposal(CandidateState({"action": C.FULL}), soundness=ENV_SOUND)
            c["widening_tried"] += 1
            c["widening_accepted"] += _strict(current, widen, M.CountingVerifier(sc, s, h)).accepted
            swap = LatticeProposal(CandidateState({"action": frozenset({a})}), soundness=ENV_SOUND)
            c["swap_accepted"] += _strict(current, swap, M.CountingVerifier(sc, s, h)).accepted
    out["nonmonotone"] = dict(c)

    # 6-7. Empty domains, invalid suggestions, adversarial and random proposers.
    c = Counter()
    empty = bounded_recurrent_repair(CandidateState({"action": frozenset()}), lambda _c: "wait",
                                     lambda _a: True, lambda *_: ENV_SOUND, None, max_iterations=5)
    c["empty_initial_domain_executed"] += empty.action is not None
    for sc, s in dom:
        S = M.certified_set(sc, s, h)
        v = M.CountingVerifier(sc, s, h)
        for nominee in ("teleport", None):
            def propose(cur, n=nominee):
                if n is not None:
                    return n
                return next(iter(sorted(C.FULL - cur.domains["action"])), "wait")
            res = bounded_recurrent_repair(CandidateState({"action": C.FULL}), propose, v,
                                           C.elimination_verifier(v), M.context(sc, s, h), max_iterations=5)
            c["invalid_suggestion_executed_uncertified"] += res.action is not None and res.action not in S
    for name, prop in (("adversarial", FnProposer(adversarial)), ("random_7", FnProposer(seeded(7))),
                       ("random_11", FnProposer(seeded(11))), ("constant", FnProposer(lambda *_: [0.0] * 5))):
        for arm in ("A1", "A2id", "A2tr", "A3", "A4", "A5"):
            for sc, s in dom:
                S = M.certified_set(sc, s, h)
                d = C.decide(arm, prop, sc, s, h)
                c[f"{name}_{arm}_unsafe"] += d.executed is not None and d.executed not in S
                c[f"{name}_{arm}_abstain_with_nonempty_S"] += d.executed is None and bool(S)
    out["proposer_independence"] = dict(c)
    return out
