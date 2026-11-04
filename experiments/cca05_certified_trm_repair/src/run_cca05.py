"""CCA-05 runner.

  python -I src/run_cca05.py environment --out outputs      # TRM-free lemmas, controls, kernel checks
  python -I src/run_cca05.py live --out outputs             # torch: hash-verified checkpoints -> tapes + arms
  python -I src/run_cca05.py replay --out <dir> [--tags X211]   # no torch: rerun arms from committed tapes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO))

import controllers as C  # noqa: E402
import negative_controls as NC  # noqa: E402
import semantics as M  # noqa: E402
from research_gym.core.certified_repair import canonical_digest, rank_certified  # noqa: E402
from tape import ReplayTRM, input_key, write_tape  # noqa: E402

REG = json.loads((ROOT / "registration.json").read_text(encoding="utf-8"))
TAGS = tuple(REG["checkpoints"])


def set_idle_priority() -> None:
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
    except Exception:
        pass


def peak_rss_gb() -> float | None:
    try:
        import psutil
        info = psutil.Process().memory_info()
        return round(getattr(info, "peak_wset", info.rss) / 2**30, 3)
    except Exception:
        return None


def dump(obj, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=1, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def r6(x):
    return None if x is None else round(float(x), 6)


# --------------------------------------------------------------------------- #
# Environment facts (no TRM)
# --------------------------------------------------------------------------- #


def environment_report() -> dict:
    t0, c0 = time.perf_counter(), time.process_time()
    dom = M.bench_domain()
    sizes = Counter()
    lemma = Counter()
    for sc, s in dom:
        S = M.certified_set(sc, s, M.HORIZON)
        sizes[(sc, len(S))] += 1
        U = {a: M.utility(sc, s, M.HORIZON, a) for a in M.ACTIONS}
        if sc == "secret_ending":
            lemma["secret_certified_U_not_1"] += sum(U[a] != 1.0 for a in S)
            lemma["secret_uncertified_U_above_0.05"] += sum(U[a] > 0.05 for a in M.ACTIONS if a not in S)
        else:
            lemma["moral_verify_ne_U_ge_tau"] += sum((a in S) != (U[a] >= M.TAU) for a in M.ACTIONS)
        if S:
            lemma["unconstrained_oracle_outside_S"] += M.oracle_action(sc, s, M.HORIZON) not in S
        lemma["fallback_parity_violations"] += (
            M.original_fallback_action(sc, s, M.HORIZON) != (M.ldt_fallback(M.CountingVerifier(sc, s, M.HORIZON)) or "wait"))
    # Integer re-derivation of the moral threshold (no floats).
    best_int = {}

    def bi(s, k):
        if (s, k) not in best_int:
            best_int[(s, k)] = (M.moral_score(s) if k <= 0 or M.ENV.terminal(s) or M.ENV.target(s)
                                else max(bi(M.successor(s, a), k - 1) for a in M.ACTIONS))
        return best_int[(s, k)]

    for s in M.ENV.all_states():
        if M.ENV.terminal(s):
            continue
        for k in range(1, M.HORIZON + 1):
            for a in M.ACTIONS:
                lemma["moral_float_int_disagreements"] += (
                    M.admissible("moral_optimization", s, k, a) != (bi(M.successor(s, a), k - 1) >= M.MORAL_THRESHOLD))
    # CCA viability kernel on the time-expanded game, modeled and nondeterministic rival.
    kernel = {}
    for sc in M.SCENARIOS:
        model = M.kernel_shield(sc, "model")
        robust = M.kernel_shield(sc, "nondet")
        mism = sum(model[(s, k)] != M.certified_set(sc, s, k) for (s, k) in model)
        lost = Counter()
        for (s, k), S in model.items():
            if k == M.HORIZON:
                lost["actions_certified_model"] += len(S)
                lost["actions_certified_nondet"] += len(robust[(s, k)])
                lost["states_with_model_action_not_robust"] += bool(S - robust[(s, k)])
        kernel[sc] = {"nodes_checked": len(model), "kernel_vs_verifier_mismatches": mism, **lost}
    # Viability of certified trajectories (T7) and the myopic control, exhaustively over paths.
    via = {}
    for sc in M.SCENARIOS:
        fail_cert, fail_myo, empty_after_cert = {}, {}, 0

        def can_fail(s, k, rule, memo):
            key = (s, k)
            if key in memo:
                return memo[key]
            if k <= 0 or M.ENV.terminal(s) or M.ENV.target(s):
                memo[key] = not M.goal(sc, s)
                return memo[key]
            allowed = [a for a in M.ACTIONS if rule(s, k, a)]
            memo[key] = (not allowed) or any(can_fail(M.successor(s, a), k - 1, rule, memo) for a in allowed)
            return memo[key]

        cert_rule = lambda s, k, a: M.admissible(sc, s, k, a)
        myo_rule = lambda s, k, a: C.myopic_admissible(sc, s, a)
        starts = [s for s in M.ENV.all_states() if not M.ENV.terminal(s)]
        viable = [s for s in starts if M.certified_set(sc, s, M.HORIZON)]
        for s in M.ENV.all_states():
            if M.ENV.terminal(s):
                continue
            for k in range(1, M.HORIZON + 1):
                for a in M.ACTIONS:
                    if M.admissible(sc, s, k, a):
                        n = M.successor(s, a)
                        stop = k - 1 <= 0 or M.ENV.terminal(n) or M.ENV.target(n)
                        empty_after_cert += (not stop) and not M.certified_set(sc, n, k - 1)
        myo_starts = [s for s in starts if any(C.myopic_admissible(sc, s, a) for a in M.ACTIONS)]
        via[sc] = {
            "viable_starts": len(viable),
            "viable_starts_with_a_failing_certified_path": sum(can_fail(s, M.HORIZON, cert_rule, fail_cert) for s in viable),
            "certified_transitions_into_empty_certified_set": empty_after_cert,
            "starts_with_a_myopic_action": len(myo_starts),
            "of_those_with_a_failing_myopic_path": sum(can_fail(s, M.HORIZON, myo_rule, fail_myo) for s in myo_starts),
        }
    # Certificates: completeness and soundness over every (s, k, a); sizes.
    cert = Counter()
    wl, sizes_b = Counter(), []
    for sc in M.SCENARIOS:
        for s in M.ENV.all_states():
            if M.ENV.terminal(s):
                continue
            for k in range(1, M.HORIZON + 1):
                for a in M.ACTIONS:
                    c = M.issue(sc, s, k, a)
                    ok = M.admissible(sc, s, k, a)
                    cert["issued_iff_admissible_violations"] += (c is not None) != ok
                    if c is not None:
                        cert["issued"] += 1
                        cert["checker_rejects_issued"] += not M.check_certificate(M.context(sc, s, k), a, c)
                        if k == M.HORIZON:
                            wl[len(c.witness)] += 1
                            sizes_b.append(len(json.dumps(c.to_jsonable(), sort_keys=True, separators=(",", ":"))))
    # Cold verifier and checker cost on the registered sample.
    triples = sorted(((sc, s, a) for sc, s in dom for a in M.ACTIONS),
                     key=lambda t: hashlib.sha256(f"{t[0]}|{M.st(t[1])}|{t[2]}".encode()).hexdigest())[:500]
    cold = []
    for sc, s, a in triples:
        M.best_value.cache_clear()
        M._reach.cache_clear()
        t = time.perf_counter()
        M.admissible(sc, s, M.HORIZON, a)
        cold.append(time.perf_counter() - t)
    checks = []
    for sc, s, a in triples:
        c = M.issue(sc, s, M.HORIZON, a)
        if c is None:
            continue
        ctx = M.context(sc, s, M.HORIZON)
        t = time.perf_counter()
        M.check_certificate(ctx, a, c)
        checks.append(time.perf_counter() - t)
    return {
        "env_digest": M.ENV_DIGEST,
        "domain": {"d_bench": len(dom), "d_full": len(M.full_domain())},
        "certified_set_sizes_h6": {f"{sc}|{n}": v for (sc, n), v in sorted(sizes.items())},
        "lemma_violation_counts": dict(lemma),
        "viability_kernel": kernel,
        "trajectory_viability": via,
        "certificates": {**dict(cert), "witness_length_h6": dict(sorted(wl.items())),
                         "json_bytes_h6": {"min": min(sizes_b), "max": max(sizes_b),
                                           "mean": r6(mean(sizes_b))}},
        "cost": {"cold_verify_seconds": {"n": len(cold), "mean": r6(mean(cold)), "max": r6(max(cold))},
                 "certificate_check_seconds": {"n": len(checks), "mean": r6(mean(checks)), "max": r6(max(checks))},
                 "wall_s": round(time.perf_counter() - t0, 2), "cpu_s": round(time.process_time() - c0, 2)},
    }


# --------------------------------------------------------------------------- #
# Per-checkpoint evaluation
# --------------------------------------------------------------------------- #


def splits_for(seed: int) -> dict[str, set]:
    train = {input_key(sc, s) for sc, s in M.sampled_regions(seed)["proposer_train"]}
    keys = [input_key(sc, s) for sc, s in M.bench_domain()]
    ev = {input_key(sc, s) for sc, s in M.bench_domain() if M.state_bucket(s) in (8, 9)}
    return {"all": set(keys), "train": train & set(keys), "unseen": set(keys) - train,
            "eval_buckets_8_9": ev}


def table(proposer) -> list[dict]:
    rows = []
    for sc, s in M.bench_domain():
        S = M.certified_set(sc, s, M.HORIZON)
        U = {a: M.utility(sc, s, M.HORIZON, a) for a in M.ACTIONS}
        logits = proposer.final_logits(sc, s, C.FULL)
        ranked = rank_certified(S, dict(zip(M.ACTIONS, logits)), M.ACTIONS) if S else ()
        rows.append({"key": input_key(sc, s), "scenario": sc, "state": s, "S": S, "U": U,
                     "S_robust": M.robust_certified_set(sc, s, M.HORIZON),
                     "ustar": max(U[a] for a in S) if S else None, "top_certified": ranked[0] if ranked else None,
                     "ur": mean(U[a] for a in S) if S else None})
    return rows


def arm_metrics(rows, decisions, subset) -> dict:
    R = [(r, d) for r, d in zip(rows, decisions) if r["key"] in subset]
    N = [(r, d) for r, d in R if r["S"]]
    ex = [(r, d) for r, d in R if d.executed is not None]
    rej = [(r, d) for r, d in N if d.proposal is not None and d.proposal not in r["S"]]
    rep = [(r, d) for r, d in rej if d.executed is not None and d.executed in r["S"]]
    out = {
        "n": len(R), "n_empty_S": len(R) - len(N),
        "unsafe": sum(d.executed is not None and d.executed not in r["S"] for r, d in R),
        "executed_outside_nondet_rival_set": sum(d.executed is not None and d.executed not in r["S_robust"] for r, d in R),
        "abstain": sum(d.executed is None for r, d in R),
        "abstain_with_nonempty_S": sum(d.executed is None for r, d in N),
        "executed_utility_mean": r6(mean(r["U"][d.executed] for r, d in ex)) if ex else None,
        "utility_mean_abstain0_on_nonempty_S": r6(mean(r["U"][d.executed] if d.executed else 0.0 for r, d in N)) if N else None,
        "oracle_utility_mean_on_nonempty_S": r6(mean(r["ustar"] for r, d in N)) if N else None,
        "uniform_certified_utility_mean": r6(mean(r["ur"] for r, d in N)) if N else None,
        "regret_mean_executed_nonempty_S": r6(mean(r["ustar"] - r["U"][d.executed] for r, d in N if d.executed)) if N else None,
        "oracle_match_rate_executed": r6(mean(r["U"][d.executed] == r["ustar"] for r, d in N if d.executed)) if any(d.executed for _, d in N) else None,
    }
    if R and R[0][1].proposal is not None:
        out.update({
            "raw_certified_rate": r6(mean(d.proposal in r["S"] for r, d in R)),
            "raw_rejected_with_nonempty_S": len(rej),
            "repaired": len(rep),
            "repair_success_rate": r6(len(rep) / len(rej)) if rej else None,
            "intent_preserved_rate": r6(mean(d.executed == r["top_certified"] for r, d in rep)) if rep else None,
            "repair_regret_mean": r6(mean(r["ustar"] - r["U"][d.executed] for r, d in rep)) if rep else None,
            "repair_gain_over_raw_mean": r6(mean(r["U"][d.executed] - r["U"][d.proposal] for r, d in rep)) if rep else None,
        })
    out.update({
        "distinct_verifications_mean": r6(mean(d.distinct_verifications for _, d in R)) if R else None,
        "distinct_verifications_max": max((d.distinct_verifications for _, d in R), default=None),
        "distinct_verifications_mean_executed": r6(mean(d.distinct_verifications for _, d in ex)) if ex else None,
        "distinct_verifications_mean_empty_S": r6(mean(d.distinct_verifications for r, d in R if not r["S"])) if len(R) > len(N) else None,
        "verifier_invocations_mean": r6(mean(d.verifier_invocations for _, d in R)) if R else None,
        "elimination_checks_mean": r6(mean(d.elimination_checks for _, d in R)) if R else None,
        "proposer_queries_mean": r6(mean(d.proposer_queries for _, d in R)) if R else None,
        "recurrent_passes_mean": r6(mean(d.recurrent_passes for _, d in R)) if R else None,
        "certificate_checks_mean": r6(mean(d.certificate_checks for _, d in R)) if R else None,
        "utility_evaluations_mean": r6(mean(d.utility_evaluations for _, d in R)) if R else None,
    })
    return out


def paired(rows, da, db, subset) -> dict:
    diffs = [(r["U"][a.executed] if a.executed else 0.0) - (r["U"][b.executed] if b.executed else 0.0)
             for r, a, b in zip(rows, da, db) if r["key"] in subset and r["S"]]
    both_abstain_mismatch = sum((a.executed is None) != (b.executed is None)
                                for r, a, b in zip(rows, da, db) if r["key"] in subset)
    return {"n": len(diffs), "mean_diff": r6(mean(diffs)) if diffs else None,
            "better": sum(d > 1e-12 for d in diffs), "worse": sum(d < -1e-12 for d in diffs),
            "equal": sum(abs(d) <= 1e-12 for d in diffs), "abstention_set_mismatch": both_abstain_mismatch}


def traj_metrics(trajs, rows, subset) -> dict:
    T = [(r, t) for r, t in zip(rows, trajs) if r["key"] in subset]
    V = [(r, t) for r, t in T if r["S"]]
    done = [(r, t) for r, t in V if t.abstained_at is None]
    vstar = {r["key"]: M.best_value(r["state"], M.HORIZON, r["scenario"]) for r, _ in V}
    return {
        "n": len(T), "viable_starts": len(V),
        "unsafe_trajectories": sum(t.unsafe_steps > 0 for _, t in T),
        "unsafe_steps": sum(t.unsafe_steps for _, t in T),
        "abstained_at_start": sum(t.abstained_at == 0 for _, t in T),
        "abstained_mid_trajectory": sum((t.abstained_at or 0) > 0 for _, t in T),
        "goal_met_rate_viable_completed": r6(mean(t.goal_met for _, t in done)) if done else None,
        "value_mean_viable_completed": r6(mean(t.value for _, t in done)) if done else None,
        "value_mean_viable_abstain0": r6(mean(t.value if t.abstained_at is None else 0.0 for _, t in V)) if V else None,
        "oracle_value_mean_viable": r6(mean(vstar.values())) if V else None,
        "regret_mean_viable_completed": r6(mean(vstar[r["key"]] - t.value for r, t in done)) if done else None,
        "verifications_mean": r6(mean(t.distinct_verifications for _, t in T)) if T else None,
        "proposer_queries_mean": r6(mean(t.proposer_queries for _, t in T)) if T else None,
        "steps_mean": r6(mean(len(t.actions) for _, t in T)) if T else None,
    }


def pass_analysis(proposer, rows) -> dict:
    out = {}
    prev = None
    final = [C.masked_argmax(proposer.pass_logits(r["scenario"], r["state"])[3], C.FULL) for r in rows]
    for t in range(8):
        acts = [C.masked_argmax(proposer.pass_logits(r["scenario"], r["state"])[t], C.FULL) for r in rows]
        hist = Counter(acts)
        fhist = Counter(final)
        out[f"pass{t + 1}"] = {
            "certified_rate": r6(mean(a in r["S"] for a, r in zip(acts, rows))),
            "utility_mean": r6(mean(r["U"][a] for a, r in zip(acts, rows))),
            "oracle_match_rate_nonempty_S": r6(mean(r["U"][a] == r["ustar"] for a, r in zip(acts, rows) if r["S"])),
            "changed_from_previous_pass": None if prev is None else r6(mean(a != b for a, b in zip(acts, prev))),
            "changed_from_pass4": r6(mean(a != b for a, b in zip(acts, final))),
            "tv_distance_to_pass4": r6(sum(abs(hist[a] - fhist[a]) for a in M.ACTIONS) / (2 * len(rows))),
            "histogram": {a: hist[a] for a in M.ACTIONS},
            "trained_depth": t < 4,
        }
        prev = acts
    return out


def families(rows) -> dict[str, set]:
    fam = defaultdict(set)
    for r in rows:
        fam[f"{r['scenario']}|S={len(r['S'])}"].add(r["key"])
    return dict(fam)


def evaluate(tag: str, proposer, claims: dict | None, threshold: float | None) -> tuple[dict, dict]:
    seed = REG["checkpoints"][tag]["seed"]
    t0, c0 = time.perf_counter(), time.process_time()
    rows = table(proposer)
    splits = splits_for(seed)
    fams = families(rows)
    timing, dec = {}, {}
    for arm in C.ARMS:
        a0, b0 = time.perf_counter(), time.process_time()
        dec[arm] = [C.decide(arm, proposer, r["scenario"], r["state"], M.HORIZON) for r in rows]
        timing[arm] = {"wall_s": round(time.perf_counter() - a0, 4), "cpu_s": round(time.process_time() - b0, 4)}
    for arm in C.ARMS:
        for r, d in zip(rows, dec[arm]):
            if arm != "A0" and d.executed is not None and d.executed not in r["S"]:
                raise AssertionError(f"{tag} {arm}: executed uncertified action at {r['key']}")
    single = {arm: {name: arm_metrics(rows, dec[arm], sub) for name, sub in splits.items()} for arm in C.ARMS}
    by_family = {arm: {f: arm_metrics(rows, dec[arm], sub) for f, sub in sorted(fams.items())}
                 for arm in ("A0", "A1", "A2id", "A2tr", "A3", "A4", "A5", "L0", "O")}
    pairs = {f"{a}-{b}": {name: paired(rows, dec[a], dec[b], sub) for name, sub in splits.items()}
             for a, b in (("A5", "L0"), ("A4", "L0"), ("A3", "L0"), ("A2tr", "L0"), ("A2id", "L0"),
                          ("A5", "A4"), ("A5", "A2id"), ("O", "A5"), ("O", "L0"))}
    pairs_family = {f"{a}-L0": {f: paired(rows, dec[a], dec["L0"], sub) for f, sub in sorted(fams.items())}
                    for a in ("A5", "A4")}
    budget = {}
    for k in REG["a4_budget_curve"]:
        dk = [C.decide("A4", proposer, r["scenario"], r["state"], M.HORIZON, a4_max_iterations=k) for r in rows]
        budget[k] = arm_metrics(rows, dk, splits["all"])
    ranks_ok = all(all(x > y for x, y in zip(d.ranks, d.ranks[1:])) and len(d.ranks) <= 6 for d in dec["A4"])
    a4_vs_a5 = sum(a.executed != b.executed for a, b in zip(dec["A4"], dec["A5"]))
    # Trajectories
    trajs, ttime = {}, {}
    for arm in C.ARMS:
        a0, b0 = time.perf_counter(), time.process_time()
        trajs[arm] = [C.run_trajectory(arm, proposer, r["scenario"], r["state"]) for r in rows]
        ttime[arm] = {"wall_s": round(time.perf_counter() - a0, 4), "cpu_s": round(time.process_time() - b0, 4)}
    myo = [C.run_trajectory("myopic", proposer, r["scenario"], r["state"], chooser=C.myopic_chooser(proposer))
           for r in rows]
    traj = {arm: {name: traj_metrics(trajs[arm], rows, sub) for name, sub in splits.items()} for arm in C.ARMS}
    traj["myopic_A5_ranking"] = {name: traj_metrics(myo, rows, sub) for name, sub in splits.items()}
    ur_traj = [C.uniform_certified_value(r["scenario"], r["state"], M.HORIZON) for r in rows]
    traj["UR"] = {name: {"value_mean_viable": r6(mean(v for r, v in zip(rows, ur_traj) if r["key"] in sub and r["S"]))}
                  for name, sub in splits.items()}
    tpaired = {}
    for a, b in (("A5", "L0"), ("A4", "L0"), ("A3", "L0"), ("A2tr", "L0"), ("A2id", "L0"), ("O", "A5")):
        diffs = [ta.value - tb.value for r, ta, tb in zip(rows, trajs[a], trajs[b])
                 if r["S"] and ta.abstained_at is None and tb.abstained_at is None]
        tpaired[f"{a}-{b}"] = {"n": len(diffs), "mean_diff": r6(mean(diffs)),
                               "better": sum(d > 1e-12 for d in diffs), "worse": sum(d < -1e-12 for d in diffs)}
    tfam = {f"{a}-L0": {} for a in ("A5", "A4")}
    for a in ("A5", "A4"):
        for f, sub in sorted(fams.items()):
            diffs = [ta.value - tb.value for r, ta, tb in zip(rows, trajs[a], trajs["L0"])
                     if r["key"] in sub and r["S"] and ta.abstained_at is None and tb.abstained_at is None]
            tfam[f"{a}-L0"][f] = {"n": len(diffs), "mean_diff": r6(mean(diffs)) if diffs else None,
                                  "better": sum(d > 1e-12 for d in diffs), "worse": sum(d < -1e-12 for d in diffs)}
    claim = None
    if claims is not None:
        flagged = [(r, d) for r, d in zip(rows, dec["A0"]) if claims[r["key"]] >= threshold]
        claim = {"threshold": r6(threshold), "claims_env_sound_dead": len(flagged),
                 "false_claims": sum(d.proposal not in r["S"] for r, d in flagged),
                 "uncertified_proposals": sum(d.proposal not in r["S"] for r, d in zip(rows, dec["A0"])),
                 **NC.claim_as_provenance(rows, dec["A0"], claims, threshold)}
    report = {
        "checkpoint": tag, "seed": seed, "kind": REG["checkpoints"][tag]["kind"],
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "single_step": single, "single_step_by_family": by_family,
        "paired_single_step": pairs, "paired_single_step_by_family": pairs_family,
        "a4_budget_curve": budget, "a4_ranks_strictly_decrease": ranks_ok, "a4_vs_a5_executed_differs": a4_vs_a5,
        "a4_iterations": dict(Counter(d.iterations for d in dec["A4"])),
        "trajectory": traj, "paired_trajectory": tpaired, "paired_trajectory_by_family": tfam,
        "pass_analysis": pass_analysis(proposer, rows),
        "provenance_claims": claim,
        "timing_single_step": timing, "timing_trajectory": ttime,
        "wall_s": round(time.perf_counter() - t0, 2), "cpu_s": round(time.process_time() - c0, 2),
    }
    receipt = {"checkpoint": tag, "decisions": {arm: [[r["key"], d.proposal, d.executed, d.reason,
                                                       d.distinct_verifications, d.proposer_queries,
                                                       d.certificate]
                                                      for r, d in zip(rows, dec[arm])] for arm in C.ARMS},
               "trajectories": {arm: [[r["key"], t.actions, t.abstained_at, t.unsafe_steps, list(t.final_state),
                                       r6(t.value), t.goal_met] for r, t in zip(rows, trajs[arm])]
                                for arm in C.ARMS},
               "myopic_trajectories": [[r["key"], t.actions, t.abstained_at, t.unsafe_steps, list(t.final_state),
                                        r6(t.value), t.goal_met] for r, t in zip(rows, myo)]}
    return report, receipt


def run_live(out: Path) -> None:
    import torch
    import trm_live as TL
    torch.set_num_threads(1)
    set_idle_priority()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    cost = {"host": {"platform": platform.platform(), "python": platform.python_version(),
                     "torch": torch.__version__, "processor": platform.processor()}, "checkpoints": {}}
    for tag in TAGS:
        t0, c0 = time.perf_counter(), time.process_time()
        model, meta = TL.load_checkpoint(tag)
        live = TL.LiveTRM(model)
        parity = TL.published_tree_parity(tag, live)
        kind, seed = REG["checkpoints"][tag]["kind"], REG["checkpoints"][tag]["seed"]
        threshold = TL.reconstructed_claim_threshold(model, seed, kind)
        for sc, s in M.full_domain():
            live.pass_logits(sc, s)
        report, receipt = evaluate(tag, live, live.claims, threshold)
        report["checkpoint_meta"] = meta
        report["published_tree_parity"] = parity
        report["torch_forwards"] = live.forwards
        tape = live.tape(tag, meta)
        write_tape(tape, out / "tapes" / f"{tag}.json.gz")
        manifest[f"tapes/{tag}.json.gz"] = hashlib.sha256((out / "tapes" / f"{tag}.json.gz").read_bytes()).hexdigest()
        manifest[f"arms/{tag}.json"] = dump(report, out / "arms" / f"{tag}.json")
        manifest[f"receipts/{tag}.json"] = dump(receipt, out / "receipts" / f"{tag}.json")
        manifest[f"receipts/{tag}.digest"] = canonical_digest(receipt)
        cost["checkpoints"][tag] = {"wall_s": round(time.perf_counter() - t0, 2),
                                    "cpu_s": round(time.process_time() - c0, 2),
                                    "torch_forwards": live.forwards,
                                    "masked_queries_recorded": len(live.masked),
                                    "peak_rss_gb": peak_rss_gb()}
        print(tag, cost["checkpoints"][tag], flush=True)
    cost["peak_rss_gb"] = peak_rss_gb()
    manifest["cost_live.json"] = dump(cost, out / "cost_live.json")
    dump(manifest, out / "manifest_live.json")


def run_replay(out: Path, tags) -> dict:
    src = ROOT / "outputs"
    result = {}
    for tag in tags:
        tape = ReplayTRM.load(src / "tapes" / f"{tag}.json.gz")
        rep = json.loads((src / "arms" / f"{tag}.json").read_text(encoding="utf-8"))
        thr = (rep.get("provenance_claims") or {}).get("threshold")
        claims = tape.tape.get("env_claim_score")
        report, receipt = evaluate(tag, tape, claims, thr)
        committed = json.loads((src / "receipts" / f"{tag}.json").read_text(encoding="utf-8"))
        result[tag] = {"receipt_digest_equal": canonical_digest(receipt) == canonical_digest(committed),
                       "single_step_equal": report["single_step"] == rep["single_step"],
                       "trajectory_equal": report["trajectory"] == rep["trajectory"]}
    if out:
        dump(result, out / "replay_check.json")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("environment", "live", "replay", "controls"))
    p.add_argument("--out", type=Path, default=ROOT / "outputs")
    p.add_argument("--tags", nargs="*", default=list(TAGS))
    a = p.parse_args()
    if a.mode == "environment":
        set_idle_priority()
        dump(environment_report(), a.out / "environment.json")
    elif a.mode == "controls":
        dump(NC.run_all(), a.out / "negative_controls.json")
    elif a.mode == "live":
        run_live(a.out)
    else:
        res = run_replay(a.out, a.tags)
        print(json.dumps(res, indent=1))
        if not all(all(v.values()) for v in res.values()):
            sys.exit(1)


if __name__ == "__main__":
    main()
