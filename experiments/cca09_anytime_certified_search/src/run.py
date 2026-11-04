"""CCA-09: genuine CCA-05 TRM-logit replay with proof-checked anytime search.

No torch or inference: the checkpoint tapes were already frozen by CCA-05.
This study is retrospective/descriptive; oracle utilities are used to score
outcomes, never to order the proposed actions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[3]
CCA05 = REPO / "experiments" / "cca05_certified_trm_repair"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CCA05 / "src"))
import semantics as M  # noqa: E402
from tape import ReplayTRM, input_key  # noqa: E402
from research_gym.core.cca_anytime_search import (  # noqa: E402
    ActionVerdict, UpperCertificate, certified_anytime_search,
)
from research_gym.core.certified_repair import (  # noqa: E402
    ActionCertificate, bind, execute_certified,
)
from research_gym.core.hybrid import CandidateState  # noqa: E402

METHOD = "CCA09-host-global-upper-1.0"

def upper_bound(ctx, action):
    return UpperCertificate(action, 1.0, bind(ctx),
                            {"proof_rule": METHOD, "environment": M.ENV_DIGEST})

def check_bound(ctx, action, cert):
    # For this one frozen environment, terminal values lie in [0,1].
    # Max aggregation therefore preserves the upper limit of one.
    return (ctx.get("env") == M.ENV_DIGEST and action in M.ACTIONS and
            cert.action == action and cert.upper == 1.0 and
            cert.binding == bind(ctx) and
            cert.evidence == {"proof_rule": METHOD, "environment": M.ENV_DIGEST})

def evaluate(ctx, action):
    sc, state, horizon = ctx["scenario"], M.StoryState(*ctx["state"]), ctx["horizon"]
    cert = M.issue(sc, state, horizon, action)
    if cert is None:
        return ActionVerdict(action, False, None, "host-negative-solver")
    return ActionVerdict(action, True, M.utility(sc, state, horizon, action), cert)

def check_verdict(ctx, action, verdict):
    sc, state, horizon = ctx["scenario"], M.StoryState(*ctx["state"]), ctx["horizon"]
    if verdict.action != action or action not in M.ACTIONS:
        return False
    if not verdict.feasible:
        return (verdict.utility is None and
                verdict.evidence == "host-negative-solver" and
                not M.admissible(sc, state, horizon, action))
    if verdict.utility is None or not math.isfinite(verdict.utility):
        return False
    if not isinstance(verdict.evidence, ActionCertificate):
        return False
    return (execute_certified(
                ctx, action, verdict.evidence, M.check_certificate,
                domain=CandidateState({"action": frozenset(M.ACTIONS)})
            ) == action and
            math.isclose(verdict.utility, M.utility(sc, state, horizon, action),
                         rel_tol=0, abs_tol=1e-12))

def stable_domain(limit):
    groups = defaultdict(list)
    for sc, s in M.bench_domain():
        digest = hashlib.sha256(input_key(sc, s).encode()).hexdigest()
        groups[sc].append((digest, sc, s))
    if limit:
        quotas = {"secret_ending": (limit + 1)//2,
                  "moral_optimization": limit//2}
        return [(sc, s) for name in M.SCENARIOS
                for _, sc, s in sorted(groups[name])[:quotas[name]]]
    return [(sc, s) for name in M.SCENARIOS
            for _, sc, s in sorted(groups[name])]

def run(outdir, tags, limit, budgets, epsilon):
    outdir.mkdir(parents=True, exist_ok=True)
    domain = stable_domain(limit)
    aggregates = defaultdict(list)
    tape_hashes = {}
    for tag in tags:
        tape_path = CCA05 / "outputs" / "tapes" / f"{tag}.json.gz"
        tape_hashes[tag] = hashlib.sha256(tape_path.read_bytes()).hexdigest()
        tape = ReplayTRM.load(tape_path)
        for sc, state in domain:
            logits = tape.full[input_key(sc, state)][3]
            neural = tuple(sorted(M.ACTIONS, key=lambda a:
                (-logits[M.ACTIONS.index(a)], M.ACTIONS.index(a))))
            allowed = M.certified_set(sc, state, M.HORIZON)
            best = max((M.utility(sc, state, M.HORIZON, a) for a in allowed),
                       default=None)
            ctx = M.context(sc, state, M.HORIZON)
            bounds = {a: upper_bound(ctx, a) for a in M.ACTIONS}
            for name, order in (("trm_pass4", neural),
                                ("fixed_priority", M.MORAL_PRIORITY)):
                for budget in budgets:
                    result = certified_anytime_search(
                        ctx, M.ACTIONS, order, bounds,
                        check_bound=check_bound, evaluate=evaluate,
                        check_verdict=check_verdict,
                        max_evaluations=budget, epsilon=epsilon,
                    )
                    if result.action is not None and result.action not in allowed:
                        raise AssertionError("unauthorized selected action")
                    regret = None
                    if best is not None and result.lower is not None:
                        regret = best - result.lower
                        if result.upper is None or regret > result.gap + 1e-9:
                            raise AssertionError("false certified regret bound")
                    if budget == len(M.ACTIONS):
                        if (bool(allowed) and result.status != "epsilon_optimal") or (
                            not allowed and result.status != "proven_empty"
                        ):
                            raise AssertionError("incomplete exhaustive verdict")
                    aggregates[(tag, sc, name, budget)].append({
                        "checks": len(result.checked),
                        "viable": bool(allowed),
                        "selected": result.action is not None,
                        "certified": result.status == "epsilon_optimal",
                        "empty_proved": result.status == "proven_empty",
                        "utility": result.lower if result.lower is not None else 0.0,
                        "regret": regret,
                        "gap": result.gap,
                    })
    rows = []
    for (tag, sc, name, budget), records in sorted(aggregates.items()):
        viable = [x for x in records if x["viable"]]
        selected = [x for x in viable if x["selected"]]
        rows.append({
            "tag": tag, "scenario": sc, "strategy": name, "budget": budget,
            "cases": len(records), "viable": len(viable),
            "mean_checks": mean(x["checks"] for x in records),
            "viable_abstentions": len(viable) - len(selected),
            "viable_mean_utility_abstain_zero": (
                mean(x["utility"] for x in viable) if viable else None
            ),
            "executed_mean_regret": (
                mean(x["regret"] for x in selected) if selected else None
            ),
            "exactly_certified": sum(x["certified"] for x in records),
            "proven_empty": sum(x["empty_proved"] for x in records),
            "max_certificate_gap": max((x["gap"] for x in selected), default=None),
        })
    data = {
        "status": "retrospective_descriptive_replay",
        "source": "CCA-05 real TRM checkpoint logits, no new inference",
        "evaluator": "CCA-05 modeled-rival environment-owned checker",
        "upper": METHOD, "environment_digest": M.ENV_DIGEST,
        "tape_sha256": tape_hashes,
        "sample": ("D_bench full" if not limit else f"deterministic {len(domain)}-state smoke"),
        "budgets": list(budgets), "epsilon": epsilon, "rows": rows,
    }
    path = outdir / "metrics.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    for row in rows:
        if row["budget"] in (1, max(budgets)):
            print("CCA09", row["tag"], row["scenario"], row["strategy"],
                  "K", row["budget"], "optimal", row["exactly_certified"],
                  "abstain", row["viable_abstentions"], "calls", round(row["mean_checks"], 3))
    print("WROTE", path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experiments/cca09_anytime_certified_search/outputs")
    parser.add_argument("--tags", nargs="+", default=["X211","X223","X227","R211","R223","R227"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1,2,3,4,5])
    parser.add_argument("--epsilon", type=float, default=0.0)
    args = parser.parse_args()
    if args.limit < 0 or not args.budgets or min(args.budgets)<1 or max(args.budgets)>5:
        parser.error("invalid limit or budgets")
    run(Path(args.out), args.tags, args.limit, args.budgets, args.epsilon)

if __name__ == "__main__":
    main()
