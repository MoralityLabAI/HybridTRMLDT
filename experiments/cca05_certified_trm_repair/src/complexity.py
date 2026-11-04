"""Phase 7 accounting: size of what must be trusted versus the TRM.

  python -I src/complexity.py      # writes outputs/complexity.json
"""

from __future__ import annotations

import inspect
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parents[1]))

import semantics as M  # noqa: E402
from research_gym.core import certified_repair as CR  # noqa: E402
from research_gym.core import hybrid as H  # noqa: E402
from research_gym.envs.coupled_storyworld import CoupledStoryworldEnv  # noqa: E402

COORDS = ("scenario", "trust", "evidence", "heat", "scene")


def code_lines(fn) -> int:
    """Non-blank, non-comment source lines, docstrings included."""
    return sum(1 for ln in inspect.getsource(fn).splitlines()
               if ln.strip() and not ln.strip().startswith("#"))


def tree_leaves(X: list[tuple[int, ...]], y: list) -> int:
    """Leaves of a best-first Gini tree grown to purity over integer coordinates
    (tests `coordinate >= threshold`), the grammar of complete_accounts_v1 Part B."""
    def gini(idx):
        c = Counter(y[i] for i in idx)
        n = len(idx)
        return n - sum(v * v for v in c.values()) / n if n else 0.0

    def best(idx):
        if len({y[i] for i in idx}) <= 1:
            return None
        base, top = gini(idx), None
        for f in range(len(X[0])):
            for t in sorted({X[i][f] for i in idx})[1:]:
                left = [i for i in idx if X[i][f] < t]
                right = [i for i in idx if X[i][f] >= t]
                g = base - gini(left) - gini(right)
                if top is None or g > top[0] + 1e-12:
                    top = (g, left, right)
        return top

    leaves = [list(range(len(y)))]
    splits = [best(leaves[0])]
    while any(s is not None for s in splits):
        i = max((k for k, s in enumerate(splits) if s is not None), key=lambda k: splits[k][0])
        _, left, right = splits[i]
        leaves[i:i + 1] = [left, right]
        splits[i:i + 1] = [best(left), best(right)]
    return len(leaves)


def main() -> None:
    dom = M.bench_domain()
    X = [(M.SCENARIOS.index(sc), s.trust, s.evidence, s.heat, s.scene) for sc, s in dom]
    masks = [tuple(a in M.certified_set(sc, s, M.HORIZON) for a in M.ACTIONS) for sc, s in dom]
    per_action = {a: tree_leaves(X, [m[i] for m in masks]) for i, a in enumerate(M.ACTIONS)}
    trusted = {
        "CoupledStoryworldEnv.step": code_lines(CoupledStoryworldEnv.step),
        "CoupledStoryworldEnv.other_policy": code_lines(CoupledStoryworldEnv.other_policy),
        "CoupledStoryworldEnv.target": code_lines(CoupledStoryworldEnv.target),
        "CoupledStoryworldEnv.terminal": code_lines(CoupledStoryworldEnv.terminal),
        "CoupledStoryworldEnv.clamp": code_lines(CoupledStoryworldEnv.clamp),
        "semantics.successor": code_lines(M.successor),
        "semantics.terminal_value": code_lines(M.terminal_value),
        "semantics.witness_ends_in_goal": code_lines(M.witness_ends_in_goal),
        "semantics.check_certificate": code_lines(M.check_certificate),
        "certified_repair.execute_certified": code_lines(CR.execute_certified),
        "certified_repair.bind": code_lines(CR.bind),
    }
    issuer = {
        "semantics.admissible": code_lines(M.admissible),
        "semantics.best_value": code_lines(M.best_value.__wrapped__),
        "CoupledStoryworldEnv.reachable": code_lines(CoupledStoryworldEnv.reachable),
        "semantics.issue": code_lines(M.issue),
        "semantics._moral_witness": code_lines(M._moral_witness),
    }
    membrane = {
        "certified_repair.bounded_recurrent_repair": code_lines(CR.bounded_recurrent_repair),
        "hybrid.certify_and_apply_verified": code_lines(H.certify_and_apply_verified),
        "hybrid.certify_and_apply": code_lines(H.certify_and_apply),
        "controllers.elimination_verifier": None,
    }
    import controllers as C
    membrane["controllers.elimination_verifier"] = code_lines(C.elimination_verifier)
    out = {
        "trusted_execution_path_lines": trusted, "trusted_execution_path_total": sum(trusted.values()),
        "issuer_lines": issuer, "issuer_total": sum(issuer.values()),
        "repair_membrane_lines": membrane, "repair_membrane_total": sum(membrane.values()),
        "verified_constraints": {"per_decision": "one: Verify(s, a, k) for the executed action",
                                 "parameters": {"horizon": M.HORIZON, "moral_threshold": M.MORAL_THRESHOLD}},
        "certified_set_table": {"inputs": len(dom), "distinct_masks": len(set(masks)),
                                "exact_tree_leaves_joint_mask": tree_leaves(X, masks),
                                "exact_tree_leaves_per_action": per_action},
        "trm": {"parameters": 15521, "recurrent_passes_per_query": 4,
                "published_exact_final_pass_tree_leaves_action_mode_conflict":
                    {"X": "677 to 834", "R": "371 to 588"},
                "published_exact_automaton_states": "229 to 590",
                "source": "jinn-or-beast-paper complete_accounts_v1/RESULTS.md"},
        "trusted_components": [
            "transition function (CoupledStoryworldEnv.step, clamp)",
            "rival model (other_policy)",
            "goal predicate and tau (target, terminal_value, MORAL_THRESHOLD)",
            "certificate checker (witness_ends_in_goal, check_certificate)",
            "executor (execute_certified, bind)",
            "elimination verifier inside certify_and_apply_verified (A4 only)",
        ],
        "untrusted": ["TRM weights and logits", "TRM provenance head", "issuer search (its certificates are rechecked)"],
    }
    text = json.dumps(out, indent=1, sort_keys=True) + "\n"
    (ROOT / "outputs" / "complexity.json").write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
