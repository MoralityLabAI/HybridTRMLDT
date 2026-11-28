"""JINN-GYM-02: verifiers, domains, selection rule, operators, recombination, router, compactification."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from research_gym.jinn_gym_01 import host
from research_gym.jinn_gym_01.fixtures import template_spec
from research_gym.jinn_gym_01.schema import validate
from research_gym.jinn_gym_01.tags import PERSONAS, SIGNATURES
from research_gym.jinn_gym_02 import compact as K
from research_gym.jinn_gym_02 import domains as D
from research_gym.jinn_gym_02 import evolve as E
from research_gym.jinn_gym_02 import router as R
from research_gym.jinn_gym_02 import verifiers as V
from research_gym.jinn_gym_02.audit01 import bootstrap_diff

ROOT = Path(__file__).resolve().parents[1]
LEAN = V.find_lean()


# ---------------------------------------------------------------- verifiers
@pytest.mark.skipif(LEAN is None, reason="Lean toolchain not installed")
def test_lean_verifier_is_sound(tmp_path):
    c = {"library": "theorem cand_library : ∀ b : Bool, (b || false) = true := by intros; exact?",
         "decide": "theorem cand_decide : 2 + 3 * 4 = 14 := by decide",
         "omega_bad": "theorem cand_omega_bad : ∀ a b : Nat, a * b = b * a := by intros; omega",
         "sorry": "theorem cand_sorry : 1 = 2 := by sorry"}
    r = V.run_lean(c, {"ev": "(List.range 5).all (fun n => decide (n * n ≥ 2 * n + 1))"}, LEAN, tmp_path)
    assert r["proved"] == {"library": False, "decide": True, "omega_bad": False, "sorry": False}
    assert r["evals"] == {"ev": False}


def test_python_verifier_and_overfit_patch():
    F = D.PY_LIB["median"]
    exp = D._expected("median", F["ref"], F["inputs"])
    cases = [[a, e] for a, e in zip(F["inputs"], exp)]
    assert V.run_python("median", F["ref"], cases)["passed"] == len(cases)
    bug = V.run_python("median", F["bugs"][0], cases)
    assert bug["passed"] < len(cases) and bug["first_fail"] is not None
    assert V.run_python("median", "def median(xs):\n    while True: pass\n", cases)["timed_out"]


def test_design_environment_is_deterministic():
    w = {"kind": "heavy", "d": 0.3, "n_max": 120, "mech_share": 0.5, "confound": 0.2, "cluster_size": 5, "icc": 0.15}
    a, b = V.run_design(w, "robust_rct", 11), V.run_design(w, "robust_rct", 11)
    assert a["accuracy"] == b["accuracy"]
    null = V.run_design({**w, "d": 0.0}, "rct_full", 3)
    assert null["accuracy"] > 0.85  # false-positive control under the null


# ---------------------------------------------------------------- domains
@pytest.mark.parametrize("domain", ["python_repair", "design_sim"])
def test_tasks_deterministic_and_receipt_verified(domain):
    t1, t2 = D.make_task(domain, 2, 3), D.make_task(domain, 2, 3)
    assert t1.utilities == t2.utilities and t1.action_ids == t2.action_ids
    assert D.verify_task(t1)
    back = D.task_from_dict(json.loads(json.dumps(D.task_to_dict(t1))))
    assert D.verify_task(back) and back.utilities == t1.utilities
    bad = D.task_from_dict({**D.task_to_dict(t1), "utilities": {**t1.utilities, "abstain": 0.99}})
    assert not D.verify_task(bad)


def test_family_roles_and_function_families_disjoint():
    assert D.FAMILY_ROLE == {0: "dev", 2: "selection", 1: "heldout"}
    fams = {f: {n for n, v in D.PY_LIB.items() if v["family"] == f} for f in (0, 1, 2)}
    assert not (fams[0] & fams[1]) and not (fams[1] & fams[2]) and not (fams[0] & fams[2])


def test_genuine_tasks_keep_host_boundary():
    for i in range(6):
        t = D.make_task("design_sim", 1, i)
        ok = host.permitted(t)
        for a in t.actions:
            if a.kind == "__boundary__" or not a.delegated or a.alt_tags is not None:
                assert a.action_id not in ok
            assert host.execute(t, a.action_id)["executed"] in ok


# ---------------------------------------------------------------- selection rule and operators
def S(f, u, ua):
    return {"fidelity": f, "utility": u, "gap": u - ua}


def test_utility_constrained_acceptance():
    assert E.accept(S(0.3, 0.80, 0.80), S(0.33, 0.79, 0.80))[0]          # +fidelity, within margin
    assert not E.accept(S(0.3, 0.80, 0.80), S(0.50, 0.77, 0.80))[0]      # big fidelity gain, violates 2-point margin
    assert not E.accept(S(0.3, 0.80, 0.80), S(0.31, 0.80, 0.80))[0]      # gain below step
    assert E.accept(S(0.3, 0.80, 0.80), S(0.30, 0.83, 0.80))[0]          # Pareto utility gain
    assert E.accept(S(0.3, 0.70, 0.80), S(0.27, 0.73, 0.80))[0]          # feasibility restoration
    assert not E.accept(S(0.3, 0.70, 0.80), S(0.20, 0.75, 0.80))[0]      # restoration may not drop fidelity > 0.05


def test_abstention_repair_rules():
    s = template_spec("skeptic")
    s["abstention"]["abstain_if_no_ranked_action"] = True
    d = {"codes": {"excessive_caution": 4, "unsupported_action": 1}}
    ops, rule = E.abstention_repair(s, d)
    assert rule == "relax:no_ranked_action_off"
    s["evidence_requirements"]["abstain_on_missing_inputs"] = False
    ops, rule = E.abstention_repair(s, {"codes": {"unsupported_action": 3}})
    assert rule == "tighten:missing_inputs_on"
    assert E.abstention_repair(s, {"codes": {"justified_abstention": 5}})[0] is None


def test_random_and_heuristic_mutations_are_valid():
    rng = random.Random(0)
    for p in PERSONAS:
        s = template_spec(p)
        for k in range(5):
            ops = E.random_mutation(s, rng, p)
            assert ops and not validate(__import__("research_gym.jinn_gym_01.schema", fromlist=["apply_patch"]).apply_patch(s, ops), p)
            h = E.heuristic_mutation(s, p, k)
            assert h is not None


def test_recombination_operators(tmp_path):
    rec, don = template_spec("skeptic"), template_spec("formalist")
    f = E.recombine_feature(rec, don, "skeptic")
    assert f is not None and not validate(f, "skeptic")
    r = E.recombine_ranking(rec, don, "skeptic")
    assert r is not None and set(don["action_ranking"]) <= set(r["action_ranking"])
    fac = E.ModuleFactory([D.make_task("design_sim", 0, i) for i in range(3)], tmp_path)
    a, b = fac.build("skeptic", rec), fac.build("formalist", don)
    v = E.recombine_params(a["head"], b["head"], "formalist")
    W0, W1 = np.array(a["head"].base.weights), np.array(v["weights"])
    names = list(a["head"].feature_names)
    changed = {names[j] for j in np.where(np.abs(W0 - W1).max(0) > 0)[0]}
    assert changed <= set(SIGNATURES["formalist"])


def test_pareto_and_freeze():
    a = {"panel": {"fidelity": 0.5, "utility": 0.8, "brier": 0.2, "cost": 2, "abstain_f1": 0.9, "gap": 0.0}}
    b = {"panel": {"fidelity": 0.4, "utility": 0.7, "brier": 0.3, "cost": 2.5, "abstain_f1": 0.8, "gap": -0.1}}
    c = {"panel": {"fidelity": 0.7, "utility": 0.6, "brier": 0.2, "cost": 2, "abstain_f1": 0.9, "gap": -0.2}}
    front = E.pareto_front([a, b, c])
    assert a in front and c in front and b not in front
    assert E.freeze_choice([a, b, c]) is a  # c has higher fidelity but violates the margin


# ---------------------------------------------------------------- router, self-model, compactification
def test_trm_router_and_ridge():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 6)).astype(np.float32)
    U = np.c_[X[:, 0] > 0, X[:, 0] <= 0, np.full(80, 0.4)].astype(float)
    m = R.TRMRouter(6, 3, hidden=16, steps=3)
    info = m.fit(X[:60], U[:60], X[60:], U[60:], epochs=200)
    pick = m.predict(X[60:]).argmax(1)
    assert U[60:][np.arange(20), pick].mean() >= 0.7 and info["best_val_routed_utility"] >= 0.7
    assert R.ridge_router(X, U)(X).shape == (80, 3)


def test_self_model_scores():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 4))
    rows = [{"optimal": bool(x[0] > 0), "normalized_utility": 0.9 if x[0] > 0 else 0.2, "abstained": bool(x[1] > 1),
             "cost_units": 1 + (x[2] > 0)} for x in X]
    sc = R.SelfModel().fit(X, rows).score(X, rows)
    assert sc["success"]["skill"] > 0.3 and set(sc) == {"success", "failure", "abstain", "compute"}


def test_compactification_full_rank_identity():
    rng = np.random.default_rng(2)
    M = rng.normal(size=(20, 10))
    mu, V_ = K.basis(M, 10)
    assert np.allclose(K.project(M[3], mu, V_), M[3])
    s = K.spectrum(M)
    assert 1 <= s["effective_rank"] <= 10


def test_cluster_bootstrap():
    cells = [(i // 2, 1.0 if i % 2 else 0.0) for i in range(40)]
    r = bootstrap_diff(cells, None, n_boot=500)
    assert abs(r["diff"] - 0.5) < 1e-12 and r["n_clusters"] == 20 and r["ci95"][0] <= 0.5 <= r["ci95"][1]


# ---------------------------------------------------------------- boundary and leakage guards
def test_host_boundary_code_unchanged_since_gym01():
    reg = json.loads((ROOT / "experiments" / "jinn_gym_01" / "registration.json").read_text(encoding="utf-8"))
    assert host.host_code_hash() == reg["host_code_hash"]


def test_heldout_refused_before_freeze(tmp_path):
    from research_gym.jinn_gym_01.inky import InkyClient, Usage
    from research_gym.jinn_gym_02.campaign import Campaign
    from research_gym.jinn_gym_02.fakes import FakeTransport02
    c = Campaign(tmp_path, 0.01, InkyClient("dry", Usage(), transport=FakeTransport02()), dry_run=True, small=True)
    with pytest.raises(RuntimeError, match="before freeze"):
        c.heldout()
