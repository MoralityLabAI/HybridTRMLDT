import dataclasses
import json
import math

import numpy as np
import pytest

from research_gym.trm_skill_gym import modules as M
from research_gym.trm_skill_gym_v3 import research_env as R


def _tiny(family="screen_gate_h4", root=3_200_000, budget=1.3, max_abandon=1):
    spec, truth = R.make_world(family, root)
    return dataclasses.replace(spec, budget=budget, max_abandon=max_abandon), truth


def test_determinism_and_root_ranges():
    for fam in R.FAMILY_CFG:
        a, ta = R.make_world(fam, 3_200_007)
        b, tb = R.make_world(fam, 3_200_007)
        assert np.array_equal(a.L1, b.L1) and np.array_equal(a.prior, b.prior)
        assert (ta.h, ta.ncode) == (tb.h, tb.ncode) and np.array_equal(ta.u, tb.u)
        c, _ = R.make_world(fam, 3_200_008)
        assert not np.array_equal(a.L1, c.L1)
    assert R.ROOT_RANGES["train"][0] == 3_000_000 and R.ROOT_RANGES["calibration"][0] == 3_100_000
    assert R.ROOT_RANGES["test"][0] == 3_200_000
    spec, truth = R.make_world("confounded_h5", 3_200_001)
    r1 = R.run_episode(spec, truth, R.GreedyEIG())
    r2 = R.run_episode(spec, truth, R.GreedyEIG())
    assert r1["utility"] == r2["utility"] and r1["first_action"] == r2["first_action"]


def _brute_value(spec, seq, alive, spent_c):
    """Expectimax from first principles: posterior recomputed from the whole outcome sequence."""
    hist = [0] * spec.E
    w = R.joint_prior(spec).copy()
    for j, y in seq:
        flag = 1 if spec.reuse[j] >= 0 and hist[spec.reuse[j]] != 0 else 0
        hist[j] = 1 + y + 2 * flag
        w = w * (spec.L1[j, flag] if y else 1 - spec.L1[j, flag])
    pseq = w.sum()
    w = w / pseq
    post = w.reshape(spec.H, spec.NN).sum(1)
    best = -1e9
    n_alive = bin(alive).count("1")
    for a in R.legal_actions(spec, tuple(hist), alive, spent_c):
        if a[0] == "exp":
            j = a[1]
            c = R.cost_now(spec, j, tuple(hist), n_alive)
            flag = 1 if spec.reuse[j] >= 0 and hist[spec.reuse[j]] != 0 else 0
            p1 = float(w @ spec.L1[j, flag])
            q = -c
            for y, py in ((1, p1), (0, 1 - p1)):
                if py > 1e-12:
                    q += py * _brute_value(spec, seq + [(j, y)], alive, spent_c + int(round(c * 100)))
        elif a[0] == "abandon":
            q = _brute_value(spec, seq, alive & ~(1 << a[1]), spent_c)
        else:
            q = R.terminal_value(a, post)
        best = max(best, q)
    return best


@pytest.mark.parametrize("budget,max_ab", [(1.0, 1), (1.45, 0)])
def test_planner_matches_brute_force(budget, max_ab):
    spec, _ = _tiny(budget=budget, max_abandon=max_ab)
    pl = R.Planner(spec)
    assert pl.solve()
    v = pl.memo[((0,) * spec.E, (1 << spec.H) - 1, 0)][0]
    assert v == pytest.approx(_brute_value(spec, [], (1 << spec.H) - 1, 0), abs=1e-9)
    # planner is never worse than doing nothing useful
    assert v >= R.ABSTAIN_A - 1e-9


def test_planner_intractable_is_reported_not_raised():
    spec, _ = R.make_world("deep_chain_h5", 3_200_000)
    pl = R.Planner(spec, max_states=10)
    assert pl.solve() is False and pl.status.startswith("intractable")
    assert not R.plan_world(spec, max_states=10).ok


def test_policies_never_see_hidden_truth():
    spec, truth = R.make_world("screen_gate_h4", 3_200_003)
    other = R.Truth((truth.h + 1) % spec.H, truth.ncode ^ 1, 1 - truth.u)
    o1, o2 = R.Env(spec, truth).obs(), R.Env(spec, other).obs()
    assert o1.actions == o2.actions and o1.hist == o2.hist and np.array_equal(o1.belief, o2.belief)
    assert {f.name for f in dataclasses.fields(R.Obs)} == {"spec", "hist", "alive", "spent_c", "belief", "actions"}
    assert not any(isinstance(getattr(o1, f.name), R.Truth) for f in dataclasses.fields(R.Obs))
    assert not hasattr(o1, "truth") and not hasattr(o1.spec, "truth")
    e1, e2 = R.encode(o1), R.encode(o2)
    assert all(np.array_equal(e1[k].numpy(), e2[k].numpy()) for k in e1)
    # a policy that is only handed Obs picks identical actions in worlds with different truth
    for pol in (R.GreedyEIG(), R.VarianceHeuristic(), R.ReuseFirst()):
        assert pol.act(o1) == pol.act(o2)


def test_prerequisite_enforcement():
    spec, truth = R.make_world("screen_gate_h4", 3_200_004)
    env = R.Env(spec, truth)
    acts = env.obs().actions
    assert ("exp", 3) not in acts and ("exp", 4) not in acts and ("exp", 1) in acts
    with pytest.raises(ValueError):
        env.step(("exp", 3))
    env.step(("exp", 1))
    assert ("exp", 3) in env.obs().actions and ("exp", 4) not in env.obs().actions
    deep, dt = R.make_world("deep_chain_h5", 3_200_004)
    env = R.Env(deep, dt)
    assert ("exp", 3) not in env.obs().actions
    env.step(("exp", 0))
    assert ("exp", 1) in env.obs().actions and ("exp", 3) not in env.obs().actions
    env.step(("exp", 1))
    assert ("exp", 3) in env.obs().actions and ("exp", 4) not in env.obs().actions
    with pytest.raises(ValueError):
        env.step(("exp", 0))  # run-once


def test_reuse_lowers_cost_and_sharpens():
    spec, truth = R.make_world("screen_gate_h4", 3_200_005)
    h0 = (0,) * spec.E
    h1 = (1,) + (0,) * (spec.E - 1)
    assert R.cost_now(spec, 1, h1, 4) == pytest.approx(R.cost_now(spec, 1, h0, 4) - spec.disc[1])
    assert not np.array_equal(spec.L1[1, 0], spec.L1[1, 1])


def test_abandonment_semantics():
    spec, truth = R.make_world("screen_gate_h4", 3_200_006)
    env = R.Env(spec, truth)
    before = R.cost_now(spec, 1, env.hist, spec.H)
    t = truth.h
    env.step(("abandon", t))
    obs = env.obs()
    assert env.abandoned_truth and ("conclude", t) not in obs.actions
    assert not any(a[0] == "abandon" for a in obs.actions)  # family limit reached
    assert R.cost_now(spec, 1, env.hist, spec.H - 1) == pytest.approx(before - R.CARRY)
    other = next(h for h in range(spec.H) if h != t)
    r, done = env.step(("conclude", other))
    assert done and r == -R.PENALTY_P and env.outcome == "false"
    with pytest.raises(RuntimeError):
        env.step(("abstain",))
    # cannot abandon below two alive hypotheses
    deep, dt = R.make_world("deep_chain_h5", 3_200_006)
    e = R.Env(deep, dt)
    for h in (0, 1, 2):
        if any(a == ("abandon", h) for a in e.obs().actions):
            e.step(("abandon", h))
    assert bin(e.alive).count("1") >= 2 and bin(e.alive).count("1") >= deep.H - deep.max_abandon


def test_utility_accounting_and_budget():
    spec, truth = R.make_world("confounded_h5", 3_200_009)
    env = R.Env(spec, truth)
    total, costs = 0.0, 0.0
    for a in [("exp", 0), ("exp", 1), ("exp", 2)]:
        c = R.cost_now(spec, a[1], env.hist, bin(env.alive).count("1"))
        r, done = env.step(a)
        assert r == pytest.approx(-c) and not done
        total += r
        costs += c
    assert env.spent_c == int(round(costs * 100))
    best = max((a for a in env.obs().actions if a[0] == "conclude"), key=lambda a: R.hyp_post(spec, env.belief)[a[1]])
    r, done = env.step(best)
    assert done
    assert r == (R.REWARD_R if best[1] == truth.h else -R.PENALTY_P)
    assert total + r == pytest.approx(r - costs)
    rec = R.run_episode(spec, truth, R.GreedyEIG())
    assert rec["utility"] == pytest.approx({"correct": R.REWARD_R, "false": -R.PENALTY_P, "abstain": R.ABSTAIN_A}[rec["outcome"]] - rec["cost"])
    # budget: spending is capped
    for pol in (R.RandomPolicy(1), R.ReuseFirst()):
        r = R.run_episode(spec, truth, pol)
        assert r["cost"] <= spec.budget + 1e-9


def test_baselines_complete_and_exact_dominates_in_expectation():
    worlds = [R.make_world("confounded_h5", 3_200_020 + i) for i in range(2)]
    for spec, truth in worlds:
        plan = R.plan_world(spec)
        assert plan.ok
        obs = R.Env(spec, truth).obs()
        qs = plan.lookup(obs)[1]
        for pol in (R.GreedyEIG(), R.VarianceHeuristic(), R.ReuseFirst(), R.RandomPolicy(3)):
            rec = R.run_episode(spec, truth, pol, plan)
            assert rec["outcome"] in ("correct", "false", "abstain")
            a = pol.act(obs)
            assert qs[obs.actions.index(a)] <= max(qs) + 1e-9
        ex = R.run_episode(spec, truth, R.ExactPolicy(plan), plan)
        a0 = next(a for a in obs.actions if R.action_name(a) == ex["first_action"])
        assert qs[obs.actions.index(a0)] == pytest.approx(plan.v_star())


def test_surprise_condition_misspecified_prior_contradicts():
    hits = 0
    for i in range(6):
        spec, truth = R.make_world("surprise_h4", 3_200_030 + i)
        fav = int(np.argmax(spec.prior))
        assert truth.h != fav and spec.prior[fav] > 0.3
        hits += R._contradicts(spec, truth)
    assert hits >= 4


def test_learned_policy_interface_and_param_match():
    cfgs = R.model_configs(32)
    n = {k: M.SkillModule(v).n_params() for k, v in cfgs.items()}
    assert n["trm"] == n["rlt"]
    for k in ("ff", "gru"):
        assert abs(n[k] - n["trm"]) / n["trm"] <= 0.15
    assert cfgs["rlt"]["feedback"] == "rlt" and cfgs["trm"]["feedback"] == "direct"
    spec, truth = R.make_world("deep_chain_h5", 3_200_040)
    enc = R.encode(R.Env(spec, truth).obs())
    assert enc["cand"].shape == (1, 16, 24) and enc["glob"].shape == (1, 8) and enc["rel"].shape == (1, 16, 16)
    pol = R.LearnedPolicy(M.SkillModule(cfgs["rlt"]), "rlt")
    rec = R.run_episode(spec, truth, pol)
    assert rec["flops"] > 0 and 0 <= pol.last_conf <= 1
    assert "Trigger" in R.SKILL_MD and "Termination" in R.SKILL_MD


def test_smoke_cli(tmp_path):
    out = tmp_path / "run"
    R.main(["--out", str(out), "--smoke", "--seeds", "10", "11"])
    res = json.loads((out / "results.json").read_text())
    assert res["complete_jobs"] == ["baselines", "ff_s10", "gru_s10", "rlt_s10", "trm_s10"]
    assert res["incomplete_jobs"] == []
    assert res["root_ranges"]["train"] == [3_000_000, 3_100_000]
    assert len(res["source_sha256_research_env_py"]) == 64
    assert res["steps_completed_per_job"]["trm_s10"] == 20 and "trm_s10" in res["train_cpu_seconds_per_job"]
    base = res["aggregate"]["baselines"]
    assert set(base) == {"exact_planner", "greedy_eig_per_cost", "variance_heuristic", "reuse_first", "random_valid"}
    assert "surprise_h4" in res["aggregate"]["learned"]["trm"]
    assert math.isfinite(res["aggregate"]["learned"]["rlt"]["dev_screen_gate_h4"]["rollout"]["utility"]["mean"])
    # resumable: a second invocation reuses finished jobs
    before = (out / "jobs" / "trm_s10.json").stat().st_mtime_ns
    R.main(["--out", str(out), "--smoke", "--seeds", "10"])
    assert (out / "jobs" / "trm_s10.json").stat().st_mtime_ns == before


def test_deadline_marks_incomplete(tmp_path):
    out = tmp_path / "dl"
    R.main(["--out", str(out), "--smoke", "--seeds", "10", "--hours", "0.0000001"])
    res = json.loads((out / "results.json").read_text())
    assert "baselines" in res["incomplete_jobs"] and not res["complete_jobs"]
