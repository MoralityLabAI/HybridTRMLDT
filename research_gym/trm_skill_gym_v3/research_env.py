"""TRM-SKILL-GYM-02 sequential research environment ("Baby Goblin's research skill", v3).

GYM-01 research was one-shot (pick the max EIG-minus-cost experiment) and a hand variance
heuristic matched it.  This module is genuinely sequential:

* competing hypotheses with a public prior, binary nuisance variables shared across
  experiments (correlated evidence, confounding), cheap screening / control experiments
  and expensive definitive experiments, prerequisite chains, reusable intermediate
  evidence (a finished control experiment makes later experiments cheaper and sharper),
  an option to abandon a hypothesis (it can no longer be concluded and the carrying cost
  of every later experiment drops; abandoning the true hypothesis forces a false
  conclusion or an abstain), a hard adaptive budget, and terminal actions
  ``conclude:<h>`` / ``abstain``.
* verified research utility at episode end: +R for a correct conclusion, -P for a false
  one, +A for abstaining (A < R), minus the total experiment cost.
* the hidden truth (true hypothesis, true nuisance values, outcome noise) lives only in
  ``Env._truth``; policies receive ``Obs`` (public spec, observed outcomes, availability,
  abandoned set, spent/remaining budget, Bayes belief computed from public data).
* exact finite-horizon Bayes planner (memoised DP over (outcomes, alive set, spent)),
  four hand baselines, and learned FF / GRU / TRM / RLT-TRM policies that reuse
  ``research_gym.trm_skill_gym.modules.SkillModule`` through the shared candidate-set
  interface (one candidate per legal action, ``tasks.N`` slots).

Ideas reused from DataGoblin (read-only, nothing imported):
``src/datagoblin/training_gym/rlt_gym/research_controller.py`` (public-evidence-only
``abstract`` view, paired trials on shared cases) and
``training/rlt_gym/hypergame_taste/skills/research_hillclimb/SKILL.md`` (skill layout).
Paired evaluation here means every policy sees identical roots and identical outcome noise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from research_gym.trm_skill_gym import dsl
from research_gym.trm_skill_gym import modules as M
from research_gym.trm_skill_gym import tasks as T

SKILL_MD = """# Research TRM v3: sequential experiment planning

## Trigger
A finite set of competing hypotheses (3-6) with a public prior must be resolved by running
experiments. Experiment outcomes are binary, noisy, and correlated through shared hidden
nuisance variables. Some experiments are only available after a prerequisite experiment has
run; a finished control experiment makes later experiments cheaper and sharper. There is a
hard budget.

## Objective
Maximise verified research utility: +R for concluding the true hypothesis, -P for concluding a
false one, +A (A < R) for abstaining, minus the total cost of the experiments run.

## Required inputs
Public only: prior over hypotheses, nuisance priors, per-experiment outcome likelihood tables,
costs, prerequisites, reuse sources, budget, outcomes observed so far, which hypotheses were
abandoned, remaining budget. The hidden truth and outcome noise are never visible.

## Action schema
{"action": "exp:<j>" | "abandon:<h>" | "conclude:<h>" | "abstain"}
Exactly one action per step. exp:<j> is legal when experiment j has not run, its prerequisite
has run, and its current cost fits in the remaining budget. abandon:<h> prunes an alive
hypothesis (at most the family limit, at least two stay alive); it removes conclude:<h> and
cuts the per-experiment carrying cost, but is a false conclusion in waiting if h is true.

## Procedure
1. Update the posterior over (hypothesis, nuisance) from every observed outcome.
2. Prefer cheap control/screening experiments that unlock or sharpen definitive ones; beware
   screens whose outcome is explained by a nuisance as well as by a hypothesis.
3. Run a definitive experiment only when the remaining uncertainty is worth its cost.
4. Abandon a hypothesis only when its posterior is small enough that the carrying-cost saving
   exceeds the expected loss of excluding it.
5. When an outcome contradicts the leading hypothesis, revise the plan from the new posterior
   instead of continuing the earlier sequence.
6. Conclude the best hypothesis if its expected utility beats abstaining, else abstain.

## Verifier contract
The host owns the truth. After a terminal action it computes utility = terminal reward - sum of
experiment costs, and records whether the conclusion was correct. Module confidence outputs are
scored by calibration against whether the chosen action was optimal under the exact planner.

## Termination
The episode ends at conclude:<h> or abstain, or when no experiment is affordable (only terminal
and abandon actions remain, then a terminal action must be taken).
"""

ROOT_RANGES = {"train": (3_000_000, 3_100_000), "calibration": (3_100_000, 3_200_000), "test": (3_200_000, 3_300_000)}
TRAIN0, CAL0, TEST0 = ROOT_RANGES["train"][0], ROOT_RANGES["calibration"][0], ROOT_RANGES["test"][0]

REWARD_R, PENALTY_P, ABSTAIN_A = 10.0, 8.0, 1.5
CARRY = 0.08  # per alive hypothesis per experiment

FAMILY_CFG = {
    "screen_gate_h4": dict(H=4, K=2, conf=0.9, budget=4.0, max_abandon=1, chain=1, def_cost=1.8, surprise=False),
    "confounded_h5": dict(H=5, K=2, conf=1.5, budget=4.5, max_abandon=1, chain=1, def_cost=1.8, surprise=False),
    "deep_chain_h5": dict(H=5, K=3, conf=1.8, budget=5.0, max_abandon=2, chain=2, def_cost=2.0, surprise=False),
    "surprise_h4": dict(H=4, K=2, conf=0.9, budget=4.0, max_abandon=1, chain=1, def_cost=1.8, surprise=True),
}
GROUPS = {  # report group -> family
    "dev_screen_gate_h4": "screen_gate_h4",
    "dev_confounded_h5": "confounded_h5",
    "withheld_deep_chain_h5": "deep_chain_h5",
    "surprise_h4": "surprise_h4",
}
TRAIN_FAMILIES = ("screen_gate_h4", "confounded_h5")
E_EXP = 5
KIND_NAMES = ("control", "screen", "screen", "definitive", "definitive")


def _fkey(family: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(family))


# ------------------------------------------------------------------ world
@dataclass(frozen=True)
class PublicSpec:
    family: str
    root: int
    H: int
    K: int
    prior: np.ndarray       # [H] public prior (may be misspecified in the surprise family)
    nprior: np.ndarray      # [K]
    kind: Tuple[str, ...]
    cost: Tuple[float, ...]
    disc: Tuple[float, ...]
    prereq: Tuple[int, ...]
    reuse: Tuple[int, ...]
    L1: np.ndarray          # [E, 2, J] P(outcome=1 | joint) without / with reuse flag
    S: np.ndarray           # [E, H] hypothesis loading (structure of the public model)
    W: np.ndarray           # [E, K] nuisance loading
    confs: np.ndarray       # [E] confounding scale
    budget: float
    max_abandon: int

    @property
    def E(self) -> int:
        return len(self.cost)

    @property
    def NN(self) -> int:
        return 2 ** self.K

    @property
    def J(self) -> int:
        return self.H * self.NN


@dataclass(frozen=True)
class Truth:
    h: int
    ncode: int
    u: np.ndarray           # [E] outcome noise uniforms (common random numbers across policies)


def _nuis_sign(K: int, ncode: int) -> np.ndarray:
    return np.array([2 * ((ncode >> k) & 1) - 1 for k in range(K)], np.float64)


def _build_spec(rng: np.random.Generator, family: str, root: int) -> PublicSpec:
    c = FAMILY_CFG[family]
    H, K = c["H"], c["K"]
    E, NN = E_EXP, 2 ** K
    if c["surprise"]:
        fav = int(rng.integers(H))
        prior = np.full(H, 0.60 / (H - 1))
        prior[fav] = 0.40
    else:
        prior = np.clip(rng.dirichlet(np.full(H, 1.5)), 0.06, None)
        prior = prior / prior.sum()
    nprior = rng.uniform(0.3, 0.7, size=K)
    S = np.zeros((E, H))
    W = np.zeros((E, K))
    sig = np.zeros(E)
    confs = np.zeros(E)
    for j, kind in enumerate(KIND_NAMES):
        W[j] = rng.choice([-1.0, 1.0], size=K)
        if kind == "control":
            S[j] = rng.choice([-1.0, 1.0], size=H)
            sig[j], confs[j] = 0.25, 1.6
        elif kind == "screen":
            while True:
                S[j] = rng.choice([-1.0, 1.0], size=H)
                if 0 < (S[j] > 0).sum() < H:
                    break
            sig[j], confs[j] = 1.0, c["conf"]
        else:
            S[j] = -1.0
            S[j, int(rng.integers(H))] = 1.0
            sig[j], confs[j] = 2.0, 0.8 * c["conf"]
    L1 = np.zeros((E, 2, H * NN))
    signs = [_nuis_sign(K, n) for n in range(NN)]
    for j in range(E):
        for flag in (0, 1):
            ks, kc = (1.4, 0.25) if flag else (1.0, 1.0)
            for h in range(H):
                for n in range(NN):
                    z = ks * sig[j] * S[j, h] + kc * confs[j] * float(W[j] @ signs[n]) / math.sqrt(K)
                    L1[j, flag, h * NN + n] = min(0.98, max(0.02, 1 / (1 + math.exp(-z))))
    dc = c["def_cost"]
    cost = (0.30, 0.40, 0.40, dc, dc)
    disc = (0.0, 0.15, 0.15, 0.6, 0.6)
    if c["chain"] == 1:
        prereq, reuse = (-1, -1, -1, 1, 2), (-1, 0, 0, 0, 0)
    else:
        prereq, reuse = (-1, 0, -1, 1, 3), (-1, -1, 0, 2, 2)
    return PublicSpec(family, root, H, K, prior, nprior, KIND_NAMES, cost, disc, prereq, reuse, L1, S, W, confs,
                      c["budget"], c["max_abandon"])


def joint_prior(spec: PublicSpec) -> np.ndarray:
    NN = spec.NN
    nj = np.ones(NN)
    for n in range(NN):
        for k in range(spec.K):
            q = spec.nprior[k]
            nj[n] *= q if (n >> k) & 1 else 1 - q
    return (spec.prior[:, None] * nj[None, :]).reshape(-1)


def make_world(family: str, root: int) -> Tuple[PublicSpec, Truth]:
    """Pure function of (family, root).  The surprise family misspecifies the public prior
    and rejects truth/noise draws until the first greedy-EIG experiment contradicts the
    prior-favoured hypothesis."""
    c = FAMILY_CFG[family]
    key = _fkey(family)
    attempt = 0
    while True:
        rng = np.random.default_rng([int(root), key, attempt])
        spec = _build_spec(rng, family, root)
        for _ in range(60 if c["surprise"] else 1):
            if c["surprise"]:
                fav = int(np.argmax(spec.prior))
                h = int(rng.choice([x for x in range(spec.H) if x != fav]))
            else:
                h = int(rng.choice(spec.H, p=spec.prior))
            nc = int(sum(int(rng.random() < spec.nprior[k]) << k for k in range(spec.K)))
            truth = Truth(h, nc, rng.random(spec.E))
            if not c["surprise"] or _contradicts(spec, truth):
                return spec, truth
        attempt += 1
        if attempt >= 40:  # deterministic fallback: misspecified prior, no contradiction guarantee
            return spec, truth


def _contradicts(spec: PublicSpec, truth: Truth) -> bool:
    env = Env(spec, truth)
    obs = env.obs()
    acts = [a for a in obs.actions if a[0] == "exp"]
    if not acts:
        return False
    best = max(acts, key=lambda a: exp_eig(spec, obs.belief, obs.hist, a[1]) / max(cost_now(spec, a[1], obs.hist, spec.H), 1e-6))
    env.step(best)
    return int(np.argmax(hyp_post(spec, env.obs().belief))) != int(np.argmax(spec.prior))


# ------------------------------------------------------------------ model helpers
def hyp_post(spec: PublicSpec, belief: np.ndarray) -> np.ndarray:
    return belief.reshape(spec.H, spec.NN).sum(1)


def entropy_bits(p: np.ndarray) -> float:
    p = p[p > 1e-12]
    return float(-(p * np.log2(p)).sum())


def _flag(spec: PublicSpec, hist: Tuple[int, ...], j: int) -> int:
    return 1 if spec.reuse[j] >= 0 and hist[spec.reuse[j]] != 0 else 0


def cost_now(spec: PublicSpec, j: int, hist: Tuple[int, ...], n_alive: int) -> float:
    base = spec.cost[j] - (spec.disc[j] if _flag(spec, hist, j) else 0.0)
    return round(max(0.05, base) + CARRY * n_alive, 2)


def legal_actions(spec: PublicSpec, hist: Tuple[int, ...], alive: int, spent_c: int) -> List[Tuple]:
    n_alive = bin(alive).count("1")
    acts: List[Tuple] = []
    for j in range(spec.E):
        if hist[j] == 0 and (spec.prereq[j] < 0 or hist[spec.prereq[j]] != 0):
            if spent_c + int(round(cost_now(spec, j, hist, n_alive) * 100)) <= int(round(spec.budget * 100)):
                acts.append(("exp", j))
    if spec.H - n_alive < spec.max_abandon and n_alive > 2:
        acts += [("abandon", h) for h in range(spec.H) if alive >> h & 1]
    acts += [("conclude", h) for h in range(spec.H) if alive >> h & 1]
    acts.append(("abstain",))
    return acts


def action_name(a: Tuple) -> str:
    return "abstain" if a[0] == "abstain" else f"{'exp' if a[0] == 'exp' else a[0]}:{a[1]}"


def terminal_value(a: Tuple, post: np.ndarray) -> float:
    if a[0] == "abstain":
        return ABSTAIN_A
    p = float(post[a[1]])
    return REWARD_R * p - PENALTY_P * (1 - p)


def outcome_update(spec: PublicSpec, belief: np.ndarray, hist: Tuple[int, ...], j: int):
    """Returns [(y, prob, new_belief, new_hist_value)] for outcomes with positive probability."""
    flag = _flag(spec, hist, j)
    L = spec.L1[j, flag]
    p1 = float(belief @ L)
    out = []
    for y, py, lik in ((0, 1 - p1, 1 - L), (1, p1, L)):
        if py > 1e-12:
            out.append((y, py, belief * lik / py, 1 + y + 2 * flag))
    return out


def exp_eig(spec: PublicSpec, belief: np.ndarray, hist: Tuple[int, ...], j: int) -> float:
    """Mutual information (bits) between the outcome and the hypothesis."""
    flag = _flag(spec, hist, j)
    cond = belief.reshape(spec.H, spec.NN)
    marg = cond.sum(1)
    p1h = (cond * spec.L1[j, flag].reshape(spec.H, spec.NN)).sum(1) / np.maximum(marg, 1e-12)
    pbar = float(marg @ p1h)

    def hb(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return float(hb(pbar) - marg @ hb(p1h))


def exp_p1h(spec: PublicSpec, belief: np.ndarray, hist: Tuple[int, ...], j: int) -> np.ndarray:
    flag = _flag(spec, hist, j)
    cond = belief.reshape(spec.H, spec.NN)
    marg = cond.sum(1)
    return (cond * spec.L1[j, flag].reshape(spec.H, spec.NN)).sum(1) / np.maximum(marg, 1e-12)


# ------------------------------------------------------------------ environment
@dataclass(frozen=True)
class Obs:
    """Everything a policy may see.  No hidden truth."""
    spec: PublicSpec
    hist: Tuple[int, ...]       # per experiment: 0 not run, else 1 + outcome + 2 * reuse_flag
    alive: int                  # bitmask of non-abandoned hypotheses
    spent_c: int                # spent budget in hundredths
    belief: np.ndarray          # Bayes posterior over (hypothesis, nuisance) from public data
    actions: Tuple[Tuple, ...]

    @property
    def key(self):
        return (self.hist, self.alive, self.spent_c)

    @property
    def remaining(self) -> float:
        return self.spec.budget - self.spent_c / 100


class Env:
    def __init__(self, spec: PublicSpec, truth: Truth):
        self.spec = spec
        self._truth = truth  # host-side only; never passed to policies
        self.hist = (0,) * spec.E
        self.alive = (1 << spec.H) - 1
        self.spent_c = 0
        self.belief = joint_prior(spec)
        self.done = False
        self.outcome: Optional[str] = None
        self.abandoned_truth = False

    def obs(self) -> Obs:
        acts = tuple(legal_actions(self.spec, self.hist, self.alive, self.spent_c))
        assert len(acts) <= T.N
        return Obs(self.spec, self.hist, self.alive, self.spent_c, self.belief.copy(), acts)

    def step(self, action: Tuple):
        """Returns (reward, done).  Reward is -cost for experiments, terminal reward otherwise."""
        if self.done:
            raise RuntimeError("episode finished")
        if action not in legal_actions(self.spec, self.hist, self.alive, self.spent_c):
            raise ValueError(f"illegal action {action}")
        spec, tr = self.spec, self._truth
        if action[0] == "exp":
            j = action[1]
            n_alive = bin(self.alive).count("1")
            c = cost_now(spec, j, self.hist, n_alive)
            flag = _flag(spec, self.hist, j)
            p1 = spec.L1[j, flag, tr.h * spec.NN + tr.ncode]
            y = int(tr.u[j] < p1)
            for yy, _, nb, hv in outcome_update(spec, self.belief, self.hist, j):
                if yy == y:
                    self.belief = nb
                    h = list(self.hist)
                    h[j] = hv
                    self.hist = tuple(h)
            self.spent_c += int(round(c * 100))
            return -c, False
        if action[0] == "abandon":
            self.alive &= ~(1 << action[1])
            if action[1] == tr.h:
                self.abandoned_truth = True
            return 0.0, False
        self.done = True
        if action[0] == "abstain":
            self.outcome = "abstain"
            return ABSTAIN_A, True
        if action[1] == tr.h:
            self.outcome = "correct"
            return REWARD_R, True
        self.outcome = "false"
        return -PENALTY_P, True


# ------------------------------------------------------------------ exact planner
class Intractable(RuntimeError):
    pass


class Planner:
    """Bayes-optimal finite-horizon planner: memoised DP over (outcomes, alive, spent)."""

    def __init__(self, spec: PublicSpec, max_states: int = 60_000, keep_beliefs: bool = False):
        self.spec, self.max_states, self.keep = spec, max_states, keep_beliefs
        self.memo: Dict[Tuple, Tuple[float, List[float]]] = {}
        self.beliefs: Dict[Tuple, np.ndarray] = {}
        self.status = "unsolved"

    def solve(self) -> bool:
        spec = self.spec
        try:
            self._V((0,) * spec.E, (1 << spec.H) - 1, 0, joint_prior(spec))
            self.status = "exact"
            return True
        except Intractable:
            self.memo.clear()
            self.beliefs.clear()
            self.status = f"intractable(>{self.max_states} states)"
            return False

    def _V(self, hist, alive, spent_c, belief) -> float:
        key = (hist, alive, spent_c)
        hit = self.memo.get(key)
        if hit is not None:
            return hit[0]
        if len(self.memo) >= self.max_states:
            raise Intractable()
        spec = self.spec
        post = hyp_post(spec, belief)
        n_alive = bin(alive).count("1")
        qs: List[float] = []
        for a in legal_actions(spec, hist, alive, spent_c):
            if a[0] == "exp":
                j = a[1]
                c = cost_now(spec, j, hist, n_alive)
                ci = spent_c + int(round(c * 100))
                q = -c
                for _, py, nb, hv in outcome_update(spec, belief, hist, j):
                    h2 = hist[:j] + (hv,) + hist[j + 1:]
                    q += py * self._V(h2, alive, ci, nb)
            elif a[0] == "abandon":
                q = self._V(hist, alive & ~(1 << a[1]), spent_c, belief)
            else:
                q = terminal_value(a, post)
            qs.append(q)
        self.memo[key] = (max(qs), qs)
        if self.keep:
            self.beliefs[key] = belief
        return self.memo[key][0]


class PlanTable:
    """Read-only (V, Q) lookup for one world; ``None`` table means intractable."""

    def __init__(self, table: Optional[Dict], status: str, root_key):
        self.table, self.status, self.root_key = table, status, root_key

    @property
    def ok(self) -> bool:
        return self.table is not None

    def lookup(self, obs: Obs):
        return self.table[obs.key]

    def v_star(self) -> float:
        return self.table[self.root_key][0]


def plan_world(spec: PublicSpec, max_states: int = 60_000) -> PlanTable:
    pl = Planner(spec, max_states)
    ok = pl.solve()
    return PlanTable(pl.memo if ok else None, pl.status, ((0,) * spec.E, (1 << spec.H) - 1, 0))


# ------------------------------------------------------------------ policies
class Policy:
    name = "policy"

    def act(self, obs: Obs) -> Tuple:
        raise NotImplementedError


class ExactPolicy(Policy):
    name = "exact_planner"

    def __init__(self, plan: PlanTable):
        self.plan = plan

    def act(self, obs: Obs) -> Tuple:
        qs = self.plan.lookup(obs)[1]
        return obs.actions[int(np.argmax(qs))]


def _best_terminal(obs: Obs) -> Tuple:
    post = hyp_post(obs.spec, obs.belief)
    best = max((a for a in obs.actions if a[0] != "exp" and a[0] != "abandon"), key=lambda a: terminal_value(a, post))
    return best


class _StopRule(Policy):
    stop_post = 0.85

    def choose(self, obs: Obs, exps: List[Tuple]) -> Optional[Tuple]:
        raise NotImplementedError

    def act(self, obs: Obs) -> Tuple:
        post = hyp_post(obs.spec, obs.belief)
        exps = [a for a in obs.actions if a[0] == "exp"]
        if float(post.max()) >= self.stop_post or not exps:
            return _best_terminal(obs)
        a = self.choose(obs, exps)
        return a if a is not None else _best_terminal(obs)


class GreedyEIG(_StopRule):
    name = "greedy_eig_per_cost"

    def choose(self, obs, exps):
        n_alive = bin(obs.alive).count("1")
        sc = [(exp_eig(obs.spec, obs.belief, obs.hist, a[1]) / cost_now(obs.spec, a[1], obs.hist, n_alive), a) for a in exps]
        best = max(sc, key=lambda x: x[0])
        return best[1] if exp_eig(obs.spec, obs.belief, obs.hist, best[1][1]) >= 0.02 else None


class VarianceHeuristic(_StopRule):
    """Port of the GYM-01 idea: belief-weighted variance of the outcome likelihoods
    (marginalised over nuisance, hence fooled by confounding) minus a cost weight."""
    name = "variance_heuristic"
    cost_weight = 1.0

    def choose(self, obs, exps):
        n_alive = bin(obs.alive).count("1")
        marg = hyp_post(obs.spec, obs.belief)
        best, bs = None, 0.0
        for a in exps:
            p1h = exp_p1h(obs.spec, obs.belief, obs.hist, a[1])
            mean = float(marg @ p1h)
            var = float(marg @ (p1h - mean) ** 2)
            s = 20 * var - self.cost_weight * cost_now(obs.spec, a[1], obs.hist, n_alive)
            if s > bs:
                best, bs = a, s
        return best


class ReuseFirst(GreedyEIG):
    """Run experiments that unlock or feed other experiments first (cheapest first)."""
    name = "reuse_first"

    def choose(self, obs, exps):
        spec = obs.spec
        n_alive = bin(obs.alive).count("1")
        feeders = [a for a in exps if _unlocks(spec, obs.hist, a[1]) > 0]
        if feeders:
            return min(feeders, key=lambda a: cost_now(spec, a[1], obs.hist, n_alive))
        return super().choose(obs, exps)


class RandomPolicy(Policy):
    name = "random_valid"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: Obs) -> Tuple:
        return obs.actions[int(self.rng.integers(len(obs.actions)))]


def _random_factory():
    counter = [0]

    def make():
        counter[0] += 1
        return RandomPolicy(counter[0])
    return make


def _unlocks(spec: PublicSpec, hist: Tuple[int, ...], j: int) -> int:
    return sum(1 for k in range(spec.E) if hist[k] == 0 and k != j and (spec.prereq[k] == j or spec.reuse[k] == j))


# ------------------------------------------------------------------ candidate-set encoding
def encode(obs: Obs) -> Dict[str, torch.Tensor]:
    """Public observation -> one candidate per legal action (tasks.N slots, tasks.FC features)."""
    spec = obs.spec
    n_alive = bin(obs.alive).count("1")
    post = hyp_post(spec, obs.belief)
    cand = np.zeros((T.N, T.FC), np.float32)
    rel = np.zeros((T.N, T.N), np.float32)
    mask = np.zeros(T.N, np.float32)
    acts = obs.actions
    for i, a in enumerate(acts):
        mask[i] = 1
        if a[0] == "exp":
            j = a[1]
            p1h = exp_p1h(spec, obs.belief, obs.hist, j)
            cand[i, 0] = 1
            cand[i, 4] = cost_now(spec, j, obs.hist, n_alive) / 2
            cand[i, 5] = float(post @ p1h)
            cand[i, 6] = _flag(spec, obs.hist, j)
            cand[i, 7] = _unlocks(spec, obs.hist, j) / 2
            cand[i, 8 + ("control", "screen", "definitive").index(spec.kind[j])] = 1
            cand[i, 11] = spec.confs[j] / 2
            cand[i, 12:12 + spec.H] = p1h
        elif a[0] == "abandon":
            cand[i, 1] = 1
            cand[i, 12 + a[1]] = 1
            cand[i, 18] = post[a[1]]
            cand[i, 19] = CARRY / REWARD_R
        elif a[0] == "conclude":
            cand[i, 2] = 1
            cand[i, 12 + a[1]] = 1
            cand[i, 18] = post[a[1]]
            cand[i, 19] = terminal_value(a, post) / REWARD_R
        else:
            cand[i, 3] = 1
            cand[i, 19] = ABSTAIN_A / REWARD_R
    for i, a in enumerate(acts):
        for k, b in enumerate(acts):
            if a[0] == "exp" and b[0] == "exp" and i != k:
                if spec.prereq[b[1]] == a[1] or spec.reuse[b[1]] == a[1]:
                    rel[i, k] = 1.0
                elif abs(float(spec.W[a[1]] @ spec.W[b[1]])) > 0.3:
                    rel[i, k] = 0.5
            elif a[0] == "exp" and b[0] in ("conclude", "abandon") and spec.S[a[1], b[1]] > 0:
                rel[i, k] = 0.5
    glob = np.zeros(T.G, np.float32)
    glob[0] = entropy_bits(post) / math.log2(6)
    glob[1] = post.max()
    glob[2] = obs.remaining / spec.budget
    glob[3] = n_alive / spec.H
    glob[4] = sum(1 for v in obs.hist if v) / spec.E
    glob[5] = ABSTAIN_A / REWARD_R
    glob[6] = (spec.H - n_alive) / max(spec.max_abandon, 1)
    glob[7] = min(float((post * np.log(np.maximum(post, 1e-12) / np.maximum(spec.prior, 1e-12))).sum()), 2.0) / 2
    return {"cand": torch.from_numpy(cand)[None], "rel": torch.from_numpy(rel)[None],
            "glob": torch.from_numpy(glob)[None], "mask": torch.from_numpy(mask)[None]}


def impl_label(cfg: Dict) -> str:
    return dsl.implementation_label(cfg) + ("+rlt_feedback" if cfg["core"] == "trm" and cfg["feedback"] == "rlt" else "")


class LearnedPolicy(Policy):
    def __init__(self, model: M.SkillModule, name: str):
        self.model, self.name = model, name
        self.last_conf = 0.0
        self.flops = 0

    def act(self, obs: Obs) -> Tuple:
        b = encode(obs)
        with torch.no_grad():
            M.flops_reset()
            st = self.model.init_state(b)
            R = self.model.cfg["visits"]
            for _ in range(R):
                logits, conf, _, st = self.model.visit(b["cand"], b, st, R)
            self.flops += M.flops_read()
        lg = logits[0].masked_fill(b["mask"][0] == 0, -1e9)
        i = int(lg.argmax())
        self.last_conf = float(torch.sigmoid(conf[0]))
        return obs.actions[i]


# ------------------------------------------------------------------ episodes and metrics
def run_episode(spec: PublicSpec, truth: Truth, policy: Policy, plan: Optional[PlanTable] = None,
                continuation: Optional[Policy] = None) -> Dict:
    """Roll one episode.  If ``continuation`` is given, ``policy`` chooses only the first action."""
    env = Env(spec, truth)
    p0 = entropy_bits(spec.prior)
    total, steps, n_exp = 0.0, 0, 0
    cost_total = 0.0
    conf_pairs: List[Tuple[float, int]] = []
    contra = None
    fav = int(np.argmax(spec.prior))
    cpu0 = time.process_time()
    flops0 = getattr(policy, "flops", 0)
    first_action = None
    while True:
        obs = env.obs()
        pol = policy if (continuation is None or steps == 0) else continuation
        a = pol.act(obs)
        if steps == 0:
            first_action = a
        if plan is not None and plan.ok and isinstance(pol, LearnedPolicy):
            qs = plan.lookup(obs)[1]
            conf_pairs.append((pol.last_conf, int(qs[obs.actions.index(a)] >= max(qs) - 1e-6)))
        r, done = env.step(a)
        total += r
        steps += 1
        if a[0] == "exp":
            n_exp += 1
            cost_total += -r
            if contra is None and int(np.argmax(hyp_post(spec, env.belief))) != fav:
                contra = {"key": (env.hist, env.alive, env.spent_c), "reward_before": total}
        if done:
            terminal = a
            break
    post = hyp_post(spec, env.belief)
    exp_term = terminal_value(terminal, post)
    rec = {
        "utility": total, "expected_utility": total - (r) + exp_term, "cost": cost_total, "n_exp": n_exp,
        "bits": p0 - entropy_bits(post), "outcome": env.outcome, "correct": env.outcome == "correct",
        "false": env.outcome == "false", "abstain": env.outcome == "abstain",
        "abandoned_truth": env.abandoned_truth, "n_abandon": spec.H - bin(env.alive).count("1"),
        "cpu_s": time.process_time() - cpu0, "flops": getattr(policy, "flops", 0) - flops0,
        "first_action": action_name(first_action), "conf_pairs": conf_pairs,
        "post_conf": (float(post[terminal[1]]) if terminal[0] == "conclude" else None),
        "contra": None,
    }
    if contra is not None:
        rtg = total - contra["reward_before"]
        rec["contra"] = {"rtg": rtg}
        if plan is not None and plan.ok:
            rec["contra"]["v_star"] = plan.table[contra["key"]][0]
    return rec


def ece(pairs: Sequence[Tuple[float, float]], bins: int = 10) -> Optional[float]:
    if not pairs:
        return None
    arr = np.array(pairs, np.float64)
    idx = np.minimum((arr[:, 0] * bins).astype(int), bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(arr[m, 0].mean() - arr[m, 1].mean())
    return float(out)


def summarize(recs: List[Dict], planner_recs: Optional[List[Dict]], vstar: Optional[List[float]]) -> Dict:
    n = len(recs)
    mean = lambda k: float(np.mean([r[k] for r in recs]))  # noqa: E731
    out = {
        "n_worlds": n, "utility": mean("utility"), "expected_utility": mean("expected_utility"),
        "evidence_experiments": mean("n_exp"), "evidence_bits": mean("bits"), "research_cost": mean("cost"),
        "false_conclusion_rate": mean("false"), "completed_objective_rate": mean("correct"),
        "abstain_rate": mean("abstain"), "abandon_rate": float(np.mean([r["n_abandon"] > 0 for r in recs])),
        "abandoned_truth_rate": mean("abandoned_truth"),
        "cpu_s_per_episode": mean("cpu_s"), "flops_per_episode": mean("flops"),
        "ece_posterior": ece([(r["post_conf"], float(r["correct"])) for r in recs if r["post_conf"] is not None]),
        "ece_conf": ece([p for r in recs for p in r["conf_pairs"]]),
    }
    if planner_recs is not None:
        out["regret_realized_vs_planner"] = float(np.mean([p["utility"] - r["utility"] for p, r in zip(planner_recs, recs)]))
    if vstar is not None:
        out["regret_expected_vs_vstar"] = float(np.mean([v - r["expected_utility"] for v, r in zip(vstar, recs)]))
    c = [r["contra"] for r in recs if r["contra"] is not None]
    out["contradiction_rate"] = len(c) / n
    if c:
        out["post_contradiction_return"] = float(np.mean([x["rtg"] for x in c]))
        vs = [x["v_star"] - x["rtg"] for x in c if "v_star" in x]
        if vs:
            out["post_contradiction_regret"] = float(np.mean(vs))
    return out


# ------------------------------------------------------------------ datasets
def group_roots(group: str, n: int, base: int = TEST0) -> List[int]:
    return [base + i for i in range(n)]


def build_examples(specs_truths: List[Tuple[PublicSpec, Truth]], states_per_world: int, seed: int, max_states: int):
    """Plan each world exactly; sample states from epsilon-greedy planner rollouts and uniformly
    from the reachable set.  Labels are exact Q-values (utility units)."""
    cands, rels, globs, masks, qs_all, skipped = [], [], [], [], [], 0
    for spec, truth in specs_truths:
        pl = Planner(spec, max_states, keep_beliefs=True)
        if not pl.solve():
            skipped += 1
            continue
        rng = np.random.default_rng([seed, spec.root])
        keys = []
        env = Env(spec, truth)
        for _ in range(states_per_world // 2):
            env = Env(spec, Truth(truth.h, truth.ncode, rng.random(spec.E)))
            while not env.done:
                o = env.obs()
                keys.append(o.key)
                qs = pl.memo[o.key][1]
                i = int(rng.integers(len(qs))) if rng.random() < 0.3 else int(np.argmax(qs))
                env.step(o.actions[i])
        all_keys = list(pl.memo)
        keys = keys[: states_per_world // 2 * 3]
        pick = rng.choice(len(all_keys), size=min(states_per_world // 2, len(all_keys)), replace=False)
        keys += [all_keys[k] for k in pick]
        for k in dict.fromkeys(keys):
            if k not in pl.beliefs:
                continue
            o = Obs(spec, k[0], k[1], k[2], pl.beliefs[k], tuple(legal_actions(spec, k[0], k[1], k[2])))
            e = encode(o)
            q = torch.zeros(T.N)
            q[: len(pl.memo[k][1])] = torch.tensor(pl.memo[k][1], dtype=torch.float32)
            cands.append(e["cand"][0]); rels.append(e["rel"][0]); globs.append(e["glob"][0]); masks.append(e["mask"][0])
            qs_all.append(q)
    data = {"cand": torch.stack(cands), "rel": torch.stack(rels), "glob": torch.stack(globs),
            "mask": torch.stack(masks), "q": torch.stack(qs_all)} if cands else None
    return data, skipped


def train_worlds(n: int, base: int) -> List[Tuple[PublicSpec, Truth]]:
    return [make_world(TRAIN_FAMILIES[i % len(TRAIN_FAMILIES)], base + i) for i in range(n)]


# ------------------------------------------------------------------ training
def model_configs(hidden: int = 32) -> Dict[str, Dict]:
    base = dict(hidden=hidden, visits=3, latent_steps=2, cycles=1, grad_cycles=1, verifier_visits=0, supervision="deep")
    trm = dsl.validate(dict(base, core="trm", feedback="direct"))
    rlt = dsl.validate(dict(base, core="trm", feedback="rlt"))
    target = M.SkillModule(trm).n_params()
    gru = M.match_hidden(dsl.validate(dict(base, core="gru")), target)
    ff = M.match_hidden(dsl.validate(dict(base, core="ff", visits=1)), target)
    return {"ff": ff, "gru": gru, "trm": trm, "rlt": rlt}


def _loss(model: M.SkillModule, b: Dict[str, torch.Tensor]) -> torch.Tensor:
    R = model.cfg["visits"]
    mask, q = b["mask"], b["q"] / REWARD_R
    qmask = q.masked_fill(mask == 0, -1e9)
    best = qmask.argmax(-1)
    st = model.init_state(b)
    loss = 0.0
    for _ in range(R):
        logits, conf, _, st = model.visit(b["cand"], b, st, R)
        reg = (((logits - q) ** 2) * mask).sum() / mask.sum()
        ce = F.cross_entropy((logits * 4).masked_fill(mask == 0, -1e9), best)
        pick = logits.detach().masked_fill(mask == 0, -1e9).argmax(-1)
        gap = qmask.max(-1).values - qmask.gather(1, pick[:, None]).squeeze(1)
        cb = F.binary_cross_entropy_with_logits(conf, (gap < 1e-6).float())
        loss = loss + (reg * 5 + ce + 0.2 * cb) / R
    return loss


def train_model(cfg: Dict, data: Dict, cal: Optional[Dict], steps: int, seed: int, deadline: float, batch: int = 64):
    torch.manual_seed(seed)
    model = M.SkillModule(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    g = torch.Generator().manual_seed(seed)
    n = data["cand"].shape[0]
    cpu0 = time.process_time()
    best_state, best_cal, done_steps = None, float("inf"), 0
    eval_every = max(steps // 5, 1)
    for s in range(steps):
        if time.time() > deadline:
            break
        idx = torch.randint(0, n, (min(batch, n),), generator=g)
        b = {k: v[idx] for k, v in data.items()}
        loss = _loss(model, b)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        done_steps = s + 1
        if cal is not None and (done_steps % eval_every == 0 or done_steps == steps):
            with torch.no_grad():
                cl = float(_loss(model, cal))
            if cl < best_cal:
                best_cal = cl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"train_cpu_seconds": time.process_time() - cpu0, "steps_completed": done_steps,
                   "steps_requested": steps, "calibration_loss": None if best_state is None else best_cal,
                   "final_train_loss": float(loss.detach()) if done_steps else None}


# ------------------------------------------------------------------ evaluation
def evaluate_policy(make_policy, worlds: List[Tuple[PublicSpec, Truth]], plans: List[PlanTable], planner_recs):
    """Returns (rollout summary, first-action-ranking summary)."""
    recs, matched, q_reg, top1 = [], [], [], []
    for (spec, truth), plan in zip(worlds, plans):
        pol = make_policy()
        recs.append(run_episode(spec, truth, pol, plan))
        if plan.ok:
            pol2 = make_policy()
            matched.append(run_episode(spec, truth, pol2, plan, continuation=ExactPolicy(plan)))
            obs = Env(spec, truth).obs()
            qs = plan.lookup(obs)[1]
            names = [action_name(a) for a in obs.actions]
            q_reg.append(max(qs) - qs[names.index(matched[-1]["first_action"])])
            top1.append(float(q_reg[-1] < 1e-6))
    ok = [p.ok for p in plans]
    pr = [p for p, o in zip(planner_recs, ok) if o] if planner_recs else None
    vs = [p.v_star() for p in plans if p.ok]
    roll = summarize(recs, None, None)
    if any(ok):
        roll_ok = summarize([r for r, o in zip(recs, ok) if o], pr, vs)
        roll["tractable_subset"] = roll_ok
    first = None
    if matched:
        first = summarize(matched, pr, vs)
        first["first_action_q_regret"] = float(np.mean(q_reg))
        first["first_action_top1_optimal"] = float(np.mean(top1))
    return roll, first, recs


def exact_records(worlds, plans):
    return [run_episode(s, t, ExactPolicy(p), p) if p.ok else None for (s, t), p in zip(worlds, plans)]


# ------------------------------------------------------------------ jobs / CLI
def _sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), newline="\n")
    os.replace(tmp, path)


def test_worlds(n: int) -> Dict[str, List[Tuple[PublicSpec, Truth]]]:
    return {g: [make_world(fam, TEST0 + i) for i in range(n)] for g, fam in GROUPS.items()}


def load_plans(out: Path, worlds, max_states: int):
    f = out / "plans.pkl"
    if f.exists():
        cache = pickle.loads(f.read_bytes())
    else:
        cache = {}
        for g, ws in worlds.items():
            cache[g] = [(lambda p: (p.table, p.status))(plan_world(s, max_states)) for s, _ in ws]
        f.write_bytes(pickle.dumps(cache))
    return {g: [PlanTable(t, st, ((0,) * ws[0][0].E, (1 << ws[0][0].H) - 1, 0)) for t, st in cache[g]]
            for g, ws in worlds.items()}


def run(args) -> Dict:
    torch.set_num_threads(1)
    t_start = time.time()
    hours = min(float(args.hours), 8.0)
    deadline = t_start + hours * 3600
    out = Path(args.out)
    (out / "jobs").mkdir(parents=True, exist_ok=True)
    src = Path(__file__)
    worlds = test_worlds(args.test_worlds)
    plans = load_plans(out, worlds, args.max_states)
    planner_status = {g: {"tractable": sum(p.ok for p in ps), "total": len(ps), "status": sorted({p.status for p in ps})} for g, ps in plans.items()}
    prec = {g: exact_records(worlds[g], plans[g]) for g in worlds}
    cfgs = model_configs(args.hidden)
    jobs: Dict[str, Dict] = {}
    incomplete: List[str] = []

    def job_file(name):
        return out / "jobs" / f"{name}.json"

    def timed_out(name):
        if time.time() > deadline:
            incomplete.append(name)
            return True
        return False

    # baselines and exact planner (paired on the same worlds)
    if not job_file("baselines").exists() and not timed_out("baselines"):
        res = {}
        cpu0 = time.process_time()
        for pname, mk in (("exact_planner", None), ("greedy_eig_per_cost", GreedyEIG), ("variance_heuristic", VarianceHeuristic),
                          ("reuse_first", ReuseFirst), ("random_valid", _random_factory())):
            res[pname] = {}
            for g in worlds:
                if mk is None:
                    recs = [r for r in prec[g] if r is not None]
                    res[pname][g] = {"rollout": summarize(recs, None, [p.v_star() for p in plans[g] if p.ok]) if recs else None, "first_action": None}
                    continue
                roll, first, _ = evaluate_policy(lambda: mk(), worlds[g], plans[g], prec[g])
                res[pname][g] = {"rollout": roll, "first_action": first}
        _write_json(job_file("baselines"), {"complete": True, "policies": res, "cpu_seconds": time.process_time() - cpu0})

    # learned policies
    train_data = cal_data = None
    data_f = out / "train_data.pt"
    for seed in args.seeds:
        for kind in ("ff", "gru", "trm", "rlt"):
            name = f"{kind}_s{seed}"
            if job_file(name).exists():
                continue
            if timed_out(name):
                continue
            if train_data is None:
                if data_f.exists():
                    blob = torch.load(data_f)
                    train_data, cal_data, skipped = blob["train"], blob["cal"], blob["skipped"]
                else:
                    train_data, sk1 = build_examples(train_worlds(args.train_worlds, TRAIN0), args.states_per_world, 1, args.max_states)
                    cal_data, sk2 = build_examples(train_worlds(args.cal_worlds, CAL0), args.states_per_world, 2, args.max_states)
                    skipped = sk1 + sk2
                    torch.save({"train": train_data, "cal": cal_data, "skipped": skipped}, data_f)
            model, info = train_model(cfgs[kind], train_data, cal_data, args.steps, seed, deadline)
            if info["steps_completed"] < args.steps:
                incomplete.append(name)
                continue
            res = {}
            cpu0 = time.process_time()
            for g in worlds:
                roll, first, _ = evaluate_policy(lambda: LearnedPolicy(model, kind), worlds[g], plans[g], prec[g])
                res[g] = {"rollout": roll, "first_action": first}
            info.update({"complete": True, "kind": kind, "seed": seed, "implementation": impl_label(cfgs[kind]),
                         "n_params": model.n_params(), "eval_cpu_seconds": time.process_time() - cpu0,
                         "n_train_examples": int(train_data["cand"].shape[0]), "groups": res})
            _write_json(job_file(name), info)

    # aggregate
    for f in sorted((out / "jobs").glob("*.json")):
        jobs[f.stem] = json.loads(f.read_text())
    agg = aggregate(jobs, list(worlds))
    results = {
        "complete_jobs": sorted(k for k, v in jobs.items() if v.get("complete")),
        "incomplete_jobs": sorted(set(incomplete)),
        "note_incomplete": "incomplete jobs are excluded from every comparison",
        "args": {k: (v if not isinstance(v, Path) else str(v)) for k, v in vars(args).items()},
        "root_ranges": ROOT_RANGES, "families": FAMILY_CFG, "groups": GROUPS, "train_families": list(TRAIN_FAMILIES),
        "reward": {"correct": REWARD_R, "false": -PENALTY_P, "abstain": ABSTAIN_A, "carry_per_alive_hypothesis": CARRY},
        "model_configs": {k: {"cfg": v, "n_params": M.SkillModule(v).n_params(), "implementation": impl_label(v)} for k, v in cfgs.items()},
        "planner_status": planner_status,
        "source_sha256_research_env_py": _sha256_lf(src),
        "train_cpu_seconds_per_job": {k: v.get("train_cpu_seconds") for k, v in jobs.items() if "train_cpu_seconds" in v},
        "steps_completed_per_job": {k: v.get("steps_completed") for k, v in jobs.items() if "steps_completed" in v},
        "wall_seconds": time.time() - t_start, "aggregate": agg,
    }
    _write_json(out / "results.json", results)
    return results


def aggregate(jobs: Dict[str, Dict], groups: List[str]) -> Dict:
    agg: Dict = {"baselines": jobs.get("baselines", {}).get("policies"), "learned": {}, "surprise_revision": None}
    for kind in ("ff", "gru", "trm", "rlt"):
        js = [v for k, v in jobs.items() if k.startswith(kind + "_s") and v.get("complete")]
        if not js:
            continue
        agg["learned"][kind] = {"n_seeds": len(js), "n_params": js[0]["n_params"], "implementation": js[0]["implementation"]}
        for g in groups:
            row = {}
            for part in ("rollout", "first_action"):
                keys = [k for k, v in js[0]["groups"][g][part].items() if isinstance(v, (int, float)) and v is not None] if js[0]["groups"][g][part] else []
                row[part] = {k: {"mean": float(np.mean([j["groups"][g][part][k] for j in js if j["groups"][g][part].get(k) is not None])),
                                 "sd": float(np.std([j["groups"][g][part][k] for j in js if j["groups"][g][part].get(k) is not None]))}
                             for k in keys}
            agg["learned"][kind][g] = row
    if "trm" in agg["learned"] and "rlt" in agg["learned"] and "surprise_h4" in groups:
        seeds = sorted({int(k.split("_s")[1]) for k in jobs if k.startswith("trm_s") and jobs[k].get("complete") and f"rlt_s{k.split('_s')[1]}" in jobs and jobs[f"rlt_s{k.split('_s')[1]}"].get("complete")})
        diffs = {}
        for key in ("post_contradiction_return", "post_contradiction_regret", "utility"):
            d = []
            for s in seeds:
                a = jobs[f"trm_s{s}"]["groups"]["surprise_h4"]["rollout"].get(key)
                b = jobs[f"rlt_s{s}"]["groups"]["surprise_h4"]["rollout"].get(key)
                if a is not None and b is not None:
                    d.append(b - a)
            if d:
                diffs[key] = {"rlt_minus_trm_mean": float(np.mean(d)), "se": float(np.std(d, ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None, "n_seeds": len(d)}
        agg["surprise_revision"] = {"paired_over_seeds": diffs,
                                    "rlt_revises_better": bool(diffs.get("post_contradiction_return", {}).get("rlt_minus_trm_mean", 0) > 0)}
    return agg


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-worlds", type=int, default=200)
    ap.add_argument("--test-worlds", type=int, default=40)
    ap.add_argument("--cal-worlds", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12, 13])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--states-per-world", type=int, default=40)
    ap.add_argument("--max-states", type=int, default=60_000)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.smoke:
        args.train_worlds, args.test_worlds, args.cal_worlds = 4, 2, 2
        args.steps, args.states_per_world, args.seeds = 20, 8, args.seeds[:1]
    if args.cal_worlds <= 0:
        args.cal_worlds = max(4, args.train_worlds // 8)
    res = run(args)
    print(json.dumps({"complete_jobs": res["complete_jobs"], "incomplete_jobs": res["incomplete_jobs"],
                      "planner_status": res["planner_status"], "wall_seconds": round(res["wall_seconds"], 1)}, indent=1))


if __name__ == "__main__":
    main()
