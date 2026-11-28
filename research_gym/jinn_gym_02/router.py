"""Compact TRM router over persona modules, routing baselines, and persona self-models.

Router input: observable research-state features only (domain, phase, public evidence flags,
the host-audited tag histogram of the menu, the size of the host-permitted set, and summary
statistics of neutral Inky's prior). No verifier outcome, hidden state or utility is a feature.

TRM: a tiny recursive model. An input embedding e and a latent z (z_0 = 0) are refined K times,
z_k = z_{k-1} + MLP([e, z_{k-1}]); after every step a linear readout predicts the verified
utility of each routing option, with deep supervision over all K readouts. Routing picks the
argmax of the final readout. CPU-only torch.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from research_gym.jinn_gym_01 import host
from research_gym.jinn_gym_01.module import host_flags
from research_gym.jinn_gym_01.tags import ALL_TAGS, ABSTAIN

from .domains import DOMAINS

PHASE_SET = ("orient", "test", "diagnose")


def features(task, prior: dict, parsed_ok: bool) -> np.ndarray:
    f = [float(task.domain == d) for d in DOMAINS]
    f += [float(task.phase == p) for p in PHASE_SET]
    fl = host_flags(task)
    f += [float(fl["inputs_missing"]), float(fl["failure_reported"]), fl["evidence_items"] / 4.0]
    n = len(task.actions)
    ok = host.permitted(task)
    f += [n / 10.0, len(ok) / n]
    hist = {t: 0.0 for t in ALL_TAGS}
    for a in task.actions:
        if a.action_id in ok and a.action_id != ABSTAIN:
            for t in a.tags:
                hist[t] += 1.0 / n
    f += [hist[t] for t in ALL_TAGS]
    p = np.array(list(prior.values()))
    f += [float(-(p * np.log(p)).sum() / math.log(n)), float(p.max()), float(parsed_ok)]
    return np.array(f, dtype=np.float32)


class TRMRouter:
    def __init__(self, n_in: int, n_out: int, hidden: int = 64, steps: int = 4, seed: int = 0):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self.torch, self.steps = torch, steps
        self.embed = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh())
        self.step = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.read = nn.Linear(hidden, n_out)
        self.params = list(self.embed.parameters()) + list(self.step.parameters()) + list(self.read.parameters())
        self.mu = self.sd = None

    def forward(self, X, steps=None):
        torch = self.torch
        e = self.embed(X)
        z = torch.zeros_like(e)
        outs = []
        for _ in range(steps or self.steps):
            z = z + self.step(torch.cat([e, z], dim=1))
            outs.append(self.read(z))
        return outs

    def fit(self, X: np.ndarray, Y: np.ndarray, Xv: np.ndarray, Yv: np.ndarray, epochs: int = 400,
            lr: float = 3e-3, wd: float = 1e-4) -> dict:
        torch = self.torch
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        Yt = torch.tensor(Y, dtype=torch.float32)
        Xvt = torch.tensor((Xv - self.mu) / self.sd, dtype=torch.float32)
        opt = torch.optim.Adam(self.params, lr=lr, weight_decay=wd)
        best, best_state, hist = -1e9, None, []
        for ep in range(epochs):
            opt.zero_grad()
            outs = self.forward(Xt)
            loss = sum(((o - Yt) ** 2).mean() for o in outs) / len(outs)
            loss.backward()
            opt.step()
            if ep % 10 == 9:
                with torch.no_grad():
                    pick = self.forward(Xvt)[-1].argmax(1).numpy()
                val = float(Yv[np.arange(len(Yv)), pick].mean())
                hist.append((ep + 1, float(loss.detach()), val))
                if val > best:
                    best = val
                    best_state = [p.detach().clone() for p in self.params]
        if best_state is not None:
            with torch.no_grad():
                for p, b in zip(self.params, best_state):
                    p.copy_(b)
        return {"best_val_routed_utility": best, "history": hist}

    def predict(self, X: np.ndarray, steps=None) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            return self.forward(torch.tensor((X - self.mu) / self.sd, dtype=torch.float32), steps)[-1].numpy()


def ridge_router(X, Y, lam: float = 1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ Y)
    return lambda Xn: np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ W


# --------------------------------------------------------------------------- self-model
class Logistic:
    def __init__(self, l2: float = 1.0):
        self.l2, self.w = l2, None

    def fit(self, X, y, iters: int = 500):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        w = np.zeros(Z.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
            w -= 0.3 * (Z.T @ (p - y) / len(y) + self.l2 * np.r_[0, w[1:]] / len(y))
        self.w = w
        return self

    def predict(self, X):
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        return 1 / (1 + np.exp(-np.clip(Z @ self.w, -30, 30)))


SELF_METRICS = {
    "success": "P(the executed action is optimal among host-permitted actions, within 1e-9 of the best verified utility)",
    "failure": "P(normalized verified utility of the executed action < 0.5)",
    "abstain": "P(the persona abstains)",
    "compute": "expected cost units of the executed action (low = 1, medium = 2, high = 3; abstain counts as its own cost, 1)",
}


def self_targets(rows: Sequence[dict]) -> dict:
    return {"success": np.array([float(r["optimal"]) for r in rows]),
            "failure": np.array([float(r["normalized_utility"] < 0.5) for r in rows]),
            "abstain": np.array([float(r["abstained"]) for r in rows]),
            "compute": np.array([r.get("cost_units", 1.0) for r in rows])}


class SelfModel:
    def fit(self, X, rows):
        T = self_targets(rows)
        self.base = {k: float(v.mean()) for k, v in T.items()}
        self.clf = {k: Logistic().fit(X, T[k]) for k in ("success", "failure", "abstain")}
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Z = np.c_[np.ones(len(X)), (X - mu) / sd]
        W = np.linalg.solve(Z.T @ Z + np.eye(Z.shape[1]), Z.T @ T["compute"])
        self.reg = (mu, sd, W)
        return self

    def predict(self, X) -> dict:
        mu, sd, W = self.reg
        out = {k: m.predict(X) for k, m in self.clf.items()}
        out["compute"] = np.c_[np.ones(len(X)), (X - mu) / sd] @ W
        return out

    def score(self, X, rows) -> dict:
        T, P = self_targets(rows), self.predict(X)
        res = {}
        for k in ("success", "failure", "abstain"):
            b = float(((P[k] - T[k]) ** 2).mean())
            b0 = float(((self.base[k] - T[k]) ** 2).mean())
            bins = np.minimum(4, (P[k] * 5).astype(int))
            ece = float(sum((bins == j).mean() * abs(P[k][bins == j].mean() - T[k][bins == j].mean())
                            for j in range(5) if (bins == j).any()))
            res[k] = {"brier": b, "brier_base_rate": b0, "skill": 1 - b / b0 if b0 > 0 else None, "ece": ece,
                      "mean_pred": float(P[k].mean()), "observed": float(T[k].mean())}
        mae = float(np.abs(P["compute"] - T["compute"]).mean())
        mae0 = float(np.abs(self.base["compute"] - T["compute"]).mean())
        res["compute"] = {"mae": mae, "mae_base_rate": mae0, "skill": 1 - mae / mae0 if mae0 > 0 else None,
                          "mean_pred": float(P["compute"].mean()), "observed": float(T["compute"].mean())}
        return res
