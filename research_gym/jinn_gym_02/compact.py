"""Population compactification of LDT heads.

Each head is flattened to its 3 x 35 weights plus 3 biases (108 numbers). Over the whole
population (every archived variant of every persona) we measure the singular spectrum,
effective rank (exponential of the spectral entropy), participation ratio, principal angles
between persona subspaces, and whether rank-r reconstructions of the frozen personas keep their
behavior (executed-action agreement with the full head) and their competence (fidelity and
utility) on held-out tasks.
"""
from __future__ import annotations

import numpy as np

from research_gym.jinn_gym_01._vendor.ldt_experiment import AdditiveCandidateHead


def flat(head) -> np.ndarray:
    b = head.base if hasattr(head, "base") else head
    return np.r_[np.array(b.weights).ravel(), np.array(b.bias)]


def unflat(vec: np.ndarray, like) -> dict:
    b = like.base if hasattr(like, "base") else like
    n = len(b.feature_names)
    v = b.to_jsonable()
    v["weights"] = vec[: 3 * n].reshape(3, n).tolist()
    v["bias"] = vec[3 * n:].tolist()
    return v


def spectrum(M: np.ndarray) -> dict:
    mu = M.mean(0)
    s = np.linalg.svd(M - mu, compute_uv=False)
    e = s ** 2 / (s ** 2).sum()
    ent = float(-(e[e > 0] * np.log(e[e > 0])).sum())
    return {"n": int(M.shape[0]), "dim": int(M.shape[1]), "singular_values": s[:20].tolist(),
            "explained": np.cumsum(e)[:20].tolist(), "effective_rank": float(np.exp(ent)),
            "participation_ratio": float((s ** 2).sum() ** 2 / (s ** 4).sum())}


def basis(M: np.ndarray, r: int):
    mu = M.mean(0)
    _, _, Vt = np.linalg.svd(M - mu, full_matrices=False)
    return mu, Vt[:r]


def project(vec: np.ndarray, mu: np.ndarray, V: np.ndarray) -> np.ndarray:
    return mu + (vec - mu) @ V.T @ V


def principal_angles(A: np.ndarray, B: np.ndarray, r: int = 3) -> list[float]:
    def sub(M):
        _, _, Vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
        return Vt[: min(r, len(M) - 1)].T
    Qa, Qb = sub(A), sub(B)
    if Qa.shape[1] == 0 or Qb.shape[1] == 0:
        return []
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1))).tolist()
