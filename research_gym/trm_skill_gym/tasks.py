"""Deterministic task generators and host-owned verifiers for the six TRM skill families.

Every skill is posed through one shared candidate-set interface so that modules,
baselines and transfer experiments see identically shaped tensors:

* ``cand``  [N, FC]  typed candidate observations (cols 0..FS-1 skill features,
                    cols FS..FC-1 the verifier feedback channel, written by the host)
* ``rel``   [N, N]   one relation channel between candidates
* ``glob``  [G]      global skill observation
* ``mask``  [N]      valid candidates
* ``label`` [N]      0/1 targets; ``single`` skills have exactly one positive
* ``aux``   [N, A], ``auxg`` [A]  hidden data visible only to the verifier / LDT certifier

Problem roots are integer seeds; an instance is a pure function of
``(skill, family, root)``.  Development roots and frozen-evaluation roots come
from disjoint integer ranges (see ``ROOT_RANGES``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

N = 16          # candidate slots
FS = 20         # skill feature columns
FB = 4          # feedback channel columns: verified, ok, diag0, diag1
FC = FS + FB
G = 8           # global observation width
A = 8           # hidden aux width

SKILLS = ("repair", "retrieval", "deduction", "research", "curation", "stopping")
LABEL_KIND = {
    "repair": "multi",
    "retrieval": "multi",
    "deduction": "multi",
    "research": "single",
    "curation": "multi",
    "stopping": "single",
}
# two development families per skill, one family withheld until the frozen evaluation
FAMILIES = {
    "repair": {"dev": ["tree_region", "cycle_region"], "withheld": ["dense_large_region"]},
    "retrieval": {"dev": ["layered_shallow", "layered_wide"], "withheld": ["layered_deep"]},
    "deduction": {"dev": ["kernel_h4", "kernel_h5"], "withheld": ["kernel_h7"]},
    "research": {"dev": ["uniform_prior", "peaked_prior"], "withheld": ["sharp_costly"]},
    "curation": {"dev": ["clusters3", "clusters4_dups"], "withheld": ["noisy_overlap"]},
    "stopping": {"dev": ["horizon2", "horizon4"], "withheld": ["horizon6"]},
}
ROOT_RANGES = {
    "train": (0, 400_000),
    "dev_eval": (400_000, 500_000),       # held-out-within-development cases
    "frozen_eval": (900_000, 1_000_000),  # fresh roots after architectures are frozen
}


@dataclass
class Instance:
    cand: np.ndarray
    rel: np.ndarray
    glob: np.ndarray
    mask: np.ndarray
    label: np.ndarray
    aux: np.ndarray
    auxg: np.ndarray


def _empty() -> Instance:
    return Instance(
        cand=np.zeros((N, FC), np.float32),
        rel=np.zeros((N, N), np.float32),
        glob=np.zeros(G, np.float32),
        mask=np.zeros(N, np.float32),
        label=np.zeros(N, np.float32),
        aux=np.zeros((N, A), np.float32),
        auxg=np.zeros(A, np.float32),
    )


def _rng(skill: str, family: str, root: int) -> np.random.Generator:
    key = sum((i + 1) * ord(c) for i, c in enumerate(f"{skill}/{family}"))
    return np.random.default_rng([int(root), key])


# ---------------------------------------------------------------- repair
def _gen_repair(rng: np.random.Generator, family: str) -> Instance:
    """Bipartite 2-colouring repair.  A rejected candidate colouring has a connected
    region flipped; the host verifier reports per-node conflict counts.  The target is
    the minimum flip set (the region or its complement, whichever is smaller)."""
    if family == "tree_region":
        n, extra, kmax = int(rng.integers(10, 15)), 0, 4
    elif family == "cycle_region":
        n, extra, kmax = int(rng.integers(10, 15)), 3, 4
    else:  # dense_large_region
        n, extra, kmax = 16, 6, 7
    col = np.zeros(n, np.int64)
    adj = np.zeros((n, n), np.float32)
    for v in range(1, n):
        u = int(rng.integers(0, v))
        adj[u, v] = adj[v, u] = 1
        col[v] = 1 - col[u]
    tries = 0
    while extra > 0 and tries < 200:
        tries += 1
        u, v = rng.integers(0, n, size=2)
        if u != v and col[u] != col[v] and adj[u, v] == 0:
            adj[u, v] = adj[v, u] = 1
            extra -= 1
    k = int(rng.integers(1, kmax + 1))
    start = int(rng.integers(0, n))
    region = [start]
    frontier = [start]
    while len(region) < k and frontier:
        u = frontier.pop(int(rng.integers(0, len(frontier))))
        nbrs = [int(w) for w in np.nonzero(adj[u])[0] if int(w) not in region]
        rng.shuffle(nbrs)
        for w in nbrs:
            if len(region) < k:
                region.append(w)
                frontier.append(w)
    flip = np.zeros(n, np.float32)
    flip[region] = 1
    if flip.sum() > n / 2:
        flip = 1 - flip
    cand_col = (col + flip.astype(np.int64)) % 2
    same = (cand_col[:, None] == cand_col[None, :]).astype(np.float32) * adj
    conf = same.sum(1)
    deg = adj.sum(1)
    inst = _empty()
    inst.mask[:n] = 1
    inst.rel[:n, :n] = adj
    inst.label[:n] = flip
    inst.cand[:n, 0] = cand_col
    inst.cand[:n, 1] = conf / np.maximum(deg, 1)
    inst.cand[:n, 2] = deg / 6.0
    inst.cand[:n, 3] = (conf > 0)
    inst.cand[:n, 4:12] = rng.normal(0, 1, (n, 8))
    inst.glob[0] = n / N
    inst.glob[1] = conf.sum() / max(adj.sum(), 1)
    inst.aux[:n, 0] = cand_col
    inst.aux[:n, 1] = deg
    return inst


# ---------------------------------------------------------------- retrieval
def _gen_retrieval(rng: np.random.Generator, family: str) -> Instance:
    """Dependency retrieval: find every document/symbol the query transitively depends
    on.  rel[i, j] = 1 when i directly depends on j."""
    if family == "layered_shallow":
        layers, skip = 3, 0.0
    elif family == "layered_wide":
        layers, skip = 4, 0.15
    else:  # layered_deep
        layers, skip = 6, 0.1
    n = int(rng.integers(12, N + 1))
    layer = np.sort(rng.integers(0, layers, size=n))
    layer[-1] = layers - 1
    for l in range(layers):  # every layer populated
        if not (layer == l).any():
            layer[int(rng.integers(0, n - 1))] = l
    layer = np.sort(layer)
    q = n - 1
    layer[q] = layers - 1
    rel = np.zeros((n, n), np.float32)
    for i in range(n):
        lower = np.nonzero(layer == layer[i] - 1)[0]
        if len(lower):
            for j in rng.choice(lower, size=min(len(lower), int(rng.integers(1, 3))), replace=False):
                rel[i, j] = 1
        deeper = np.nonzero(layer < layer[i] - 1)[0]
        for j in deeper:
            if rng.random() < skip:
                rel[i, j] = 1
    # query keeps a single branch so that distractor siblings exist
    reach = np.zeros(n, bool)
    stack = [q]
    while stack:
        u = stack.pop()
        for v in np.nonzero(rel[u])[0]:
            if not reach[v]:
                reach[v] = True
                stack.append(int(v))
    inst = _empty()
    inst.mask[:n] = 1
    inst.rel[:n, :n] = rel
    inst.label[:n] = reach.astype(np.float32)
    inst.cand[q, 0] = 1
    lex = np.where(reach, rng.normal(0.6, 0.25, n), rng.normal(0.4, 0.25, n))
    inst.cand[:n, 1] = np.clip(lex, 0, 1)
    inst.cand[:n, 2:10] = rng.normal(0, 1, (n, 8))
    inst.cand[:n, 10] = rel.sum(1) / 4
    inst.cand[:n, 11] = rel.sum(0) / 4
    inst.glob[0] = n / N
    inst.aux[:n, 0] = 0
    inst.aux[q, 0] = 1
    return inst


# ---------------------------------------------------------------- deduction
def _gen_deduction(rng: np.random.Generator, family: str) -> Instance:
    """Candidate-lattice reduction as the kernel of a dominance DAG: a candidate
    survives iff no surviving candidate dominates it.  rel[i, j] = 1 when j dominates i."""
    height = {"kernel_h4": 4, "kernel_h5": 5, "kernel_h7": 7}[family]
    n = int(rng.integers(12, N + 1))
    layer = np.sort(rng.integers(0, height, size=n))
    for l in range(height):
        if not (layer == l).any():
            layer[int(rng.integers(0, n))] = l
    layer = np.sort(layer)
    rel = np.zeros((n, n), np.float32)
    for i in range(n):
        if layer[i] == 0:
            continue
        prev = np.nonzero(layer == layer[i] - 1)[0]
        for j in rng.choice(prev, size=min(len(prev), int(rng.integers(1, 3))), replace=False):
            rel[i, j] = 1
        older = np.nonzero(layer < layer[i] - 1)[0]
        for j in older:
            if rng.random() < 0.08:
                rel[i, j] = 1
    alive = np.zeros(n, bool)
    for i in range(n):  # nodes are in topological order (layer sorted)
        alive[i] = not any(alive[j] for j in np.nonzero(rel[i])[0])
    inst = _empty()
    inst.mask[:n] = 1
    inst.rel[:n, :n] = rel
    inst.label[:n] = alive.astype(np.float32)
    inst.cand[:n, 0] = np.clip(np.where(alive, rng.normal(0.55, 0.25, n), rng.normal(0.45, 0.25, n)), 0, 1)
    inst.cand[:n, 1] = rel.sum(1) / 3
    inst.cand[:n, 2] = rel.sum(0) / 3
    inst.cand[:n, 3] = (rel.sum(1) == 0)
    inst.cand[:n, 4:12] = rng.normal(0, 1, (n, 8))
    inst.glob[0] = n / N
    return inst


# ---------------------------------------------------------------- research
def _entropy(p: np.ndarray) -> float:
    p = p[p > 1e-12]
    return float(-(p * np.log2(p)).sum())


def _eig(prior: np.ndarray, lik: np.ndarray) -> float:
    p1 = float((prior * lik).sum())
    out = _entropy(prior)
    for po, l in ((p1, lik), (1 - p1, 1 - lik)):
        if po > 1e-12:
            out -= po * _entropy(prior * l / po)
    return out


def _gen_research(rng: np.random.Generator, family: str) -> Instance:
    """Rank proposed experiments by expected information gain minus a cost penalty."""
    H = 6
    if family == "uniform_prior":
        prior = rng.dirichlet(np.full(H, 5.0))
        lik = rng.uniform(0.05, 0.95, size=(N, H))
        cost = rng.uniform(0, 1, N)
        lam = 0.15
    elif family == "peaked_prior":
        prior = rng.dirichlet(np.full(H, 0.5))
        lik = rng.uniform(0.05, 0.95, size=(N, H))
        cost = rng.uniform(0, 1, N)
        lam = 0.1
    else:  # sharp_costly
        prior = rng.dirichlet(np.full(H, 1.5))
        lik = np.clip(rng.beta(0.3, 0.3, size=(N, H)), 0.02, 0.98)
        cost = rng.exponential(0.5, N)
        lam = 0.25
    n = int(rng.integers(8, 13))
    score = np.array([_eig(prior, lik[i]) - lam * cost[i] for i in range(n)])
    best = int(np.argmax(score))
    inst = _empty()
    inst.mask[:n] = 1
    inst.label[best] = 1
    inst.cand[:n, 0:H] = lik[:n]
    inst.cand[:n, H] = cost[:n]
    inst.cand[:n, H + 1 : 2 * H + 1] = prior[None, :]
    inst.cand[:n, 2 * H + 1] = lam
    inst.glob[:H] = prior
    inst.glob[H] = lam
    inst.aux[:n, 0] = score
    inst.auxg[0] = score.max()
    inst.auxg[1] = score.max() - score.min() + 1e-6
    # LDT certificate inputs: entropy of prior and outcome entropy bounds
    p1 = (lik[:n] * prior[None, :]).sum(1)
    hb = -(p1 * np.log2(np.clip(p1, 1e-12, 1)) + (1 - p1) * np.log2(np.clip(1 - p1, 1e-12, 1)))
    inst.aux[:n, 1] = np.minimum(hb, _entropy(prior)) - lam * cost[:n]  # sound upper bound
    inst.aux[:n, 2] = -lam * cost[:n]                                   # sound lower bound
    return inst


# ---------------------------------------------------------------- curation
def _gen_curation(rng: np.random.Generator, family: str) -> Instance:
    """Flag inconsistent (label disagrees with its cluster majority) or low-value
    (exact duplicate of an earlier example) training examples."""
    if family == "clusters3":
        C, rho, dup_p, spread = 3, 0.15, 0.05, 0.35
    elif family == "clusters4_dups":
        C, rho, dup_p, spread = 4, 0.12, 0.2, 0.35
    else:  # noisy_overlap
        C, rho, dup_p, spread = 2, 0.28, 0.1, 0.6
    n = int(rng.integers(12, N + 1))
    centers = rng.normal(0, 1, (C, 6))
    ylab = rng.integers(0, 2, C)
    if C >= 2 and ylab.min() == ylab.max():
        ylab[0] = 1 - ylab[0]
    cl = rng.integers(0, C, n)
    x = centers[cl] + rng.normal(0, spread, (n, 6))
    y = ylab[cl].copy()
    flips = rng.random(n) < rho
    y[flips] = 1 - y[flips]
    dup = np.zeros(n, bool)
    for i in range(1, n):
        if rng.random() < dup_p:
            j = int(rng.integers(0, i))
            x[i], y[i], cl[i] = x[j], y[j], cl[j]
            dup[i] = True
    incons = np.zeros(n, bool)
    for c in range(C):
        idx = np.nonzero((cl == c) & ~dup)[0]
        if len(idx) == 0:
            continue
        m = y[idx].mean()
        maj = ylab[c] if abs(m - 0.5) < 1e-9 else int(m > 0.5)
        incons[(cl == c)] = y[cl == c] != maj
    flag = incons | dup
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    sim = np.exp(-d2 / (2 * spread**2 * 6))
    np.fill_diagonal(sim, 0)
    knn = np.zeros_like(sim)
    for i in range(n):
        for j in np.argsort(-sim[i])[:4]:
            knn[i, j] = knn[j, i] = 1
    same = (d2 < 1e-12).astype(np.float32)
    np.fill_diagonal(same, 0)
    rel = np.maximum(knn * sim, same).astype(np.float32)
    inst = _empty()
    inst.mask[:n] = 1
    inst.rel[:n, :n] = rel
    inst.label[:n] = flag.astype(np.float32)
    inst.cand[:n, 0:6] = x
    inst.cand[:n, 6] = y
    inst.cand[:n, 7] = np.arange(n) / N
    inst.cand[:n, 8] = (same * np.tril(np.ones((n, n)), -1)).sum(1) > 0  # has identical earlier twin
    inst.glob[0] = n / N
    inst.aux[:n, 0] = y
    inst.aux[:n, 1] = inst.cand[:n, 8]
    return inst


# ---------------------------------------------------------------- stopping
_ACTIONS = ("stop", "reason", "retrieve", "verify")


def _stop_value(c, b, k, retrieved, p) -> Tuple[float, List[float]]:
    """Exact finite-horizon value of each action from state (confidence c, budget b,
    reasoning count k, retrieved flag)."""
    vals = [c * p["R"], -1e9, -1e9, -1e9]
    if b > 0:
        g = p["g_r"] * (p["delta"] ** k)
        vals[1] = -p["k_r"] + _stop_value(c + g * (1 - c), b - 1, k + 1, retrieved, p)[0]
        if not retrieved:
            vals[2] = -p["k_t"] + _stop_value(c + p["g_t"] * (1 - c), b - 1, k, True, p)[0]
        vals[3] = -p["k_v"] + c * p["R"] + (1 - c) * _stop_value(p["c_rej"], b - 1, k, retrieved, p)[0]
    return max(vals), vals


def _gen_stopping(rng: np.random.Generator, family: str) -> Instance:
    """Decide whether further reasoning, retrieval or verification is worth its cost."""
    hmax = {"horizon2": 2, "horizon4": 4, "horizon6": 6}[family]
    b = int(rng.integers(1, hmax + 1)) if hmax < 6 else int(rng.integers(4, 7))
    p = {
        "R": 1.0,
        "g_r": float(rng.uniform(0.05, 0.5)),
        "delta": float(rng.uniform(0.3, 0.95)),
        "g_t": float(rng.uniform(0.05, 0.7)),
        "k_r": float(rng.uniform(0.0, 0.12)),
        "k_t": float(rng.uniform(0.0, 0.2)),
        "k_v": float(rng.uniform(0.0, 0.15)),
        "c_rej": float(rng.uniform(0.05, 0.6)),
    }
    c = float(rng.uniform(0.05, 0.95))
    retrieved = bool(rng.random() < 0.3)
    _, vals = _stop_value(c, b, 0, retrieved, p)
    vals = np.array(vals)
    best = int(np.argmax(vals))
    inst = _empty()
    inst.mask[:4] = 1
    if retrieved:
        inst.mask[2] = 0
    inst.label[best] = 1
    for a in range(4):
        inst.cand[a, a] = 1
    gains = [0.0, p["g_r"], p["g_t"], 1 - p["c_rej"]]
    costs = [0.0, p["k_r"], p["k_t"], p["k_v"]]
    inst.cand[:4, 4] = costs
    inst.cand[:4, 5] = gains
    inst.cand[:4, 6] = c
    inst.cand[:4, 7] = b / 6
    inst.cand[:4, 8] = p["delta"]
    inst.cand[:4, 9] = p["c_rej"]
    inst.cand[:4, 10] = float(retrieved)
    inst.glob[:7] = [c, b / 6, p["g_r"], p["delta"], p["g_t"], p["c_rej"], float(retrieved)]
    finite = np.where(vals > -1e8, vals, np.nan)
    inst.aux[:4, 0] = np.nan_to_num(finite, nan=-1.0)
    inst.auxg[0] = np.nanmax(finite)
    inst.auxg[1] = np.nanmax(finite) - np.nanmin(finite) + 1e-6
    return inst


GENERATORS: Dict[str, Callable[[np.random.Generator, str], Instance]] = {
    "repair": _gen_repair,
    "retrieval": _gen_retrieval,
    "deduction": _gen_deduction,
    "research": _gen_research,
    "curation": _gen_curation,
    "stopping": _gen_stopping,
}


def make_instance(skill: str, family: str, root: int) -> Instance:
    return GENERATORS[skill](_rng(skill, family, root), family)


def make_batch(skill: str, families: List[str], split: str, start: int, count: int) -> Dict[str, torch.Tensor]:
    """Deterministic batch: roots ``lo+start .. lo+start+count-1`` of ``split``,
    families interleaved round-robin."""
    lo, hi = ROOT_RANGES[split]
    assert lo + start + count <= hi
    insts = [make_instance(skill, families[i % len(families)], lo + start + i) for i in range(count)]
    return stack(insts)


def stack(insts: List[Instance]) -> Dict[str, torch.Tensor]:
    return {
        k: torch.from_numpy(np.stack([getattr(i, k) for i in insts]).astype(np.float32))
        for k in ("cand", "rel", "glob", "mask", "label", "aux", "auxg")
    }


def shuffle_labels(batch: Dict[str, torch.Tensor], seed: int) -> Dict[str, torch.Tensor]:
    """Shuffled-supervision ablation: permute labels across instances (masks follow
    the labels' instances only where shapes allow; invalid positions are zeroed)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(batch["label"].shape[0], generator=g)
    out = dict(batch)
    out["label"] = batch["label"][perm] * batch["mask"]
    return out


# ================================================================= verifiers
def _prop_reach(rel: torch.Tensor, seed: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Nodes reachable from ``seed`` along rel edges restricted to ``allowed`` nodes."""
    r = seed.clone()
    for _ in range(N):
        nxt = torch.clamp(r + (torch.einsum("bi,bij->bj", r, rel) > 0).float() * allowed, max=1)
        if torch.equal(nxt, r):
            break
        r = nxt
    return r


def verify(skill: str, batch: Dict[str, torch.Tensor], pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Host-owned verifier.  ``pred`` is a 0/1 [B, N] decision.  Returns
    ``ok`` [B] (bool) and per-candidate structured diagnostics [B, N, 2].
    Diagnostics only use checkable structure; ``ok`` is the receipt verdict."""
    m, rel, aux, lab = batch["mask"], batch["rel"], batch["aux"], batch["label"]
    pred = pred * m
    diag = torch.zeros(pred.shape + (2,))
    if skill == "repair":
        col = (aux[..., 0] + pred) % 2
        same = (col[:, :, None] == col[:, None, :]).float() * rel
        conf = same.sum(-1)
        deg = aux[..., 1].clamp(min=1)
        diag[..., 0] = conf / deg
        diag[..., 1] = pred
        ok = (conf * m).sum(-1) == 0
    elif skill == "retrieval":
        q = aux[..., 0]
        have = torch.clamp(pred + q, max=1)
        frontier = ((torch.einsum("bi,bij->bj", have, rel) > 0).float() * (1 - have)) * m
        reach = _prop_reach(rel, q, pred)
        orphan = pred * (1 - reach)
        diag[..., 0] = frontier
        diag[..., 1] = orphan
        ok = ((pred - lab).abs() * m).sum(-1) == 0
    elif skill == "deduction":
        dom_alive = (torch.einsum("bij,bj->bi", rel, pred) > 0).float()
        diag[..., 0] = pred * dom_alive              # independence violation
        diag[..., 1] = (1 - pred) * (1 - dom_alive) * m  # absorption violation
        ok = ((pred - lab).abs() * m).sum(-1) == 0
    elif skill == "curation":
        y = aux[..., 0]
        keep = (1 - pred) * m
        w = rel * keep[:, None, :]
        vote = torch.einsum("bij,bj->bi", w, y) / w.sum(-1).clamp(min=1e-6)
        diag[..., 0] = pred
        diag[..., 1] = (y - vote).abs() * (w.sum(-1) > 0).float()
        ok = ((pred - lab).abs() * m).sum(-1) == 0
    elif skill in ("research", "stopping"):
        score = aux[..., 0]
        best, spread = batch["auxg"][:, 0], batch["auxg"][:, 1]
        chosen = (score * pred).sum(-1)
        diag[..., 0] = pred
        diag[..., 1] = pred * ((best[:, None] - score) / spread[:, None])  # normalised regret of verified choice
        ok = (pred.sum(-1) == 1) & (chosen >= best - 1e-9)
    else:
        raise KeyError(skill)
    return ok, diag * m[..., None]


def decide(skill: str, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Turn candidate scores into a 0/1 decision."""
    if LABEL_KIND[skill] == "single":
        masked = logits.masked_fill(mask == 0, -1e9)
        return torch.nn.functional.one_hot(masked.argmax(-1), N).float()
    return (logits > 0).float() * mask


def exact(skill: str, batch: Dict[str, torch.Tensor], pred: torch.Tensor) -> torch.Tensor:
    """Exact task success.  Repair: minimum flip set; single-choice skills: verifier
    acceptance (ties count); multi-label skills: every slot correct."""
    if LABEL_KIND[skill] == "single":
        return verify(skill, batch, pred)[0].float()
    return (((pred - batch["label"]).abs() * batch["mask"]).sum(-1) == 0).float()


# ================================================================= LDT certification
def certify(skill: str, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Sound one-step symbolic certificates.  Returns [B, N] with +1 certified-in,
    -1 certified-out, 0 unknown.  Soundness is checked against labels in tests and
    receipts (``certificate_violations`` must be 0)."""
    m, rel, aux = batch["mask"], batch["rel"], batch["aux"]
    cert = torch.zeros_like(m)
    if skill == "retrieval":
        q = aux[..., 0]
        direct = (torch.einsum("bi,bij->bj", q, rel) > 0).float()
        cert = cert + direct
        und = torch.clamp(rel + rel.transpose(1, 2), max=1)
        comp = _prop_reach(und, q, m)
        cert = cert - (1 - comp) * m
        cert = cert - q  # the query is never its own dependency
    elif skill == "deduction":
        src = (rel.sum(-1) == 0).float() * m
        dom_by_src = (torch.einsum("bij,bj->bi", rel, src) > 0).float()
        cert = src - dom_by_src
    elif skill == "research":
        ub, lb = aux[..., 1], aux[..., 2]
        best_lb = lb.masked_fill(m == 0, -1e9).max(-1).values
        cert = -((ub < best_lb[:, None] - 1e-9).float() * m)
    elif skill == "curation":
        cert = aux[..., 1] * m
    elif skill == "stopping":
        cert = torch.zeros_like(m)
    elif skill == "repair":
        cert = torch.zeros_like(m)  # no sound unary one-step rule; reported as zero coverage
    return cert.clamp(-1, 1) * m


def certificate_violations(skill: str, batch: Dict[str, torch.Tensor]) -> int:
    cert = certify(skill, batch)
    lab = batch["label"]
    bad = ((cert > 0) & (lab < 0.5)) | ((cert < 0) & (lab > 0.5))
    return int((bad & (batch["mask"] > 0)).sum())


# ================================================================= arm A: existing skill heuristics
def heuristic_scores(skill: str, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """The existing skill's rule as written in its SKILL.md procedure, without any
    trainable module.  Returns logits (positive = select)."""
    c, rel, m = batch["cand"], batch["rel"], batch["mask"]
    if skill == "repair":  # flip nodes where most incident edges conflict
        return (c[..., 1] - 0.5) * 4 + 1e-3
    if skill == "retrieval":  # direct dependencies of the query plus lexical overlap
        q = c[..., 0]
        direct = (torch.einsum("bi,bij->bj", q, rel) > 0).float()
        return (direct + (c[..., 1] > 0.65).float() - 0.5) * 4
    if skill == "deduction":  # keep undominated candidates
        return (c[..., 3] - 0.5) * 4
    if skill == "research":  # prior-weighted likelihood variance minus cost
        lik, prior, lam = c[..., 0:6], batch["glob"][:, None, 0:6], batch["glob"][:, 6:7]
        mean = (lik * prior).sum(-1, keepdim=True)
        var = ((lik - mean) ** 2 * prior).sum(-1)
        return var * 8 - lam * c[..., 6]
    if skill == "curation":  # neighbour-vote disagreement or earlier twin
        y = c[..., 6]
        vote = torch.einsum("bij,bj->bi", rel, y) / rel.sum(-1).clamp(min=1e-6)
        return (((y - vote).abs() > 0.5).float() + c[..., 8] - 0.5) * 4
    if skill == "stopping":  # myopic: act once then stop
        cc, gain, cost = c[..., 6], c[..., 5], c[..., 4]
        val = cc + gain * (1 - cc) - cost
        val[..., 0] = cc[..., 0]
        val[..., 3] = cc[..., 3] + (1 - cc[..., 3]) * c[..., 3, 9] - cost[..., 3]
        return val
    raise KeyError(skill)


def heuristic_repair_step(skill: str, batch: Dict[str, torch.Tensor], pred: torch.Tensor, diag: torch.Tensor) -> torch.Tensor:
    """Arm A's use of verifier diagnostics: apply the local fix the diagnostics name."""
    m = batch["mask"]
    if skill == "repair":
        flag = (diag[..., 0] > 0.5).float()
        return (pred + flag - 2 * pred * flag) * m
    if skill == "retrieval":
        return torch.clamp(pred + diag[..., 0] - diag[..., 1], 0, 1) * m
    if skill == "deduction":
        return torch.clamp(pred - diag[..., 0] + diag[..., 1], 0, 1) * m
    return pred
