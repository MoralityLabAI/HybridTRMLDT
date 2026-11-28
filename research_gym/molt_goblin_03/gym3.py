"""Seven additional token-sequence families for the molt-goblin 03 campaign.

Reuses the molt_goblin_02 task protocol. Every label depends only on tokens at the same or earlier
positions, and tokens are self-describing (no absolute-position reliance).
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from research_gym.molt_goblin_02 import gym
from research_gym.molt_goblin_02.gym import MAX_T, Batch, Family, TaskSpec, score

TABLE = np.random.Generator(np.random.PCG64(20261128)).integers(0, 7, (8, 7))
FINALS_ONLY_FAMILIES = ("research_plan",)


def _fill_positions(rng, T, n):
    return set(rng.choice(np.arange(1, T), size=n, replace=False).tolist()) if n else set()


def _chain_edges(rng, i, chain, cap):
    """Draw two dependency offsets for position i, zeroing any edge that would exceed the depth cap."""
    ds = [int(v) for v in rng.choice(4, size=2, p=[0.3, 0.3, 0.2, 0.2])]
    for e in range(2):
        if ds[e] > i or (ds[e] and chain[i - ds[e]] + 1 > cap):
            ds[e] = 0
    chain.append(max((chain[i - d] + 1 for d in ds if d), default=0))
    return ds


class LocalMap(Family):
    name, vocab, n_out = "local_map", 7, 7
    blurb = ("No-recurrence control. Tokens are digits 0..6; the label at position i is TABLE[prev][x_i] for a "
             "fixed random table, where prev is the previous token (7 at position 0).")

    def _check(self, spec):
        if not 4 <= spec.length <= 24:
            raise ValueError("local_map: length must be in 4..24")

    def total_len(self, spec):
        return spec.length

    def _row(self, spec, rng):
        x = [int(v) for v in rng.integers(0, 7, spec.length)]
        y = [int(TABLE[7 if i == 0 else x[i - 1]][x[i]]) for i in range(len(x))]
        return x, y

    def solve(self, x_row):
        out, prev = [], 7
        for v in x_row:
            out.append(int(TABLE[prev, v]))
            prev = v
        return out

    def _adm_row(self, x_row):
        return {i: [int(TABLE[7 if i == 0 else x_row[i - 1]][x_row[i]])] for i in range(len(x_row))}


TABLE3 = np.random.Generator(np.random.PCG64(20261129)).integers(0, 7, (8, 8, 7))


class LocalMap3(Family):
    """Window-3 lookup: one pass suffices (relative offsets 1 and 2); 100 contexts keep it off the ceiling.

    Calibration 2026-11-28: with digits 0..6 (392 contexts) both A and B sat near the floor, so inputs use 0..3.
    """
    name, vocab, n_out = "local_map3", 4, 7
    blurb = ("No-recurrence control. Tokens are digits 0..3; the label at position i is TABLE3[x_{i-2}][x_{i-1}][x_i] "
             "for a fixed random table, with 7 standing in for positions before the start.")

    def _check(self, spec):
        if not 4 <= spec.length <= 24:
            raise ValueError("local_map3: length must be in 4..24")

    def total_len(self, spec):
        return spec.length

    @staticmethod
    def _lab(x, i):
        a = 7 if i < 2 else x[i - 2]
        b = 7 if i < 1 else x[i - 1]
        return int(TABLE3[a][b][x[i]])

    def _row(self, spec, rng):
        x = [int(v) for v in rng.integers(0, 4, spec.length)]
        return x, [self._lab(x, i) for i in range(len(x))]

    def solve(self, x_row):
        out, p2, p1 = [], 7, 7
        for v in x_row:
            out.append(int(TABLE3[p2, p1, v]))
            p2, p1 = p1, v
        return out

    def _adm_row(self, x_row):
        return {i: list(range(7)) for i in range(len(x_row))}


class Cumsum3(Family):
    name, vocab, n_out = "cumsum3", 4, 3
    blurb = ("Digits 0..2 and filler token 3 (adds nothing). Predict the running sum mod 3 at every position. "
             "length is the number of digits; distract is the number of fillers inserted after position 0.")

    def _check(self, spec):
        if not 2 <= spec.length <= 24:
            raise ValueError("cumsum3: length must be in 2..24")

    def total_len(self, spec):
        return spec.length + spec.distract

    def _row(self, spec, rng):
        T = self.total_len(spec)
        x = [int(v) for v in rng.integers(0, 3, T)]
        for t in _fill_positions(rng, T, spec.distract):
            x[t] = 3
        y, s = [], 0
        for v in x:
            s = (s + v) % 3 if v < 3 else s
            y.append(s)
        return x, y

    def solve(self, x_row):
        return [sum(v for v in x_row[: i + 1] if v != 3) % 3 for i in range(len(x_row))]

    def _adm_row(self, x_row):
        return {0: [x_row[0]]} if x_row[0] < 3 else {}


class ProofResolve(Family):
    name, vocab, n_out = "proof_resolve", 64, 2
    blurb = ("Lemma token ok*32+mode*16+d1*4+d2. mode 0 is AND, mode 1 is OR over the dependencies d positions "
             "back (0 means none). A lemma is proven iff ok=1 and its dependencies resolve (all proven for AND, "
             "any proven for OR; trivially true with no dependencies). depth caps dependency-chain length.")

    def _check(self, spec):
        if spec.depth < 1:
            raise ValueError("proof_resolve: depth must be >= 1")

    def total_len(self, spec):
        return spec.length

    def _row(self, spec, rng):
        x, y, chain = [], [], []
        for i in range(spec.length):
            ok = int(rng.random() < 0.85)
            mode = int(rng.integers(0, 2))
            ds = _chain_edges(rng, i, chain, spec.depth)
            live = [y[i - d] for d in ds if d]
            res = (all(live) if mode == 0 else any(live)) if live else True
            x.append(ok * 32 + mode * 16 + ds[0] * 4 + ds[1])
            y.append(int(bool(ok and res)))
        return x, y

    @staticmethod
    def _deps(tok, i):
        return [i - d for d in ((tok % 16) // 4, tok % 4) if d and i - d >= 0]

    def solve(self, x_row):
        proven = []
        for i, tok in enumerate(x_row):
            ok, mode = tok // 32, (tok // 16) % 2
            vals = [proven[j] for j in self._deps(tok, i)]
            if not vals:
                res = 1
            elif mode:
                res = int(sum(vals) > 0)
            else:
                res = int(min(vals))
            proven.append(int(ok == 1 and res == 1))
        return proven

    def _adm_row(self, x_row):
        adm = {}
        for i, tok in enumerate(x_row):
            if tok < 32:
                adm[i] = [0]
            elif not self._deps(tok, i):
                adm[i] = [1]
        return adm


class ExperimentPortfolio(Family):
    name, vocab, n_out = "experiment_portfolio", 66, 2
    blurb = ("Budget token 48+(B-1), then `length` (3..6) item tokens (cost-1)*12+value*3+pre, then one answer "
             "slot 60+j per item. Item j requires item j-pre when pre>0. At slot j predict 1 iff item j is in the "
             "optimal prerequisite-closed subset with total cost <= B and maximal total value (ties: lower total "
             "cost, then lexicographically smaller index tuple).")

    def _check(self, spec):
        if not 3 <= spec.length <= 6:
            raise ValueError("experiment_portfolio: length must be in 3..6")

    def total_len(self, spec):
        return 1 + 2 * spec.length

    def _row(self, spec, rng):
        L = spec.length
        B = int(rng.integers(1, 13))
        cost = [int(v) for v in rng.integers(1, 5, L)]
        val = [int(v) for v in rng.integers(0, 4, L)]
        pre = [int(v) for v in rng.integers(0, 3, L)]
        pre = [p if j - p >= 0 else 0 for j, p in enumerate(pre)]
        best = None
        for r in range(L + 1):
            for sub in itertools.combinations(range(L), r):
                s = set(sub)
                if any(pre[j] and j - pre[j] not in s for j in sub):
                    continue
                c = sum(cost[j] for j in sub)
                if c > B:
                    continue
                key = (-sum(val[j] for j in sub), c, sub)
                if best is None or key < best:
                    best = key
        chosen = set(best[2])
        x = [48 + B - 1] + [(cost[j] - 1) * 12 + val[j] * 3 + pre[j] for j in range(L)] + [60 + j for j in range(L)]
        return x, [None] * (L + 1) + [int(j in chosen) for j in range(L)]

    @staticmethod
    def _items(x_row):
        L = (len(x_row) - 1) // 2
        return x_row[0] - 47, [(t // 12 + 1, (t % 12) // 3, t % 3) for t in x_row[1 : 1 + L]]

    def solve(self, x_row):
        B, items = self._items(x_row)
        L = len(items)
        cands = []
        for mask in range(1 << L):
            sub = [j for j in range(L) if mask >> j & 1]
            if any(items[j][2] and not mask >> (j - items[j][2]) & 1 for j in sub):
                continue
            c = sum(items[j][0] for j in sub)
            if c <= B:
                cands.append((-sum(items[j][1] for j in sub), c, tuple(sub)))
        best = min(cands)[2]
        return [None] * (L + 1) + [int(j in best) for j in range(L)]

    def _adm_row(self, x_row):
        B, items = self._items(x_row)
        L = len(items)
        return {1 + L + j: [0] for j, it in enumerate(items) if it[0] > B}


class RevisionRetract(Family):
    name, vocab, n_out = "revision_retract", 33, 7
    blurb = ("Four keys hold values 0..5. Assign tokens key*6+value push a value, retract tokens 24+key pop the "
             "key's latest assignment (reverting to the previous one; no-op if none), query tokens 28+key ask for "
             "the current value (6 if no live assignment), 32 is filler. length is the number of events; "
             "distract is the number of fillers inserted after position 0.")

    def _check(self, spec):
        if spec.length < 2:
            raise ValueError("revision_retract: length must be >= 2")

    def total_len(self, spec):
        return spec.length + spec.distract

    def _row(self, spec, rng):
        ev = []
        for _ in range(spec.length):
            u, k = rng.random(), int(rng.integers(0, 4))
            ev.append(k * 6 + int(rng.integers(0, 6)) if u < 0.5 else 24 + k if u < 0.7 else 28 + k)
        if not any(e >= 28 for e in ev):
            ev[-1] = 28 + int(rng.integers(0, 4))
        T = self.total_len(spec)
        fill = _fill_positions(rng, T, spec.distract)
        it = iter(ev)
        x = [32 if t in fill else next(it) for t in range(T)]
        stacks, y = {k: [] for k in range(4)}, []
        for tok in x:
            y.append(None)
            if tok < 24:
                stacks[tok // 6].append(tok % 6)
            elif tok < 28:
                if stacks[tok - 24]:
                    stacks[tok - 24].pop()
            elif tok < 32:
                s = stacks[tok - 28]
                y[-1] = s[-1] if s else 6
        return x, y

    @staticmethod
    def _live(prefix, key):
        st = []
        for t in prefix:
            if t < 24 and t // 6 == key:
                st.append(t % 6)
            elif t == 24 + key and st:
                st.pop()
        return st

    def solve(self, x_row):
        out = []
        for i, tok in enumerate(x_row):
            if 28 <= tok < 32:
                st = self._live(x_row[:i], tok - 28)
                out.append(st[-1] if st else 6)
            else:
                out.append(None)
        return out

    def _adm_row(self, x_row):
        adm = {}
        for i, tok in enumerate(x_row):
            if 28 <= tok < 32:
                ever = {t % 6 for t in x_row[:i] if t < 24 and t // 6 == tok - 28}
                adm[i] = sorted(ever | {6})
        return adm


class CexRepair(Family):
    name, vocab, n_out = "cex_repair", 56, 5
    blurb = ("Hidden f(x)=(a*x+b) mod 5. A wrong candidate token a'*5+b', then 2+depth counterexample tokens "
             "25+x*5+f(x) (first two with distinct x), then `length` (1..5) query tokens 50+x; token 55 is "
             "filler. Predict f(x) at each query. distract is the number of fillers inserted after position 0.")

    def _check(self, spec):
        if not 1 <= spec.length <= 5:
            raise ValueError("cex_repair: length must be in 1..5")
        if spec.depth > 8:
            raise ValueError("cex_repair: depth must be <= 8")

    def total_len(self, spec):
        return 1 + 2 + spec.depth + spec.length + spec.distract

    def _row(self, spec, rng):
        a, b = int(rng.integers(1, 5)), int(rng.integers(0, 5))
        while True:
            ca, cb = int(rng.integers(0, 5)), int(rng.integers(0, 5))
            if (ca, cb) != (a, b):
                break
        k = 2 + spec.depth
        xs = [int(v) for v in rng.choice(5, size=2, replace=False)] + [int(v) for v in rng.integers(0, 5, k - 2)]
        qs = [int(v) for v in rng.integers(0, 5, spec.length)]
        core = [ca * 5 + cb] + [25 + x * 5 + (a * x + b) % 5 for x in xs] + [50 + q for q in qs]
        T = self.total_len(spec)
        fill = _fill_positions(rng, T, spec.distract)
        it = iter(core)
        x = [55 if t in fill else next(it) for t in range(T)]
        return x, [(a * (t - 50) + b) % 5 if 50 <= t < 55 else None for t in x]

    @staticmethod
    def _points(x_row):
        return [((t - 25) // 5, (t - 25) % 5) for t in x_row if 25 <= t < 50]

    def solve(self, x_row):
        pts = self._points(x_row)
        fits = [(a, b) for a in range(1, 5) for b in range(5) if all((a * px + b) % 5 == py for px, py in pts)]
        a, b = fits[0]
        return [(a * (t - 50) + b) % 5 if 50 <= t < 55 else None for t in x_row]

    def _adm_row(self, x_row):
        seen = {}
        for px, py in self._points(x_row):
            seen[px] = py
        return {i: [seen[t - 50]] for i, t in enumerate(x_row) if 50 <= t < 55 and t - 50 in seen}


class ResearchPlan(Family):
    name, vocab, n_out, withheld = "research_plan", 48, 16, True
    blurb = ("Tasks with duration 1..3 and two prerequisite offsets (0 none), token dur-1+3*(p1+4*p2). The "
             "label is the earliest finish time min(finish,15), where finish = duration + latest prerequisite "
             "finish. depth caps prerequisite-chain length.")

    def _check(self, spec):
        if not 2 <= spec.length <= 16:
            raise ValueError("research_plan: length must be in 2..16")
        if spec.depth < 1:
            raise ValueError("research_plan: depth must be >= 1")

    def total_len(self, spec):
        return spec.length

    def _row(self, spec, rng):
        x, fin, chain = [], [], []
        for i in range(spec.length):
            dur = int(rng.integers(1, 4))
            ds = _chain_edges(rng, i, chain, spec.depth)
            fin.append(dur + max((fin[i - d] for d in ds if d), default=0))
            x.append(dur - 1 + 3 * (ds[0] + 4 * ds[1]))
        return x, [min(f, 15) for f in fin]

    @staticmethod
    def _parse(tok, i):
        q = tok // 3
        return tok % 3 + 1, [i - p for p in (q % 4, q // 4) if p and i - p >= 0]

    def solve(self, x_row):
        fin = []
        for i, tok in enumerate(x_row):
            dur, deps = self._parse(tok, i)
            fin.append(dur + max([fin[j] for j in deps], default=0))
        return [min(f, 15) for f in fin]

    def _adm_row(self, x_row):
        adm = {}
        for i, tok in enumerate(x_row):
            dur, deps = self._parse(tok, i)
            adm[i] = list(range(dur, 16)) if deps else [dur]
        return adm


NEW_FAMILIES = {f.name: f for f in (LocalMap(), Cumsum3(), ProofResolve(), ExperimentPortfolio(),
                                    RevisionRetract(), CexRepair(), ResearchPlan(), LocalMap3())}
FAMILIES3 = {**gym.FAMILIES, **NEW_FAMILIES}


def verify_spec3(spec, n=64, seed=12345):
    reasons = []
    try:
        fam = FAMILIES3[spec.family]
        b = fam.generate(spec, n, seed)
    except (ValueError, KeyError) as e:
        return {"ok": False, "reasons": [f"invalid spec: {e}"], "T": 0, "majority_exact": 1.0}
    T = b.x.shape[1]
    if T > MAX_T:
        reasons.append(f"T={T} exceeds MAX_T")
    if bool((b.x < 0).any() or (b.x >= fam.vocab).any()):
        reasons.append("token outside vocab")
    if not bool(b.mask.any(dim=1).all()):
        reasons.append("row with no scored position")
    for i in range(n):
        sol = fam.solve(b.x[i].tolist())
        msk = b.mask[i].tolist()
        got = [int(v) if m else None for v, m in zip(b.y[i].tolist(), msk)]
        if [s if m else None for s, m in zip(sol, msk)] != got or any((s is None) == bool(m) for s, m in zip(sol, msk)):
            reasons.append(f"solve disagrees with generate on row {i}")
            break
    adm = fam.admissible(b.x)
    hit = adm.gather(-1, b.y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    if not bool((hit | ~b.mask).all()):
        reasons.append("admissible excludes a true label")
    majority = max(((b.y == c) | ~b.mask).all(dim=1).float().mean().item() for c in range(fam.n_out))
    if majority >= 0.5:
        reasons.append(f"majority-class exact {majority:.2f} >= 0.5")
    return {"ok": not reasons, "reasons": reasons, "T": T, "majority_exact": majority}


verify_spec = verify_spec3
__all__ = ["NEW_FAMILIES", "FAMILIES3", "FINALS_ONLY_FAMILIES", "score", "verify_spec", "verify_spec3",
           "TaskSpec", "Batch", "MAX_T", "TABLE"]
