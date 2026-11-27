"""Deterministic token-sequence task families for tiny causal recurrent transformers.

Every label depends only on tokens at the same or earlier positions.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch

MAX_T = 96
DEV_FAMILIES = ("cumsum", "pointer_chase", "revision", "repair", "proof_deps", "shallow_copy")
WITHHELD_FAMILIES = ("bounded_walk", "experiment_select")


@dataclass(frozen=True)
class TaskSpec:
    family: str
    length: int
    depth: int = 0
    distract: int = 0
    variant: str = "clean"

    def key(self) -> str:
        return f"{self.family}:L{self.length}:D{self.depth}:X{self.distract}:{self.variant}"


@dataclass
class Batch:
    x: torch.LongTensor
    y: torch.LongTensor
    mask: torch.BoolTensor


class Family:
    name = ""
    vocab = 0
    n_out = 0
    withheld = False
    variants: tuple = ("clean",)
    blurb = ""

    # ---- hooks ----
    def total_len(self, spec):
        raise NotImplementedError

    def _check(self, spec):
        pass

    def _row(self, spec, rng):
        """Return (tokens, labels) with None labels at unscored positions."""
        raise NotImplementedError

    def solve(self, x_row):
        raise NotImplementedError

    def _adm_row(self, x_row):
        """Return {position: allowed labels}; unlisted positions are unrestricted."""
        return {}

    # ---- shared API ----
    def describe(self):
        return self.blurb

    def validate(self, spec):
        if spec.family != self.name:
            raise ValueError(f"spec family {spec.family!r} does not match {self.name!r}")
        if spec.variant not in self.variants:
            raise ValueError(f"{self.name}: variant {spec.variant!r} not in {self.variants}")
        if spec.length < 1 or spec.depth < 0 or spec.distract < 0:
            raise ValueError(f"{self.name}: negative or empty size in {spec.key()}")
        self._check(spec)
        if self.total_len(spec) > MAX_T:
            raise ValueError(f"{self.name}: T={self.total_len(spec)} exceeds MAX_T={MAX_T}")

    def generate(self, spec, n, seed):
        self.validate(spec)
        rng = np.random.Generator(np.random.PCG64(seed))
        xs, ys = [], []
        for _ in range(n):
            x, y = self._row(spec, rng)
            xs.append([int(v) for v in x])
            ys.append([-100 if v is None else int(v) for v in y])
        x = torch.tensor(xs, dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)
        return Batch(x=x, y=y, mask=y != -100)

    def admissible(self, x):
        n, T = x.shape
        out = torch.ones(n, T, self.n_out, dtype=torch.bool)
        for i in range(n):
            for t, allowed in self._adm_row(x[i].tolist()).items():
                out[i, t] = False
                out[i, t, list(allowed)] = True
        return out


class Cumsum(Family):
    name, vocab, n_out = "cumsum", 8, 7
    blurb = ("Tokens 0..6 are digits and token 7 is a filler that adds nothing. At every position, "
             "predict the running sum of all digits so far modulo 7. length is the number of digits; "
             "distract is the number of fillers inserted after position 0.")

    def total_len(self, spec):
        return spec.length + spec.distract

    def _row(self, spec, rng):
        T = self.total_len(spec)
        x = rng.integers(0, 7, T)
        if spec.distract:
            x[rng.choice(np.arange(1, T), size=spec.distract, replace=False)] = 7
        y, s = [], 0
        for v in x:
            if v < 7:
                s = (s + int(v)) % 7
            y.append(s)
        return x, y

    def solve(self, x_row):
        return [sum(v for v in x_row[: i + 1] if v < 7) % 7 for i in range(len(x_row))]

    def _adm_row(self, x_row):
        return {0: [x_row[0]]} if x_row[0] < 7 else {}


class PointerChase(Family):
    name, vocab, n_out = "pointer_chase", 130, 8
    variants = ("clean", "shuffled")
    blurb = ("Eight definition tokens (i*8+f(i)) give a function f over nodes 0..7, then a separator (64) "
             "and filler tokens (129). Each query token 65+s*8+(k-1) asks for f applied k times to node s; "
             "predict the resulting node at query positions. depth is the maximum hop count (1..8); "
             "length is the number of queries. The shuffled variant permutes the definition order.")

    def _check(self, spec):
        if not 1 <= spec.depth <= 8:
            raise ValueError("pointer_chase: depth must be in 1..8")

    def total_len(self, spec):
        return 9 + spec.distract + spec.length

    def _row(self, spec, rng):
        f = rng.integers(0, 8, 8)
        s = rng.integers(0, 8, spec.length)
        k = rng.integers(1, spec.depth + 1, spec.length)
        perm = rng.permutation(8)
        order = perm if spec.variant == "shuffled" else np.arange(8)
        x = [int(i) * 8 + int(f[i]) for i in order] + [64] + [129] * spec.distract
        y = [None] * len(x)
        for si, ki in zip(s, k):
            node = int(si)
            for _ in range(int(ki)):
                node = int(f[node])
            x.append(65 + int(si) * 8 + int(ki) - 1)
            y.append(node)
        return x, y

    @staticmethod
    def _defs(x_row):
        f = {}
        for tok in x_row:
            if tok < 64:
                f[tok // 8] = tok % 8
        return f

    def solve(self, x_row):
        f = self._defs(x_row)
        out = []
        for tok in x_row:
            if 65 <= tok <= 128:
                node, hops = (tok - 65) // 8, (tok - 65) % 8 + 1
                while hops:
                    node, hops = f[node], hops - 1
                out.append(node)
            else:
                out.append(None)
        return out

    def _adm_row(self, x_row):
        image = sorted(set(self._defs(x_row).values()))
        return {t: image for t, tok in enumerate(x_row) if 65 <= tok <= 128}


class Revision(Family):
    name, vocab, n_out = "revision", 29, 7
    variants = ("clean", "reordered")
    blurb = ("Four keys hold values 0..5. Assign tokens (key*6+value) overwrite a key; query tokens (24+key) "
             "ask for the key's current value; token 28 is filler. At each query predict the most recent "
             "value assigned to that key, or 6 if it was never assigned. depth is the minimum number of "
             "reassignments each key receives before the first query (clean variant); length is the number "
             "of events; the reordered variant randomly permutes the events.")

    def _check(self, spec):
        if spec.length < 2:
            raise ValueError("revision: length must be >= 2")
        if spec.depth > 0 and spec.length < 4 * (spec.depth + 1) + 2:
            raise ValueError("revision: length too small for depth")

    def total_len(self, spec):
        return spec.length + spec.distract

    def _row(self, spec, rng):
        ev, last = [], {}
        if spec.depth > 0:
            for _ in range(spec.depth + 1):
                for k in range(4):
                    v = int(rng.integers(0, 6))
                    while v == last.get(k):
                        v = int(rng.integers(0, 6))
                    last[k] = v
                    ev.append(k * 6 + v)
        while len(ev) < spec.length:
            k = int(rng.integers(0, 4))
            ev.append(k * 6 + int(rng.integers(0, 6)) if rng.random() < 0.6 else 24 + k)
        if all(e < 24 for e in ev):
            ev[-1] = 24 + int(rng.integers(0, 4))
        if spec.variant == "reordered":
            ev = [ev[i] for i in rng.permutation(len(ev))]
        T = self.total_len(spec)
        fill = set(rng.choice(np.arange(1, T), size=spec.distract, replace=False).tolist()) if spec.distract else set()
        it = iter(ev)
        x = [28 if t in fill else next(it) for t in range(T)]
        cur, y = {}, []
        for tok in x:
            if tok < 24:
                cur[tok // 6] = tok % 6
                y.append(None)
            elif tok < 28:
                y.append(cur.get(tok - 24, 6))
            else:
                y.append(None)
        return x, y

    def solve(self, x_row):
        out = []
        for i, tok in enumerate(x_row):
            if 24 <= tok < 28:
                vals = [t % 6 for t in x_row[:i] if t < 24 and t // 6 == tok - 24]
                out.append(vals[-1] if vals else 6)
            else:
                out.append(None)
        return out

    def _adm_row(self, x_row):
        adm = {}
        for i, tok in enumerate(x_row):
            if 24 <= tok < 28:
                vals = {t % 6 for t in x_row[:i] if t < 24 and t // 6 == tok - 24}
                adm[i] = sorted(vals) or [6]
        return adm


class Repair(Family):
    name, vocab, n_out = "repair", 22, 7
    variants = ("flagged", "unflagged")
    blurb = ("A prefix of `length` digits (0..6), a separator (7), then `length` candidate tokens "
             "8+c+7*flag proposing the running sum mod 7 at each prefix position. depth candidates are "
             "corrupted. At each candidate position predict the correct running sum. In the flagged "
             "variant corrupted candidates carry flag=1; in the unflagged variant no flags appear. "
             "depth=0 is a copying control.")

    def _check(self, spec):
        if spec.depth > spec.length:
            raise ValueError("repair: depth must be <= length")

    def total_len(self, spec):
        return 2 * spec.length + 1

    def _row(self, spec, rng):
        L = spec.length
        d = rng.integers(0, 7, L)
        true = np.cumsum(d) % 7
        cand = true.copy()
        bad = rng.choice(L, size=spec.depth, replace=False) if spec.depth else []
        flag = np.zeros(L, dtype=int)
        for j in bad:
            cand[j] = (cand[j] + rng.integers(1, 7)) % 7
            flag[j] = 1 if spec.variant == "flagged" else 0
        x = list(d) + [7] + [8 + int(c) + 7 * int(f) for c, f in zip(cand, flag)]
        return x, [None] * (L + 1) + [int(v) for v in true]

    def solve(self, x_row):
        sep = x_row.index(7)
        out = [None] * (sep + 1)
        for j in range(len(x_row) - sep - 1):
            out.append(sum(x_row[: j + 1]) % 7)
        return out

    def _adm_row(self, x_row):
        sep = x_row.index(7)
        return {t: [c for c in range(7) if c != x_row[t] - 15]
                for t in range(sep + 1, len(x_row)) if x_row[t] >= 15}


class ProofDeps(Family):
    name, vocab, n_out = "proof_deps", 32, 2
    blurb = ("Each token is a lemma id ok*16+d1*4+d2. ok=1 means the lemma's own check passed; d>0 means "
             "it depends on the lemma d positions earlier. A lemma is proven iff ok=1 and every dependency "
             "is proven. Predict the proven bit at every position. depth caps the length of dependency chains.")

    def _check(self, spec):
        if spec.depth < 1:
            raise ValueError("proof_deps: depth must be >= 1")

    def total_len(self, spec):
        return spec.length

    def _row(self, spec, rng):
        x, y, chain = [], [], []
        for i in range(spec.length):
            ok = int(rng.random() < 0.85)
            ds = [int(v) for v in rng.choice(4, size=2, p=[0.3, 0.3, 0.2, 0.2])]
            for e in range(2):
                if ds[e] > i or (ds[e] and chain[i - ds[e]] + 1 > spec.depth):
                    ds[e] = 0
            chain.append(max((chain[i - d] + 1 for d in ds if d), default=0))
            proven = ok and all(y[i - d] for d in ds if d)
            x.append(ok * 16 + ds[0] * 4 + ds[1])
            y.append(int(bool(proven)))
        return x, y

    @staticmethod
    def _deps(tok, i):
        return [i - d for d in ((tok % 16) // 4, tok % 4) if d and i - d >= 0]

    def solve(self, x_row):
        proven = []
        for i, tok in enumerate(x_row):
            proven.append(int(tok >= 16 and all(proven[j] for j in self._deps(tok, i))))
        return proven

    def _adm_row(self, x_row):
        adm = {}
        for i, tok in enumerate(x_row):
            if tok < 16:
                adm[i] = [0]
            elif not self._deps(tok, i):
                adm[i] = [1]
        return adm


class ShallowCopy(Family):
    name, vocab, n_out = "shallow_copy", 7, 7
    blurb = ("Matched control where recurrence should not help: tokens are digits 0..6 and the label at "
             "each position is (3*x+1) mod 7 of the same token.")

    def total_len(self, spec):
        return spec.length

    def _row(self, spec, rng):
        x = rng.integers(0, 7, spec.length)
        return x, (3 * x + 1) % 7

    def solve(self, x_row):
        return [(x + x + x + 1) % 7 for x in x_row]


class BoundedWalk(Family):
    name, vocab, n_out, withheld = "bounded_walk", 6, 6, True
    blurb = ("A position starts at 0 on the segment [0,5]. Tokens 0..4 move it by -2..+2 and token 5 is a "
             "wait. Positions are clipped to the segment. Predict the position after every token. length is "
             "the number of moves; distract is the number of waits mixed in.")

    def total_len(self, spec):
        return spec.length + spec.distract

    def _row(self, spec, rng):
        T = self.total_len(spec)
        x = rng.integers(0, 5, T)
        if spec.distract:
            x[rng.choice(T, size=spec.distract, replace=False)] = 5
        y, p = [], 0
        for m in x:
            p = int(np.clip(p + (0 if m == 5 else m - 2), 0, 5))
            y.append(p)
        return x, y

    def solve(self, x_row):
        out, p = [], 0
        for m in x_row:
            step = 0 if m == 5 else m - 2
            p = min(5, max(0, p + step))
            out.append(p)
        return out

    def _adm_row(self, x_row):
        return {0: [self.solve(x_row[:1])[0]]}


class ExperimentSelect(Family):
    name, vocab, n_out, withheld = "experiment_select", 32, 2, True
    blurb = ("The first token encodes a budget B (16+B-1). Then `length` (4..6) item tokens (cost-1)*4+value "
             "with cost 1..4 and value 0..3, then one answer slot token (26+j) per item. At slot j predict 1 "
             "if item j belongs to the optimal subset: maximum total value within total cost <= B, ties broken "
             "by lower total cost, then lexicographically smaller index tuple.")

    def _check(self, spec):
        if not 4 <= spec.length <= 6:
            raise ValueError("experiment_select: length must be in 4..6")

    def total_len(self, spec):
        return 1 + 2 * spec.length

    def _row(self, spec, rng):
        L = spec.length
        B = int(rng.integers(1, 11))
        cost = rng.integers(1, 5, L)
        val = rng.integers(0, 4, L)
        best = None
        for r in range(L + 1):
            for sub in itertools.combinations(range(L), r):
                c = sum(int(cost[i]) for i in sub)
                if c > B:
                    continue
                key = (-sum(int(val[i]) for i in sub), c, sub)
                if best is None or key < best:
                    best = key
        chosen = set(best[2])
        x = [16 + B - 1] + [(int(c) - 1) * 4 + int(v) for c, v in zip(cost, val)] + [26 + j for j in range(L)]
        y = [None] * (L + 1) + [int(j in chosen) for j in range(L)]
        return x, y

    @staticmethod
    def _items(x_row):
        L = (len(x_row) - 1) // 2
        B = x_row[0] - 15
        return B, [(t // 4 + 1, t % 4) for t in x_row[1 : 1 + L]]

    def solve(self, x_row):
        B, items = self._items(x_row)
        L = len(items)
        scored = []
        for mask in range(1 << L):
            sub = tuple(i for i in range(L) if mask >> i & 1)
            c = sum(items[i][0] for i in sub)
            if c <= B:
                scored.append((sum(items[i][1] for i in sub), c, sub))
        best = sorted(scored, key=lambda s: (-s[0], s[1], s[2]))[0][2]
        return [None] * (L + 1) + [int(j in best) for j in range(L)]

    def _adm_row(self, x_row):
        B, items = self._items(x_row)
        L = len(items)
        return {1 + L + j: [0] for j, (c, _) in enumerate(items) if c > B}


FAMILIES = {f.name: f for f in (Cumsum(), PointerChase(), Revision(), Repair(), ProofDeps(),
                                ShallowCopy(), BoundedWalk(), ExperimentSelect())}


def score(batch, logits_or_pred):
    p = logits_or_pred
    pred = p.argmax(-1) if p.dim() == 3 else p
    right = (pred == batch.y) | ~batch.mask
    per = right.all(dim=1).long()
    n_masked = int(batch.mask.sum())
    tok = float(((pred == batch.y) & batch.mask).sum()) / max(1, n_masked)
    return {"exact": float(per.float().mean()), "token_acc": tok, "per_example": per.tolist()}


def verify_spec(spec, n=64, seed=12345):
    reasons, T, majority = [], 0, 1.0
    try:
        fam = FAMILIES[spec.family]
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
        got = [int(v) if m else None for v, m in zip(b.y[i].tolist(), b.mask[i].tolist())]
        if [s if m else None for s, m in zip(sol, b.mask[i].tolist())] != got or \
                any((s is None) == bool(m) for s, m in zip(sol, b.mask[i].tolist())):
            reasons.append(f"solve disagrees with generate on row {i}")
            break
    adm = fam.admissible(b.x)
    hit = adm.gather(-1, b.y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    if not bool((hit | ~b.mask).all()):
        reasons.append("admissible excludes a true label")
    majority = 0.0
    for c in range(fam.n_out):
        eq = ((b.y == c) | ~b.mask).all(dim=1).float().mean().item()
        majority = max(majority, eq)
    if majority >= 0.5:
        reasons.append(f"majority-class exact {majority:.2f} >= 0.5")
    return {"ok": not reasons, "reasons": reasons, "T": T, "majority_exact": majority}
