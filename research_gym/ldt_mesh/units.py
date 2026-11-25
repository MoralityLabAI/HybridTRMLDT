"""Mesh units: typed, checkable per-block state plus the inference operators.

World: ``K`` latent bits split into ``U`` units of ``UNIT_BITS`` bits. A block (one or
more units) keeps an exact log-belief table over its own bits. Noisy sensor factors
that fall inside one block are absorbed into its table; factors that span blocks stay
pending and appear as dependencies. Pure Python, no third-party imports.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

K = 16
UNIT_BITS = 4
U = K // UNIT_BITS
OPERATORS = ("local", "bp", "exact", "abstain")
KINDS = {0: "xor", 1: "and", 2: "maj"}

Block = Tuple[int, ...]


def popcount(x: int) -> int:
    return x.bit_count()


def unit_of(bit: int) -> int:
    return bit // UNIT_BITS


def unit_mask(u: int) -> int:
    return ((1 << UNIT_BITS) - 1) << (UNIT_BITS * u)


def block_mask(block: Block) -> int:
    return sum(unit_mask(u) for u in block)


def sensor_value(x: int, mask: int, kind: int) -> int:
    """Noiseless sensor: parity (0), all-ones (1) or strict majority (2) of x on mask."""
    c = popcount(x & mask)
    a = popcount(mask)
    if kind == 0:
        return c % 2
    if kind == 1:
        return int(c == a)
    return int(2 * c > a)


@dataclass(frozen=True)
class Factor:
    """Noisy binary observation y of sensor(x, mask, kind) with flip rate e."""

    mask: int
    kind: int
    y: int
    e: float

    def log_lik(self, x: int) -> float:
        v = sensor_value(x, self.mask, self.kind)
        p_on = self.e + (1.0 - 2.0 * self.e) * v
        return math.log(p_on if self.y else 1.0 - p_on)

    def units(self) -> Block:
        return tuple(u for u in range(U) if self.mask & unit_mask(u))


def kl_bound(e: float) -> float:
    """ln 2 - H(e): the most information a flip-rate-e binary sensor can carry."""
    e = min(max(e, 1e-9), 1 - 1e-9)
    return math.log(2) + e * math.log(e) + (1 - e) * math.log(1 - e)


def parity_prob(log_w: Sequence[float], states: Sequence[int], q: int) -> float:
    """P(parity of the q-bits = 1) under unnormalised log weights over global-bit states."""
    m = max(log_w)
    num = den = 0.0
    for lw, s in zip(log_w, states):
        w = math.exp(lw - m)
        den += w
        if popcount(s & q) & 1:
            num += w
    return num / den


def _spread(index: int, units: Block) -> int:
    """Global-bit encoding of the block-local state ``index``."""
    s = 0
    for i, u in enumerate(units):
        s |= ((index >> (UNIT_BITS * i)) & ((1 << UNIT_BITS) - 1)) << (UNIT_BITS * u)
    return s


def block_states(block: Block) -> List[int]:
    return [_spread(i, block) for i in range(1 << (UNIT_BITS * len(block)))]


# ----------------------------------------------------------- exposed unit state

@dataclass(frozen=True)
class LocalState:
    units: Block
    log_table: Tuple[float, ...]


@dataclass(frozen=True)
class Verification:
    """status is unverified (never run exact), stale (new factors since) or verified."""

    status: str
    verified_at: int
    factors_since: int


@dataclass(frozen=True)
class MeshUnit:
    """The seven fields a router may read (the a_t side of the mesh)."""

    local_state: LocalState
    dependencies: Tuple[Block, ...]
    uncertainty: float
    operators: Tuple[str, ...]
    cost_estimates: Dict[str, int]
    verification: Verification
    info_lost: float

    def check(self) -> None:
        """Raise ValueError if the exposed state violates its typed invariants."""
        ls = self.local_state
        if len(ls.log_table) != 1 << (UNIT_BITS * len(ls.units)):
            raise ValueError("local table size does not match block size")
        if list(ls.units) != sorted(set(ls.units)) or not all(0 <= u < U for u in ls.units):
            raise ValueError("block units must be sorted, unique and in range")
        if any(d == ls.units for d in self.dependencies):
            raise ValueError("a unit cannot depend on itself")
        if not (-1e-9 <= self.uncertainty <= UNIT_BITS * len(ls.units) * math.log(2) + 1e-9):
            raise ValueError("entropy out of range")
        if not set(self.operators) <= set(OPERATORS):
            raise ValueError("unknown operator")
        if set(self.cost_estimates) != set(self.operators) or any(c < 0 for c in self.cost_estimates.values()):
            raise ValueError("cost estimates must cover exactly the operators and be non-negative")
        if self.verification.status not in ("unverified", "stale", "verified") or self.info_lost < 0:
            raise ValueError("bad verification status or negative information loss")


# ------------------------------------------------------------------------ mesh

class Mesh:
    """Blocks of units with exact block tables and pending cross-block factors."""

    def __init__(self) -> None:
        self.blocks: List[Block] = [(u,) for u in range(U)]
        self.factors: List[Factor] = []
        self._states: Dict[Block, List[int]] = {}
        self._tables: Dict[Block, List[float]] = {}
        self._verified: Dict[Block, int] = {}
        for b in self.blocks:
            self._init_block(b)

    # -- structure
    def _init_block(self, b: Block) -> None:
        st = block_states(b)
        self._states[b] = st
        table = [0.0] * len(st)
        bm = block_mask(b)
        for f in self.factors:
            if f.mask & bm == f.mask:
                for i, s in enumerate(st):
                    table[i] += f.log_lik(s)
        self._tables[b] = table

    def block_of(self, u: int) -> Block:
        return next(b for b in self.blocks if u in b)

    def _set_blocks(self, blocks: Iterable[Iterable[int]]) -> None:
        blocks = [tuple(sorted(b)) for b in blocks]
        if sorted(u for b in blocks for u in b) != list(range(U)):
            raise ValueError("blocks must partition the units")
        for b in blocks:
            if b not in self._tables:
                self._init_block(b)
        self.blocks = blocks
        self._tables = {b: self._tables[b] for b in blocks}
        self._states = {b: self._states[b] for b in blocks}
        self._verified = {b: v for b, v in self._verified.items() if b in self._tables}

    def merge(self, b1: Block, b2: Block) -> Block:
        """Merge two blocks; factors spanning exactly them move into the new table."""
        if b1 == b2 or b1 not in self.blocks or b2 not in self.blocks:
            raise ValueError("merge needs two distinct existing blocks")
        new = tuple(sorted(b1 + b2))
        self._set_blocks([b for b in self.blocks if b not in (b1, b2)] + [new])
        return new

    def split(self, b: Block) -> List[Block]:
        """Split a block into single units; spanning factors become pending."""
        if b not in self.blocks:
            raise ValueError("unknown block")
        parts = [(u,) for u in b]
        self._set_blocks([x for x in self.blocks if x != b] + parts)
        return parts

    # -- evidence
    def absorb(self, f: Factor) -> None:
        if len(f.units()) > 2:
            raise ValueError("a factor may touch at most two units")
        self.factors.append(f)
        blocks = {self.block_of(u) for u in f.units()}
        if len(blocks) == 1:
            b = next(iter(blocks))
            t = self._tables[b]
            for i, s in enumerate(self._states[b]):
                t[i] += f.log_lik(s)

    def cross_factors(self) -> List[Tuple[Factor, Tuple[Block, ...]]]:
        out = []
        for f in self.factors:
            bl = tuple(sorted({self.block_of(u) for u in f.units()}))
            if len(bl) > 1:
                out.append((f, bl))
        return out

    def relevant(self, q: int) -> List[Block]:
        """Blocks of the query joined through pending cross factors (its component)."""
        cr = self.cross_factors()
        todo = [self.block_of(unit_of(v)) for v in range(K) if (q >> v) & 1]
        comp = set(todo)
        while todo:
            b = todo.pop()
            for _, bl in cr:
                if b in bl:
                    for c in bl:
                        if c not in comp:
                            comp.add(c)
                            todo.append(c)
        return sorted(comp)

    def _component_factors(self, comp: Sequence[Block]) -> List[Factor]:
        cm = sum(block_mask(b) for b in comp)
        return [f for f in self.factors if f.mask & cm == f.mask]

    # -- exposed interface
    def expose(self, b: Block) -> MeshUnit:
        table = self._tables[b]
        m = max(table)
        w = [math.exp(t - m) for t in table]
        z = sum(w)
        ent = -sum((x / z) * math.log(max(x / z, 1e-300)) for x in w)
        cr = [(f, bl) for f, bl in self.cross_factors() if b in bl]
        deps = tuple(sorted({c for _, bl in cr for c in bl if c != b}))
        comp = self.relevant(block_mask(b))
        comp_units = sum(len(c) for c in comp)
        nf = len(self._component_factors(comp))
        size = 1 << (UNIT_BITS * len(b))
        bp_work = 5 * sum(size * (1 << (UNIT_BITS * len(next(c for c in bl if c != b)))) for _, bl in cr)
        n = len(self.factors)
        at = self._verified.get(b, -1)
        status = "unverified" if at < 0 else ("verified" if at == n else "stale")
        unit = MeshUnit(
            local_state=LocalState(b, tuple(table)),
            dependencies=deps,
            uncertainty=ent,
            operators=OPERATORS,
            cost_estimates={"local": size, "bp": bp_work,
                            "exact": (1 << (UNIT_BITS * comp_units)) * max(1, nf), "abstain": 0},
            verification=Verification(status, at, n - max(at, 0)),
            info_lost=sum(kl_bound(f.e) for f, _ in cr),
        )
        unit.check()
        return unit

    # -- operators
    def local(self, q: int) -> Tuple[float, int]:
        """Answer from block tables alone, treating blocks as independent."""
        sign, ops = 1.0, 0
        for b in self.blocks:
            qb = q & block_mask(b)
            if qb:
                sign *= 1 - 2 * parity_prob(self._tables[b], self._states[b], qb)
                ops += len(self._states[b])
        return (1 - sign) / 2, ops

    def bp(self, q: int, iters: int = 5, damp: float = 0.5) -> Tuple[float, int]:
        """Cluster belief propagation over pending cross factors, then local read-out."""
        cr = self.cross_factors()
        if any(len(bl) != 2 for _, bl in cr):
            raise ValueError("cluster BP needs every cross factor to span exactly two blocks")
        beliefs = {b: list(t) for b, t in self._tables.items()}
        msgs: Dict[Tuple[int, Block], List[float]] = {}
        ops = 0
        for _ in range(iters):
            new: Dict[Tuple[int, Block], List[float]] = {}
            for i, (f, (b1, b2)) in enumerate(cr):
                s1, s2 = self._states[b1], self._states[b2]
                lik = [[math.exp(f.log_lik(a | c)) for c in s2] for a in s1]
                for src, dst in ((b2, b1), (b1, b2)):
                    cav = [x - y for x, y in zip(beliefs[src], msgs.get((i, src), [0.0] * len(beliefs[src])))]
                    mx = max(cav)
                    cw = [math.exp(x - mx) for x in cav]
                    if dst == b1:
                        m = [sum(lik[a][c] * cw[c] for c in range(len(s2))) for a in range(len(s1))]
                    else:
                        m = [sum(lik[a][c] * cw[a] for a in range(len(s1))) for c in range(len(s2))]
                    lm = [math.log(max(x, 1e-300)) for x in m]
                    top = max(lm)
                    lm = [x - top for x in lm]
                    old = msgs.get((i, dst), [0.0] * len(lm))
                    new[(i, dst)] = [damp * o + (1 - damp) * x for o, x in zip(old, lm)]
                    ops += len(s1) * len(s2)
            for b in self.blocks:
                beliefs[b] = list(self._tables[b])
                for (i, d), v in new.items():
                    if d == b:
                        beliefs[b] = [x + y for x, y in zip(beliefs[b], v)]
            msgs = new
        sign = 1.0
        for b in self.blocks:
            qb = q & block_mask(b)
            if qb:
                sign *= 1 - 2 * parity_prob(beliefs[b], self._states[b], qb)
        return (1 - sign) / 2, ops + 1

    def exact_cost(self, q: int) -> Tuple[int, List[Block]]:
        comp = self.relevant(q)
        units = sum(len(b) for b in comp)
        return (1 << (UNIT_BITS * units)) * max(1, len(self._component_factors(comp))), comp

    def exact(self, q: int) -> Tuple[float, int]:
        """Exact posterior parity, enumerating only the units of the relevant component.

        Units outside the component share no pending factor with it, so the posterior
        over the component's bits is independent of every other unit.
        """
        cost, comp = self.exact_cost(q)
        units = tuple(sorted(u for b in comp for u in b))
        facs = self._component_factors(comp)
        states = block_states(units)
        log_w = [sum(f.log_lik(s) for f in facs) for s in states]
        for b in comp:
            self._verified[b] = len(self.factors)
        return parity_prob(log_w, states, q), cost
