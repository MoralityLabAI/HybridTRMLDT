"""Eight-family retrieval curriculum and distribution-shift mixtures for MCP-07.

Every query exposes only observable geometry (anchor coordinates, coarse
direction signs, coarse 3-level per-axis scales) and retrieval outcomes.
Target coordinates, target record IDs, and family labels are privileged to
verification, analysis, and 16K teacher trace generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .memory import SharedPhysicalMemory

FAMILIES: Tuple[str, ...] = (
    "horizontal",
    "vertical",
    "diagonal",
    "isotropic",
    "sparse_long_range",
    "multihop",
    "regime_shift",
    "corrupted_replica",
)
FAMILY_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(FAMILIES)}

# Distribution-shift protocol mixtures (sum to 1.0)
SHIFT_TRAIN_WEIGHTS: Dict[str, float] = {
    "horizontal": 0.20,
    "isotropic": 0.20,
    "sparse_long_range": 0.20,
    "multihop": 0.20,
    "vertical": 0.05,
    "diagonal": 0.05,
    "regime_shift": 0.05,
    "corrupted_replica": 0.05,
}
SHIFT_EVAL_WEIGHTS: Dict[str, float] = {
    "horizontal": 0.05,
    "isotropic": 0.05,
    "sparse_long_range": 0.05,
    "multihop": 0.05,
    "vertical": 0.20,
    "diagonal": 0.20,
    "regime_shift": 0.20,
    "corrupted_replica": 0.20,
}


def coarse_scale(delta: int) -> float:
    """Quantize a coordinate offset into a 3-level observable coarse scale bin."""
    ad = abs(int(delta))
    if ad == 0:
        return 0.0
    if ad <= 18:
        return 0.5
    return 1.0


def _sign(v: int) -> int:
    return 1 if v > 0 else (-1 if v < 0 else 0)


@dataclass(frozen=True)
class HopSpec:
    """One retrieval hop (never exposes target_record or target coords at inference)."""

    anchor: int
    target_record: int
    hint_x: int
    hint_y: int
    scale_x: float
    scale_y: float
    decoy_slot: int = -1
    mirror_hint_x: int = 0
    mirror_hint_y: int = 0
    mirror_scale_x: float = 0.0
    mirror_scale_y: float = 0.0


@dataclass(frozen=True)
class QueryRoot:
    """One paired retrieval problem over SharedPhysicalMemory."""

    family: str
    family_id: int
    hops: Tuple[HopSpec, ...]

    @property
    def num_hops(self) -> int:
        return len(self.hops)


def _sample_horizontal(rng: np.random.Generator, w: int, h: int, lo: int = 12, hi: int = 62) -> HopSpec:
    ax = int(rng.integers(0, w))
    ay = int(rng.integers(0, h))
    for _ in range(16):
        sign = int(rng.choice([-1, 1]))
        mag = int(rng.integers(lo, hi + 1))
        tx = ax + sign * mag
        if 0 <= tx < w and (tx // 2) != (ax // 2):
            break
    else:
        tx = ax + lo if ax + lo < w else ax - lo
    ty = ay
    return HopSpec(
        anchor=ay * w + ax,
        target_record=(ty * w + tx) // 2,
        hint_x=_sign(tx - ax),
        hint_y=0,
        scale_x=coarse_scale(tx - ax),
        scale_y=0.0,
    )


def _sample_vertical(rng: np.random.Generator, w: int, h: int, lo: int = 12, hi: int = 62) -> HopSpec:
    ax = int(rng.integers(0, w))
    ay = int(rng.integers(0, h))
    for _ in range(16):
        sign = int(rng.choice([-1, 1]))
        mag = int(rng.integers(lo, hi + 1))
        ty = ay + sign * mag
        if 0 <= ty < h:
            break
    else:
        ty = ay + lo if ay + lo < h else ay - lo
    tx = ax
    return HopSpec(
        anchor=ay * w + ax,
        target_record=(ty * w + tx) // 2,
        hint_x=0,
        hint_y=_sign(ty - ay),
        scale_x=0.0,
        scale_y=coarse_scale(ty - ay),
    )


def _sample_diagonal(rng: np.random.Generator, w: int, h: int) -> HopSpec:
    """Anti-diagonal Z-corridor across 16x16 or 32x32 dyadic tiles (Q1 <-> Q2)."""
    sz = int(rng.choice([16, 32]))
    half = sz // 2
    bx = int(rng.integers(0, w // sz)) * sz
    by = int(rng.integers(0, h // sz)) * sz
    # Q1 bottom-right corner and Q2 top-left corner are adjacent across Morton's Z-stroke
    q1x = bx + half + int(rng.integers(half - 4, half))
    q1y = by + int(rng.integers(half - 4, half))
    q2x = bx + int(rng.integers(0, 4))
    q2y = by + half + int(rng.integers(0, 4))
    if int(rng.integers(0, 2)) == 0:
        ax, ay, tx, ty = q1x, q1y, q2x, q2y
    else:
        ax, ay, tx, ty = q2x, q2y, q1x, q1y
    return HopSpec(
        anchor=ay * w + ax,
        target_record=(ty * w + tx) // 2,
        hint_x=_sign(tx - ax),
        hint_y=_sign(ty - ay),
        scale_x=coarse_scale(tx - ax),
        scale_y=coarse_scale(ty - ay),
    )


def _sample_isotropic(rng: np.random.Generator, w: int, h: int, lo: int = 3, hi: int = 9) -> HopSpec:
    ax = int(rng.integers(hi + 2, w - hi - 2))
    ay = int(rng.integers(hi + 2, h - hi - 2))
    dx = int(rng.choice([-1, 1])) * int(rng.integers(lo, hi + 1))
    dy = int(rng.choice([-1, 1])) * int(rng.integers(lo, hi + 1))
    tx, ty = ax + dx, ay + dy
    return HopSpec(
        anchor=ay * w + ax,
        target_record=(ty * w + tx) // 2,
        hint_x=_sign(dx),
        hint_y=_sign(dy),
        scale_x=coarse_scale(dx),
        scale_y=coarse_scale(dy),
    )


def _pick_far_offset(rng: np.random.Generator, pos: int, limit: int) -> int:
    """Choose a 2-page-width far jump (+/-64) + jitter (-5..+5) staying in-bounds."""
    candidates = []
    for base in (-64, 64):
        if 6 <= pos + base < limit - 6:
            candidates.append(base)
    if not candidates:
        for base in (-48, 48, -32, 32):
            if 6 <= pos + base < limit - 6:
                candidates.append(base)
    base = int(rng.choice(candidates))
    jitter = int(rng.integers(-5, 6))
    return int(np.clip(pos + base + jitter, 0, limit - 1)) - pos


def _sample_anchor_for_far(rng: np.random.Generator, limit: int) -> int:
    """Sample an anchor coordinate guaranteed to have room for a +/-64 jump even on 128-wide grids."""
    if limit <= 140:
        # Either [8..55] (can jump +64) or [72..limit-9] (can jump -64)
        if int(rng.integers(0, 2)) == 0:
            return int(rng.integers(8, limit - 72))
        return int(rng.integers(72, limit - 8))
    return int(rng.integers(8, limit - 8))


def _sample_sparse_long_range(rng: np.random.Generator, w: int, h: int) -> HopSpec:
    ax = _sample_anchor_for_far(rng, w)
    ay = _sample_anchor_for_far(rng, h)
    dx = _pick_far_offset(rng, ax, w)
    dy = _pick_far_offset(rng, ay, h)
    tx, ty = ax + dx, ay + dy
    return HopSpec(
        anchor=ay * w + ax,
        target_record=(ty * w + tx) // 2,
        hint_x=_sign(dx),
        hint_y=_sign(dy),
        scale_x=coarse_scale(dx),
        scale_y=coarse_scale(dy),
    )


def _sample_regime_shift(rng: np.random.Generator, w: int, h: int) -> HopSpec:
    """Two-regime L-shaped target: one 64-slot 1D axis + one local 2D/orthogonal residual."""
    if int(rng.integers(0, 2)) == 0:
        ax = _sample_anchor_for_far(rng, w)
        ay = int(rng.integers(12, h - 12))
        dx = _pick_far_offset(rng, ax, w)
        dy = int(rng.choice([-1, 1])) * int(rng.integers(4, 11))
    else:
        ax = int(rng.integers(12, w - 12))
        ay = _sample_anchor_for_far(rng, h)
        dx = int(rng.choice([-1, 1])) * int(rng.integers(4, 11))
        dy = _pick_far_offset(rng, ay, h)
    tx = int(np.clip(ax + dx, 0, w - 1))
    ty = int(np.clip(ay + dy, 0, h - 1))
    return HopSpec(
        anchor=ay * w + ax,
        target_record=(ty * w + tx) // 2,
        hint_x=_sign(tx - ax),
        hint_y=_sign(ty - ay),
        scale_x=coarse_scale(tx - ax),
        scale_y=coarse_scale(ty - ay),
    )


def _sample_basic_hop(rng: np.random.Generator, w: int, h: int, kind: str) -> HopSpec:
    if kind == "horizontal":
        return _sample_horizontal(rng, w, h, lo=16, hi=56)
    if kind == "vertical":
        return _sample_vertical(rng, w, h, lo=16, hi=56)
    if kind == "diagonal":
        return _sample_diagonal(rng, w, h)
    if kind == "isotropic":
        return _sample_isotropic(rng, w, h, lo=3, hi=8)
    raise ValueError(f"Unknown basic hop kind: {kind}")


def _sample_multihop(rng: np.random.Generator, w: int, h: int) -> Tuple[HopSpec, HopSpec]:
    """Two-hop pointer chain with contrasting geometries across hops."""
    pairs = (
        ("horizontal", "vertical"),
        ("vertical", "horizontal"),
        ("horizontal", "diagonal"),
        ("vertical", "diagonal"),
        ("diagonal", "horizontal"),
        ("diagonal", "vertical"),
        ("isotropic", "horizontal"),
        ("isotropic", "vertical"),
        ("horizontal", "isotropic"),
        ("vertical", "isotropic"),
    )
    g1, g2 = pairs[int(rng.integers(0, len(pairs)))]
    hop1 = _sample_basic_hop(rng, w, h, g1)
    hop2 = _sample_basic_hop(rng, w, h, g2)
    # Ensure distinct target records across the two hops
    tries = 0
    while hop2.target_record == hop1.target_record and tries < 8:
        hop2 = _sample_basic_hop(rng, w, h, g2)
        tries += 1
    return hop1, hop2


def _sample_corrupted_replica(rng: np.random.Generator, w: int, h: int) -> HopSpec:
    """Stale primary replica near anchor -> probing it reveals coarse orthogonal mirror hint."""
    # Primary stale copy along one geometry; healthy mirror displaced along orthogonal axis
    primary_mode = int(rng.integers(0, 3))  # 0: horiz->vert, 1: vert->horiz, 2: diag->vert/horiz
    if primary_mode == 0:
        primary = _sample_horizontal(rng, w, h, lo=16, hi=48)
        decoy_slot = 2 * primary.target_record
        sx, sy = decoy_slot % w, decoy_slot // w
        dy = _pick_far_offset(rng, sy, h)
        tx, ty = sx, sy + dy
    elif primary_mode == 1:
        primary = _sample_vertical(rng, w, h, lo=16, hi=48)
        decoy_slot = 2 * primary.target_record
        sx, sy = decoy_slot % w, decoy_slot // w
        dx = _pick_far_offset(rng, sx, w)
        tx, ty = sx + dx, sy
    else:
        primary = _sample_diagonal(rng, w, h)
        decoy_slot = 2 * primary.target_record
        sx, sy = decoy_slot % w, decoy_slot // w
        if int(rng.integers(0, 2)) == 0:
            dx = _pick_far_offset(rng, sx, w)
            tx, ty = sx + dx, sy
        else:
            dy = _pick_far_offset(rng, sy, h)
            tx, ty = sx, sy + dy

    target_rec = (ty * w + tx) // 2
    if target_rec == decoy_slot // 2:
        ty = (ty + 32) % h
        target_rec = (ty * w + tx) // 2

    return HopSpec(
        anchor=primary.anchor,
        target_record=target_rec,
        hint_x=primary.hint_x,
        hint_y=primary.hint_y,
        scale_x=primary.scale_x,
        scale_y=primary.scale_y,
        decoy_slot=decoy_slot,
        mirror_hint_x=_sign(tx - sx),
        mirror_hint_y=_sign(ty - sy),
        mirror_scale_x=coarse_scale(tx - sx),
        mirror_scale_y=coarse_scale(ty - sy),
    )


def sample_family_roots(
    mem: SharedPhysicalMemory,
    family: str,
    count: int,
    salt: int,
) -> List[QueryRoot]:
    """Deterministically generate `count` QueryRoots for one query family."""
    fid = FAMILY_TO_ID[family]
    rng = np.random.default_rng(mem.seed * 131 + salt * 17 + fid * 10007)
    w, h = mem.width, mem.height
    out: List[QueryRoot] = []
    for _ in range(count):
        if family == "horizontal":
            hops = (_sample_horizontal(rng, w, h),)
        elif family == "vertical":
            hops = (_sample_vertical(rng, w, h),)
        elif family == "diagonal":
            hops = (_sample_diagonal(rng, w, h),)
        elif family == "isotropic":
            hops = (_sample_isotropic(rng, w, h),)
        elif family == "sparse_long_range":
            hops = (_sample_sparse_long_range(rng, w, h),)
        elif family == "multihop":
            hops = _sample_multihop(rng, w, h)
        elif family == "regime_shift":
            hops = (_sample_regime_shift(rng, w, h),)
        elif family == "corrupted_replica":
            hops = (_sample_corrupted_replica(rng, w, h),)
        else:
            raise ValueError(f"Unknown family: {family}")
        out.append(QueryRoot(family=family, family_id=fid, hops=hops))
    return out


def sample_mixed_roots(
    mem: SharedPhysicalMemory,
    total_count: int,
    salt: int,
    weights: Dict[str, float] | None = None,
) -> List[QueryRoot]:
    """Generate a deterministic mixture of roots across all eight families."""
    if weights is None:
        weights = {f: 1.0 / len(FAMILIES) for f in FAMILIES}
    probs = np.asarray([weights[f] for f in FAMILIES], dtype=np.float64)
    probs /= probs.sum()
    counts = np.floor(probs * total_count).astype(int)
    rem = total_count - int(counts.sum())
    order = np.argsort(-(probs * total_count - counts))
    for idx in order[:rem]:
        counts[idx] += 1

    roots: List[QueryRoot] = []
    for fam, cnt in zip(FAMILIES, counts):
        if cnt > 0:
            roots.extend(sample_family_roots(mem, fam, int(cnt), salt))
    rng = np.random.default_rng(mem.seed + salt + 7919)
    perm = rng.permutation(len(roots))
    return [roots[int(i)] for i in perm]
