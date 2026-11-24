"""Shared physical memory and multi-projection address layouts for MCP-07.

All six projections share the exact same underlying physical record slots,
keys, values, and corruption draws. All index storage, preprocessing, and
runtime lookup/switch costs are explicitly accounted for, and shared physical
slot IDs are tracked across projection switches.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

PAGE: int = 32
CAPACITIES: Tuple[int, ...] = (16384, 24576, 32768, 65536, 131072)
STRUCTURED_PROJECTIONS: Tuple[str, ...] = ("row", "col", "hilbert", "morton")
ALL_PROJECTIONS: Tuple[str, ...] = (
    "row",
    "col",
    "hilbert",
    "morton",
    "rand_coloc",
    "rand_full",
)
PROJ_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(ALL_PROJECTIONS)}

# 13-action hierarchical stride set; index 6 (stride=0) is legal only on a projection switch.
STRIDES: Tuple[int, ...] = (-32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32)
STRIDES_NP: np.ndarray = np.asarray(STRIDES, dtype=np.int32)
ZERO_STRIDE_IDX: int = 6
FINE_MASK_NP: np.ndarray = np.abs(STRIDES_NP) <= 4
COARSE_MASK_NP: np.ndarray = np.abs(STRIDES_NP) >= 8

# Conservative read-equivalent cost parameters
ANCHOR_LOOKUP_COST: int = 1
SWITCH_OVERHEAD_COST: int = 2


def grid_shape(capacity: int) -> Tuple[int, int]:
    """Return canonical (width, height) lattice dimensions for a physical capacity."""
    width = 128 if capacity <= 32768 else 256
    if capacity % width != 0 or capacity % PAGE != 0:
        raise ValueError(f"Invalid capacity {capacity}: must be divisible by {width} and {PAGE}")
    height = capacity // width
    if height % PAGE != 0 or width % PAGE != 0:
        raise ValueError(f"Lattice ({width}, {height}) must have dimensions divisible by {PAGE}")
    return width, height


def hilbert_xy_scalar(side: int, d: int) -> Tuple[int, int]:
    """Scalar reference 2D Hilbert curve decoder (identical to MCP-06)."""
    x = 0
    y = 0
    s = 1
    while s < side:
        rx = (d // 2) & 1
        ry = (d ^ rx) & 1
        if ry == 0:
            if rx:
                x, y = s - 1 - x, s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d //= 4
        s *= 2
    return x, y


def hilbert_order(width: int, height: int) -> np.ndarray:
    """Vectorized 2D Hilbert-curve slot permutation clipped to (width, height)."""
    side = 1 << (max(width, height) - 1).bit_length()
    total = side * side
    d = np.arange(total, dtype=np.int64)
    x = np.zeros(total, dtype=np.int64)
    y = np.zeros(total, dtype=np.int64)
    s = 1
    t = d.copy()
    while s < side:
        rx = (t >> 1) & 1
        ry = (t ^ rx) & 1
        rot0 = ry == 0
        flip = rot0 & (rx == 1)
        x[flip] = (s - 1) - x[flip]
        y[flip] = (s - 1) - y[flip]
        tmp_x = x[rot0].copy()
        x[rot0] = y[rot0]
        y[rot0] = tmp_x
        x += s * rx
        y += s * ry
        t >>= 2
        s <<= 1
    in_bounds = (x < width) & (y < height)
    slots = (y[in_bounds] * width + x[in_bounds]).astype(np.int32)
    return slots


def _spread_bits_16(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.int64)
    v = (v | (v << 8)) & 0x00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F
    v = (v | (v << 2)) & 0x33333333
    v = (v | (v << 1)) & 0x55555555
    return v


def morton_order(width: int, height: int) -> np.ndarray:
    """Vectorized 2D Morton / Z-order curve clipped to (width, height)."""
    xs = np.arange(width, dtype=np.int64)
    ys = np.arange(height, dtype=np.int64)
    gx, gy = np.meshgrid(xs, ys)
    codes = _spread_bits_16(gx) | (_spread_bits_16(gy) << 1)
    slots = (gy * width + gx).ravel()
    order = np.argsort(codes.ravel(), kind="stable")
    return slots[order].astype(np.int32)


def projection_order(capacity: int, layout: str, seed: int) -> Tuple[np.ndarray, int]:
    """Return (slot_permutation, analytical_preprocessing_ops) for a projection."""
    width, height = grid_shape(capacity)
    if layout == "row":
        return np.arange(capacity, dtype=np.int32), capacity
    if layout == "col":
        arr = np.arange(capacity, dtype=np.int32).reshape(height, width).T.ravel()
        return np.ascontiguousarray(arr), capacity
    if layout == "hilbert":
        side = 1 << (max(width, height) - 1).bit_length()
        ops = int(side * side * int(math.log2(side)))
        return hilbert_order(width, height), ops
    if layout == "morton":
        ops = int(capacity * int(math.ceil(math.log2(capacity))))
        return morton_order(width, height), ops
    if layout == "rand_coloc":
        rng = np.random.default_rng(seed)
        rec_perm = rng.permutation(capacity // 2).astype(np.int32)
        slots = np.empty(capacity, dtype=np.int32)
        slots[0::2] = 2 * rec_perm
        slots[1::2] = 2 * rec_perm + 1
        return slots, capacity
    if layout == "rand_full":
        rng = np.random.default_rng(seed + 17)
        return rng.permutation(capacity).astype(np.int32), capacity
    raise ValueError(f"Unknown layout: {layout}")


def stride_bfs_distances(max_pages: int) -> np.ndarray:
    """Precompute minimum step count to bridge page offset delta using non-zero STRIDES."""
    dist = np.full(max_pages, 10000, dtype=np.int16)
    dist[0] = 0
    nonzero = [abs(d) for d in STRIDES if d > 0]
    queue = [0]
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        nd = dist[cur] + 1
        for step in nonzero:
            for nxt in (cur + step, cur - step):
                if 0 <= nxt < max_pages and dist[nxt] > nd:
                    dist[nxt] = nd
                    queue.append(nxt)
    return dist


@dataclass
class ProjectionView:
    """One address projection over a SharedPhysicalMemory instance."""

    name: str
    cells: np.ndarray          # int32[M, PAGE]: physical slot IDs on each page
    inv: np.ndarray            # int32[N]: physical slot -> page ID
    centers: np.ndarray        # float32[M, 2]: physical centroid (cx, cy) of each page
    center_slot: np.ndarray    # int32[M]: deterministic physical slot closest to centroid
    storage_bytes: int         # exact index memory footprint in bytes
    preprocessing_ops: int     # analytical preprocessing op count
    preprocessing_seconds: float  # measured build wall time


class SharedPhysicalMemory:
    """Shared physical record store equipped with all six MCP-07 address projections."""

    def __init__(self, capacity: int, seed: int, projections: Tuple[str, ...] = ALL_PROJECTIONS):
        self.n = int(capacity)
        self.width, self.height = grid_shape(self.n)
        self.pages = self.n // PAGE
        self.num_records = self.n // 2
        self.seed = int(seed)

        rng = np.random.default_rng(self.seed)
        self.keys = rng.permutation(self.num_records).astype(np.int32)
        self.values = rng.integers(0, 2**30, size=self.num_records, dtype=np.int64)
        self.stale = rng.random(self.n, dtype=np.float32)

        self.views: Dict[str, ProjectionView] = {}
        for name in projections:
            self.views[name] = self._build_view(name)

        self.stride_dist = stride_bfs_distances(self.pages)

    def _build_view(self, name: str) -> ProjectionView:
        t0 = time.perf_counter()
        perm, ops = projection_order(self.n, name, self.seed + 901)
        cells = perm.reshape(self.pages, PAGE)
        inv = np.empty(self.n, dtype=np.int32)
        inv[perm] = np.repeat(np.arange(self.pages, dtype=np.int32), PAGE)

        xs = (cells % self.width).astype(np.float32)
        ys = (cells // self.width).astype(np.float32)
        cx = xs.mean(axis=1)
        cy = ys.mean(axis=1)
        centers = np.stack([cx, cy], axis=1).astype(np.float32)

        # Deterministic physical cursor slot: slot in page closest (L1) to page centroid
        l1 = np.abs(xs - cx[:, None]) + np.abs(ys - cy[:, None])
        best_idx = np.argmin(l1, axis=1)
        center_slot = cells[np.arange(self.pages), best_idx].astype(np.int32)

        elapsed = time.perf_counter() - t0
        storage_bytes = int(
            cells.nbytes + inv.nbytes + centers.nbytes + center_slot.nbytes
        )
        return ProjectionView(
            name=name,
            cells=cells,
            inv=inv,
            centers=centers,
            center_slot=center_slot,
            storage_bytes=storage_bytes,
            preprocessing_ops=ops,
            preprocessing_seconds=elapsed,
        )

    def index_cost_for_projections(self, proj_names: Tuple[str, ...]) -> Dict[str, float]:
        """Return total index storage bytes, preprocessing ops, and build seconds for an arm."""
        return {
            "num_projections": len(proj_names),
            "index_storage_bytes": int(sum(self.views[p].storage_bytes for p in proj_names)),
            "preprocessing_ops": int(sum(self.views[p].preprocessing_ops for p in proj_names)),
            "preprocessing_seconds": float(sum(self.views[p].preprocessing_seconds for p in proj_names)),
        }

    def probe_single(
        self,
        proj: str,
        page: int,
        key: int,
        corruption: float,
        decoy_slot: int = -1,
    ) -> Tuple[int | None, bool, int, np.ndarray]:
        """Probe one page in projection `proj`.

        Returns (valid_value_or_None, stale_hit_flag, good_replica_count, physical_slots_read).
        """
        slots = self.views[proj].cells[page]
        recs = slots >> 1
        matched_mask = self.keys[recs] == key
        matched_slots = slots[matched_mask]
        good_mask = self.stale[matched_slots] >= corruption
        good_slots = matched_slots[good_mask]

        stale_hit = bool(np.any(~good_mask))
        if decoy_slot >= 0 and bool(np.any(slots == decoy_slot)):
            stale_hit = True

        val = int(self.values[good_slots[0] >> 1]) if len(good_slots) > 0 else None
        return val, stale_hit, int(len(good_slots)), slots

    def probe_batch(
        self,
        projs: np.ndarray,
        pages: np.ndarray,
        target_keys: np.ndarray,
        corruption: float,
        decoy_slots: np.ndarray,
        proj_names: Tuple[str, ...] = ALL_PROJECTIONS,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized batch probe across mixed projections.

        Args:
            projs: int32[B] projection indices into `proj_names`
            pages: int32[B] page indices
            target_keys: int32[B] requested keys
            corruption: float in [0, 1]
            decoy_slots: int32[B] (-1 when no injected stale decoy)
            proj_names: tuple mapping index -> projection name

        Returns:
            hit_valid: bool[B]
            values: int64[B] (-1 on miss)
            stale_hit: bool[B]
            good_count: int32[B]
            slots_batch: int32[B, PAGE] physical slot IDs touched
        """
        bsz = len(pages)
        slots_batch = np.empty((bsz, PAGE), dtype=np.int32)
        for pid in np.unique(projs):
            mask = projs == pid
            slots_batch[mask] = self.views[proj_names[int(pid)]].cells[pages[mask]]

        recs = slots_batch >> 1
        matched = self.keys[recs] == target_keys[:, None]
        if corruption > 0.0:
            not_corrupt = self.stale[slots_batch] >= corruption
        else:
            not_corrupt = np.ones_like(matched, dtype=bool)

        good = matched & not_corrupt
        bad_natural = matched & (~not_corrupt)
        decoy_hit = (decoy_slots[:, None] >= 0) & (slots_batch == decoy_slots[:, None])

        good_count = good.sum(axis=1).astype(np.int32)
        hit_valid = good_count > 0
        stale_hit = np.any(bad_natural, axis=1) | np.any(decoy_hit, axis=1)

        values = np.full(bsz, -1, dtype=np.int64)
        if np.any(hit_valid):
            first_col = np.argmax(good[hit_valid], axis=1)
            win_slots = slots_batch[hit_valid, first_col]
            values[hit_valid] = self.values[win_slots >> 1]

        return hit_valid, values, stale_hit, good_count, slots_batch
