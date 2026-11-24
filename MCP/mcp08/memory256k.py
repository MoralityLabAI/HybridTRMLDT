"""Extended shared physical memory (16K -> 256K) and charged transposition cache for MCP-08.

Reuses MCP-07's projection kernels without modifying any MCP-07 file, extending
capacity support to 256K (262,144 physical slots = 512 x 512 lattice, 8,192 pages)
and providing a bounded LRU TranspositionCache with explicit byte and lookup charging.
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from MCP.mcp07.memory import (
    ALL_PROJECTIONS,
    ANCHOR_LOOKUP_COST,
    COARSE_MASK_NP,
    FINE_MASK_NP,
    PAGE,
    PROJ_TO_ID,
    STRIDES,
    STRIDES_NP,
    STRUCTURED_PROJECTIONS,
    SWITCH_OVERHEAD_COST,
    ZERO_STRIDE_IDX,
    ProjectionView,
    hilbert_order,
    morton_order,
    stride_bfs_distances,
)

CAPACITIES_MCP08: Tuple[int, ...] = (16384, 32768, 65536, 131072, 262144)
CACHE_ENTRY_BYTES: int = 64
CACHE_LOOKUP_READ_COST: int = 1
CHECKPOINT_FRAME_BYTES: int = 128


def grid_shape_256k(capacity: int) -> Tuple[int, int]:
    """Canonical (width, height) lattice dimensions supporting 16K up to 256K."""
    if capacity <= 32768:
        width = 128
    elif capacity <= 131072:
        width = 256
    elif capacity == 262144:
        width = 512
    else:
        raise ValueError(f"Unsupported capacity: {capacity}")
    if capacity % width != 0 or capacity % PAGE != 0:
        raise ValueError(f"Invalid capacity {capacity}")
    height = capacity // width
    if height % PAGE != 0 or width % PAGE != 0:
        raise ValueError(f"Dimensions ({width}, {height}) must be divisible by {PAGE}")
    return width, height


def projection_order_256k(capacity: int, layout: str, seed: int) -> Tuple[np.ndarray, int]:
    """Return (slot_permutation, analytical_ops) supporting up to 262,144 slots."""
    width, height = grid_shape_256k(capacity)
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


class ExtendedPhysicalMemory:
    """Drop-in shared physical memory supporting 16K through 256K physical slots."""

    def __init__(
        self,
        capacity: int,
        seed: int,
        projections: Tuple[str, ...] = ALL_PROJECTIONS,
    ):
        self.n = int(capacity)
        self.width, self.height = grid_shape_256k(self.n)
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

    def clone_shallow(self) -> "ExtendedPhysicalMemory":
        """Share read-only projection views/keys/stale/stride_dist while isolating mutable `values`."""
        obj = object.__new__(ExtendedPhysicalMemory)
        obj.n = self.n
        obj.seed = self.seed
        obj.width = self.width
        obj.height = self.height
        obj.pages = self.pages
        obj.num_records = self.num_records
        obj.keys = self.keys
        obj.values = self.values.copy()
        obj.stale = self.stale
        obj.views = self.views
        obj.stride_dist = self.stride_dist
        return obj

    def _build_view(self, name: str) -> ProjectionView:
        t0 = time.perf_counter()
        perm, ops = projection_order_256k(self.n, name, self.seed + 901)
        cells = perm.reshape(self.pages, PAGE)
        inv = np.empty(self.n, dtype=np.int32)
        inv[perm] = np.repeat(np.arange(self.pages, dtype=np.int32), PAGE)

        xs = (cells % self.width).astype(np.float32)
        ys = (cells // self.width).astype(np.float32)
        cx = xs.mean(axis=1)
        cy = ys.mean(axis=1)
        centers = np.stack([cx, cy], axis=1).astype(np.float32)

        l1 = np.abs(xs - cx[:, None]) + np.abs(ys - cy[:, None])
        best_idx = np.argmin(l1, axis=1)
        center_slot = cells[np.arange(self.pages), best_idx].astype(np.int32)

        elapsed = time.perf_counter() - t0
        storage_bytes = int(cells.nbytes + inv.nbytes + centers.nbytes + center_slot.nbytes)
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
    ) -> Tuple[Optional[int], bool, int, np.ndarray]:
        slots = self.views[proj].cells[page]
        recs = slots >> 1
        matched_mask = self.keys[recs] == key
        matched_slots = slots[matched_mask]
        good_mask = self.stale[matched_slots] >= corruption if corruption > 0.0 else np.ones(len(matched_slots), dtype=bool)
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
        proj_names: Tuple[str, ...] = STRUCTURED_PROJECTIONS,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


@dataclass
class CacheEntry:
    """State entry stored in the bounded transposition cache."""

    status: str  # "dead_end", "visited", "solved"
    tried_mask: int = 0
    best_score: float = -1e9
    depth: int = 0


class TranspositionCache:
    """Bounded LRU transposition cache with explicit lookup & storage cost accounting."""

    def __init__(self, max_entries: int = 1024, enabled: bool = True):
        self.max_entries = int(max_entries)
        self.enabled = bool(enabled)
        self._store: OrderedDict[Any, CacheEntry] = OrderedDict()
        self.lookups: int = 0
        self.hits: int = 0
        self.misses: int = 0
        self.writes: int = 0
        self.pruned_states: int = 0
        self.evictions: int = 0

    @property
    def storage_bytes(self) -> int:
        return int(self.max_entries * CACHE_ENTRY_BYTES) if self.enabled else 0

    @property
    def charged_lookup_reads(self) -> int:
        return int(self.lookups * CACHE_LOOKUP_READ_COST)

    def lookup(self, key: Any) -> Optional[CacheEntry]:
        if not self.enabled:
            return None
        self.lookups += 1
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        self._store.move_to_end(key)
        if entry.status == "dead_end":
            self.pruned_states += 1
        return entry

    def store(self, key: Any, entry: CacheEntry) -> None:
        if not self.enabled:
            return
        self.writes += 1
        if key in self._store:
            self._store[key] = entry
            self._store.move_to_end(key)
            return
        if len(self._store) >= self.max_entries:
            self._store.popitem(last=False)
            self.evictions += 1
        self._store[key] = entry

    def mark_dead_end(self, key: Any, depth: int = 0) -> None:
        self.store(key, CacheEntry(status="dead_end", tried_mask=0b111, best_score=-1e9, depth=depth))

    def stats(self) -> Dict[str, int]:
        return {
            "cache_lookups": self.lookups,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_writes": self.writes,
            "cache_pruned_states": self.pruned_states,
            "cache_evictions": self.evictions,
            "cache_storage_bytes": self.storage_bytes,
            "cache_charged_reads": self.charged_lookup_reads,
        }
