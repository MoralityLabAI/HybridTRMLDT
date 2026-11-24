"""Batched vectorized evaluation harness, deterministic baselines, and gate checker for MCP-07.

Implements all experimental arms (A-H), strict physical-slot ID deduplication,
conservative cost accounting, paired bootstrap confidence intervals, and
intervention hooks shared with `mechanistic.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .curriculum import FAMILIES, QueryRoot
from .memory import (
    ALL_PROJECTIONS,
    ANCHOR_LOOKUP_COST,
    PAGE,
    PROJ_TO_ID,
    STRIDES,
    STRIDES_NP,
    STRUCTURED_PROJECTIONS,
    SWITCH_OVERHEAD_COST,
    ZERO_STRIDE_IDX,
    SharedPhysicalMemory,
)
from .models import (
    NUM_STRIDES,
    NUM_STRUCT_PROJS,
    AdaptiveGeometryController,
    build_cand_feats_batch,
    build_init_obs_batch,
    build_step_obs_batch,
)
from .train import _choose_init_proj_set, _goal_info, _state_cost_to_go

DETERMINISTIC_FIXED_ARMS: Tuple[str, ...] = (
    "det_row",
    "det_col",
    "det_hilbert",
    "det_morton",
    "det_rand_coloc",
    "det_rand_full",
)
LEARNED_FIXED_ARMS: Tuple[str, ...] = (
    "fixed_row",
    "fixed_col",
    "fixed_hilbert",
    "fixed_morton",
    "fixed_rand_coloc",
    "fixed_rand_full",
)
CORE_ROUTED_ARMS: Tuple[str, ...] = (
    "det_router",
    "router_single",
    "recurrent_switch",
    "hier_adaptive",
    "rand_routing",
    "oracle_proj",
)
ALL_REGISTERED_ARMS: Tuple[str, ...] = (
    DETERMINISTIC_FIXED_ARMS + ("det_router",) + LEARNED_FIXED_ARMS + (
        "router_single",
        "recurrent_switch",
        "hier_adaptive",
        "rand_routing",
        "oracle_proj",
    )
)


@dataclass
class InterventionSpec:
    """Optional Section 7 mechanistic intervention applied during rollout."""

    name: str = "none"
    freeze_router: bool = False
    shuffle_slow_state: bool = False
    disable_comm: bool = False
    force_wrong_proj: bool = False
    patch_slow_donor: Optional[np.ndarray] = None  # [T+1, B, d_s]
    seed: int = 0


def _det_target_span(scale: float) -> float:
    if scale <= 0.1:
        return 0.0
    if scale <= 0.75:
        return 8.0
    return 44.0


def _det_router_choose_proj(
    ax: int,
    ay: int,
    dx_ref: float,
    dy_ref: float,
    hx: int,
    hy: int,
    sx: float,
    sy: float,
    step: int,
    width: int,
    height: int,
) -> int:
    """Geometry-aware deterministic router (Arm B) using strictly observable features."""
    if sy < 0.1:
        return PROJ_TO_ID["row"]
    if sx < 0.1:
        return PROJ_TO_ID["col"]
    if sx > 0.75 and (dx_ref * float(hx)) < 48.0:
        return PROJ_TO_ID["row"]
    if sy > 0.75 and (dy_ref * float(hy)) < 48.0:
        return PROJ_TO_ID["col"]
    # Anti-diagonal Z-corner detection from purely observable anchor phase + signs
    if hx * hy < 0 and sx <= 0.75 and sy <= 0.75:
        x16, y16 = ax % 16, ay % 16
        x32, y32 = ax % 32, ay % 32
        near_q1_16 = (x16 >= 12) and (4 <= y16 < 8)
        near_q2_16 = (x16 < 4) and (8 <= y16 < 12)
        near_q1_32 = (x32 >= 28) and (12 <= y32 < 16)
        near_q2_32 = (x32 < 4) and (16 <= y32 < 20)
        if near_q1_16 or near_q2_16 or near_q1_32 or near_q2_32:
            return PROJ_TO_ID["morton"]
    return PROJ_TO_ID["hilbert"]


def _det_choose_stride_batch(
    mem: SharedPhysicalMemory,
    next_projs: np.ndarray,
    cands: np.ndarray,
    legal: np.ndarray,
    ref_xy: np.ndarray,
    hx: np.ndarray,
    hy: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    anchor_pages: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Vectorized non-oracle deterministic centroid-progress stride selector."""
    bsz = len(next_projs)
    chosen = np.full(bsz, -1, dtype=np.int32)
    act_idx = np.flatnonzero(active_mask & legal.any(axis=1))
    if len(act_idx) == 0:
        return chosen

    span_x = np.where(sx[act_idx] <= 0.1, 0.0, np.where(sx[act_idx] <= 0.75, 8.0, 56.0))
    span_y = np.where(sy[act_idx] <= 0.1, 0.0, np.where(sy[act_idx] <= 0.75, 8.0, 56.0))
    est_tx = np.clip(ref_xy[act_idx, 0] + hx[act_idx] * span_x, 0.0, float(mem.width - 1))
    est_ty = np.clip(ref_xy[act_idx, 1] + hy[act_idx] * span_y, 0.0, float(mem.height - 1))
    wx = np.where(sx[act_idx] < 0.1, 4.0, 1.0)
    wy = np.where(sy[act_idx] < 0.1, 4.0, 1.0)

    safe_cands = np.clip(cands[act_idx], 0, mem.pages - 1)  # [A, 13]
    cx = np.empty_like(safe_cands, dtype=np.float32)
    cy = np.empty_like(safe_cands, dtype=np.float32)
    sub_projs = next_projs[act_idx]
    for pid in np.unique(sub_projs):
        m = sub_projs == pid
        ctrs = mem.views[ALL_PROJECTIONS[int(pid)]].centers[safe_cands[m]]
        cx[m] = ctrs[:, :, 0]
        cy[m] = ctrs[:, :, 1]

    dist = (
        wx[:, None] * np.abs(cx - est_tx[:, None])
        + wy[:, None] * np.abs(cy - est_ty[:, None])
        + 0.02 * np.abs(safe_cands - anchor_pages[act_idx, None])
    )
    dist = np.where(legal[act_idx], dist, 1e18)
    chosen[act_idx] = np.argmin(dist, axis=1).astype(np.int32)
    return chosen


def _oracle_choose_next_proj(
    mem: SharedPhysicalMemory,
    root: QueryRoot,
    hop_idx: int,
    stale_seen: bool,
    cur_proj_name: str,
    cur_page: int,
    visited: Dict[str, set],
    reanchor_slot: int,
) -> int:
    """Privileged oracle projection selector (Arm H upper bound ONLY)."""
    hop = root.hops[hop_idx]
    goal_slots, tx, ty = _goal_info(mem, hop, stale_seen=stale_seen)
    best_pid = 0
    best_cost = 1e18
    for p_name in STRUCTURED_PROJECTIONS:
        if reanchor_slot >= 0:
            base_p = int(mem.views[p_name].inv[reanchor_slot])
            allow_zero = True
        elif p_name != cur_proj_name:
            cslot = int(mem.views[cur_proj_name].center_slot[cur_page])
            base_p = int(mem.views[p_name].inv[cslot])
            allow_zero = True
        else:
            base_p = cur_page
            allow_zero = False
        seen = visited[p_name]
        p_min = 1e18
        for d_idx, d_val in enumerate(STRIDES):
            if d_idx == ZERO_STRIDE_IDX and not allow_zero:
                continue
            q = base_p + d_val
            if 0 <= q < mem.pages and q not in seen:
                c = 1.0 + 0.15 * float(abs(d_val)) + _state_cost_to_go(
                    mem, p_name, q, goal_slots, tx, ty, STRUCTURED_PROJECTIONS
                )
                if c < p_min:
                    p_min = c
        if p_min < best_cost:
            best_cost = p_min
            best_pid = PROJ_TO_ID[p_name]
    return best_pid


def _batch_compute_candidates(
    mem: SharedPhysicalMemory,
    cur_projs: np.ndarray,
    cur_pages: np.ndarray,
    next_projs: np.ndarray,
    visited_list: List[Dict[int, set]],
    reanchor_slots: np.ndarray,
    active_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized candidate page and legal mask computation for a batch of B episodes."""
    bsz = len(cur_pages)
    base_pages = np.zeros(bsz, dtype=np.int32)
    allow_zero = np.zeros(bsz, dtype=bool)
    views = [mem.views[p] for p in ALL_PROJECTIONS]

    for i in np.flatnonzero(active_mask):
        ii = int(i)
        np_id = int(next_projs[ii])
        if reanchor_slots[ii] >= 0:
            base_pages[ii] = int(views[np_id].inv[int(reanchor_slots[ii])])
            allow_zero[ii] = True
        elif np_id != int(cur_projs[ii]):
            cslot = int(views[int(cur_projs[ii])].center_slot[int(cur_pages[ii])])
            base_pages[ii] = int(views[np_id].inv[cslot])
            allow_zero[ii] = True
        else:
            base_pages[ii] = int(cur_pages[ii])
            allow_zero[ii] = False

    cands = base_pages[:, None] + STRIDES_NP[None, :]  # [B, 13]
    legal = (cands >= 0) & (cands < mem.pages)
    legal[~allow_zero, ZERO_STRIDE_IDX] = False

    for i in np.flatnonzero(active_mask):
        ii = int(i)
        seen = visited_list[ii][int(next_projs[ii])]
        if seen:
            row_c = cands[ii]
            row_l = legal[ii]
            for d in range(NUM_STRIDES):
                if row_l[d] and int(row_c[d]) in seen:
                    row_l[d] = False
    return cands, legal, base_pages


@torch.inference_mode()
def rollout_batch(
    mem: SharedPhysicalMemory,
    roots: Sequence[QueryRoot],
    arm: str,
    model: Optional[AdaptiveGeometryController],
    budget: int,
    corruption: float,
    matched_rand_profile: Optional[Dict[str, Any]] = None,
    intervention: Optional[InterventionSpec] = None,
    record_trajectories: bool = False,
    rng_seed: int = 0,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, np.ndarray]]]:
    """Run batched rollout of `roots` under `arm` with full physical-ID & cost tracking."""
    bsz = len(roots)
    if bsz == 0:
        return [], None

    iv = intervention if intervention is not None else InterventionSpec()
    is_det_fixed = arm in DETERMINISTIC_FIXED_ARMS
    is_det_router = arm == "det_router"
    is_det = is_det_fixed or is_det_router
    is_learned_fixed = arm in LEARNED_FIXED_ARMS
    is_single_router = arm == "router_single"
    is_adaptive = arm == "hier_adaptive"
    is_rand_routing = arm == "rand_routing"
    is_oracle = arm == "oracle_proj"

    fixed_proj_id = -1
    if is_det_fixed:
        fixed_proj_id = PROJ_TO_ID[arm[len("det_") :]]
    elif is_learned_fixed:
        fixed_proj_id = PROJ_TO_ID[arm[len("fixed_") :]]

    views = [mem.views[p] for p in ALL_PROJECTIONS]
    rng = np.random.default_rng(rng_seed + iv.seed + 4242)

    # Matched random routing profile (defaults to uniform + empirical switch rate ~0.25 if omitted)
    if matched_rand_profile is not None:
        rand_init_probs = np.asarray(matched_rand_profile["init_probs"], dtype=np.float64)
        rand_init_probs = rand_init_probs / max(1e-12, rand_init_probs.sum())
        rand_switch_prob = float(matched_rand_profile["switch_prob"])
        rand_step_probs = np.asarray(matched_rand_profile["step_probs"], dtype=np.float64)
        rand_step_probs = rand_step_probs / max(1e-12, rand_step_probs.sum())
    else:
        rand_init_probs = np.full(NUM_STRUCT_PROJS, 0.25, dtype=np.float64)
        rand_switch_prob = 0.25
        rand_step_probs = np.full(NUM_STRUCT_PROJS, 0.25, dtype=np.float64)

    # Episode state arrays
    hop_idx = np.zeros(bsz, dtype=np.int32)
    num_hops = np.array([r.num_hops for r in roots], dtype=np.int32)
    cur_anchors = np.array([r.hops[0].anchor for r in roots], dtype=np.int32)
    cur_targets = np.array([r.hops[0].target_record for r in roots], dtype=np.int32)
    cur_keys = mem.keys[cur_targets]
    cur_decoys = np.array([r.hops[0].decoy_slot for r in roots], dtype=np.int32)
    cur_hx = np.array([r.hops[0].hint_x for r in roots], dtype=np.int32)
    cur_hy = np.array([r.hops[0].hint_y for r in roots], dtype=np.int32)
    cur_sx = np.array([r.hops[0].scale_x for r in roots], dtype=np.float32)
    cur_sy = np.array([r.hops[0].scale_y for r in roots], dtype=np.float32)
    ref_xy = np.stack(
        [(cur_anchors % mem.width).astype(np.float32), (cur_anchors // mem.width).astype(np.float32)],
        axis=1,
    )

    # Step 0: initial projection selection
    init_obs_np = build_init_obs_batch(mem, cur_anchors, cur_hx, cur_hy, cur_sx, cur_sy, budget)
    state = None
    init_router_probs = np.zeros((bsz, NUM_STRUCT_PROJS), dtype=np.float32)

    if model is not None:
        state = model.initial_state(bsz)
        init_obs_t = torch.from_numpy(init_obs_np)
        ov_s0 = torch.from_numpy(iv.patch_slow_donor[0]) if iv.patch_slow_donor is not None else None
        r0_logits, _, state = model.step_recurrent(
            init_obs_t, state, t=0, active_proj=None, disable_comm=iv.disable_comm, override_slow=ov_s0
        )
        if iv.shuffle_slow_state and bsz > 1:
            perm = torch.from_numpy(rng.permutation(bsz))
            state.s = state.s[perm].clone()
            # Recompute router logits from shuffled slow state
            comm0 = model._compute_comm(state.p_mem, None, disable_comm=iv.disable_comm)
            r0_logits = model.router_head(
                torch.cat((state.s, state.h, comm0, init_obs_t[:, 10:26]), dim=-1)
            )
        init_router_probs = torch.softmax(r0_logits, dim=-1).numpy()

    if fixed_proj_id >= 0:
        cur_projs = np.full(bsz, fixed_proj_id, dtype=np.int32)
    elif is_det_router:
        cur_projs = np.array(
            [
                _det_router_choose_proj(
                    int(cur_anchors[i] % mem.width),
                    int(cur_anchors[i] // mem.width),
                    0.0,
                    0.0,
                    int(cur_hx[i]),
                    int(cur_hy[i]),
                    float(cur_sx[i]),
                    float(cur_sy[i]),
                    0,
                    mem.width,
                    mem.height,
                )
                for i in range(bsz)
            ],
            dtype=np.int32,
        )
    elif is_rand_routing:
        cur_projs = rng.choice(NUM_STRUCT_PROJS, size=bsz, p=rand_init_probs).astype(np.int32)
    elif is_oracle:
        cur_projs = np.array(
            [_choose_init_proj_set(mem, r, STRUCTURED_PROJECTIONS, False)[1] for r in roots],
            dtype=np.int32,
        )
    else:
        assert model is not None
        if iv.force_wrong_proj:
            cur_projs = np.argmin(init_router_probs, axis=1).astype(np.int32)
        else:
            cur_projs = np.argmax(init_router_probs, axis=1).astype(np.int32)

    init_projs_saved = cur_projs.copy()

    anchor_pages = np.array([views[int(cur_projs[i])].inv[int(cur_anchors[i])] for i in range(bsz)], dtype=np.int32)
    cur_pages = anchor_pages.copy()
    prev_pages = anchor_pages.copy()

    active = np.ones(bsz, dtype=bool)
    solved_all = np.zeros(bsz, dtype=bool)
    stale_seen_hop = np.zeros(bsz, dtype=bool)
    stale_ever_ep = np.zeros(bsz, dtype=bool)
    premature_stop = np.zeros(bsz, dtype=bool)

    probes = np.zeros(bsz, dtype=np.int32)
    matched_good = np.zeros(bsz, dtype=np.int32)
    switches = np.zeros(bsz, dtype=np.int32)
    anchor_lookups = np.ones(bsz, dtype=np.int32)
    fast_updates = np.ones(bsz, dtype=np.int32) if model is not None else np.zeros(bsz, dtype=np.int32)
    slow_updates = np.ones(bsz, dtype=np.int32) if model is not None else np.zeros(bsz, dtype=np.int32)
    proj_counts = np.zeros((bsz, len(ALL_PROJECTIONS)), dtype=np.int32)

    visited_pages: List[Dict[int, set]] = [{p: set() for p in range(len(ALL_PROJECTIONS))} for _ in range(bsz)]
    physical_slots_seen: List[set] = [set() for _ in range(bsz)]
    final_values = np.full(bsz, -1, dtype=np.int64)

    # Optional trajectory buffers for Section 7 mechanistic analysis
    traj_router_probs = None
    traj_chosen_proj = None
    traj_slow = None
    traj_fast = None
    traj_valid = None
    traj_switch = None
    traj_solved = None
    traj_hop = None
    if record_trajectories and model is not None:
        d_s = model.cfg.d_s
        d_h = model.cfg.d_h
        traj_router_probs = np.zeros((budget + 1, bsz, NUM_STRUCT_PROJS), dtype=np.float32)
        traj_chosen_proj = np.full((budget + 1, bsz), -1, dtype=np.int32)
        traj_slow = np.zeros((budget + 1, bsz, d_s), dtype=np.float32)
        traj_fast = np.zeros((budget + 1, bsz, d_h), dtype=np.float32)
        traj_valid = np.zeros((budget + 1, bsz), dtype=bool)
        traj_switch = np.zeros((budget + 1, bsz), dtype=bool)
        traj_solved = np.zeros((budget + 1, bsz), dtype=bool)
        traj_hop = np.zeros((budget + 1, bsz), dtype=np.int32)

        traj_router_probs[0] = init_router_probs
        traj_chosen_proj[0] = cur_projs
        traj_slow[0] = state.s.numpy()
        traj_fast[0] = state.h.numpy()
        traj_valid[0] = True

    for t in range(budget):
        if not active.any():
            break
        act_idx = np.flatnonzero(active)

        # Probe active episodes in a single vectorized call
        hit_valid_a, vals_a, stale_a, good_a, slots_a = mem.probe_batch(
            cur_projs[act_idx],
            cur_pages[act_idx],
            cur_keys[act_idx],
            corruption,
            cur_decoys[act_idx],
            ALL_PROJECTIONS,
        )

        hit_cur = np.zeros(bsz, dtype=bool)
        stale_cur = np.zeros(bsz, dtype=bool)
        hop_adv = np.zeros(bsz, dtype=bool)
        reanchor_slots = np.full(bsz, -1, dtype=np.int32)

        for local_k, gi in enumerate(act_idx):
            i = int(gi)
            pid = int(cur_projs[i])
            pg = int(cur_pages[i])
            visited_pages[i][pid].add(pg)
            probes[i] += 1
            proj_counts[i, pid] += 1
            matched_good[i] += int(good_a[local_k])
            physical_slots_seen[i].update(slots_a[local_k].tolist())

            hv = bool(hit_valid_a[local_k]) and (int(vals_a[local_k]) == int(mem.values[int(cur_targets[i])]))
            st = bool(stale_a[local_k])
            hit_cur[i] = hv
            stale_cur[i] = st
            if st:
                stale_ever_ep[i] = True

            if hv and not solved_all[i]:
                if hop_idx[i] + 1 < num_hops[i]:
                    hop_idx[i] += 1
                    nxt_hop = roots[i].hops[int(hop_idx[i])]
                    hop_adv[i] = True
                    stale_seen_hop[i] = False
                    cur_anchors[i] = nxt_hop.anchor
                    cur_targets[i] = nxt_hop.target_record
                    cur_keys[i] = mem.keys[nxt_hop.target_record]
                    cur_decoys[i] = nxt_hop.decoy_slot
                    cur_hx[i] = nxt_hop.hint_x
                    cur_hy[i] = nxt_hop.hint_y
                    cur_sx[i] = nxt_hop.scale_x
                    cur_sy[i] = nxt_hop.scale_y
                    ref_xy[i, 0] = float(nxt_hop.anchor % mem.width)
                    ref_xy[i, 1] = float(nxt_hop.anchor // mem.width)
                    reanchor_slots[i] = nxt_hop.anchor
                    anchor_lookups[i] += ANCHOR_LOOKUP_COST
                    visited_pages[i] = {p: set() for p in range(len(ALL_PROJECTIONS))}
                else:
                    solved_all[i] = True
                    final_values[i] = int(vals_a[local_k])
            elif st and cur_decoys[i] >= 0 and not stale_seen_hop[i] and not solved_all[i]:
                stale_seen_hop[i] = True
                h_spec = roots[i].hops[int(hop_idx[i])]
                cur_hx[i] = h_spec.mirror_hint_x
                cur_hy[i] = h_spec.mirror_hint_y
                cur_sx[i] = h_spec.mirror_scale_x
                cur_sy[i] = h_spec.mirror_scale_y
                c = views[pid].centers[pg]
                ref_xy[i, 0] = float(c[0])
                ref_xy[i, 1] = float(c[1])

        # Deterministic arms halt immediately on verified episode completion
        if is_det:
            active[solved_all] = False
            if not active.any() or t == budget - 1:
                break
            next_projs = cur_projs.copy()
            if is_det_router:
                for i in np.flatnonzero(active):
                    ii = int(i)
                    c = views[int(cur_projs[ii])].centers[int(cur_pages[ii])]
                    dx_r = 0.0 if hop_adv[ii] else float(c[0]) - float(ref_xy[ii, 0])
                    dy_r = 0.0 if hop_adv[ii] else float(c[1]) - float(ref_xy[ii, 1])
                    next_projs[ii] = _det_router_choose_proj(
                        int(cur_anchors[ii] % mem.width),
                        int(cur_anchors[ii] // mem.width),
                        dx_r,
                        dy_r,
                        int(cur_hx[ii]),
                        int(cur_hy[ii]),
                        float(cur_sx[ii]),
                        float(cur_sy[ii]),
                        t + 1,
                        mem.width,
                        mem.height,
                    )
            cands, legal, bpages = _batch_compute_candidates(
                mem, cur_projs, cur_pages, next_projs, visited_pages, reanchor_slots, active
            )
            next_anchor_pages = anchor_pages.copy()
            for i in np.flatnonzero(active):
                ii = int(i)
                next_anchor_pages[ii] = int(views[int(next_projs[ii])].inv[int(cur_anchors[ii])])
            chosen_strides = _det_choose_stride_batch(
                mem,
                next_projs,
                cands,
                legal,
                ref_xy,
                cur_hx,
                cur_hy,
                cur_sx,
                cur_sy,
                next_anchor_pages,
                active,
            )
            for i in np.flatnonzero(active):
                ii = int(i)
                d_i = int(chosen_strides[ii])
                if d_i < 0:
                    active[ii] = False
                    continue
                np_id = int(next_projs[ii])
                if np_id != int(cur_projs[ii]):
                    switches[ii] += 1
                    prev_pages[ii] = int(bpages[ii])
                else:
                    prev_pages[ii] = int(cur_pages[ii])
                cur_projs[ii] = np_id
                cur_pages[ii] = int(cands[ii, d_i])
                anchor_pages[ii] = int(next_anchor_pages[ii])
            continue

        # Neural arms: build step observation and step coupled recurrent controller
        assert model is not None and state is not None
        step_obs_np = build_step_obs_batch(
            mem,
            cur_projs,
            cur_pages,
            anchor_pages,
            prev_pages,
            cur_anchors,
            ref_xy,
            cur_hx,
            cur_hy,
            cur_sx,
            cur_sy,
            np.full(bsz, budget - t, dtype=np.int32),
            t,
            hit_cur,
            solved_all,
            stale_cur,
            stale_seen_hop,
            hop_idx,
            hop_adv,
            ALL_PROJECTIONS,
        )
        step_obs_t = torch.from_numpy(step_obs_np)
        act_proj_t = torch.from_numpy(np.minimum(cur_projs, NUM_STRUCT_PROJS - 1).astype(np.int64))
        ov_st = torch.from_numpy(iv.patch_slow_donor[t + 1]) if iv.patch_slow_donor is not None else None

        prev_slow_cnt = state.slow_updates
        r_logits, stop_logit, state = model.step_recurrent(
            step_obs_t,
            state,
            t=t + 1,
            active_proj=act_proj_t,
            disable_comm=iv.disable_comm,
            override_slow=ov_st,
        )
        did_slow = state.slow_updates > prev_slow_cnt
        fast_updates[active] += 1
        if did_slow:
            slow_updates[active] += 1

        if iv.shuffle_slow_state and bsz > 1:
            perm = torch.from_numpy(rng.permutation(bsz))
            state.s = state.s[perm].clone()
            comm_now = model._compute_comm(state.p_mem, act_proj_t, disable_comm=iv.disable_comm)
            r_logits = model.router_head(
                torch.cat((state.s, state.h, comm_now, step_obs_t[:, 10:26]), dim=-1)
            )
            stop_ctx = torch.stack(
                (step_obs_t[:, 28], step_obs_t[:, 29], step_obs_t[:, 30], step_obs_t[:, 33]), dim=-1
            )
            stop_logit = model.stop_head(torch.cat((state.h, state.s, stop_ctx), dim=-1)).squeeze(-1)

        r_probs = torch.softmax(r_logits, dim=-1).numpy()

        # Adaptive halting (Arm F)
        if is_adaptive:
            stop_now = active & (stop_logit.numpy() >= 0.0)
            if stop_now.any():
                premature_stop[stop_now & (~solved_all)] = True
                active[stop_now] = False

        # Choose next projection
        if fixed_proj_id >= 0:
            next_projs = np.full(bsz, fixed_proj_id, dtype=np.int32)
        elif is_single_router or iv.freeze_router:
            next_projs = init_projs_saved.copy()
        elif is_rand_routing:
            next_projs = cur_projs.copy()
            do_sw = active & (rng.random(bsz) < rand_switch_prob)
            if do_sw.any():
                for ii in np.flatnonzero(do_sw):
                    # Pick a different projection proportional to matched marginal
                    p_vec = rand_step_probs.copy()
                    p_vec[int(cur_projs[ii])] = 0.0
                    s_p = p_vec.sum()
                    if s_p <= 1e-12:
                        p_vec = np.ones(NUM_STRUCT_PROJS) / float(NUM_STRUCT_PROJS)
                    else:
                        p_vec /= s_p
                    next_projs[ii] = int(rng.choice(NUM_STRUCT_PROJS, p=p_vec))
        elif is_oracle:
            next_projs = cur_projs.copy()
            for ii in np.flatnonzero(active):
                i = int(ii)
                if solved_all[i]:
                    continue
                vis_str = {ALL_PROJECTIONS[k]: v for k, v in visited_pages[i].items()}
                next_projs[i] = _oracle_choose_next_proj(
                    mem,
                    roots[i],
                    int(hop_idx[i]),
                    bool(stale_seen_hop[i]),
                    ALL_PROJECTIONS[int(cur_projs[i])],
                    int(cur_pages[i]),
                    vis_str,
                    int(reanchor_slots[i]),
                )
        else:
            if iv.force_wrong_proj:
                next_projs = np.argmin(r_probs, axis=1).astype(np.int32)
            else:
                next_projs = np.argmax(r_probs, axis=1).astype(np.int32)

        if record_trajectories and traj_router_probs is not None:
            traj_router_probs[t + 1] = r_probs
            traj_chosen_proj[t + 1] = next_projs
            traj_slow[t + 1] = state.s.numpy()
            traj_fast[t + 1] = state.h.numpy()
            traj_valid[t + 1] = active
            traj_switch[t + 1] = active & (next_projs != cur_projs)
            traj_solved[t + 1] = solved_all
            traj_hop[t + 1] = hop_idx

        if not active.any() or t == budget - 1:
            break

        cands, legal, bpages = _batch_compute_candidates(
            mem, cur_projs, cur_pages, next_projs, visited_pages, reanchor_slots, active
        )
        # Fallback if a switched projection has no legal moves: allow staying/any legal
        dead = active & (~legal.any(axis=1))
        if dead.any():
            active[dead] = False
        if not active.any():
            break

        next_proj_t = torch.from_numpy(np.minimum(next_projs, NUM_STRUCT_PROJS - 1).astype(np.int64))
        legal_t = torch.from_numpy(legal)
        cand_feats_np = build_cand_feats_batch(
            mem, cur_projs, cur_pages, next_projs, cands, legal, ref_xy, cur_hx, cur_hy, ALL_PROJECTIONS
        )
        s_logits, _ = model.predict_stride(
            state, step_obs_t, next_proj_t, torch.from_numpy(cand_feats_np)
        )
        masked_s = s_logits.masked_fill(~legal_t, -1e9)
        chosen_stride = masked_s.argmax(dim=-1).numpy()

        for ii in np.flatnonzero(active):
            i = int(ii)
            np_id = int(next_projs[i])
            if np_id != int(cur_projs[i]):
                switches[i] += 1
                prev_pages[i] = int(bpages[i])
            else:
                prev_pages[i] = int(cur_pages[i])
            cur_projs[i] = np_id
            cur_pages[i] = int(cands[i, int(chosen_stride[i])])
            anchor_pages[i] = int(views[np_id].inv[int(cur_anchors[i])])

    # Assemble per-root receipts
    receipts: List[Dict[str, Any]] = []
    for i in range(bsz):
        raw_reads = int(PAGE * probes[i])
        uniq_slots = len(physical_slots_seen[i])
        uniq_recs = len({s >> 1 for s in physical_slots_seen[i]})
        charged = int(raw_reads + anchor_lookups[i] + SWITCH_OVERHEAD_COST * switches[i])
        receipts.append(
            {
                "family": roots[i].family,
                "success": bool(solved_all[i]),
                "value": int(final_values[i]) if solved_all[i] else None,
                "probes": int(probes[i]),
                "raw_slot_reads": raw_reads,
                "unique_physical_slots": uniq_slots,
                "overlap_slot_reads": int(raw_reads - uniq_slots),
                "unique_logical_records": uniq_recs,
                "anchor_lookups": int(anchor_lookups[i]),
                "switches": int(switches[i]),
                "charged_reads": charged,
                "matched_good": int(matched_good[i]),
                "stale_seen": bool(stale_ever_ep[i]),
                "recovered": bool(stale_ever_ep[i] and solved_all[i]),
                "premature_stop": bool(premature_stop[i]),
                "fast_updates": int(fast_updates[i]),
                "slow_updates": int(slow_updates[i]),
                "recurrent_updates": int(fast_updates[i] + slow_updates[i]),
                "init_proj": int(init_projs_saved[i]),
                "proj_counts": [int(x) for x in proj_counts[i, :4]],
            }
        )

    traj = None
    if record_trajectories and traj_router_probs is not None:
        traj = {
            "router_probs": traj_router_probs,
            "chosen_proj": traj_chosen_proj,
            "slow": traj_slow,
            "fast": traj_fast,
            "valid": traj_valid,
            "switch": traj_switch,
            "solved": traj_solved,
            "hop": traj_hop,
        }
    return receipts, traj


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 20261124,
    relative: bool = False,
) -> Dict[str, float]:
    """Compute paired bootstrap mean difference (or relative reduction) with 95% CI."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape and len(a) > 0
    n = len(a)
    if relative:
        point = float((a.mean() - b.mean()) / max(1e-12, a.mean()))
    else:
        point = float(np.mean(a - b))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    if relative:
        ma = a[idx].mean(axis=1)
        mb = b[idx].mean(axis=1)
        samples = (ma - mb) / np.maximum(1e-12, ma)
    else:
        samples = (a[idx] - b[idx]).mean(axis=1)

    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {
        "point": point,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "prob_gt_zero": float(np.mean(samples > 0.0)),
        "n_pairs": int(n),
    }


def extract_matched_rand_profile(receipts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute empirical initial-projection and switch-rate profile from Arm E receipts."""
    init_c = np.zeros(NUM_STRUCT_PROJS, dtype=np.float64)
    step_c = np.zeros(NUM_STRUCT_PROJS, dtype=np.float64)
    total_sw = 0
    total_steps = 0
    for rec in receipts:
        ip = int(rec["init_proj"])
        if 0 <= ip < NUM_STRUCT_PROJS:
            init_c[ip] += 1.0
        step_c += np.asarray(rec["proj_counts"], dtype=np.float64)
        total_sw += int(rec["switches"])
        total_steps += max(1, int(rec["probes"]) - 1)
    init_probs = (init_c + 1e-6) / (init_c.sum() + 4e-6)
    step_probs = (step_c + 1e-6) / (step_c.sum() + 4e-6)
    return {
        "init_probs": init_probs.tolist(),
        "step_probs": step_probs.tolist(),
        "switch_prob": float(min(0.95, total_sw / max(1, total_steps))),
    }
