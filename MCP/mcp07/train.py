"""Verified compact-span shortest-path trace teacher, 16K DAgger aggregation, and trainer for MCP-07.

Supports:
- Optimal-action-set supervision (`action_set`) and single-best cross-entropy (`single_best`)
- On-policy 16K verified trace aggregation (DAgger) applied identically across all learned arms
- Fixed-layout learned controllers (Arm C)
- Single-selection-per-episode learned router (Arm D)
- Full recurrent projection-switching metacontroller (Arm E / F)
- Parameter-matched 54K scale and scaled 116K (<=128K) variant
- Distribution-shift training mixtures
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .curriculum import QueryRoot, sample_mixed_roots
from .memory import (
    ALL_PROJECTIONS,
    FINE_MASK_NP,
    PROJ_TO_ID,
    STRIDES,
    STRIDES_NP,
    STRUCTURED_PROJECTIONS,
    SharedPhysicalMemory,
)
from .models import (
    CAND_FEAT_DIM,
    CONFIG_54K,
    CONFIG_54K_FIXED,
    NUM_STRIDES,
    NUM_STRUCT_PROJS,
    OBS_DIM,
    AdaptiveGeometryController,
    ModelConfig,
    build_cand_feats_single,
    compute_candidates_single,
    optimal_action_set_loss,
)

# Weight on raw 1D page span so compact capacity-invariant geometries beat lucky large oracle jumps
SPAN_WEIGHT: float = 0.15
# Realistic switch penalty on non-event steps to prevent noisy oracle projection thrashing
SWITCH_PENALTY: float = 0.75
# Cost tolerance (in step-equivalents) for declaring two projections/moves tied in optimal set A*
SET_TIE_TOL: float = 0.25


def _goal_info(mem: SharedPhysicalMemory, hop, stale_seen: bool) -> Tuple[List[int], int, int]:
    """Return causally consistent active goal slots and target (tx, ty) for teacher."""
    if hop.decoy_slot >= 0 and not stale_seen:
        s = hop.decoy_slot
        return [s], s % mem.width, s // mem.width
    s0 = 2 * hop.target_record
    return [s0, s0 + 1], s0 % mem.width, s0 // mem.width


def _min_page_and_stride(
    mem: SharedPhysicalMemory, proj: str, page: int, goal_slots: List[int]
) -> Tuple[int, int]:
    inv = mem.views[proj].inv
    sd = mem.stride_dist
    best_dp = 100000
    best_st = 10000
    for gs in goal_slots:
        dp = abs(page - int(inv[gs]))
        st = int(sd[dp])
        if dp < best_dp:
            best_dp = dp
        if st < best_st:
            best_st = st
    return best_dp, best_st


def _compact_1d_cost(dp: int, steps: int) -> float:
    if dp == 0:
        return 0.0
    return float(steps) + SPAN_WEIGHT * float(dp)


def _state_cost_to_go(
    mem: SharedPhysicalMemory,
    proj: str,
    page: int,
    goal_slots: List[int],
    tx: int,
    ty: int,
    allowed_projs: Tuple[str, ...],
) -> float:
    """Estimate compact-span cost-to-go from (proj, page) to active goal."""
    dp1, st1 = _min_page_and_stride(mem, proj, page, goal_slots)
    if dp1 == 0:
        return 0.0

    best = _compact_1d_cost(dp1, st1)
    if len(allowed_projs) == 1:
        return best

    cslot = int(mem.views[proj].center_slot[page])
    for other in allowed_projs:
        if other == proj:
            continue
        op = int(mem.views[other].inv[cslot])
        dp_o, st_o = _min_page_and_stride(mem, other, op, goal_slots)
        cand = 1.0 if dp_o == 0 else 0.35 + _compact_1d_cost(dp_o, st_o)
        if cand < best:
            best = cand

    cx, cy = mem.views[proj].centers[page]
    dx = abs(float(tx) - float(cx))
    dy = abs(float(ty) - float(cy))
    x_pages = int(math.ceil(max(0.0, dx - 10.0) / 32.0))
    y_pages = int(math.ceil(max(0.0, dy - 10.0) / 32.0))
    rem_local = 0.0
    if dx > 2.0 or dy > 2.0:
        rem_local = 1.15 + 0.03 * (min(dx, 14.0) + min(dy, 14.0))
    two_axis = float(x_pages) * 1.15 + float(y_pages) * 1.15 + rem_local + 0.015 * (dx + dy)
    if two_axis < best:
        best = two_axis

    return best


def _choose_init_proj_set(
    mem: SharedPhysicalMemory,
    root: QueryRoot,
    allowed_projs: Tuple[str, ...],
    single_selection_only: bool,
) -> Tuple[np.ndarray, int]:
    """Return (optimal_proj_mask[4], chosen_proj_idx_in_ALL) for step 0."""
    hop = root.hops[0]
    goal_slots, tx, ty = _goal_info(mem, hop, stale_seen=False)

    scores: Dict[str, float] = {}
    for p_name in allowed_projs:
        p0 = int(mem.views[p_name].inv[hop.anchor])
        if single_selection_only:
            dp0, st0 = _min_page_and_stride(mem, p_name, p0, goal_slots)
            tot = _compact_1d_cost(dp0, st0)
            if root.num_hops > 1:
                h2 = root.hops[1]
                g2, _, _ = _goal_info(mem, h2, stale_seen=False)
                dp2, st2 = _min_page_and_stride(
                    mem, p_name, int(mem.views[p_name].inv[h2.anchor]), g2
                )
                tot += _compact_1d_cost(dp2, st2)
            elif hop.decoy_slot >= 0:
                gm, _, _ = _goal_info(mem, hop, stale_seen=True)
                dpm, stm = _min_page_and_stride(
                    mem, p_name, int(mem.views[p_name].inv[hop.decoy_slot]), gm
                )
                tot += _compact_1d_cost(dpm, stm)
            scores[p_name] = tot
        else:
            dp0, st0 = _min_page_and_stride(mem, p_name, p0, goal_slots)
            if dp0 == 0:
                scores[p_name] = 0.0
            else:
                best_after = 1e9
                for d_val in STRIDES:
                    if d_val == 0:
                        continue
                    q = p0 + d_val
                    if 0 <= q < mem.pages:
                        move_cost = 1.0 + SPAN_WEIGHT * float(abs(d_val))
                        rem = _state_cost_to_go(mem, p_name, q, goal_slots, tx, ty, allowed_projs)
                        c = move_cost + rem
                        if c < best_after:
                            best_after = c
                scores[p_name] = best_after

    min_sc = min(scores.values())
    opt_mask = np.zeros(NUM_STRUCT_PROJS, dtype=bool)
    best_name = min(allowed_projs, key=lambda pn: scores[pn])
    for p_name in allowed_projs:
        if scores[p_name] <= min_sc + SET_TIE_TOL and p_name in STRUCTURED_PROJECTIONS:
            opt_mask[PROJ_TO_ID[p_name]] = True
    if not opt_mask.any():
        opt_mask[0] = True
    return opt_mask, PROJ_TO_ID[best_name]


@dataclass
class TraceData:
    """Packed numpy tensors for a batch of verified traces."""

    init_obs: np.ndarray            # float32[N, 32]
    init_proj_opt: np.ndarray       # bool[N, 4]
    init_proj_best: np.ndarray      # int64[N]
    step_obs: np.ndarray            # float32[N, T, 32]
    active_proj: np.ndarray         # int64[N, T]
    next_proj_opt: np.ndarray       # bool[N, T, 4]
    next_proj_best: np.ndarray      # int64[N, T]
    stride_proj: np.ndarray         # int64[N, T] projection matching cand_feats/stride_opt
    cand_feats: np.ndarray          # float32[N, T, 13, 6]
    stride_legal: np.ndarray        # bool[N, T, 13]
    stride_opt: np.ndarray          # bool[N, T, 13]
    stride_best: np.ndarray         # int64[N, T]
    res_opt: np.ndarray             # bool[N, T, 2]
    stop_target: np.ndarray         # float32[N, T]
    step_valid: np.ndarray          # bool[N, T]
    move_valid: np.ndarray          # bool[N, T]
    teacher_verified_rate: float


def concat_traces(a: TraceData, b: TraceData) -> TraceData:
    return TraceData(
        init_obs=np.concatenate([a.init_obs, b.init_obs], axis=0),
        init_proj_opt=np.concatenate([a.init_proj_opt, b.init_proj_opt], axis=0),
        init_proj_best=np.concatenate([a.init_proj_best, b.init_proj_best], axis=0),
        step_obs=np.concatenate([a.step_obs, b.step_obs], axis=0),
        active_proj=np.concatenate([a.active_proj, b.active_proj], axis=0),
        next_proj_opt=np.concatenate([a.next_proj_opt, b.next_proj_opt], axis=0),
        next_proj_best=np.concatenate([a.next_proj_best, b.next_proj_best], axis=0),
        stride_proj=np.concatenate([a.stride_proj, b.stride_proj], axis=0),
        cand_feats=np.concatenate([a.cand_feats, b.cand_feats], axis=0),
        stride_legal=np.concatenate([a.stride_legal, b.stride_legal], axis=0),
        stride_opt=np.concatenate([a.stride_opt, b.stride_opt], axis=0),
        stride_best=np.concatenate([a.stride_best, b.stride_best], axis=0),
        res_opt=np.concatenate([a.res_opt, b.res_opt], axis=0),
        stop_target=np.concatenate([a.stop_target, b.stop_target], axis=0),
        step_valid=np.concatenate([a.step_valid, b.step_valid], axis=0),
        move_valid=np.concatenate([a.move_valid, b.move_valid], axis=0),
        teacher_verified_rate=a.teacher_verified_rate,
    )


def _fill_init_obs_row(
    out: np.ndarray,
    w: int,
    h: int,
    anchor: int,
    hx: int,
    hy: int,
    sx: float,
    sy: float,
    budget: int,
) -> None:
    ax = float(anchor % w)
    ay = float(anchor // w)
    out[8] = ax / float(w)
    out[9] = ay / float(h)
    out[10] = (ax % 16.0) / 16.0
    out[11] = (ay % 16.0) / 16.0
    out[12] = (ax % 32.0) / 32.0
    out[13] = (ay % 32.0) / 32.0
    out[18] = float(hx)
    out[19] = float(hy)
    out[20] = float(sx)
    out[21] = float(sy)
    out[22] = float(hx * hy)
    out[23] = abs(float(sx) - float(sy))
    out[26] = float(budget) / 32.0


def _fill_step_obs_row(
    out: np.ndarray,
    mem: SharedPhysicalMemory,
    proj_id: int,
    proj_name: str,
    cur_page: int,
    anchor_page: int,
    prev_page: int,
    anchor: int,
    ref_x: float,
    ref_y: float,
    hx: int,
    hy: int,
    sx: float,
    sy: float,
    remaining: int,
    step: int,
    hit_cur: bool,
    solved_all: bool,
    stale_cur: bool,
    stale_ever: bool,
    hop_idx: int,
    hop_adv: bool,
) -> None:
    out[min(proj_id, NUM_STRUCT_PROJS - 1)] = 1.0
    inv_m = 1.0 / float(mem.pages)
    out[4] = float(cur_page) * inv_m
    out[5] = float(anchor_page) * inv_m
    d_ap = float(cur_page - anchor_page) / 32.0
    out[6] = -4.0 if d_ap < -4.0 else (4.0 if d_ap > 4.0 else d_ap)
    d_pp = float(cur_page - prev_page) / 32.0
    out[7] = -4.0 if d_pp < -4.0 else (4.0 if d_pp > 4.0 else d_pp)

    ax = float(anchor % mem.width)
    ay = float(anchor // mem.width)
    out[8] = ax / float(mem.width)
    out[9] = ay / float(mem.height)
    out[10] = (ax % 16.0) / 16.0
    out[11] = (ay % 16.0) / 16.0
    out[12] = (ax % 32.0) / 32.0
    out[13] = (ay % 32.0) / 32.0

    c = mem.views[proj_name].centers[cur_page]
    dx_r = float(c[0]) - ref_x
    dy_r = float(c[1]) - ref_y
    v14 = dx_r / 32.0
    v15 = dy_r / 32.0
    v16 = dx_r / 8.0
    v17 = dy_r / 8.0
    out[14] = -4.0 if v14 < -4.0 else (4.0 if v14 > 4.0 else v14)
    out[15] = -4.0 if v15 < -4.0 else (4.0 if v15 > 4.0 else v15)
    out[16] = -4.0 if v16 < -4.0 else (4.0 if v16 > 4.0 else v16)
    out[17] = -4.0 if v17 < -4.0 else (4.0 if v17 > 4.0 else v17)

    out[18] = float(hx)
    out[19] = float(hy)
    out[20] = float(sx)
    out[21] = float(sy)
    out[22] = float(hx * hy)
    out[23] = abs(float(sx) - float(sy))
    p24 = (dx_r * float(hx)) / 32.0
    p25 = (dy_r * float(hy)) / 32.0
    out[24] = -4.0 if p24 < -4.0 else (4.0 if p24 > 4.0 else p24)
    out[25] = -4.0 if p25 < -4.0 else (4.0 if p25 > 4.0 else p25)

    out[26] = float(remaining) / 32.0
    out[27] = float(step) / 32.0
    out[28] = 1.0 if hit_cur else 0.0
    out[29] = 1.0 if solved_all else 0.0
    out[30] = 1.0 if stale_cur else 0.0
    out[31] = 1.0 if stale_ever else 0.0
    out[32] = float(hop_idx)
    out[33] = 1.0 if hop_adv else 0.0


def _label_step_actions(
    mem: SharedPhysicalMemory,
    cur_proj_name: str,
    cur_page: int,
    ep_allowed: Tuple[str, ...],
    visited: Dict[str, set],
    reanchor_slot: int,
    goal_slots: List[int],
    tx: int,
    ty: int,
    event_switch_free: bool,
    force_proj_name: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, int, str, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, int]]:
    """Compute teacher optimal projection set, optimal stride set, and best action from current state."""
    proj_best_cost: Dict[str, float] = {}
    proj_stride_costs: Dict[str, List[Tuple[float, int]]] = {}
    proj_cands_map: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}
    global_min_cost = 1e9

    for p_name in ep_allowed:
        cands, legal, bpage = compute_candidates_single(
            mem, cur_proj_name, cur_page, p_name, visited, reanchor_slot=reanchor_slot
        )
        proj_cands_map[p_name] = (cands, legal, bpage)
        if not legal.any():
            continue
        sw_pen = SWITCH_PENALTY if (p_name != cur_proj_name and not event_switch_free) else 0.0
        s_list: List[Tuple[float, int]] = []
        p_min = 1e9
        for d_idx in np.flatnonzero(legal):
            d_i = int(d_idx)
            step_c = 1.0 + SPAN_WEIGHT * float(abs(STRIDES[d_i])) + sw_pen
            rem_c = _state_cost_to_go(mem, p_name, int(cands[d_i]), goal_slots, tx, ty, ep_allowed)
            tot_c = step_c + rem_c
            s_list.append((tot_c, d_i))
            if tot_c < p_min:
                p_min = tot_c
        proj_best_cost[p_name] = p_min
        proj_stride_costs[p_name] = s_list
        if p_min < global_min_cost:
            global_min_cost = p_min

    if not proj_best_cost:
        return None

    best_p_name = min(proj_best_cost.keys(), key=lambda pn: proj_best_cost[pn])
    p_opt = np.zeros(NUM_STRUCT_PROJS, dtype=bool)
    for pn, cst in proj_best_cost.items():
        if cst <= global_min_cost + SET_TIE_TOL and pn in STRUCTURED_PROJECTIONS:
            p_opt[PROJ_TO_ID[pn]] = True
    if not p_opt.any():
        p_opt[0] = True

    # Condition stride supervision on `force_proj_name` (if supplied and legal) else `best_p_name`
    stride_p_name = (
        force_proj_name
        if (force_proj_name is not None and force_proj_name in proj_stride_costs)
        else best_p_name
    )
    cands, legal, bpage = proj_cands_map[stride_p_name]
    s_list = proj_stride_costs[stride_p_name]
    min_s_cost = min(item[0] for item in s_list)
    s_opt = np.zeros(NUM_STRIDES, dtype=bool)
    for cst, d_i in s_list:
        if cst <= min_s_cost + SET_TIE_TOL:
            s_opt[d_i] = True
    best_d_i = min(s_list, key=lambda item: item[0])[1]
    r_opt = np.array(
        [bool((s_opt & FINE_MASK_NP).any()), bool((s_opt & ~FINE_MASK_NP).any())],
        dtype=bool,
    )
    return (
        p_opt,
        min(PROJ_TO_ID[best_p_name], NUM_STRUCT_PROJS - 1),
        stride_p_name,
        cands,
        legal,
        s_opt,
        best_d_i,
        r_opt,
        bpage,
    )


@torch.inference_mode()
def generate_verified_traces(
    mem: SharedPhysicalMemory,
    roots: Sequence[QueryRoot],
    allowed_projs: Tuple[str, ...],
    single_selection_only: bool = False,
    horizon: int = 16,
    policy_model: Optional[AdaptiveGeometryController] = None,
) -> TraceData:
    """Generate verified traces using teacher rollout (policy_model=None) or 16K DAgger on-policy rollout."""
    if policy_model is not None:
        policy_model.eval()
    return _generate_verified_traces_impl(
        mem, roots, allowed_projs, single_selection_only, horizon, policy_model
    )


def _generate_verified_traces_impl(
    mem: SharedPhysicalMemory,
    roots: Sequence[QueryRoot],
    allowed_projs: Tuple[str, ...],
    single_selection_only: bool,
    horizon: int,
    policy_model: Optional[AdaptiveGeometryController],
) -> TraceData:
    n_roots = len(roots)
    init_obs = np.zeros((n_roots, OBS_DIM), dtype=np.float32)
    init_proj_opt = np.zeros((n_roots, NUM_STRUCT_PROJS), dtype=bool)
    init_proj_best = np.zeros(n_roots, dtype=np.int64)

    step_obs = np.zeros((n_roots, horizon, OBS_DIM), dtype=np.float32)
    active_proj = np.zeros((n_roots, horizon), dtype=np.int64)
    next_proj_opt = np.zeros((n_roots, horizon, NUM_STRUCT_PROJS), dtype=bool)
    next_proj_best = np.zeros((n_roots, horizon), dtype=np.int64)
    stride_proj = np.zeros((n_roots, horizon), dtype=np.int64)
    cand_feats = np.zeros((n_roots, horizon, NUM_STRIDES, CAND_FEAT_DIM), dtype=np.float32)
    stride_legal = np.zeros((n_roots, horizon, NUM_STRIDES), dtype=bool)
    stride_opt = np.zeros((n_roots, horizon, NUM_STRIDES), dtype=bool)
    stride_best = np.full((n_roots, horizon), -1, dtype=np.int64)
    res_opt = np.zeros((n_roots, horizon, 2), dtype=bool)
    stop_target = np.zeros((n_roots, horizon), dtype=np.float32)
    step_valid = np.zeros((n_roots, horizon), dtype=bool)
    move_valid = np.zeros((n_roots, horizon), dtype=bool)

    verified_solved = 0
    views = mem.views
    keys = mem.keys
    values = mem.values
    w, h = mem.width, mem.height
    is_fixed = len(allowed_projs) == 1

    for idx, root in enumerate(roots):
        h0 = root.hops[0]
        _fill_init_obs_row(
            init_obs[idx], w, h, h0.anchor, h0.hint_x, h0.hint_y, h0.scale_x, h0.scale_y, horizon
        )

        p0_mask, p0_id_teacher = _choose_init_proj_set(mem, root, allowed_projs, single_selection_only)
        init_proj_opt[idx] = p0_mask
        init_proj_best[idx] = min(p0_id_teacher, NUM_STRUCT_PROJS - 1)

        p_state = None
        if policy_model is not None:
            p_state = policy_model.initial_state(1)
            r0_logits, _, p_state = policy_model.step_recurrent(
                torch.from_numpy(init_obs[idx : idx + 1]), p_state, t=0, active_proj=None
            )
            if is_fixed:
                cur_proj_name = allowed_projs[0]
            else:
                cur_proj_name = STRUCTURED_PROJECTIONS[int(r0_logits.argmax(dim=-1).item())]
        else:
            cur_proj_name = ALL_PROJECTIONS[p0_id_teacher]

        ep_allowed = (cur_proj_name,) if single_selection_only else allowed_projs

        hop_idx = 0
        hop = root.hops[0]
        stale_seen = False
        cur_hx, cur_hy = hop.hint_x, hop.hint_y
        cur_sx, cur_sy = hop.scale_x, hop.scale_y
        cur_anchor = hop.anchor
        ref_x, ref_y = float(cur_anchor % w), float(cur_anchor // w)
        anchor_page = int(views[cur_proj_name].inv[cur_anchor])
        cur_page = anchor_page
        prev_page = anchor_page
        visited: Dict[str, set] = {p: set() for p in ep_allowed}
        solved_all = False

        for t in range(horizon):
            visited[cur_proj_name].add(cur_page)
            slots = views[cur_proj_name].cells[cur_page]
            target_key = int(keys[hop.target_record])
            matched = keys[slots >> 1] == target_key
            hit_valid = bool(matched.any()) and int(values[hop.target_record]) == int(
                values[slots[matched][0] >> 1]
            )
            stale_hit = hop.decoy_slot >= 0 and bool((slots == hop.decoy_slot).any())

            hop_just_adv = False
            reanchor_slot = -1
            if hit_valid:
                if hop_idx + 1 < root.num_hops:
                    hop_idx += 1
                    hop = root.hops[hop_idx]
                    hop_just_adv = True
                    stale_seen = False
                    cur_hx, cur_hy = hop.hint_x, hop.hint_y
                    cur_sx, cur_sy = hop.scale_x, hop.scale_y
                    cur_anchor = hop.anchor
                    ref_x, ref_y = float(cur_anchor % w), float(cur_anchor // w)
                    reanchor_slot = cur_anchor
                    visited = {p: set() for p in ep_allowed}
                else:
                    solved_all = True
            elif stale_hit and hop.decoy_slot >= 0 and not stale_seen:
                stale_seen = True
                cur_hx, cur_hy = hop.mirror_hint_x, hop.mirror_hint_y
                cur_sx, cur_sy = hop.mirror_scale_x, hop.mirror_scale_y
                c_s = views[cur_proj_name].centers[cur_page]
                ref_x, ref_y = float(c_s[0]), float(c_s[1])

            _fill_step_obs_row(
                step_obs[idx, t],
                mem,
                PROJ_TO_ID[cur_proj_name],
                cur_proj_name,
                cur_page,
                anchor_page,
                prev_page,
                cur_anchor,
                ref_x,
                ref_y,
                cur_hx,
                cur_hy,
                cur_sx,
                cur_sy,
                horizon - t,
                t,
                hit_valid,
                solved_all,
                stale_hit,
                stale_seen,
                hop_idx,
                hop_just_adv,
            )

            active_proj[idx, t] = min(PROJ_TO_ID[cur_proj_name], NUM_STRUCT_PROJS - 1)
            step_valid[idx, t] = True
            stop_target[idx, t] = 1.0 if solved_all else 0.0

            if solved_all:
                verified_solved += 1
                break

            # Advance policy state if running DAgger on-policy rollout
            pol_next_proj_name: Optional[str] = None
            if policy_model is not None and p_state is not None:
                obs_t = torch.from_numpy(step_obs[idx : idx + 1, t])
                ap_t = torch.tensor([active_proj[idx, t]], dtype=torch.long)
                r_logits, _, p_state = policy_model.step_recurrent(obs_t, p_state, t=t + 1, active_proj=ap_t)
                if is_fixed or single_selection_only:
                    pol_next_proj_name = cur_proj_name
                else:
                    pol_next_proj_name = STRUCTURED_PROJECTIONS[int(r_logits.argmax(dim=-1).item())]

            goal_slots, tx, ty = _goal_info(mem, hop, stale_seen=stale_seen)
            labeled = _label_step_actions(
                mem,
                cur_proj_name,
                cur_page,
                ep_allowed,
                visited,
                reanchor_slot,
                goal_slots,
                tx,
                ty,
                event_switch_free=(hop_just_adv or stale_hit),
                force_proj_name=pol_next_proj_name,
            )
            if labeled is None:
                break

            p_opt, best_p_id, stride_p_name, cands, legal, s_opt, best_d_i, r_opt, bpage = labeled
            cf_t = build_cand_feats_single(
                mem, cur_proj_name, cur_page, stride_p_name, cands, legal, ref_x, ref_y, cur_hx, cur_hy
            )

            next_proj_opt[idx, t] = p_opt
            next_proj_best[idx, t] = best_p_id
            stride_proj[idx, t] = min(PROJ_TO_ID[stride_p_name], NUM_STRUCT_PROJS - 1)
            cand_feats[idx, t] = cf_t
            stride_legal[idx, t] = legal
            stride_opt[idx, t] = s_opt
            stride_best[idx, t] = best_d_i
            res_opt[idx, t] = r_opt
            move_valid[idx, t] = True

            # Choose actual execution transition (on-policy under DAgger, teacher otherwise)
            if policy_model is not None and p_state is not None:
                obs_t = torch.from_numpy(step_obs[idx : idx + 1, t])
                np_t = torch.tensor([min(PROJ_TO_ID[stride_p_name], NUM_STRUCT_PROJS - 1)], dtype=torch.long)
                s_logits, _ = policy_model.predict_stride(
                    p_state, obs_t, np_t, torch.from_numpy(cf_t[None])
                )
                exec_d_i = int(s_logits[0].masked_fill(~torch.from_numpy(legal), -1e9).argmax().item())
                exec_p_name = stride_p_name
            else:
                exec_d_i = best_d_i
                exec_p_name = stride_p_name

            prev_page = cur_page if exec_p_name == cur_proj_name else bpage
            cur_proj_name = exec_p_name
            cur_page = int(cands[exec_d_i])
            anchor_page = int(views[cur_proj_name].inv[cur_anchor])

    return TraceData(
        init_obs=init_obs,
        init_proj_opt=init_proj_opt,
        init_proj_best=init_proj_best,
        step_obs=step_obs,
        active_proj=active_proj,
        next_proj_opt=next_proj_opt,
        next_proj_best=next_proj_best,
        stride_proj=stride_proj,
        cand_feats=cand_feats,
        stride_legal=stride_legal,
        stride_opt=stride_opt,
        stride_best=stride_best,
        res_opt=res_opt,
        stop_target=stop_target,
        step_valid=step_valid,
        move_valid=move_valid,
        teacher_verified_rate=float(verified_solved) / float(max(1, n_roots)),
    )


def _train_epochs(
    model: AdaptiveGeometryController,
    opt: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    traces: TraceData,
    is_fixed: bool,
    single_sel: bool,
    supervision: str,
    epochs: int,
    batch_size: int,
    horizon: int,
    rng: np.random.Generator,
) -> float:
    n_samples = len(traces.init_obs)
    t_init_obs = torch.from_numpy(traces.init_obs)
    t_init_opt = torch.from_numpy(traces.init_proj_opt)
    t_init_best = torch.from_numpy(traces.init_proj_best)
    t_step_obs = torch.from_numpy(traces.step_obs)
    t_act_proj = torch.from_numpy(traces.active_proj)
    t_next_opt = torch.from_numpy(traces.next_proj_opt)
    t_next_best = torch.from_numpy(traces.next_proj_best)
    t_stride_proj = torch.from_numpy(traces.stride_proj)
    t_cand_feats = torch.from_numpy(traces.cand_feats)
    t_s_legal = torch.from_numpy(traces.stride_legal)
    t_s_opt = torch.from_numpy(traces.stride_opt)
    t_s_best = torch.from_numpy(traces.stride_best)
    t_r_opt = torch.from_numpy(traces.res_opt)
    t_stop = torch.from_numpy(traces.stop_target)
    t_svalid = torch.from_numpy(traces.step_valid)
    t_mvalid = torch.from_numpy(traces.move_valid)

    # Penalize false premature stops 2x harder than a single extra probe
    pos_weight = torch.tensor(0.5)
    n_batches = max(1, math.ceil(n_samples / batch_size))
    last_loss = 0.0

    model.train()
    for _epoch in range(epochs):
        perm = rng.permutation(n_samples)
        epoch_loss = 0.0
        for b_idx in np.array_split(perm, n_batches):
            idx = torch.from_numpy(b_idx)
            bsz = len(idx)
            state = model.initial_state(bsz)
            total_loss = torch.tensor(0.0)
            denom = 0.0

            init_r_logits, _, state = model.step_recurrent(t_init_obs[idx], state, t=0, active_proj=None)
            if not is_fixed:
                if supervision == "action_set":
                    l_init = optimal_action_set_loss(init_r_logits, t_init_opt[idx]).sum()
                else:
                    l_init = F.cross_entropy(init_r_logits, t_init_best[idx], reduction="sum")
                total_loss = total_loss + 1.5 * l_init
                denom += float(bsz)

            for t in range(horizon):
                s_mask = t_svalid[idx, t]
                if not bool(s_mask.any()):
                    break
                r_logits, stop_logit, state = model.step_recurrent(
                    t_step_obs[idx, t],
                    state,
                    t=t + 1,
                    active_proj=t_act_proj[idx, t],
                )
                l_stop = F.binary_cross_entropy_with_logits(
                    stop_logit[s_mask], t_stop[idx, t][s_mask], pos_weight=pos_weight, reduction="sum"
                )
                total_loss = total_loss + 1.0 * l_stop
                denom += float(s_mask.sum().item())

                m_mask = t_mvalid[idx, t]
                if bool(m_mask.any()):
                    if (not is_fixed) and (not single_sel):
                        if supervision == "action_set":
                            l_rout = optimal_action_set_loss(
                                r_logits[m_mask], t_next_opt[idx, t][m_mask]
                            ).sum()
                        else:
                            l_rout = F.cross_entropy(
                                r_logits[m_mask], t_next_best[idx, t][m_mask], reduction="sum"
                            )
                        total_loss = total_loss + 1.4 * l_rout

                    chosen_p = t_stride_proj[idx, t][m_mask]
                    sub_state = type(state)(
                        h=state.h[m_mask],
                        s=state.s[m_mask],
                        p_mem=state.p_mem[m_mask],
                    )
                    s_logits, res_logits = model.predict_stride(
                        sub_state, t_step_obs[idx, t][m_mask], chosen_p, t_cand_feats[idx, t][m_mask]
                    )
                    if supervision == "action_set":
                        l_stride = optimal_action_set_loss(
                            s_logits,
                            t_s_opt[idx, t][m_mask],
                            legal_mask=t_s_legal[idx, t][m_mask],
                        ).sum()
                        l_res = optimal_action_set_loss(
                            res_logits, t_r_opt[idx, t][m_mask]
                        ).sum()
                    else:
                        masked_s = s_logits.masked_fill(~t_s_legal[idx, t][m_mask], -1e9)
                        l_stride = F.cross_entropy(
                            masked_s, t_s_best[idx, t][m_mask], reduction="sum"
                        )
                        res_target = (~torch.from_numpy(FINE_MASK_NP)[t_s_best[idx, t][m_mask]]).long()
                        l_res = F.cross_entropy(res_logits, res_target, reduction="sum")
                    total_loss = total_loss + l_stride + 0.25 * l_res

            loss = total_loss / max(1.0, denom)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.item())

        scheduler.step()
        last_loss = epoch_loss / n_batches
    return last_loss


def train_controller(
    mem_16k: SharedPhysicalMemory,
    arm_kind: str,
    seed: int,
    num_roots: int = 2048,
    epochs: int = 14,
    batch_size: int = 128,
    lr: float = 3.0e-3,
    horizon: int = 16,
    supervision: str = "action_set",
    cfg: Optional[ModelConfig] = None,
    mixture_weights: Optional[Dict[str, float]] = None,
    salt: int = 37,
) -> Tuple[AdaptiveGeometryController, Dict[str, float]]:
    """Train a controller exclusively on 16K memory with teacher traces + 16K DAgger aggregation."""
    if mem_16k.n != 16384:
        raise ValueError(f"Training must occur exclusively at 16K (16384), got {mem_16k.n}")

    is_fixed = arm_kind.startswith("fixed_")
    single_sel = arm_kind == "router_single"
    if is_fixed:
        layout = arm_kind[len("fixed_") :]
        allowed_projs: Tuple[str, ...] = (layout,)
        model_cfg = cfg if cfg is not None else CONFIG_54K_FIXED
    else:
        allowed_projs = STRUCTURED_PROJECTIONS
        model_cfg = cfg if cfg is not None else CONFIG_54K

    roots = sample_mixed_roots(mem_16k, num_roots, salt=salt, weights=mixture_weights)
    traces_teacher = generate_verified_traces(
        mem_16k,
        roots,
        allowed_projs=allowed_projs,
        single_selection_only=single_sel,
        horizon=horizon,
        policy_model=None,
    )

    torch.manual_seed(seed)
    model = AdaptiveGeometryController(model_cfg)
    rng = np.random.default_rng(seed + 101)

    phase1_epochs = max(1, (epochs * 3) // 5)
    phase2_epochs = max(0, epochs - phase1_epochs)

    opt1 = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=phase1_epochs, eta_min=lr * 0.25)
    last_loss = _train_epochs(
        model, opt1, sch1, traces_teacher, is_fixed, single_sel, supervision,
        phase1_epochs, batch_size, horizon, rng,
    )

    dagger_rate = 0.0
    if phase2_epochs > 0:
        # On-policy 16K verified trace aggregation (DAgger) strictly on the 16K training world
        dagger_count = min(768, max(64, num_roots // 3))
        dagger_roots = sample_mixed_roots(mem_16k, dagger_count, salt=salt + 991, weights=mixture_weights)
        traces_dagger = generate_verified_traces(
            mem_16k,
            dagger_roots,
            allowed_projs=allowed_projs,
            single_selection_only=single_sel,
            horizon=horizon,
            policy_model=model,
        )
        dagger_rate = traces_dagger.teacher_verified_rate
        combined = concat_traces(traces_teacher, traces_dagger)
        opt2 = torch.optim.AdamW(model.parameters(), lr=lr * 0.65, weight_decay=1e-4)
        sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=phase2_epochs, eta_min=lr * 0.05)
        last_loss = _train_epochs(
            model, opt2, sch2, combined, is_fixed, single_sel, supervision,
            phase2_epochs, batch_size, horizon, rng,
        )

    model.eval()
    stats = {
        "teacher_verified_rate": traces_teacher.teacher_verified_rate,
        "dagger_16k_onpolicy_rate": dagger_rate,
        "final_train_loss": last_loss,
        **model.parameter_summary(active_router=not is_fixed),
    }
    return model, stats
