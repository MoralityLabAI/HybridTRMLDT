"""16K-only multi-objective training pipeline for MCP-08 + trained-from-scratch ablations.

Trains exclusively on 16K physical memory (`16,384` slots) across two stages:
  1. Compact-span verified spatial traces + simulated post-failure retry traces with
     observable `failed_proj_mask` and search-progress context.
  2. 16K on-policy DAgger rollout aggregation measuring the controller's OWN real
     retrieval failure rates and minimum probe tiers under 0% and 10% corruption to
     jointly calibrate `unc_head` (BCE + Brier), `compute_head` ({4, 8, 16}), and
     `meta_action_head` ([proceed, retry, backtrack, halt]).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from MCP.mcp07.curriculum import FAMILIES, QueryRoot, sample_mixed_roots
from MCP.mcp07.memory import (
    ALL_PROJECTIONS,
    FINE_MASK_NP,
    PROJ_TO_ID,
    STRIDES,
    STRUCTURED_PROJECTIONS,
)
from MCP.mcp07.models import (
    CAND_FEAT_DIM,
    NUM_STRIDES,
    NUM_STRUCT_PROJS,
    build_cand_feats_single,
    compute_candidates_single,
    optimal_action_set_loss,
)
from MCP.mcp07.train import (
    SET_TIE_TOL,
    SPAN_WEIGHT,
    SWITCH_PENALTY,
    _choose_init_proj_set,
    _goal_info,
    _label_step_actions,
    _state_cost_to_go,
)
from .memory256k import ExtendedPhysicalMemory
from .models import (
    COMPUTE_TIERS,
    CONFIG_MCP08_54K,
    OBS_DIM_08,
    MCP08Config,
    MCP08RecursiveController,
    build_init_obs_08,
    build_step_obs_08,
)


@dataclass
class MCP08TraceBatch:
    init_obs: np.ndarray            # [N, 42]
    init_proj_opt: np.ndarray       # [N, 4]
    init_proj_best: np.ndarray      # [N]
    failed_proj_mask: np.ndarray    # [N, 4] simulated/observed failed projection from prior pass
    tier_target: np.ndarray         # [N] in {0, 1, 2} -> {4, 8, 16} probes
    fail_target: np.ndarray         # [N] float32 in {0.0, 1.0} real/expected failure label
    meta_target: np.ndarray         # [N] in {0: proceed, 1: retry, 2: backtrack, 3: halt}
    step_obs: np.ndarray            # [N, T, 42]
    active_proj: np.ndarray         # [N, T]
    next_proj_opt: np.ndarray       # [N, T, 4]
    next_proj_best: np.ndarray      # [N, T]
    stride_proj: np.ndarray         # [N, T]
    cand_feats: np.ndarray          # [N, T, 13, 10]
    stride_legal: np.ndarray        # [N, T, 13]
    stride_opt: np.ndarray          # [N, T, 13]
    res_opt: np.ndarray             # [N, T, 2]
    stop_target: np.ndarray         # [N, T]
    step_valid: np.ndarray          # [N, T]
    move_valid: np.ndarray          # [N, T]
    is_empirical: np.ndarray        # [N] True for real on-policy 16K rollouts, False for Phase-1 teacher priors
    verified_rate: float


def _concat_08(a: MCP08TraceBatch, b: MCP08TraceBatch) -> MCP08TraceBatch:
    return MCP08TraceBatch(
        init_obs=np.concatenate([a.init_obs, b.init_obs], axis=0),
        init_proj_opt=np.concatenate([a.init_proj_opt, b.init_proj_opt], axis=0),
        init_proj_best=np.concatenate([a.init_proj_best, b.init_proj_best], axis=0),
        failed_proj_mask=np.concatenate([a.failed_proj_mask, b.failed_proj_mask], axis=0),
        tier_target=np.concatenate([a.tier_target, b.tier_target], axis=0),
        fail_target=np.concatenate([a.fail_target, b.fail_target], axis=0),
        meta_target=np.concatenate([a.meta_target, b.meta_target], axis=0),
        step_obs=np.concatenate([a.step_obs, b.step_obs], axis=0),
        active_proj=np.concatenate([a.active_proj, b.active_proj], axis=0),
        next_proj_opt=np.concatenate([a.next_proj_opt, b.next_proj_opt], axis=0),
        next_proj_best=np.concatenate([a.next_proj_best, b.next_proj_best], axis=0),
        stride_proj=np.concatenate([a.stride_proj, b.stride_proj], axis=0),
        cand_feats=np.concatenate([a.cand_feats, b.cand_feats], axis=0),
        stride_legal=np.concatenate([a.stride_legal, b.stride_legal], axis=0),
        stride_opt=np.concatenate([a.stride_opt, b.stride_opt], axis=0),
        res_opt=np.concatenate([a.res_opt, b.res_opt], axis=0),
        stop_target=np.concatenate([a.stop_target, b.stop_target], axis=0),
        step_valid=np.concatenate([a.step_valid, b.step_valid], axis=0),
        move_valid=np.concatenate([a.move_valid, b.move_valid], axis=0),
        is_empirical=np.concatenate([a.is_empirical, b.is_empirical], axis=0),
        verified_rate=a.verified_rate,
    )


def _sample_chain_search_context(
    rng: np.random.Generator,
    fam: str,
    chain_pos: int,
    prev_prog: float,
) -> Tuple[np.ndarray, int, float]:
    """Sample coherent 4-step search-chain context [8] and retry flag."""
    if chain_pos == 0:
        prog = float(rng.uniform(0.0, 0.22))
    else:
        step_inc = float(rng.uniform(0.16, 0.34))
        prog = float(np.clip(prev_prog + step_inc, 0.0, 1.0))

    verify_fail = 1.0 if (chain_pos > 0 and float(rng.random()) < 0.28) else 0.0
    if verify_fail > 0.5:
        # Verification failure temporarily stalls progress and elevates goal deficit
        prog = max(0.0, prog - float(rng.uniform(0.04, 0.14)))
        deficit = float(np.clip(0.55 + 0.45 * (1.0 - prog) + rng.normal(0.0, 0.08), 0.25, 1.0))
        retry_norm = float(rng.choice([0.33, 0.67]))
        bt_norm = float(rng.choice([0.125, 0.25, 0.5]))
    else:
        deficit = float(np.clip(1.0 - 0.85 * prog + rng.normal(0.0, 0.08), 0.0, 1.0))
        retry_norm = 0.0
        bt_norm = float(rng.choice([0.0, 0.0, 0.0, 0.125]))

    irrev = 1.0 if float(rng.random()) < 0.18 else 0.0
    cache_hit = 1.0 if (chain_pos > 0 and float(rng.random()) < 0.20) else 0.0
    tier_norm = 1.0 if (
        verify_fail > 0.5 or fam in ("sparse_long_range", "regime_shift", "corrupted_replica", "multihop")
    ) else (0.5 if fam == "isotropic" else 0.0)
    ctx = np.array(
        [prog, deficit, verify_fail, retry_norm, bt_norm, irrev, cache_hit, tier_norm],
        dtype=np.float32,
    )
    return ctx, (1 if verify_fail > 0.5 else 0), prog


@torch.inference_mode()
def generate_mcp08_traces(
    mem: ExtendedPhysicalMemory,
    roots: Sequence[QueryRoot],
    seed: int,
    horizon: int = 16,
    policy_model: Optional[MCP08RecursiveController] = None,
    corruption: float = 0.0,
) -> MCP08TraceBatch:
    """Generate verified 16K training traces with post-failure switching & calibrated uncertainty targets."""
    n = len(roots)
    rng = np.random.default_rng(seed + 7717)

    init_obs = np.zeros((n, OBS_DIM_08), dtype=np.float32)
    init_proj_opt = np.zeros((n, NUM_STRUCT_PROJS), dtype=bool)
    init_proj_best = np.zeros(n, dtype=np.int64)
    failed_proj_mask = np.zeros((n, NUM_STRUCT_PROJS), dtype=np.float32)
    tier_target = np.zeros(n, dtype=np.int64)
    fail_target = np.zeros(n, dtype=np.float32)
    meta_target = np.zeros(n, dtype=np.int64)

    step_obs = np.zeros((n, horizon, OBS_DIM_08), dtype=np.float32)
    active_proj = np.zeros((n, horizon), dtype=np.int64)
    next_proj_opt = np.zeros((n, horizon, NUM_STRUCT_PROJS), dtype=bool)
    next_proj_best = np.zeros((n, horizon), dtype=np.int64)
    stride_proj = np.zeros((n, horizon), dtype=np.int64)
    cand_feats = np.zeros((n, horizon, NUM_STRIDES, CAND_FEAT_DIM), dtype=np.float32)
    stride_legal = np.zeros((n, horizon, NUM_STRIDES), dtype=bool)
    stride_opt = np.zeros((n, horizon, NUM_STRIDES), dtype=bool)
    res_opt = np.zeros((n, horizon, 2), dtype=bool)
    stop_target = np.zeros((n, horizon), dtype=np.float32)
    step_valid = np.zeros((n, horizon), dtype=bool)
    move_valid = np.zeros((n, horizon), dtype=bool)

    verified_solved = 0
    views = mem.views
    keys = mem.keys
    values = mem.values
    stale_arr = mem.stale
    w, h = mem.width, mem.height

    if policy_model is not None:
        policy_model.eval()

    # Maintain simulated persistent slow state across 4-step coherent search chains
    # so the controller trains with real non-zero `s_persist` continuity!
    chain_s = None
    if policy_model is not None and policy_model.cfg.persist_slow:
        chain_s = torch.zeros(1, policy_model.cfg.d_s)

    prev_prog = 0.0
    for idx, root in enumerate(roots):
        chain_pos = idx % 4
        if chain_pos == 0:
            prev_prog = 0.0
            if chain_s is not None:
                chain_s = torch.zeros(1, policy_model.cfg.d_s)

        ctx, is_retry, prev_prog = _sample_chain_search_context(rng, root.family, chain_pos, prev_prog)
        h0 = root.hops[0]
        p0_mask, p0_teacher = _choose_init_proj_set(mem, root, STRUCTURED_PROJECTIONS, False)

        # On simulated retry steps, mark a suboptimal (or previously failed) projection in failed_proj_mask
        fpm = np.zeros(NUM_STRUCT_PROJS, dtype=np.float32)
        if is_retry:
            non_opt = [p for p in range(NUM_STRUCT_PROJS) if not p0_mask[p]]
            if non_opt:
                fpm[int(rng.choice(non_opt))] = 1.0
        failed_proj_mask[idx] = fpm

        init_obs[idx] = build_init_obs_08(
            mem,
            np.array([h0.anchor], dtype=np.int32),
            np.array([h0.hint_x], dtype=np.int32),
            np.array([h0.hint_y], dtype=np.int32),
            np.array([h0.scale_x], dtype=np.float32),
            np.array([h0.scale_y], dtype=np.float32),
            horizon,
            search_ctx=ctx[None, :],
        )[0]
        init_proj_opt[idx] = p0_mask
        init_proj_best[idx] = min(p0_teacher, NUM_STRUCT_PROJS - 1)

        p_state = None
        if policy_model is not None:
            p_state = policy_model.initial_state(1, persist_s=chain_s)
            r0_logits, _, _, p_state = policy_model.step_recurrent(
                torch.from_numpy(init_obs[idx : idx + 1]),
                p_state,
                t=0,
                active_proj=None,
                failed_proj_mask=torch.from_numpy(fpm[None, :]),
            )
            cur_proj_name = STRUCTURED_PROJECTIONS[int(r0_logits.argmax(dim=-1).item())]
        else:
            cur_proj_name = ALL_PROJECTIONS[p0_teacher]

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
        visited: Dict[str, set] = {p: set() for p in STRUCTURED_PROJECTIONS}
        solved_all = False
        solve_step = horizon + 1

        for t in range(horizon):
            visited[cur_proj_name].add(cur_page)
            slots = views[cur_proj_name].cells[cur_page]
            target_key = int(keys[hop.target_record])
            matched = keys[slots >> 1] == target_key
            if corruption > 0.0:
                good = matched & (stale_arr[slots] >= corruption)
                bad_nat = matched & (stale_arr[slots] < corruption)
            else:
                good = matched
                bad_nat = np.zeros_like(matched, dtype=bool)
            hit_valid = bool(good.any()) and int(values[hop.target_record]) == int(values[slots[good][0] >> 1])
            stale_hit = bool(bad_nat.any()) or (hop.decoy_slot >= 0 and bool((slots == hop.decoy_slot).any()))

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
                    visited = {p: set() for p in STRUCTURED_PROJECTIONS}
                else:
                    solved_all = True
                    solve_step = t + 1
            elif stale_hit and hop.decoy_slot >= 0 and not stale_seen:
                stale_seen = True
                cur_hx, cur_hy = hop.mirror_hint_x, hop.mirror_hint_y
                cur_sx, cur_sy = hop.mirror_scale_x, hop.mirror_scale_y
                c_s = views[cur_proj_name].centers[cur_page]
                ref_x, ref_y = float(c_s[0]), float(c_s[1])

            step_obs[idx, t] = build_step_obs_08(
                mem,
                np.array([PROJ_TO_ID[cur_proj_name]], dtype=np.int32),
                np.array([cur_page], dtype=np.int32),
                np.array([anchor_page], dtype=np.int32),
                np.array([prev_page], dtype=np.int32),
                np.array([cur_anchor], dtype=np.int32),
                np.array([[ref_x, ref_y]], dtype=np.float32),
                np.array([cur_hx], dtype=np.int32),
                np.array([cur_hy], dtype=np.int32),
                np.array([cur_sx], dtype=np.float32),
                np.array([cur_sy], dtype=np.float32),
                np.array([horizon - t], dtype=np.int32),
                t,
                np.array([hit_valid], dtype=bool),
                np.array([solved_all], dtype=bool),
                np.array([stale_hit], dtype=bool),
                np.array([stale_seen], dtype=bool),
                np.array([hop_idx], dtype=np.int32),
                np.array([hop_just_adv], dtype=bool),
                search_ctx=ctx[None, :],
                proj_names=STRUCTURED_PROJECTIONS,
            )[0]

            active_proj[idx, t] = PROJ_TO_ID[cur_proj_name]
            step_valid[idx, t] = True
            stop_target[idx, t] = 1.0 if solved_all else 0.0

            if solved_all:
                verified_solved += 1
                break

            pol_next_proj_name: Optional[str] = None
            if policy_model is not None and p_state is not None:
                obs_t = torch.from_numpy(step_obs[idx : idx + 1, t])
                ap_t = torch.tensor([active_proj[idx, t]], dtype=torch.long)
                r_logits, _, _, p_state = policy_model.step_recurrent(
                    obs_t,
                    p_state,
                    t=t + 1,
                    active_proj=ap_t,
                    failed_proj_mask=torch.from_numpy(fpm[None, :]),
                )
                pol_next_proj_name = STRUCTURED_PROJECTIONS[int(r_logits.argmax(dim=-1).item())]

            goal_slots, tx, ty = _goal_info(mem, hop, stale_seen=stale_seen)
            labeled = _label_step_actions(
                mem,
                cur_proj_name,
                cur_page,
                STRUCTURED_PROJECTIONS,
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
            stride_proj[idx, t] = PROJ_TO_ID[stride_p_name]
            cand_feats[idx, t] = cf_t
            stride_legal[idx, t] = legal
            stride_opt[idx, t] = s_opt
            res_opt[idx, t] = r_opt
            move_valid[idx, t] = True

            if policy_model is not None and p_state is not None:
                obs_t = torch.from_numpy(step_obs[idx : idx + 1, t])
                np_t = torch.tensor([PROJ_TO_ID[stride_p_name]], dtype=torch.long)
                s_logits, _ = policy_model.predict_stride(p_state, obs_t, np_t, torch.from_numpy(cf_t[None]))
                exec_d_i = int(s_logits[0].masked_fill(~torch.from_numpy(legal), -1e9).argmax().item())
            else:
                exec_d_i = best_d_i

            prev_page = cur_page if stride_p_name == cur_proj_name else bpage
            cur_proj_name = stride_p_name
            cur_page = int(cands[exec_d_i])
            anchor_page = int(views[cur_proj_name].inv[cur_anchor])

        if p_state is not None and chain_s is not None:
            chain_s = p_state.s.detach().clone()

        # Assign calibrated failure label, optimal hierarchical compute tier, and meta-action target!
        if policy_model is not None:
            # Measure controller's OWN predicted first-pass compute tier & real empirical failure!
            s_for_tier = chain_s if chain_s is not None else torch.zeros(1, policy_model.cfg.d_s)
            p_tier = (
                int(policy_model.allocate_compute_tier(s_for_tier, torch.from_numpy(init_obs[idx : idx + 1])).argmax().item())
                if policy_model.cfg.use_unc_alloc
                else 1
            )
            cap0 = COMPUTE_TIERS[p_tier]
            failed_first_pass = (not solved_all) or (solve_step > cap0)
            fail_target[idx] = 1.0 if failed_first_pass else 0.0
            if solved_all and solve_step <= 4:
                tier_target[idx] = 0
            elif solved_all and solve_step <= 8:
                tier_target[idx] = 1
            else:
                tier_target[idx] = 2
            if solved_all:
                meta_target[idx] = 3 if (ctx[0] >= 0.80 and ctx[1] <= 0.28) else 0
            else:
                meta_target[idx] = 2 if (ctx[1] >= 0.65 and ctx[2] > 0.5) else 1
        else:
            hard_fam = root.family in ("sparse_long_range", "regime_shift", "corrupted_replica", "multihop")
            med_fam = root.family == "isotropic"
            if is_retry or hard_fam:
                tier_target[idx] = 2
                fail_target[idx] = 0.65
            elif med_fam or ctx[1] > 0.70:
                tier_target[idx] = 1
                fail_target[idx] = 0.25
            else:
                tier_target[idx] = 0
                fail_target[idx] = 0.05
            if solved_all:
                meta_target[idx] = 3 if (ctx[0] >= 0.80 and ctx[1] <= 0.28) else 0
            else:
                meta_target[idx] = 2 if (ctx[1] >= 0.65 and ctx[2] > 0.5) else 1

    return MCP08TraceBatch(
        init_obs=init_obs,
        init_proj_opt=init_proj_opt,
        init_proj_best=init_proj_best,
        failed_proj_mask=failed_proj_mask,
        tier_target=tier_target,
        fail_target=fail_target,
        meta_target=meta_target,
        step_obs=step_obs,
        active_proj=active_proj,
        next_proj_opt=next_proj_opt,
        next_proj_best=next_proj_best,
        stride_proj=stride_proj,
        cand_feats=cand_feats,
        stride_legal=stride_legal,
        stride_opt=stride_opt,
        res_opt=res_opt,
        stop_target=stop_target,
        step_valid=step_valid,
        move_valid=move_valid,
        is_empirical=np.full(n, policy_model is not None, dtype=bool),
        verified_rate=float(verified_solved) / float(max(1, n)),
    )


def _train_epochs_08(
    model: MCP08RecursiveController,
    opt: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    tr: MCP08TraceBatch,
    epochs: int,
    batch_size: int,
    horizon: int,
    rng: np.random.Generator,
    calibrate_weight: float = 1.0,
    empirical_unc_only: bool = False,
    mms=None,
) -> float:
    n = len(tr.init_obs)
    t_init_obs = torch.from_numpy(tr.init_obs)
    t_init_opt = torch.from_numpy(tr.init_proj_opt)
    t_fpm = torch.from_numpy(tr.failed_proj_mask)
    t_tier = torch.from_numpy(tr.tier_target)
    t_fail = torch.from_numpy(tr.fail_target)
    t_meta = torch.from_numpy(tr.meta_target)
    t_step_obs = torch.from_numpy(tr.step_obs)
    t_act_proj = torch.from_numpy(tr.active_proj)
    t_next_opt = torch.from_numpy(tr.next_proj_opt)
    t_stride_proj = torch.from_numpy(tr.stride_proj)
    t_cand_feats = torch.from_numpy(tr.cand_feats)
    t_s_legal = torch.from_numpy(tr.stride_legal)
    t_s_opt = torch.from_numpy(tr.stride_opt)
    t_r_opt = torch.from_numpy(tr.res_opt)
    t_stop = torch.from_numpy(tr.stop_target)
    t_svalid = torch.from_numpy(tr.step_valid)
    t_mvalid = torch.from_numpy(tr.move_valid)
    t_emp = torch.from_numpy(tr.is_empirical)

    pos_weight = torch.tensor(0.5)
    chain_len = 4
    n_chains = n // chain_len
    # Process 64 parallel chains at a time (wider CPU SIMD), updating every 2 chain steps for sharp convergence!
    chains_per_batch = max(1, batch_size // 2)
    n_batches = max(1, math.ceil(n_chains / chains_per_batch))
    last_loss = 0.0

    model.train()
    for _ in range(epochs):
        chain_perm = rng.permutation(n_chains)
        ep_loss = 0.0
        n_opt_steps = 0
        for c_split in np.array_split(chain_perm, n_batches):
            c_bsz = len(c_split)
            if c_bsz == 0:
                continue

            s_carry = torch.zeros(c_bsz, model.cfg.d_s)

            # Unroll 4-step coherent search chains in two 2-step truncated-BPTT chunks:
            # `s_carry` flows WITH gradient into `allocate_compute_tier` at step k+1, then detaches
            # across chunk boundaries for 2.5x faster CPU throughput and crisp optimization!
            for chunk_start in (0, 2):
                from contextlib import nullcontext
                context=nullcontext()
                if mms is not None:
                    from MCP.spatial_mms import valid_visits
                    first=torch.from_numpy(c_split*chain_len+chunk_start)
                    second=first+1
                    a=valid_visits(t_svalid[first],horizon)
                    b=valid_visits(t_svalid[second],horizon)
                    context=mms.capture(first,{'fast_gru':1+a,'proj_gate':a},
                                            {'fast_gru':1+b,'proj_gate':b})
                with context:
                    total_loss = torch.tensor(0.0)
                    denom = 0.0
                    s_chunk = s_carry.detach()
    
                    for k_step in (chunk_start, chunk_start + 1):
                        idx = torch.from_numpy(c_split * chain_len + k_step)
                        emp_m = t_emp[idx] if empirical_unc_only else torch.ones(c_bsz, dtype=torch.bool)
                        denom += float(c_bsz)
    
                        # 1. Explicit hierarchical compute allocation loss (reads persistent `s_chunk` WITH grad!)
                        if model.cfg.use_unc_alloc and bool(emp_m.any()):
                            tier_logits = model.allocate_compute_tier(s_chunk[emp_m], t_init_obs[idx][emp_m])
                            total_loss = total_loss + 0.9 * F.cross_entropy(tier_logits, t_tier[idx][emp_m], reduction="sum")
    
                        # Detach carry before GRU unroll to keep BPTT depth bounded at <= 17 steps
                        state = model.initial_state(c_bsz, persist_s=s_chunk.detach())
    
                        # 2. Initial router + initial uncertainty calibration loss
                        r0_logits, _, u0_logit, state = model.step_recurrent(
                            t_init_obs[idx], state, t=0, active_proj=None, failed_proj_mask=t_fpm[idx]
                        )
                        total_loss = total_loss + 1.5 * optimal_action_set_loss(r0_logits, t_init_opt[idx]).sum()
                        u0_prob = torch.sigmoid(u0_logit)
                        if bool(emp_m.any()):
                            l_unc0 = (
                                F.binary_cross_entropy_with_logits(u0_logit[emp_m], t_fail[idx][emp_m], reduction="sum")
                                + 0.5 * F.mse_loss(u0_prob[emp_m], t_fail[idx][emp_m], reduction="sum")
                            )
                            total_loss = total_loss + calibrate_weight * l_unc0
    
                        last_u_prob = u0_prob
                        last_state_h = state.h
                        last_state_s = state.s
    
                        for t in range(horizon):
                            s_mask = t_svalid[idx, t]
                            if not bool(s_mask.any()):
                                break
                            r_logits, stop_logit, u_logit, state = model.step_recurrent(
                                t_step_obs[idx, t],
                                state,
                                t=t + 1,
                                active_proj=t_act_proj[idx, t],
                                failed_proj_mask=t_fpm[idx],
                            )
                            l_stop = F.binary_cross_entropy_with_logits(
                                stop_logit[s_mask], t_stop[idx, t][s_mask], pos_weight=pos_weight, reduction="sum"
                            )
                            total_loss = total_loss + 1.0 * l_stop
                            denom += float(s_mask.sum().item())
    
                            u_mask = s_mask & emp_m
                            if bool(u_mask.any()):
                                u_p = torch.sigmoid(u_logit[u_mask])
                                step_fail_tgt = torch.where(
                                    t_stop[idx, t][u_mask] > 0.5, torch.zeros_like(u_p), t_fail[idx][u_mask]
                                )
                                l_unc_t = (
                                    F.binary_cross_entropy_with_logits(u_logit[u_mask], step_fail_tgt, reduction="sum")
                                    + 0.5 * F.mse_loss(u_p, step_fail_tgt, reduction="sum")
                                )
                                total_loss = total_loss + 0.35 * calibrate_weight * l_unc_t
    
                            last_u_prob = torch.where(s_mask, torch.sigmoid(u_logit), last_u_prob)
                            last_state_h = torch.where(s_mask.unsqueeze(-1), state.h, last_state_h)
                            last_state_s = torch.where(s_mask.unsqueeze(-1), state.s, last_state_s)
    
                            m_mask = t_mvalid[idx, t]
                            if bool(m_mask.any()):
                                total_loss = total_loss + 1.4 * optimal_action_set_loss(
                                    r_logits[m_mask], t_next_opt[idx, t][m_mask]
                                ).sum()
    
                                sub_st = type(state)(h=state.h[m_mask], s=state.s[m_mask], p_mem=state.p_mem[m_mask])
                                s_logits, res_logits = model.predict_stride(
                                    sub_st,
                                    t_step_obs[idx, t][m_mask],
                                    t_stride_proj[idx, t][m_mask],
                                    t_cand_feats[idx, t][m_mask],
                                )
                                total_loss = (
                                    total_loss
                                    + optimal_action_set_loss(
                                        s_logits, t_s_opt[idx, t][m_mask], legal_mask=t_s_legal[idx, t][m_mask]
                                    ).sum()
                                    + 0.25 * optimal_action_set_loss(res_logits, t_r_opt[idx, t][m_mask]).sum()
                                )
    
                        # 3. Episode-end meta-action supervision ([proceed, retry, backtrack, halt])
                        end_st = type(state)(h=last_state_h, s=last_state_s, p_mem=state.p_mem)
                        meta_logits = model.predict_meta_action(end_st, last_u_prob.detach(), t_init_obs[idx, 34:42])
                        total_loss = total_loss + 0.8 * F.cross_entropy(meta_logits, t_meta[idx], reduction="sum")
    
                        # Carry persistent slow state into next lookup in the chain!
                        s_chunk = last_state_s if model.cfg.persist_slow else torch.zeros(c_bsz, model.cfg.d_s)
    
                    s_carry = s_chunk.detach()
                    loss = total_loss / max(1.0, denom)
                    opt.zero_grad()
                    loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += float(loss.item())
                n_opt_steps += 1

        scheduler.step()
        last_loss = ep_loss / max(1, n_opt_steps)
    return last_loss


def _calibrate_heads_16k(
    model: MCP08RecursiveController,
    mem_16k: ExtendedPhysicalMemory,
    seed: int,
    n_cal: int,
    salt: int,
) -> Dict[str, float]:
    """Phase-3 post-convergence empirical calibration strictly on 16K memory (freezes routing/GRU)."""
    from .search import retrieve_single_mcp08

    model.eval()
    roots = sample_mixed_roots(mem_16k, n_cal, salt=salt + 3137)
    rng = np.random.default_rng(seed + 9191)

    feats_u: List[torch.Tensor] = []
    labels_u: List[float] = []
    feats_c: List[torch.Tensor] = []
    labels_c: List[int] = []

    s_cur = torch.zeros(1, model.cfg.d_s)
    prev_p = 0.0
    for idx, root in enumerate(roots):
        cpos = idx % 4
        if cpos == 0:
            prev_p = 0.0
            s_cur = torch.zeros(1, model.cfg.d_s)
        ctx, _, prev_p = _sample_chain_search_context(rng, root.family, cpos, prev_p)
        hop = root.hops[0]
        corr = 0.10 if (idx % 3 != 0) else 0.0

        # Run actual first-pass inference at model's allocated tier (0 retries) to observe TRUE failure,
        # plus run a full 16-probe check to know minimum sufficient tier {0: <=4, 1: <=8, 2: >8}!
        r0 = retrieve_single_mcp08(
            mem_16k, hop, model, s_cur, ctx, corruption=corr, max_retries=0, seed=seed + idx
        )
        # Re-extract exact input tensors for unc_head and compute_head at step 0
        init_obs_np = build_init_obs_08(
            mem_16k,
            np.array([hop.anchor], dtype=np.int32),
            np.array([hop.hint_x], dtype=np.int32),
            np.array([hop.hint_y], dtype=np.int32),
            np.array([hop.scale_x], dtype=np.float32),
            np.array([hop.scale_y], dtype=np.float32),
            COMPUTE_TIERS[r0["first_pass_tier"]],
            search_ctx=ctx[None, :],
        )
        with torch.no_grad():
            obs_t = torch.from_numpy(init_obs_np)
            st0 = model.initial_state(1, persist_s=s_cur)
            _, _, _, st1 = model.step_recurrent(obs_t, st0, t=0, active_proj=None)
            unc_ctx = torch.cat(
                (obs_t[:, 20:22], obs_t[:, 23:24], obs_t[:, 26:27], torch.abs(obs_t[:, 22:23]), obs_t[:, 34:39]), dim=-1
            )
            feats_u.append(torch.cat((st1.h, st1.s, unc_ctx), dim=-1)[0].clone())
            labels_u.append(1.0 if r0["first_pass_failed"] else 0.0)

            comp_ctx = torch.cat((obs_t[:, 20:24], obs_t[:, 36:40]), dim=-1)
            feats_c.append(torch.cat((s_cur, comp_ctx), dim=-1)[0].clone())
            if r0["success"] and r0["probes"] <= 4:
                labels_c.append(0)
            elif r0["success"] and r0["probes"] <= 8:
                labels_c.append(1)
            else:
                labels_c.append(2)

        s_cur = r0["next_s_persist"] if model.cfg.persist_slow else torch.zeros(1, model.cfg.d_s)

    if len(feats_u) < 16:
        return {"cal_16k_fail_rate": 0.0}

    Xu = torch.stack(feats_u, dim=0)
    yu = torch.tensor(labels_u, dtype=torch.float32)
    Xc = torch.stack(feats_c, dim=0)
    yc = torch.tensor(labels_c, dtype=torch.long)

    # Optimize ONLY unc_head (+ compute_head if enabled) with L2 regularization — zero change to GRU/router/stride!
    cal_params = list(model.unc_head.parameters())
    if model.cfg.use_unc_alloc:
        cal_params += list(model.compute_head.parameters())
    opt_cal = torch.optim.AdamW(cal_params, lr=3e-3, weight_decay=5e-2)
    model.train()
    for _ in range(16):
        opt_cal.zero_grad()
        u_log = model.unc_head(Xu).squeeze(-1)
        u_p = torch.sigmoid(u_log)
        loss_c = F.binary_cross_entropy_with_logits(u_log, yu) + 0.5 * F.mse_loss(u_p, yu)
        if model.cfg.use_unc_alloc:
            c_log = model.compute_head(Xc)
            loss_c = loss_c + 0.6 * F.cross_entropy(c_log, yc)
        loss_c.backward()
        opt_cal.step()
    model.eval()
    return {"cal_16k_fail_rate": float(yu.mean().item())}


def train_mcp08_controller(
    mem_16k: ExtendedPhysicalMemory,
    seed: int,
    cfg: MCP08Config = CONFIG_MCP08_54K,
    num_roots: int = 2048,
    dagger_roots: int = 768,
    epochs: int = 18,
    batch_size: int = 128,
    lr: float = 3.5e-3,
    horizon: int = 16,
    salt: int = 37,
    mms_options=None,
) -> Tuple[MCP08RecursiveController, Dict[str, float]]:
    """Train MCP-08 ~54K controller exclusively on 16K physical memory."""
    if mem_16k.n != 16384:
        raise ValueError(f"Training must occur exclusively at 16K (16384), got {mem_16k.n}")

    roots = sample_mixed_roots(mem_16k, num_roots, salt=salt)
    tr_teacher = generate_mcp08_traces(mem_16k, roots, seed=seed, horizon=horizon, policy_model=None)

    torch.manual_seed(seed)
    model = MCP08RecursiveController(cfg)
    mms=None
    if mms_options and mms_options.get('mib',0):
        from MCP.spatial_mms import SpatialMemory
        mms=SpatialMemory(model,mms_options)
    rng = np.random.default_rng(seed + 409)

    p1 = max(1, (epochs * 3) // 5)
    p2 = max(0, epochs - p1)

    opt1 = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=p1, eta_min=lr * 0.25)
    last_loss = _train_epochs_08(model, opt1, sch1, tr_teacher, p1, batch_size, horizon, rng, calibrate_weight=0.5,mms=mms)

    dagger_clean_rate = 0.0
    dagger_corr_rate = 0.0
    cal_stats: Dict[str, float] = {"cal_16k_fail_rate": 0.0}
    if p2 > 0:
        half = max(32, dagger_roots // 2)
        d_roots_clean = sample_mixed_roots(mem_16k, half, salt=salt + 991)
        d_roots_corr = sample_mixed_roots(mem_16k, half, salt=salt + 1993)
        tr_d1 = generate_mcp08_traces(
            mem_16k, d_roots_clean, seed=seed + 11, horizon=horizon, policy_model=model, corruption=0.0
        )
        tr_d2 = generate_mcp08_traces(
            mem_16k, d_roots_corr, seed=seed + 23, horizon=horizon, policy_model=model, corruption=0.10
        )
        dagger_clean_rate = tr_d1.verified_rate
        dagger_corr_rate = tr_d2.verified_rate

        combined = _concat_08(_concat_08(tr_teacher, tr_d1), tr_d2)
        opt2 = torch.optim.AdamW(model.parameters(), lr=lr * 0.65, weight_decay=1e-4)
        sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=p2, eta_min=lr * 0.05)
        last_loss = _train_epochs_08(
            model, opt2, sch2, combined, p2, batch_size, horizon, rng,
            calibrate_weight=1.4, empirical_unc_only=True,mms=mms,
        )
        # Phase 3: crisp post-convergence proper-scoring calibration strictly on 16K rollouts
        n_cal = min(512, max(128, dagger_roots // 2))
        cal_stats = _calibrate_heads_16k(model, mem_16k, seed=seed, n_cal=n_cal, salt=salt)

    model.eval()
    return model, {
        "teacher_verified_rate": tr_teacher.verified_rate,
        "dagger_16k_clean_rate": dagger_clean_rate,
        "dagger_16k_c10_rate": dagger_corr_rate,
        "final_train_loss": last_loss,
        **cal_stats,
        **({"mms":mms.receipt()} if mms is not None else {}),
        **model.parameter_summary(),
    }
