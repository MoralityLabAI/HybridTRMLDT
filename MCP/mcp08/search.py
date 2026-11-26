"""Verifier-guided recursive search engine, 256K error-recovery evaluator, and all MCP-08 baselines.

Implements:
  - Single-query error-recovery rollout with selective retries, hierarchical compute allocation,
    post-failure adaptive projection switching, and calibrated uncertainty telemetry (16K..256K).
  - Full multi-step verifier-guided planning engine across Storyworld, Canvas, and Lean-style
    domains with bounded backtracking, verified state restoration honoring irreversible locks,
    and charged TranspositionCache.
  - All 7 required comparison arms:
      1. `mcp07_orig` (original frozen MCP-07 checkpoints, zero retraining)
      2. `det_router_greedy`
      3. `uniform_retries`
      4. `greedy_planning`
      5. `beam_search`
      6. `bounded_symbolic`
      7. `mcp08_full`
  - All 7 inference-time lesions and slow-state activation-patching hooks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from MCP.mcp07.curriculum import HopSpec, QueryRoot
from MCP.mcp07.eval_harness import _det_choose_stride_batch, _det_router_choose_proj, rollout_batch as mcp07_rollout_batch
from MCP.mcp07.memory import (
    ALL_PROJECTIONS,
    ANCHOR_LOOKUP_COST,
    PAGE,
    PROJ_TO_ID,
    STRIDES,
    STRIDES_NP,
    STRUCTURED_PROJECTIONS,
    SWITCH_OVERHEAD_COST,
    ZERO_STRIDE_IDX,
)
from MCP.mcp07.models import (
    AdaptiveGeometryController,
    NUM_STRIDES,
    NUM_STRUCT_PROJS,
    build_cand_feats_single,
    compute_candidates_single,
)
from .memory256k import (
    CHECKPOINT_FRAME_BYTES,
    CacheEntry,
    ExtendedPhysicalMemory,
    TranspositionCache,
    CACHE_ENTRY_BYTES,
)
from .models import (
    COMPUTE_TIERS,
    MCP08RecursiveController,
    build_init_obs_08,
    build_step_obs_08,
)

CORE_PLANNING_ARMS: Tuple[str, ...] = (
    "mcp07_orig",
    "det_router_greedy",
    "uniform_retries",
    "greedy_planning",
    "beam_search",
    "bounded_symbolic",
    "mcp08_full",
)


@dataclass
class SearchLesionSpec:
    """Inference-time lesion or cross-domain slow-state activation-patch spec."""

    name: str = "none"
    reset_slow_per_step: bool = False
    shuffle_slow_donor: Optional[np.ndarray] = None  # [D] or [K, D]
    uniform_compute_tier: Optional[int] = None       # e.g. 1 (8 probes)
    disable_postfail_switch: bool = False
    disable_backtrack: bool = False
    disable_cache: bool = False
    corrupt_uncertainty: bool = False                # replace u_prob with 1 - u_prob
    disable_early_stop: bool = False                 # fixed-budget deep tier comparison (like MCP-07 Arm E)
    patch_slow_vec: Optional[np.ndarray] = None      # [d_s] cross-domain donor slow vector
    patch_alpha: float = 1.0


@torch.inference_mode()
def retrieve_single_mcp08(
    mem: ExtendedPhysicalMemory,
    hop: HopSpec,
    model: MCP08RecursiveController,
    s_persist: torch.Tensor,
    search_ctx: np.ndarray,
    corruption: float,
    max_retries: int = 2,
    uniform_tier: Optional[int] = None,
    uniform_blind_retry: bool = False,
    disable_early_stop: bool = False,
    lesion: Optional[SearchLesionSpec] = None,
    read_budget_remaining: int = 100000,
    retry_unc_threshold: float = 0.20,
    seed: int = 0,
) -> Dict[str, Any]:
    """Execute one verifier-guided memory lookup with hierarchical compute & selective retries."""
    ls = lesion if lesion is not None else SearchLesionSpec()
    s_in = (
        torch.zeros_like(s_persist)
        if (ls.reset_slow_per_step or not model.cfg.persist_slow)
        else s_persist.clone()
    )
    if ls.patch_slow_vec is not None:
        donor_t = torch.from_numpy(ls.patch_slow_vec.astype(np.float32))[None, :]
        s_in = (1.0 - ls.patch_alpha) * s_in + ls.patch_alpha * donor_t

    failed_mask = np.zeros(NUM_STRUCT_PROJS, dtype=np.float32)
    total_probes = 0
    total_raw = 0
    total_anchor_lookups = 0
    total_switches = 0
    total_fast = 0
    total_slow = 0
    retries_used = 0
    postfail_switches = 0
    stale_seen_any = False
    phys_slots_seen: set = set()
    final_val: Optional[int] = None
    first_pass_u_prob: float = 0.5
    first_pass_failed: bool = False
    first_pass_tier: int = 1
    first_pass_init_proj: int = 0
    retry_init_proj: int = -1
    cur_s = s_in
    last_h = torch.zeros(1, model.cfg.d_h)

    disable_pf_sw = ls.disable_postfail_switch or (not model.cfg.use_postfail_switch) or uniform_blind_retry
    # Preserve visited pages across adaptive retries (uniform_retries stays stateless/blind!)
    shared_visited: Dict[str, set] = {p: set() for p in STRUCTURED_PROJECTIONS}
    prev_attempt_end_page: Optional[int] = None
    prev_attempt_proj_id: int = 0
    prev_attempt_cap: int = 0
    prev_attempt_stale: bool = False
    prev_attempt_ref: Optional[Tuple[float, float]] = None
    prev_attempt_hints: Optional[Tuple[int, int, float, float]] = None

    for attempt in range(max_retries + 1):
        if read_budget_remaining <= 34:
            break
        ctx_att = search_ctx.copy()
        if attempt > 0:
            ctx_att[2] = 1.0  # verify_fail_cur
            ctx_att[3] = min(1.0, float(attempt) / 3.0)

        # Allocate explicit hierarchical compute tier
        init_obs_pre = build_init_obs_08(
            mem,
            np.array([hop.anchor], dtype=np.int32),
            np.array([hop.hint_x], dtype=np.int32),
            np.array([hop.hint_y], dtype=np.int32),
            np.array([hop.scale_x], dtype=np.float32),
            np.array([hop.scale_y], dtype=np.float32),
            16,
            search_ctx=ctx_att[None, :],
        )
        if uniform_tier is not None:
            tier = int(uniform_tier)
        elif ls.uniform_compute_tier is not None or (not model.cfg.use_unc_alloc):
            tier = 1 if ls.uniform_compute_tier is None else int(ls.uniform_compute_tier)
        else:
            t_logits = model.allocate_compute_tier(cur_s, torch.from_numpy(init_obs_pre))
            tier = int(t_logits.argmax(dim=-1).item())
            if attempt > 0:
                tier = min(2, max(tier, 1) + 1)  # escalate tier on retry

        if attempt == 0:
            first_pass_tier = tier

        probe_cap = min(COMPUTE_TIERS[tier], max(1, (read_budget_remaining - 2) // PAGE))
        ctx_att[7] = float(tier) / 2.0
        init_obs = build_init_obs_08(
            mem,
            np.array([hop.anchor], dtype=np.int32),
            np.array([hop.hint_x], dtype=np.int32),
            np.array([hop.hint_y], dtype=np.int32),
            np.array([hop.scale_x], dtype=np.float32),
            np.array([hop.scale_y], dtype=np.float32),
            probe_cap,
            search_ctx=ctx_att[None, :],
        )

        st = model.initial_state(1, persist_s=cur_s)
        fpm_t = torch.from_numpy(failed_mask[None, :])
        r0_logits, _, u0_logit, st = model.step_recurrent(
            torch.from_numpy(init_obs),
            st,
            t=0,
            active_proj=None,
            failed_proj_mask=fpm_t,
            disable_postfail_switch=disable_pf_sw,
        )
        total_fast += 1
        total_slow += 1

        if attempt == 0:
            u_raw = float(torch.sigmoid(u0_logit).item())
            first_pass_u_prob = (1.0 - u_raw) if ls.corrupt_uncertainty else u_raw

        # Resume in same optimal projection if Attempt 0 was a shallow tier-0 timeout without stale hit;
        # otherwise switch to an untried complementary projection on deep/stale failure!
        shallow_resume = (
            (attempt == 1)
            and (not uniform_blind_retry)
            and (prev_attempt_cap <= 4)
            and (not prev_attempt_stale)
            and (prev_attempt_end_page is not None)
        )
        if attempt > 0 and disable_pf_sw:
            cur_proj_id = first_pass_init_proj
        elif shallow_resume:
            cur_proj_id = prev_attempt_proj_id
        else:
            cur_proj_id = int(r0_logits.argmax(dim=-1).item())

        if attempt == 0:
            first_pass_init_proj = cur_proj_id
        else:
            retry_init_proj = cur_proj_id
            if cur_proj_id != first_pass_init_proj:
                postfail_switches += 1

        cur_proj_name = STRUCTURED_PROJECTIONS[cur_proj_id]
        cur_anchor = hop.anchor
        cur_hx, cur_hy = hop.hint_x, hop.hint_y
        cur_sx, cur_sy = hop.scale_x, hop.scale_y
        ref_x, ref_y = float(cur_anchor % mem.width), float(cur_anchor // mem.width)
        anchor_page = int(mem.views[cur_proj_name].inv[cur_anchor])
        visited = {p: set() for p in STRUCTURED_PROJECTIONS} if uniform_blind_retry else shared_visited
        stale_seen_hop = False

        if shallow_resume and prev_attempt_end_page is not None:
            # Continue from the last frontier page reached on the shallow 4-probe pass!
            cands_res, legal_res, _ = compute_candidates_single(
                mem, cur_proj_name, prev_attempt_end_page, cur_proj_name, visited
            )
            if legal_res.any():
                cf_res = build_cand_feats_single(
                    mem, cur_proj_name, prev_attempt_end_page, cur_proj_name, cands_res, legal_res,
                    ref_x, ref_y, cur_hx, cur_hy
                )
                s_l, _ = model.predict_stride(
                    st, torch.from_numpy(init_obs), torch.tensor([cur_proj_id], dtype=torch.long), torch.from_numpy(cf_res[None])
                )
                d_r = int(s_l[0].masked_fill(~torch.from_numpy(legal_res), -1e9).argmax().item())
                cur_page = int(cands_res[d_r])
                prev_page = prev_attempt_end_page
            else:
                cur_page = anchor_page
                prev_page = anchor_page
                total_anchor_lookups += ANCHOR_LOOKUP_COST
        else:
            cur_page = anchor_page
            prev_page = anchor_page
            total_anchor_lookups += ANCHOR_LOOKUP_COST
            if attempt > 0 and (not uniform_blind_retry) and prev_attempt_stale and prev_attempt_ref is not None and prev_attempt_hints is not None:
                # Preserve discovered mirror hint after a stale decoy hit across retry!
                stale_seen_hop = True
                ref_x, ref_y = prev_attempt_ref
                cur_hx, cur_hy, cur_sx, cur_sy = prev_attempt_hints
                if prev_attempt_end_page is not None:
                    prev_p_name = STRUCTURED_PROJECTIONS[prev_attempt_proj_id]
                    cands_br, legal_br, bpage_br = compute_candidates_single(
                        mem, prev_p_name, prev_attempt_end_page, cur_proj_name, visited
                    )
                    if legal_br.any():
                        cf_br = build_cand_feats_single(
                            mem, prev_p_name, prev_attempt_end_page, cur_proj_name, cands_br, legal_br,
                            ref_x, ref_y, cur_hx, cur_hy
                        )
                        s_l, _ = model.predict_stride(
                            st, torch.from_numpy(init_obs), torch.tensor([cur_proj_id], dtype=torch.long), torch.from_numpy(cf_br[None])
                        )
                        d_br = int(s_l[0].masked_fill(~torch.from_numpy(legal_br), -1e9).argmax().item())
                        cur_page = int(cands_br[d_br])
                        prev_page = bpage_br
                        total_switches += 1
                        read_budget_remaining -= SWITCH_OVERHEAD_COST
            elif attempt > 0 and (not uniform_blind_retry) and (cur_page in visited[cur_proj_name]):
                # Step immediately to an unvisited neighbor of anchor_page instead of re-reading anchor_page
                cands_u, legal_u, _ = compute_candidates_single(
                    mem, cur_proj_name, anchor_page, cur_proj_name, visited
                )
                if legal_u.any():
                    cf_u = build_cand_feats_single(
                        mem, cur_proj_name, anchor_page, cur_proj_name, cands_u, legal_u,
                        ref_x, ref_y, cur_hx, cur_hy
                    )
                    s_l, _ = model.predict_stride(
                        st, torch.from_numpy(init_obs), torch.tensor([cur_proj_id], dtype=torch.long), torch.from_numpy(cf_u[None])
                    )
                    d_u = int(s_l[0].masked_fill(~torch.from_numpy(legal_u), -1e9).argmax().item())
                    cur_page = int(cands_u[d_u])
                    prev_page = anchor_page

        solved = False
        target_key = int(mem.keys[hop.target_record])

        for t in range(probe_cap):
            visited[cur_proj_name].add(cur_page)
            val, stale_hit, _, slots = mem.probe_single(
                cur_proj_name, cur_page, target_key, corruption, decoy_slot=hop.decoy_slot
            )
            total_probes += 1
            total_raw += PAGE
            phys_slots_seen.update(slots.tolist())
            read_budget_remaining -= PAGE
            if stale_hit:
                stale_seen_any = True

            hit_valid = val is not None and val == int(mem.values[hop.target_record])
            if hit_valid:
                solved = True
                final_val = int(val)
            elif stale_hit and hop.decoy_slot >= 0 and not stale_seen_hop:
                stale_seen_hop = True
                cur_hx, cur_hy = hop.mirror_hint_x, hop.mirror_hint_y
                cur_sx, cur_sy = hop.mirror_scale_x, hop.mirror_scale_y
                c_s = mem.views[cur_proj_name].centers[cur_page]
                ref_x, ref_y = float(c_s[0]), float(c_s[1])

            s_obs = build_step_obs_08(
                mem,
                np.array([cur_proj_id], dtype=np.int32),
                np.array([cur_page], dtype=np.int32),
                np.array([anchor_page], dtype=np.int32),
                np.array([prev_page], dtype=np.int32),
                np.array([cur_anchor], dtype=np.int32),
                np.array([[ref_x, ref_y]], dtype=np.float32),
                np.array([cur_hx], dtype=np.int32),
                np.array([cur_hy], dtype=np.int32),
                np.array([cur_sx], dtype=np.float32),
                np.array([cur_sy], dtype=np.float32),
                np.array([probe_cap - t], dtype=np.int32),
                t,
                np.array([hit_valid], dtype=bool),
                np.array([solved], dtype=bool),
                np.array([stale_hit], dtype=bool),
                np.array([stale_seen_hop], dtype=bool),
                np.array([0], dtype=np.int32),
                np.array([False], dtype=bool),
                search_ctx=ctx_att[None, :],
                proj_names=STRUCTURED_PROJECTIONS,
            )
            prev_sc = st.slow_updates
            r_logits, stop_logit, u_step_logit, st = model.step_recurrent(
                torch.from_numpy(s_obs),
                st,
                t=t + 1,
                active_proj=torch.tensor([cur_proj_id], dtype=torch.long),
                failed_proj_mask=fpm_t,
                disable_postfail_switch=disable_pf_sw,
            )
            total_fast += 1
            if st.slow_updates > prev_sc:
                total_slow += 1

            # Learned adaptive halting (disabled only for fixed-budget uniform_deep comparison)
            if (not disable_early_stop) and (not ls.disable_early_stop) and float(stop_logit.item()) >= 0.0:
                break
            if t == probe_cap - 1 or read_budget_remaining <= PAGE:
                break

            nxt_proj_id = int(r_logits.argmax(dim=-1).item())
            nxt_proj_name = STRUCTURED_PROJECTIONS[nxt_proj_id]
            cands, legal, bpage = compute_candidates_single(
                mem, cur_proj_name, cur_page, nxt_proj_name, visited
            )
            if not legal.any():
                # Try staying in current projection if switched projection dead-ended
                nxt_proj_id = cur_proj_id
                nxt_proj_name = cur_proj_name
                cands, legal, bpage = compute_candidates_single(
                    mem, cur_proj_name, cur_page, nxt_proj_name, visited
                )
                if not legal.any():
                    break

            cf = build_cand_feats_single(
                mem, cur_proj_name, cur_page, nxt_proj_name, cands, legal, ref_x, ref_y, cur_hx, cur_hy
            )
            s_logits, _ = model.predict_stride(
                st,
                torch.from_numpy(s_obs),
                torch.tensor([nxt_proj_id], dtype=torch.long),
                torch.from_numpy(cf[None]),
            )
            d_i = int(s_logits[0].masked_fill(~torch.from_numpy(legal), -1e9).argmax().item())
            if nxt_proj_id != cur_proj_id:
                total_switches += 1
                read_budget_remaining -= SWITCH_OVERHEAD_COST
                prev_page = bpage
            else:
                prev_page = cur_page
            cur_proj_id = nxt_proj_id
            cur_proj_name = nxt_proj_name
            cur_page = int(cands[d_i])
            anchor_page = int(mem.views[cur_proj_name].inv[cur_anchor])

        cur_s = st.s.detach().clone()
        last_h = st.h.detach().clone()
        prev_attempt_end_page = cur_page
        prev_attempt_proj_id = cur_proj_id
        prev_attempt_cap = probe_cap
        prev_attempt_stale = stale_seen_hop
        prev_attempt_ref = (ref_x, ref_y)
        prev_attempt_hints = (cur_hx, cur_hy, cur_sx, cur_sy)

        if solved:
            break

        if attempt == 0:
            first_pass_failed = True
        # Only ban initial projection after a deep (>=8 probes) or stale failure (or attempt >= 1)
        # so shallow 4-probe passes can escalate cleanly in the optimal geometry first!
        if probe_cap >= 8 or stale_seen_hop or attempt >= 1:
            failed_mask[cur_proj_id] = 1.0
            failed_mask[first_pass_init_proj] = 1.0

        # Decide whether to trigger selective retry
        if attempt >= max_retries:
            break
        if uniform_blind_retry:
            retries_used += 1
            continue
        # Selective uncertainty-guided retry: retry if calibrated uncertainty is non-trivial or critical step
        u_now = float(torch.sigmoid(u_step_logit).item()) if total_probes > 0 else first_pass_u_prob
        if ls.corrupt_uncertainty:
            u_now = 1.0 - u_now
        should_retry = (
            (u_now >= retry_unc_threshold)
            or (first_pass_u_prob >= retry_unc_threshold)
            or (search_ctx[5] > 0.5)   # irreversible risk!
            or (not model.cfg.use_unc_alloc)
        )
        if not should_retry:
            break
        retries_used += 1

    charged = total_raw + total_anchor_lookups + SWITCH_OVERHEAD_COST * total_switches
    uniq_slots = len(phys_slots_seen)
    return {
        "success": final_val is not None,
        "value": final_val,
        "probes": total_probes,
        "raw_slot_reads": total_raw,
        "unique_physical_slots": uniq_slots,
        "overlap_slot_reads": total_raw - uniq_slots,
        "anchor_lookups": total_anchor_lookups,
        "switches": total_switches,
        "charged_reads": charged,
        "fast_updates": total_fast,
        "slow_updates": total_slow,
        "recurrent_updates": total_fast + total_slow,
        "retries": retries_used,
        "postfail_switches": postfail_switches,
        "stale_seen": stale_seen_any,
        "first_pass_failed": first_pass_failed,
        "recovered_after_fail": bool(first_pass_failed and final_val is not None),
        "first_pass_u_prob": first_pass_u_prob,
        "first_pass_tier": first_pass_tier,
        "first_pass_init_proj": first_pass_init_proj,
        "retry_init_proj": retry_init_proj,
        "next_s_persist": cur_s,
        "last_h": last_h,
    }


def _retrieve_single_det_router(
    mem: ExtendedPhysicalMemory,
    hop: HopSpec,
    budget: int,
    corruption: float,
    seed: int = 0,
) -> Dict[str, Any]:
    """Deterministic geometry-aware router retrieval wrapper returning MCP-08 receipt schema."""
    root = QueryRoot(family="plan", family_id=-1, hops=(hop,))
    recs, _ = mcp07_rollout_batch(mem, [root], arm="det_router", model=None, budget=budget, corruption=corruption, rng_seed=seed)
    r = recs[0]
    return {
        "success": bool(r["success"]),
        "value": r["value"],
        "probes": int(r["probes"]),
        "raw_slot_reads": int(r["raw_slot_reads"]),
        "unique_physical_slots": int(r["unique_physical_slots"]),
        "overlap_slot_reads": int(r["overlap_slot_reads"]),
        "anchor_lookups": int(r["anchor_lookups"]),
        "switches": int(r["switches"]),
        "charged_reads": int(r["charged_reads"]),
        "fast_updates": 0,
        "slow_updates": 0,
        "recurrent_updates": 0,
        "retries": 0,
        "postfail_switches": 0,
        "stale_seen": bool(r["stale_seen"]),
        "first_pass_failed": not bool(r["success"]),
        "recovered_after_fail": False,
        "first_pass_u_prob": 0.5,
        "first_pass_tier": 1,
        "next_s_persist": None,
        "last_h": None,
    }


def _retrieve_single_mcp07_orig(
    mem: ExtendedPhysicalMemory,
    hop: HopSpec,
    mcp07_model: AdaptiveGeometryController,
    budget: int,
    corruption: float,
    seed: int = 0,
) -> Dict[str, Any]:
    """Original frozen MCP-07 `hier_adaptive` retrieval wrapper (zero modifications to MCP-07)."""
    root = QueryRoot(family="plan", family_id=-1, hops=(hop,))
    recs, _ = mcp07_rollout_batch(
        mem, [root], arm="hier_adaptive", model=mcp07_model, budget=budget, corruption=corruption, rng_seed=seed
    )
    r = recs[0]
    return {
        "success": bool(r["success"]),
        "value": r["value"],
        "probes": int(r["probes"]),
        "raw_slot_reads": int(r["raw_slot_reads"]),
        "unique_physical_slots": int(r["unique_physical_slots"]),
        "overlap_slot_reads": int(r["overlap_slot_reads"]),
        "anchor_lookups": int(r["anchor_lookups"]),
        "switches": int(r["switches"]),
        "charged_reads": int(r["charged_reads"]),
        "fast_updates": int(r["fast_updates"]),
        "slow_updates": int(r["slow_updates"]),
        "recurrent_updates": int(r["recurrent_updates"]),
        "retries": 0,
        "postfail_switches": 0,
        "stale_seen": bool(r["stale_seen"]),
        "first_pass_failed": not bool(r["success"]),
        "recovered_after_fail": False,
        "first_pass_u_prob": 0.5,
        "first_pass_tier": 1,
        "next_s_persist": None,
        "last_h": None,
    }


def solve_domain_episode(
    env: Any,
    arm: str,
    mcp08_model: Optional[MCP08RecursiveController] = None,
    mcp07_model: Optional[AdaptiveGeometryController] = None,
    corruption: float = 0.10,
    seed: int = 107,
    lesion: Optional[SearchLesionSpec] = None,
    record_slow_trace: bool = False,
    archived_witness: Optional[Sequence[Sequence[int]]] = None,
    transposition_mib: Optional[float] = None,
) -> Dict[str, Any]:
    """Run one complete long-horizon episode on Storyworld, Canvas, or Lean-style search."""
    ls = lesion if lesion is not None else SearchLesionSpec()
    rng = np.random.default_rng(seed * 9973 + env.target_horizon * 31)
    env.verifier_calls = 0

    L = env.target_horizon
    # Matched episode compute budgets across all arms
    max_reads = 2200 * L + 6000
    max_verifier_calls = 45 * L + 140
    max_backtracks = max(16, L + 6)
    max_stack_depth = 8

    use_cache = (
        arm in ("mcp08_full", "bounded_symbolic")
        and (not ls.disable_cache)
        and (mcp08_model is None or mcp08_model.cfg.use_cache or arm == "bounded_symbolic")
    )
    use_backtrack = (
        arm in ("mcp08_full", "bounded_symbolic")
        and (not ls.disable_backtrack)
        and (mcp08_model is None or mcp08_model.cfg.use_backtrack or arm == "bounded_symbolic")
    )
    entries=1024
    if transposition_mib is not None:
        import math
        if not math.isfinite(transposition_mib) or transposition_mib<0:
            raise ValueError("invalid transposition budget")
        entries=int(transposition_mib*2**20)//CACHE_ENTRY_BYTES
        use_cache=use_cache and entries>0
    tcache = TranspositionCache(max_entries=entries, enabled=use_cache)

    # Retrieval record cache within episode so backtracking never re-pays for already-decoded options
    # at the same node (realistic working-set memory; lookup still goes through node record check).
    # Wait: to be strictly conservative, ONLY cache decoded options if transposition cache is enabled!
    decoded_opt_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}

    tot_probes = 0
    tot_raw = 0
    tot_uniq_slots = 0
    tot_overlap = 0
    tot_anchor_lookups = 0
    tot_switches = 0
    tot_charged = 0
    tot_fast = 0
    tot_slow = 0
    tot_retries = 0
    tot_postfail_sw = 0
    tot_queries = 0
    tot_retrieved = 0
    tot_recovered = 0
    tot_first_fail = 0
    backtracks_used = 0
    irrev_blocks_hit = 0
    tier_counts = [0, 0, 0]
    unc_preds: List[float] = []
    unc_labels: List[int] = []
    slow_trace: List[Dict[str, Any]] = []

    d_s = mcp08_model.cfg.d_s if mcp08_model is not None else 49
    d_h = mcp08_model.cfg.d_h if mcp08_model is not None else 49
    s_persist = torch.zeros(1, d_s)
    last_h_step = torch.zeros(1, d_h)

    def fetch_options_for_node(
        cur_state: Any,
        last_vfail: float,
        bt_count: int,
        has_irrev_hint: float,
        cache_hit_flag: float,
        only_js: Optional[Sequence[int]] = None,
        escalate_recovery: bool = False,
    ) -> Dict[int, Dict[str, Any]]:
        nonlocal tot_probes, tot_raw, tot_uniq_slots, tot_overlap, tot_anchor_lookups
        nonlocal tot_switches, tot_charged, tot_fast, tot_slow, tot_retries, tot_postfail_sw
        nonlocal tot_queries, tot_retrieved, tot_recovered, tot_first_fail, s_persist, last_h_step

        nid = cur_state.node_id
        prog, deficit = env.progress_metrics(cur_state)
        s_ctx = np.array(
            [
                prog,
                deficit,
                1.0 if escalate_recovery else last_vfail,
                0.5 if escalate_recovery else 0.0,
                min(1.0, float(bt_count) / float(max(1, max_backtracks))),
                has_irrev_hint,
                cache_hit_flag,
                1.0 if escalate_recovery else 0.5,
            ],
            dtype=np.float32,
        )

        out: Dict[int, Dict[str, Any]] = {}
        allowed = set(only_js) if only_js is not None else None
        for j, hop in env.option_queries(nid):
            if allowed is not None and j not in allowed:
                continue
            if use_cache and (nid, j) in decoded_opt_cache:
                tcache.lookups += 1
                tcache.hits += 1
                out[j] = decoded_opt_cache[(nid, j)]
                continue
            rem_reads = max_reads - (tot_charged + tcache.charged_lookup_reads)
            if rem_reads <= 34 or env.verifier_calls >= max_verifier_calls:
                break

            tot_queries += 1
            if arm == "mcp07_orig":
                assert mcp07_model is not None
                rec = _retrieve_single_mcp07_orig(env.mem, hop, mcp07_model, 8, corruption, seed + nid * 7 + j)
            elif arm in ("det_router_greedy", "bounded_symbolic"):
                det_b = 16 if escalate_recovery else 8
                rec = _retrieve_single_det_router(env.mem, hop, det_b, corruption, seed + nid * 7 + j + (97 if escalate_recovery else 0))
            elif arm == "uniform_retries":
                assert mcp08_model is not None
                rec = retrieve_single_mcp08(
                    env.mem,
                    hop,
                    mcp08_model,
                    s_persist,
                    s_ctx,
                    corruption=corruption,
                    max_retries=2,
                    uniform_tier=1,
                    uniform_blind_retry=True,
                    lesion=ls,
                    read_budget_remaining=rem_reads,
                    seed=seed + nid * 7 + j,
                )
            else:
                assert mcp08_model is not None
                rec = retrieve_single_mcp08(
                    env.mem,
                    hop,
                    mcp08_model,
                    s_persist,
                    s_ctx,
                    corruption=corruption,
                    max_retries=2 if escalate_recovery else 1,
                    uniform_tier=2 if escalate_recovery else None,
                    uniform_blind_retry=False,
                    lesion=ls,
                    read_budget_remaining=rem_reads,
                    seed=seed + nid * 7 + j + (97 if escalate_recovery else 0),
                )

            tot_probes += rec["probes"]
            tot_raw += rec["raw_slot_reads"]
            tot_uniq_slots += rec["unique_physical_slots"]
            tot_overlap += rec["overlap_slot_reads"]
            tot_anchor_lookups += rec["anchor_lookups"]
            tot_switches += rec["switches"]
            tot_charged += rec["charged_reads"]
            tot_fast += rec["fast_updates"]
            tot_slow += rec["slow_updates"]
            tot_retries += rec["retries"]
            tot_postfail_sw += rec["postfail_switches"]
            tier_counts[int(rec["first_pass_tier"])] += 1
            if rec["first_pass_failed"]:
                tot_first_fail += 1
            if rec["recovered_after_fail"]:
                tot_recovered += 1
            unc_preds.append(float(rec["first_pass_u_prob"]))
            unc_labels.append(1 if rec["first_pass_failed"] else 0)
            if rec["next_s_persist"] is not None:
                s_persist = rec["next_s_persist"]
            if rec["last_h"] is not None:
                last_h_step = rec["last_h"]

            if rec["success"] and rec["value"] is not None:
                dec = env.decode_record(nid, j, int(rec["value"]))
                if dec is not None:
                    tot_retrieved += 1
                    out[j] = dec
                    if use_cache:
                        decoded_opt_cache[(nid, j)] = dec
        return out

    # =========================================================================
    # Arm 5: External-Memory Beam Search (`beam_search`, width = 3)
    # =========================================================================
    if arm == "beam_search":
        beam_w = 3
        init_st = env.initial_state()
        # Beam entry: (score, state, route_list)
        beam: List[Tuple[float, Any, List[Tuple[int, int]]]] = [(0.0, init_st, [])]
        solved_route: Optional[List[Tuple[int, int]]] = None

        for _depth in range(L + 4):
            if not beam or (tot_charged >= max_reads) or (env.verifier_calls >= max_verifier_calls):
                break
            next_beam: List[Tuple[float, Any, List[Tuple[int, int]]]] = []
            for _sc, b_st, b_route in beam:
                if b_st.node_id == env.terminal_id:
                    ver_chk = env.verify_final_route(b_route, archived_witness=archived_witness)
                    if ver_chk.get("all_five_verified", ver_chk.get("all_verified", False)):
                        solved_route = b_route
                        break
                    continue
                if b_st.node_id == env.dead_sink_id:
                    continue
                opts = fetch_options_for_node(b_st, 0.0, 0, 0.0, 0.0)
                if not opts:
                    j_fb = int(rng.integers(0, 3))
                    n_st, info = env.execute_ground_truth_action(b_st, j_fb)
                    if info["solved"]:
                        solved_route = [*b_route, (b_st.node_id, j_fb)]
                        break
                    if not info["dead_end"]:
                        next_beam.append((info["score"], n_st, [*b_route, (b_st.node_id, j_fb)]))
                else:
                    for j, opt in opts.items():
                        n_st, info = env.step_and_verify(b_st, opt)
                        if info["solved"]:
                            solved_route = [*b_route, (b_st.node_id, j)]
                            break
                        if not info["dead_end"]:
                            next_beam.append((info["score"], n_st, [*b_route, (b_st.node_id, j)]))
                    if solved_route is not None:
                        break
            if solved_route is not None:
                break
            # Keep top beam_w distinct states
            next_beam.sort(key=lambda item: item[0], reverse=True)
            seen_keys = set()
            pruned = []
            for item in next_beam:
                sk = env.state_key(item[1])
                if sk not in seen_keys:
                    seen_keys.add(sk)
                    pruned.append(item)
                if len(pruned) >= beam_w:
                    break
            beam = pruned

        final_route = solved_route if solved_route is not None else (beam[0][2] if beam else [])
        final_ver = env.verify_final_route(final_route, archived_witness=archived_witness)
        solved_flag = bool(final_ver.get("all_five_verified", final_ver.get("all_verified", False)))
        return _pack_episode_receipt(
            env, arm, seed, corruption, solved_flag, final_ver,
            tot_probes, tot_raw, tot_uniq_slots, tot_overlap, tot_anchor_lookups,
            tot_switches, tot_charged, tot_fast, tot_slow, tot_retries, tot_postfail_sw,
            tot_queries, tot_retrieved, tot_first_fail, tot_recovered,
            backtracks_used, irrev_blocks_hit, tier_counts, tcache, 0,
            unc_preds, unc_labels, slow_trace,
        )

    # =========================================================================
    # Sequential / Verifier-Guided Backtracking Search (`mcp08_full`, `bounded_symbolic`,
    # `greedy_planning`, `uniform_retries`, `det_router_greedy`, `mcp07_orig`)
    # =========================================================================
    cur_st = env.initial_state()
    route: List[Tuple[int, int]] = []
    # Bounded verified checkpoint stack: each frame stores
    # (state, route_len, remaining_candidates_sorted, s_persist_snapshot, stack_step_idx)
    checkpoint_stack: List[Dict[str, Any]] = []
    lock_route_len: int = 0  # Irreversible barrier: cannot restore any checkpoint with route_len < lock_route_len!
    last_vfail = 0.0
    cur_state_dead = False
    max_iters = L * 5 + 60
    solved_flag = False

    for _iter in range(max_iters):
        if (tot_charged + tcache.charged_lookup_reads >= max_reads) or (env.verifier_calls >= max_verifier_calls):
            break

        if cur_st.node_id == env.terminal_id:
            ver_now = env.verify_final_route(route, archived_witness=archived_witness)
            if ver_now.get("all_five_verified", ver_now.get("all_verified", False)):
                solved_flag = True
                break
            # Terminal reached but failed projector/clue/counterfactual gate -> treat as dead end!
            tcache.mark_dead_end(env.state_key(cur_st), depth=len(route))
            last_vfail = 1.0
            cur_state_dead = False
            if not use_backtrack:
                break
            # Attempt bounded backtracking to latest reversible verified checkpoint
            restored = False
            while checkpoint_stack and backtracks_used < max_backtracks:
                frame = checkpoint_stack[-1]
                if frame["route_len"] < lock_route_len:
                    irrev_blocks_hit += 1
                    checkpoint_stack.pop()
                    continue
                if (not frame["remaining"]) and (not frame.get("unret_js")):
                    tcache.mark_dead_end(env.state_key(frame["state"]), depth=frame["route_len"])
                    checkpoint_stack.pop()
                    continue
                # Restore verified checkpoint state!
                cur_st = frame["state"]
                route = route[: frame["route_len"]]
                if mcp08_model is not None and mcp08_model.cfg.persist_slow and not ls.reset_slow_per_step:
                    s_persist = 0.5 * s_persist + 0.5 * frame["s_snap"]
                backtracks_used += 1
                restored = True
                break
            if not restored:
                break
            continue

        # Check transposition cache before expanding current state
        skey = env.state_key(cur_st)
        cached = tcache.lookup(skey)
        cache_hit_flag = 1.0 if cached is not None else 0.0
        is_cached_dead = (
            cur_state_dead
            or (cached is not None and cached.status == "dead_end")
            or (cur_st.node_id == env.dead_sink_id)
        )

        if is_cached_dead:
            tcache.mark_dead_end(skey, depth=len(route))
            last_vfail = 1.0
            cur_state_dead = False
            if not use_backtrack:
                break
            restored = False
            while checkpoint_stack and backtracks_used < max_backtracks:
                frame = checkpoint_stack[-1]
                if frame["route_len"] < lock_route_len:
                    irrev_blocks_hit += 1
                    checkpoint_stack.pop()
                    continue
                if (not frame["remaining"]) and (not frame.get("unret_js")):
                    tcache.mark_dead_end(env.state_key(frame["state"]), depth=frame["route_len"])
                    checkpoint_stack.pop()
                    continue
                cur_st = frame["state"]
                route = route[: frame["route_len"]]
                if mcp08_model is not None and mcp08_model.cfg.persist_slow and not ls.reset_slow_per_step:
                    s_persist = 0.5 * s_persist + 0.5 * frame["s_snap"]
                backtracks_used += 1
                restored = True
                break
            if not restored:
                break
            continue

        # Check if we just restored a checkpoint frame with pre-ranked remaining untried candidates
        if checkpoint_stack and checkpoint_stack[-1]["route_len"] == len(route) and checkpoint_stack[-1]["state"] == cur_st:
            frame = checkpoint_stack[-1]
            # If all initially retrieved candidates were exhausted, run verifier-guided escalated
            # external-memory recovery on any options missed during the cheap first pass!
            if (not frame["remaining"]) and frame.get("unret_js"):
                miss_js = list(frame["unret_js"])
                frame["unret_js"] = []
                rec_opts = fetch_options_for_node(
                    cur_st, 1.0, backtracks_used, 0.0, cache_hit_flag, only_js=miss_js, escalate_recovery=True
                )
                extra_cands = []
                for rj, ropt in rec_opts.items():
                    rnxt, rinfo = env.step_and_verify(cur_st, ropt)
                    rn_key = env.state_key(rnxt)
                    rn_c = tcache.lookup(rn_key)
                    if rn_c is not None and rn_c.status == "dead_end":
                        continue
                    if rinfo["dead_end"]:
                        tcache.mark_dead_end(rn_key, depth=len(route) + 1)
                        continue
                    extra_cands.append((rinfo["score"] + float(rng.uniform(-0.01, 0.01)), rj, ropt, rnxt, rinfo))
                extra_cands.sort(key=lambda item: item[0], reverse=True)
                frame["remaining"].extend([(c[1], c[2], c[3], c[4]) for c in extra_cands])

            # Skip any remaining candidate whose target state was subsequently marked dead in cache!
            picked = None
            while frame["remaining"]:
                cand = frame["remaining"].pop(0)
                c_cached = tcache.lookup(env.state_key(cand[2]))
                if c_cached is not None and c_cached.status == "dead_end":
                    continue
                picked = cand
                break
            if picked is None:
                tcache.mark_dead_end(skey, depth=len(route))
                checkpoint_stack.pop()
                last_vfail = 1.0
                cur_state_dead = True
                continue
            chosen_j, chosen_opt, nxt_st, nxt_info = picked
            if nxt_info["irreversible"]:
                lock_route_len = len(route) + 1
            route.append((cur_st.node_id, chosen_j))
            cur_st = nxt_st
            cur_state_dead = bool(nxt_info["dead_end"])
            last_vfail = 1.0 if cur_state_dead else 0.0
            continue

        # Retrieve scene options from external physical memory
        opts = fetch_options_for_node(cur_st, last_vfail, backtracks_used, 0.0, cache_hit_flag)

        if record_slow_trace and mcp08_model is not None:
            prog_now, def_now = env.progress_metrics(cur_st)
            slow_trace.append(
                {
                    "domain": env.domain,
                    "world_id": env.world_id,
                    "step": len(route),
                    "horizon": L,
                    "progress": prog_now,
                    "deficit": def_now,
                    "verify_fail": last_vfail,
                    "backtracks": backtracks_used,
                    "u_prob": unc_preds[-1] if unc_preds else 0.5,
                    "s": s_persist[0].numpy().copy(),
                    "h": last_h_step[0].numpy().copy(),
                }
            )

        # Rank retrieved candidates using native verifier preview (and cache dead-end checks!)
        candidates: List[Tuple[float, int, Dict[str, Any], Any, Dict[str, Any]]] = []
        for j, opt in opts.items():
            nxt_st, info = env.step_and_verify(cur_st, opt)
            nxt_key = env.state_key(nxt_st)
            nxt_cached = tcache.lookup(nxt_key)
            if nxt_cached is not None and nxt_cached.status == "dead_end":
                continue
            if info["dead_end"]:
                tcache.mark_dead_end(nxt_key, depth=len(route) + 1)
                continue
            # Extra caution before irreversible transitions when non-irreversible viable options exist
            sc = info["score"] + float(rng.uniform(-0.01, 0.01))
            candidates.append((sc, j, opt, nxt_st, info))

        candidates.sort(key=lambda item: item[0], reverse=True)

        unret_js = [j for j in range(3) if j not in opts]
        if candidates:
            _, chosen_j, chosen_opt, nxt_st, nxt_info = candidates[0]
            # Push remaining reversible alternatives (and any unretrieved option slots for
            # verifier-guided recovery on backtrack!) onto bounded checkpoint stack!
            if use_backtrack and (len(candidates) > 1 or len(unret_js) > 0):
                rem_list = [(c[1], c[2], c[3], c[4]) for c in candidates[1:]]
                checkpoint_stack.append(
                    {
                        "state": cur_st,
                        "route_len": len(route),
                        "remaining": rem_list,
                        "unret_js": unret_js,
                        "s_snap": s_persist.clone(),
                    }
                )
                if len(checkpoint_stack) > max_stack_depth:
                    checkpoint_stack.pop(0)
            if nxt_info["irreversible"]:
                lock_route_len = len(route) + 1
            route.append((cur_st.node_id, chosen_j))
            cur_st = nxt_st
            cur_state_dead = bool(nxt_info["dead_end"])
            last_vfail = 1.0 if cur_state_dead else 0.0
        else:
            # Either all retrieved options were dead ends OR retrieval missed all options!
            if len(opts) == 3:
                # All 3 options verified dead ends -> mark state dead & backtrack!
                tcache.mark_dead_end(skey, depth=len(route))
                last_vfail = 1.0
                if not use_backtrack:
                    # Forced to step anyway if greedy
                    j_fb = int(rng.integers(0, 3))
                    nxt_st, info = env.execute_ground_truth_action(cur_st, j_fb)
                    route.append((cur_st.node_id, j_fb))
                    cur_st = nxt_st
                    cur_state_dead = bool(info["dead_end"])
                else:
                    restored = False
                    while checkpoint_stack and backtracks_used < max_backtracks:
                        frame = checkpoint_stack[-1]
                        if frame["route_len"] < lock_route_len:
                            irrev_blocks_hit += 1
                            checkpoint_stack.pop()
                            continue
                        if (not frame["remaining"]) and (not frame.get("unret_js")):
                            tcache.mark_dead_end(env.state_key(frame["state"]), depth=frame["route_len"])
                            checkpoint_stack.pop()
                            continue
                        cur_st = frame["state"]
                        route = route[: frame["route_len"]]
                        if mcp08_model is not None and mcp08_model.cfg.persist_slow and not ls.reset_slow_per_step:
                            s_persist = 0.5 * s_persist + 0.5 * frame["s_snap"]
                        backtracks_used += 1
                        cur_state_dead = False
                        restored = True
                        break
                    if not restored:
                        break
            else:
                # Some options were NOT retrieved -> pick an unretrieved option blindly (paying the risk!)
                unret = [j for j in range(3) if j not in opts]
                j_fb = int(rng.choice(unret)) if unret else int(rng.integers(0, 3))
                nxt_st, info = env.execute_ground_truth_action(cur_st, j_fb)
                if info["irreversible"]:
                    lock_route_len = len(route) + 1
                route.append((cur_st.node_id, j_fb))
                cur_st = nxt_st
                cur_state_dead = bool(info["dead_end"])
                last_vfail = 1.0 if cur_state_dead else 0.0

    final_ver = env.verify_final_route(route, archived_witness=archived_witness)
    solved_flag = bool(final_ver.get("all_five_verified", final_ver.get("all_verified", False)))
    return _pack_episode_receipt(
        env, arm, seed, corruption, solved_flag, final_ver,
        tot_probes, tot_raw, tot_uniq_slots, tot_overlap, tot_anchor_lookups,
        tot_switches, tot_charged, tot_fast, tot_slow, tot_retries, tot_postfail_sw,
        tot_queries, tot_retrieved, tot_first_fail, tot_recovered,
        backtracks_used, irrev_blocks_hit, tier_counts, tcache,
        max_stack_depth * CHECKPOINT_FRAME_BYTES if use_backtrack else 0,
        unc_preds, unc_labels, slow_trace,
    )


def _pack_episode_receipt(
    env: Any,
    arm: str,
    seed: int,
    corruption: float,
    solved_flag: bool,
    final_ver: Dict[str, Any],
    tot_probes: int,
    tot_raw: int,
    tot_uniq_slots: int,
    tot_overlap: int,
    tot_anchor_lookups: int,
    tot_switches: int,
    tot_charged: int,
    tot_fast: int,
    tot_slow: int,
    tot_retries: int,
    tot_postfail_sw: int,
    tot_queries: int,
    tot_retrieved: int,
    tot_first_fail: int,
    tot_recovered: int,
    backtracks_used: int,
    irrev_blocks_hit: int,
    tier_counts: List[int],
    tcache: TranspositionCache,
    stack_bytes: int,
    unc_preds: List[float],
    unc_labels: List[int],
    slow_trace: List[Dict[str, Any]],
) -> Dict[str, Any]:
    c_stats = tcache.stats()
    idx_cost = env.mem.index_cost_for_projections(STRUCTURED_PROJECTIONS)
    total_charged_reads = int(tot_charged + c_stats["cache_charged_reads"])
    return {
        "domain": env.domain,
        "world_id": env.world_id,
        "bucket": env.bucket,
        "target_horizon": env.target_horizon,
        "capacity": env.mem.n,
        "arm": arm,
        "seed": seed,
        "corruption": corruption,
        "success": solved_flag,
        "verification": final_ver,
        "probes": int(tot_probes),
        "raw_slot_reads": int(tot_raw),
        "unique_physical_slots": int(tot_uniq_slots),
        "overlap_slot_reads": int(tot_overlap),
        "anchor_lookups": int(tot_anchor_lookups),
        "switches": int(tot_switches),
        "charged_memory_reads": total_charged_reads,
        "verifier_calls": int(env.verifier_calls),
        "fast_updates": int(tot_fast),
        "slow_updates": int(tot_slow),
        "recurrent_updates": int(tot_fast + tot_slow),
        "queries": int(tot_queries),
        "retrieved": int(tot_retrieved),
        "retrieval_rate": float(tot_retrieved) / float(max(1, tot_queries)),
        "first_pass_failures": int(tot_first_fail),
        "recovered_after_fail": int(tot_recovered),
        "error_recovery_rate": float(tot_recovered) / float(max(1, tot_first_fail)) if tot_first_fail > 0 else 1.0,
        "retries_triggered": int(tot_retries),
        "post_failure_switches": int(tot_postfail_sw),
        "backtracks_executed": int(backtracks_used),
        "irreversible_blocks_hit": int(irrev_blocks_hit),
        "tier_counts": list(tier_counts),
        "index_storage_bytes": int(idx_cost["index_storage_bytes"]),
        "checkpoint_stack_bytes": int(stack_bytes),
        **c_stats,
        "_unc_preds": unc_preds,
        "_unc_labels": unc_labels,
        "_slow_trace": slow_trace,
    }
