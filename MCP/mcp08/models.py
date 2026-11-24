"""MCP-08 ~54K Hierarchical Recursive Controller with Persistent Slow State & Calibration.

Extends MCP-07's coupled fast/slow recurrent controller with:
  1. Persistent slow state `s_persist` carried across sequential decisions, retries, and backtracks
  2. Calibrated uncertainty estimator `unc_head` (predicting retrieval/verification failure prob)
  3. Explicit hierarchical compute allocation `compute_head` -> tiers {4, 8, 16} probes
  4. Post-failure adaptive projection switching (router reads observable `failed_proj_mask`)
  5. Recursive meta-control `meta_action_head` -> [proceed, retry, backtrack, halt]
  6. First-class hooks for inference-time lesions and cross-domain slow-state activation patching
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from MCP.mcp07.memory import (
    ALL_PROJECTIONS,
    COARSE_MASK_NP,
    FINE_MASK_NP,
    STRIDES,
    STRIDES_NP,
    STRUCTURED_PROJECTIONS,
    ZERO_STRIDE_IDX,
)
from MCP.mcp07.models import CAND_FEAT_DIM, NUM_STRIDES, NUM_STRUCT_PROJS, optimal_action_set_loss

OBS_DIM_08: int = 42
COMPUTE_TIERS: Tuple[int, int, int] = (4, 8, 16)

# Observation layout (42 dims, strictly non-oracle):
# [0..3]   active projection one-hot (4)
# [4..7]   normalized page coords & relative strides (4)
# [8..9]   anchor_x/W, anchor_y/H (2)
# [10..13] anchor dyadic phase (ax%16, ay%16, ax%32, ay%32) (4)
# [14..17] physical centroid displacement from reference (/32 and /8) (4)
# [18..23] observable query geometry (hx, hy, sx, sy, hx*hy, |sx-sy|) (6)
# [24..25] signed progress along hinted axes (2)
# [26..33] budget, step, hit_cur, solved, stale_cur, stale_ever, hop_idx, hop_adv (8)
# [34..41] plan_progress, goal_deficit, verify_fail_cur, retry_norm, backtrack_norm,
#          irreversible_risk, cache_hit_cur, compute_tier_norm (8)
GEOM_SLICE_08 = slice(10, 26)      # 16 spatial geometry + displacement + signed-progress dims
SEARCH_CTX_SLICE = slice(34, 42)   # 8 verifier-guided recursive search dims
EVENT_STALE_IDX_08 = 30
EVENT_HOP_ADV_IDX_08 = 33
EVENT_VERIFY_FAIL_IDX_08 = 36


@dataclass(frozen=True)
class MCP08Config:
    """Architecture config sized near 54K active parameters."""

    d_obs: int = OBS_DIM_08
    d_h: int = 49
    d_s: int = 49
    d_p: int = 16
    d_router_hid: int = 32
    d_stride_hid: int = 44
    d_unc_hid: int = 20
    num_projs: int = NUM_STRUCT_PROJS
    num_strides: int = NUM_STRIDES
    use_comm: bool = True
    slow_period: int = 2
    persist_slow: bool = True
    use_unc_alloc: bool = True
    use_postfail_switch: bool = True
    use_backtrack: bool = True
    use_cache: bool = True


CONFIG_MCP08_54K = MCP08Config()


@dataclass
class MCP08State:
    h: torch.Tensor        # [B, d_h]
    s: torch.Tensor        # [B, d_s] persistent slow state
    p_mem: torch.Tensor    # [B, 4, d_p]
    fast_updates: int = 0
    slow_updates: int = 0


def build_init_obs_08(
    mem,
    anchors: np.ndarray,
    hx: np.ndarray,
    hy: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    budget: int,
    search_ctx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build strictly non-oracle [B, 42] initial observation."""
    bsz = len(anchors)
    obs = np.zeros((bsz, OBS_DIM_08), dtype=np.float32)
    ax = (anchors % mem.width).astype(np.float32)
    ay = (anchors // mem.width).astype(np.float32)
    obs[:, 8] = ax / float(mem.width)
    obs[:, 9] = ay / float(mem.height)
    obs[:, 10] = (ax % 16.0) / 16.0
    obs[:, 11] = (ay % 16.0) / 16.0
    obs[:, 12] = (ax % 32.0) / 32.0
    obs[:, 13] = (ay % 32.0) / 32.0
    obs[:, 18] = hx.astype(np.float32)
    obs[:, 19] = hy.astype(np.float32)
    obs[:, 20] = sx.astype(np.float32)
    obs[:, 21] = sy.astype(np.float32)
    obs[:, 22] = (hx * hy).astype(np.float32)
    obs[:, 23] = np.abs(sx - sy).astype(np.float32)
    obs[:, 26] = float(budget) / 32.0
    if search_ctx is not None:
        obs[:, 34:42] = search_ctx.astype(np.float32)
    return obs


def build_step_obs_08(
    mem,
    proj_ids: np.ndarray,
    pages: np.ndarray,
    anchor_pages: np.ndarray,
    prev_pages: np.ndarray,
    anchors: np.ndarray,
    ref_xy: np.ndarray,
    hx: np.ndarray,
    hy: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    remaining: np.ndarray,
    step: int,
    hit_cur: np.ndarray,
    solved_all: np.ndarray,
    stale_cur: np.ndarray,
    stale_ever: np.ndarray,
    hop_idx: np.ndarray,
    hop_adv: np.ndarray,
    search_ctx: Optional[np.ndarray] = None,
    proj_names: Tuple[str, ...] = STRUCTURED_PROJECTIONS,
) -> np.ndarray:
    """Build strictly non-oracle [B, 42] step observation."""
    bsz = len(pages)
    obs = np.zeros((bsz, OBS_DIM_08), dtype=np.float32)
    clamped_p = np.minimum(proj_ids, NUM_STRUCT_PROJS - 1)
    obs[np.arange(bsz), clamped_p] = 1.0

    inv_m = 1.0 / float(mem.pages)
    obs[:, 4] = pages.astype(np.float32) * inv_m
    obs[:, 5] = anchor_pages.astype(np.float32) * inv_m
    obs[:, 6] = np.clip((pages - anchor_pages).astype(np.float32) / 32.0, -4.0, 4.0)
    obs[:, 7] = np.clip((pages - prev_pages).astype(np.float32) / 32.0, -4.0, 4.0)

    ax = (anchors % mem.width).astype(np.float32)
    ay = (anchors // mem.width).astype(np.float32)
    obs[:, 8] = ax / float(mem.width)
    obs[:, 9] = ay / float(mem.height)
    obs[:, 10] = (ax % 16.0) / 16.0
    obs[:, 11] = (ay % 16.0) / 16.0
    obs[:, 12] = (ax % 32.0) / 32.0
    obs[:, 13] = (ay % 32.0) / 32.0

    centers = np.empty((bsz, 2), dtype=np.float32)
    for pid in np.unique(proj_ids):
        mask = proj_ids == pid
        centers[mask] = mem.views[proj_names[int(pid)]].centers[pages[mask]]

    dx_ref = centers[:, 0] - ref_xy[:, 0]
    dy_ref = centers[:, 1] - ref_xy[:, 1]
    obs[:, 14] = np.clip(dx_ref / 32.0, -4.0, 4.0)
    obs[:, 15] = np.clip(dy_ref / 32.0, -4.0, 4.0)
    obs[:, 16] = np.clip(dx_ref / 8.0, -4.0, 4.0)
    obs[:, 17] = np.clip(dy_ref / 8.0, -4.0, 4.0)

    obs[:, 18] = hx.astype(np.float32)
    obs[:, 19] = hy.astype(np.float32)
    obs[:, 20] = sx.astype(np.float32)
    obs[:, 21] = sy.astype(np.float32)
    obs[:, 22] = (hx * hy).astype(np.float32)
    obs[:, 23] = np.abs(sx - sy).astype(np.float32)
    obs[:, 24] = np.clip((dx_ref * hx.astype(np.float32)) / 32.0, -4.0, 4.0)
    obs[:, 25] = np.clip((dy_ref * hy.astype(np.float32)) / 32.0, -4.0, 4.0)

    obs[:, 26] = remaining.astype(np.float32) / 32.0
    obs[:, 27] = float(step) / 32.0
    obs[:, 28] = hit_cur.astype(np.float32)
    obs[:, 29] = solved_all.astype(np.float32)
    obs[:, 30] = stale_cur.astype(np.float32)
    obs[:, 31] = stale_ever.astype(np.float32)
    obs[:, 32] = hop_idx.astype(np.float32)
    obs[:, 33] = hop_adv.astype(np.float32)
    if search_ctx is not None:
        obs[:, 34:42] = search_ctx.astype(np.float32)
    return obs


class MCP08RecursiveController(nn.Module):
    """~54K hierarchical recurrent controller for verifier-guided search and error recovery."""

    def __init__(self, cfg: MCP08Config = CONFIG_MCP08_54K):
        super().__init__()
        self.cfg = cfg

        self.fast_gru = nn.GRUCell(cfg.d_obs + cfg.d_s + cfg.d_p, cfg.d_h)
        # Slow recurrent state directly integrates fast summary + 8 domain-agnostic search-progress channels
        self.slow_gru = nn.GRUCell(cfg.d_h + 8, cfg.d_s)

        # Projection-specific state slots + cross-projection communication
        self.proj_gate = nn.Linear(cfg.d_h + cfg.d_p, 2 * cfg.d_p)
        self.comm_proj = nn.Linear(cfg.d_p, cfg.d_p)

        # Router reads [s, h, comm, spatial_geom(16), failed_proj_mask(4)]
        self.router_head = nn.Sequential(
            nn.Linear(cfg.d_s + cfg.d_h + cfg.d_p + 16 + cfg.num_projs, cfg.d_router_hid),
            nn.ReLU(),
            nn.Linear(cfg.d_router_hid, cfg.num_projs),
        )

        # Candidate-centroid bilinear stride + fine/coarse resolution head
        ctx_in = cfg.d_h + cfg.d_s + cfg.num_projs + 12
        self.stride_ctx = nn.Linear(ctx_in, cfg.d_stride_hid)
        self.stride_cand = nn.Linear(CAND_FEAT_DIM, cfg.d_stride_hid)
        self.stride_out = nn.Linear(cfg.d_stride_hid, 1)
        self.res_head = nn.Linear(cfg.d_stride_hid, 2)

        # Adaptive stop head
        self.stop_head = nn.Linear(cfg.d_h + cfg.d_s + 4, 1)

        # Calibrated uncertainty estimator (predicts retrieval/verification failure probability)
        self.unc_head = nn.Sequential(
            nn.Linear(cfg.d_h + cfg.d_s + 10, cfg.d_unc_hid),
            nn.ReLU(),
            nn.Linear(cfg.d_unc_hid, 1),
        )

        # Explicit hierarchical compute tier allocator -> {tier_0: 4, tier_1: 8, tier_2: 16}
        # Reads persistent slow state `s_persist` + 4 query-scale cues + 4 immediate verifier flags
        # (intentionally omits raw progress/deficit bypass so `s_persist` is causally required!)
        self.compute_head = nn.Linear(cfg.d_s + 8, 3)

        # Meta-search controller -> [proceed, retry, backtrack, halt]
        # Reads [h, s, u_prob, verify_fail, retry_norm, bt_norm, irrev_risk]
        self.meta_action_head = nn.Linear(cfg.d_h + cfg.d_s + 5, 4)

        self.register_buffer("fine_mask", torch.from_numpy(FINE_MASK_NP.astype(np.float32)))
        self.register_buffer("coarse_mask", torch.from_numpy(COARSE_MASK_NP.astype(np.float32)))

    def initial_state(
        self,
        batch_size: int,
        persist_s: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> MCP08State:
        dev = device or next(self.parameters()).device
        s0 = (
            persist_s.clone()
            if (persist_s is not None and self.cfg.persist_slow)
            else torch.zeros(batch_size, self.cfg.d_s, device=dev)
        )
        return MCP08State(
            h=torch.zeros(batch_size, self.cfg.d_h, device=dev),
            s=s0,
            p_mem=torch.zeros(batch_size, self.cfg.num_projs, self.cfg.d_p, device=dev),
            fast_updates=0,
            slow_updates=0,
        )

    def parameter_summary(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        unc_p = sum(p.numel() for p in self.unc_head.parameters())
        comp_p = sum(p.numel() for p in self.compute_head.parameters())
        meta_p = sum(p.numel() for p in self.meta_action_head.parameters())
        router_p = sum(p.numel() for p in self.router_head.parameters())
        return {
            "total_params": int(total),
            "active_params": int(total),
            "router_params": int(router_p),
            "unc_params": int(unc_p),
            "compute_params": int(comp_p),
            "meta_action_params": int(meta_p),
        }

    def _compute_comm(
        self,
        p_mem: torch.Tensor,
        active_proj: Optional[torch.Tensor],
        disable_comm: bool = False,
    ) -> torch.Tensor:
        bsz = p_mem.shape[0]
        if (not self.cfg.use_comm) or disable_comm:
            return torch.zeros(bsz, self.cfg.d_p, device=p_mem.device, dtype=p_mem.dtype)
        transformed = torch.tanh(self.comm_proj(p_mem))
        if active_proj is None:
            return transformed.mean(dim=1)
        idx = torch.arange(self.cfg.num_projs, device=p_mem.device)[None, :]
        other_mask = (idx != active_proj[:, None]).unsqueeze(-1).to(p_mem.dtype)
        return (transformed * other_mask).sum(dim=1) / other_mask.sum(dim=1).clamp_min(1.0)

    def allocate_compute_tier(
        self,
        s_persist: torch.Tensor,
        init_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Predict 3-way hierarchical compute tier logits [B, 3] from persistent slow state + query/event ctx."""
        # 4 query-scale cues (sx, sy, hx*hy, |sx-sy|) + 4 immediate flags (vfail, retry, bt, irrev) = 8 dims
        ctx = torch.cat((init_obs[:, 20:24], init_obs[:, 36:40]), dim=-1)
        return self.compute_head(torch.cat((s_persist, ctx), dim=-1))

    def step_recurrent(
        self,
        obs: torch.Tensor,
        state: MCP08State,
        t: int,
        active_proj: Optional[torch.Tensor] = None,
        failed_proj_mask: Optional[torch.Tensor] = None,
        disable_comm: bool = False,
        override_slow: Optional[torch.Tensor] = None,
        disable_postfail_switch: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, MCP08State]:
        """Advance fast/slow/projection states; return (router_logits, stop_logit, unc_logit, next_state)."""
        bsz = obs.shape[0]
        comm_in = self._compute_comm(state.p_mem, active_proj, disable_comm=disable_comm)
        s_in = override_slow if override_slow is not None else state.s

        h_next = self.fast_gru(torch.cat((obs, s_in, comm_in), dim=-1), state.h)
        fast_cnt = state.fast_updates + 1

        event_trig = (
            (obs[:, EVENT_STALE_IDX_08] > 0.5)
            | (obs[:, EVENT_HOP_ADV_IDX_08] > 0.5)
            | (obs[:, EVENT_VERIFY_FAIL_IDX_08] > 0.5)
        )
        slow_in = torch.cat((h_next, obs[:, SEARCH_CTX_SLICE]), dim=-1)
        if override_slow is not None:
            s_next = override_slow
            slow_cnt = state.slow_updates
        elif t % self.cfg.slow_period == 0:
            s_next = self.slow_gru(slow_in, state.s)
            slow_cnt = state.slow_updates + 1
        elif bool(event_trig.any()):
            s_cand = self.slow_gru(slow_in, state.s)
            s_next = torch.where(event_trig.unsqueeze(-1), s_cand, state.s)
            slow_cnt = state.slow_updates + 1
        else:
            s_next = state.s
            slow_cnt = state.slow_updates

        p_mem_next = state.p_mem
        if active_proj is not None:
            clamped = active_proj.clamp(0, self.cfg.num_projs - 1)
            batch_idx = torch.arange(bsz, device=obs.device)
            cur_slot = state.p_mem[batch_idx, clamped]
            z_gate, cand = self.proj_gate(torch.cat((h_next, cur_slot), dim=-1)).chunk(2, dim=-1)
            z = torch.sigmoid(z_gate)
            p_mem_next = state.p_mem.clone()
            p_mem_next[batch_idx, clamped] = (1.0 - z) * cur_slot + z * torch.tanh(cand)

        comm_out = self._compute_comm(p_mem_next, active_proj, disable_comm=disable_comm)
        if failed_proj_mask is None or disable_postfail_switch or (not self.cfg.use_postfail_switch):
            fpm = torch.zeros(bsz, self.cfg.num_projs, device=obs.device, dtype=obs.dtype)
        else:
            fpm = failed_proj_mask.to(obs.dtype)

        router_logits = self.router_head(
            torch.cat((s_next, h_next, comm_out, obs[:, GEOM_SLICE_08], fpm), dim=-1)
        )
        # Strong post-failure mask at t=0 (to force trying an untried starting geometry on retry),
        # relaxing to a mild bias at t>0 so mid-trajectory fine-localization switches remain free!
        if (failed_proj_mask is not None) and (not disable_postfail_switch) and self.cfg.use_postfail_switch:
            not_all_failed = (fpm.sum(dim=-1, keepdim=True) < float(self.cfg.num_projs - 0.5)).to(obs.dtype)
            pf_pen = 12.0 if t == 0 else 1.5
            router_logits = router_logits - pf_pen * fpm * not_all_failed

        stop_ctx = torch.stack((obs[:, 28], obs[:, 29], obs[:, 30], obs[:, 33]), dim=-1)
        stop_logit = self.stop_head(torch.cat((h_next, s_next, stop_ctx), dim=-1)).squeeze(-1)

        # Calibrated uncertainty logit reads [h, s, sx, sy, |sx-sy|, remaining, |hx*hy|,
        #                                     plan_prog, goal_def, verify_fail, retry_norm, bt_norm]
        unc_ctx = torch.cat(
            (
                obs[:, 20:22],
                obs[:, 23:24],
                obs[:, 26:27],
                torch.abs(obs[:, 22:23]),
                obs[:, 34:39],
            ),
            dim=-1,
        )
        unc_logit = self.unc_head(torch.cat((h_next, s_next, unc_ctx), dim=-1)).squeeze(-1)

        return (
            router_logits,
            stop_logit,
            unc_logit,
            MCP08State(
                h=h_next,
                s=s_next,
                p_mem=p_mem_next,
                fast_updates=fast_cnt,
                slow_updates=slow_cnt,
            ),
        )

    def predict_stride(
        self,
        state: MCP08State,
        obs: torch.Tensor,
        chosen_proj: torch.Tensor,
        cand_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        proj_oh = F.one_hot(chosen_proj.clamp(0, self.cfg.num_projs - 1), self.cfg.num_projs).to(obs.dtype)
        direct_ctx = obs[:, 14:26]
        ctx_h = self.stride_ctx(torch.cat((state.h, state.s, proj_oh, direct_ctx), dim=-1))
        cand_h = self.stride_cand(cand_feats)
        joint = F.relu(ctx_h.unsqueeze(1) + cand_h + ctx_h.unsqueeze(1) * torch.tanh(cand_h))
        stride_raw = self.stride_out(joint).squeeze(-1)
        res_logits = self.res_head(F.relu(ctx_h))
        res_bias = (
            res_logits[:, 0:1] * self.fine_mask[None, :]
            + res_logits[:, 1:2] * self.coarse_mask[None, :]
        )
        return stride_raw + res_bias, res_logits

    def predict_meta_action(
        self,
        state: MCP08State,
        u_prob: torch.Tensor,
        search_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Predict 4-way recursive search control logits: [proceed, retry, backtrack, halt]."""
        # Use only immediate flags [verify_fail, retry_norm, bt_norm, irrev_risk] from search_ctx[:, 2:6]
        # so the meta-controller genuinely relies on persistent slow state `s` for progress/deficit!
        return self.meta_action_head(
            torch.cat((state.h, state.s, u_prob.unsqueeze(-1), search_ctx[:, 2:6]), dim=-1)
        )


def compute_ece_and_auroc(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Compute Expected Calibration Error (ECE), Brier score, and AUROC cleanly in NumPy."""
    probs = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    labels = np.asarray(labels, dtype=np.float64)
    n = len(probs)
    if n == 0:
        return {"ece": 0.0, "brier": 0.0, "auroc": 0.5, "n": 0}

    brier = float(np.mean((probs - labels) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        cnt = int(mask.sum())
        if cnt > 0:
            ece += (cnt / float(n)) * abs(float(labels[mask].mean()) - float(probs[mask].mean()))

    # Exact rank-based Mann-Whitney U AUROC
    pos = labels > 0.5
    n_pos = int(pos.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        auroc = 0.5
    else:
        order = np.argsort(probs)
        ranks = np.empty(n, dtype=np.float64)
        # Handle ties with average ranks
        i = 0
        while i < n:
            j = i + 1
            while j < n and probs[order[j]] == probs[order[i]]:
                j += 1
            avg_rank = 0.5 * (i + 1 + j)
            ranks[order[i:j]] = avg_rank
            i = j
        sum_pos_ranks = float(ranks[pos].sum())
        auroc = (sum_pos_ranks - 0.5 * n_pos * (n_pos + 1)) / float(n_pos * n_neg)

    return {"ece": float(ece), "brier": brier, "auroc": float(auroc), "n": n}
