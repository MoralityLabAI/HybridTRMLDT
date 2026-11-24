"""Coupled two-timescale geometry-adaptive recurrent controller for MCP-07.

Implements:
- Fast recurrent state (updates every step)
- Slow recurrent state (updates on slow clockwork + structural events)
- Projection-specific state memory with optional cross-projection communication
- Learned address-projection router
- Factored learned resolution (fine/coarse) + candidate-centroid-aware 13-stride head
- Learned adaptive stop head
- First-class hooks for Section 7 causal mechanistic interventions
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .memory import (
    ALL_PROJECTIONS,
    COARSE_MASK_NP,
    FINE_MASK_NP,
    STRIDES,
    STRIDES_NP,
    STRUCTURED_PROJECTIONS,
    ZERO_STRIDE_IDX,
    SharedPhysicalMemory,
)

OBS_DIM: int = 34
CAND_FEAT_DIM: int = 10
NUM_STRUCT_PROJS: int = len(STRUCTURED_PROJECTIONS)  # 4
NUM_STRIDES: int = len(STRIDES)  # 13

# Indices inside the 34-dim observation vector:
# [0..3]   active projection one-hot (4)
# [4..7]   page normalized coords & relative strides (4)
# [8..9]   anchor_x / W, anchor_y / H (2)
# [10..13] anchor dyadic phase (ax%16, ay%16, ax%32, ay%32) (4)
# [14..17] physical centroid displacement from reference (coarse /32 and fine /8) (4)
# [18..23] observable query geometry (hx, hy, sx, sy, hx*hy, |sx-sy|) (6)
# [24..25] signed progress along hinted axes ((cx-ref_x)*hx/32, (cy-ref_y)*hy/32) (2)
# [26..33] budget, step, hit_cur, solved, stale_cur, stale_ever, hop_idx, hop_adv (8)
GEOM_SLICE = slice(10, 26)  # 16 observable geometry/displacement/progress dims
EVENT_STALE_IDX = 30
EVENT_HOP_ADV_IDX = 33


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters with explicit parameter-count control."""

    d_obs: int = OBS_DIM
    d_h: int = 52
    d_s: int = 52
    d_p: int = 16
    d_router_hid: int = 32
    d_stride_hid: int = 48
    num_projs: int = NUM_STRUCT_PROJS
    num_strides: int = NUM_STRIDES
    use_comm: bool = True
    slow_period: int = 2


# Parameter-matched preset configs (~54K matching MCP-06's 54,345; fixed >= switch active params)
CONFIG_54K = ModelConfig(d_h=52, d_s=52, d_p=16, d_router_hid=32, d_stride_hid=48, use_comm=True)
CONFIG_54K_NOCOMM = ModelConfig(d_h=52, d_s=52, d_p=16, d_router_hid=32, d_stride_hid=48, use_comm=False)
CONFIG_54K_FIXED = ModelConfig(d_h=56, d_s=56, d_p=16, d_router_hid=32, d_stride_hid=48, use_comm=False)
# Scaled variant strictly capped <= 128K parameters (~116K)
CONFIG_116K = ModelConfig(d_h=80, d_s=80, d_p=20, d_router_hid=48, d_stride_hid=64, use_comm=True)


@dataclass
class ControllerState:
    """Recurrent hidden state container passed across steps."""

    h: torch.Tensor          # [B, d_h] fast state
    s: torch.Tensor          # [B, d_s] slow state
    p_mem: torch.Tensor      # [B, num_projs, d_p] projection-specific state slots
    fast_updates: int = 0
    slow_updates: int = 0


def build_init_obs_batch(
    mem: SharedPhysicalMemory,
    anchors: np.ndarray,
    hx: np.ndarray,
    hy: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    budget: int,
) -> np.ndarray:
    """Build strictly non-oracle [B, 32] initial observation prior to first probe."""
    bsz = len(anchors)
    obs = np.zeros((bsz, OBS_DIM), dtype=np.float32)
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
    return obs


def build_step_obs_batch(
    mem: SharedPhysicalMemory,
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
    proj_names: Tuple[str, ...] = ALL_PROJECTIONS,
) -> np.ndarray:
    """Build strictly non-oracle [B, 32] step observation after probing (proj_ids, pages)."""
    bsz = len(pages)
    obs = np.zeros((bsz, OBS_DIM), dtype=np.float32)
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
    return obs


def build_cand_feats_single(
    mem: SharedPhysicalMemory,
    cur_proj: str,
    cur_page: int,
    next_proj: str,
    cands: np.ndarray,
    legal: np.ndarray,
    ref_x: float,
    ref_y: float,
    hx: int,
    hy: int,
) -> np.ndarray:
    """Build strictly observable [13, 10] page-centroid displacement & directional alignment features."""
    feats = np.zeros((NUM_STRIDES, CAND_FEAT_DIM), dtype=np.float32)
    safe_c = np.clip(cands, 0, mem.pages - 1)
    cur_c = mem.views[cur_proj].centers[cur_page]
    nxt_c = mem.views[next_proj].centers[safe_c]  # [13, 2]
    dx_s = np.clip((nxt_c[:, 0] - float(cur_c[0])) / 32.0, -4.0, 4.0)
    dy_s = np.clip((nxt_c[:, 1] - float(cur_c[1])) / 32.0, -4.0, 4.0)
    dx_r = np.clip((nxt_c[:, 0] - ref_x) / 32.0, -4.0, 4.0)
    dy_r = np.clip((nxt_c[:, 1] - ref_y) / 32.0, -4.0, 4.0)
    feats[:, 0] = dx_s
    feats[:, 1] = dy_s
    feats[:, 2] = dx_r
    feats[:, 3] = dy_r
    feats[:, 4] = dx_r * float(hx)
    feats[:, 5] = dy_r * float(hy)
    feats[:, 6] = np.abs(dx_r)
    feats[:, 7] = np.abs(dy_r)
    feats[:, 8] = STRIDES_NP.astype(np.float32) / 32.0
    feats[:, 9] = legal.astype(np.float32)
    feats[~legal, :8] = 0.0
    return feats


def build_cand_feats_batch(
    mem: SharedPhysicalMemory,
    cur_projs: np.ndarray,
    cur_pages: np.ndarray,
    next_projs: np.ndarray,
    cands: np.ndarray,
    legal: np.ndarray,
    ref_xy: np.ndarray,
    hx: np.ndarray,
    hy: np.ndarray,
    proj_names: Tuple[str, ...] = ALL_PROJECTIONS,
) -> np.ndarray:
    """Vectorized [B, 13, 10] observable page-centroid & directional alignment features."""
    bsz = len(cur_pages)
    feats = np.zeros((bsz, NUM_STRIDES, CAND_FEAT_DIM), dtype=np.float32)
    cur_c = np.empty((bsz, 2), dtype=np.float32)
    for pid in np.unique(cur_projs):
        m = cur_projs == pid
        cur_c[m] = mem.views[proj_names[int(pid)]].centers[cur_pages[m]]

    safe_c = np.clip(cands, 0, mem.pages - 1)
    nxt_c = np.empty((bsz, NUM_STRIDES, 2), dtype=np.float32)
    for pid in np.unique(next_projs):
        m = next_projs == pid
        nxt_c[m] = mem.views[proj_names[int(pid)]].centers[safe_c[m]]

    dx_s = np.clip((nxt_c[:, :, 0] - cur_c[:, 0:1]) / 32.0, -4.0, 4.0)
    dy_s = np.clip((nxt_c[:, :, 1] - cur_c[:, 1:2]) / 32.0, -4.0, 4.0)
    dx_r = np.clip((nxt_c[:, :, 0] - ref_xy[:, 0:1]) / 32.0, -4.0, 4.0)
    dy_r = np.clip((nxt_c[:, :, 1] - ref_xy[:, 1:2]) / 32.0, -4.0, 4.0)
    feats[:, :, 0] = dx_s
    feats[:, :, 1] = dy_s
    feats[:, :, 2] = dx_r
    feats[:, :, 3] = dy_r
    feats[:, :, 4] = dx_r * hx[:, None].astype(np.float32)
    feats[:, :, 5] = dy_r * hy[:, None].astype(np.float32)
    feats[:, :, 6] = np.abs(dx_r)
    feats[:, :, 7] = np.abs(dy_r)
    feats[:, :, 8] = STRIDES_NP[None, :].astype(np.float32) / 32.0
    feats[:, :, 9] = legal.astype(np.float32)
    feats[:, :, :8] *= legal[:, :, None].astype(np.float32)
    return feats


def compute_candidates_single(
    mem: SharedPhysicalMemory,
    cur_proj: str,
    cur_page: int,
    next_proj: str,
    visited: Dict[str, set],
    reanchor_slot: int = -1,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute (candidate_pages[13], legal_mask[13], base_page) for one episode."""
    if reanchor_slot >= 0:
        base_page = int(mem.views[next_proj].inv[reanchor_slot])
        allow_zero = True
    elif next_proj != cur_proj:
        cursor_slot = int(mem.views[cur_proj].center_slot[cur_page])
        base_page = int(mem.views[next_proj].inv[cursor_slot])
        allow_zero = True
    else:
        base_page = cur_page
        allow_zero = False

    cands = base_page + STRIDES_NP
    seen = visited[next_proj]
    legal = (cands >= 0) & (cands < mem.pages)
    if not allow_zero:
        legal[ZERO_STRIDE_IDX] = False
    for i in range(NUM_STRIDES):
        if legal[i] and int(cands[i]) in seen:
            legal[i] = False
    return cands, legal, base_page


class AdaptiveGeometryController(nn.Module):
    """Two-timescale recurrent metacontroller with projection router and adaptive stop."""

    def __init__(self, cfg: ModelConfig = CONFIG_54K):
        super().__init__()
        self.cfg = cfg
        self.fast_gru = nn.GRUCell(cfg.d_obs + cfg.d_s + cfg.d_p, cfg.d_h)
        self.slow_gru = nn.GRUCell(cfg.d_h, cfg.d_s)

        # Projection-specific state slot update + cross-projection communication
        self.proj_gate = nn.Linear(cfg.d_h + cfg.d_p, 2 * cfg.d_p)
        self.comm_proj = nn.Linear(cfg.d_p, cfg.d_p)

        # Learned projection router (uses slow state, fast state, comm, observable geometry+progress)
        self.router_head = nn.Sequential(
            nn.Linear(cfg.d_s + cfg.d_h + cfg.d_p + 16, cfg.d_router_hid),
            nn.ReLU(),
            nn.Linear(cfg.d_router_hid, cfg.num_projs),
        )

        # Factored candidate-centroid-aware stride scorer + learned fine/coarse resolution head
        ctx_in = cfg.d_h + cfg.d_s + cfg.num_projs + 12
        self.stride_ctx = nn.Linear(ctx_in, cfg.d_stride_hid)
        self.stride_cand = nn.Linear(CAND_FEAT_DIM, cfg.d_stride_hid)
        self.stride_out = nn.Linear(cfg.d_stride_hid, 1)
        self.res_head = nn.Linear(cfg.d_stride_hid, 2)

        # Adaptive termination head
        self.stop_head = nn.Linear(cfg.d_h + cfg.d_s + 4, 1)

        self.register_buffer("fine_mask", torch.from_numpy(FINE_MASK_NP.astype(np.float32)))
        self.register_buffer("coarse_mask", torch.from_numpy(COARSE_MASK_NP.astype(np.float32)))

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> ControllerState:
        dev = device or next(self.parameters()).device
        return ControllerState(
            h=torch.zeros(batch_size, self.cfg.d_h, device=dev),
            s=torch.zeros(batch_size, self.cfg.d_s, device=dev),
            p_mem=torch.zeros(batch_size, self.cfg.num_projs, self.cfg.d_p, device=dev),
            fast_updates=0,
            slow_updates=0,
        )

    def parameter_summary(self, active_router: bool = True) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        router_params = sum(p.numel() for p in self.router_head.parameters())
        comm_params = sum(p.numel() for p in self.comm_proj.parameters())
        active = total
        if not active_router:
            active -= router_params
        if not self.cfg.use_comm:
            active -= comm_params
        return {
            "total_params": int(total),
            "active_params": int(active),
            "router_params": int(router_params),
            "comm_params": int(comm_params),
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
        transformed = torch.tanh(self.comm_proj(p_mem))  # [B, P, d_p]
        if active_proj is None:
            return transformed.mean(dim=1)
        idx = torch.arange(self.cfg.num_projs, device=p_mem.device)[None, :]
        other_mask = (idx != active_proj[:, None]).unsqueeze(-1).to(p_mem.dtype)
        denom = other_mask.sum(dim=1).clamp_min(1.0)
        return (transformed * other_mask).sum(dim=1) / denom

    def step_recurrent(
        self,
        obs: torch.Tensor,
        state: ControllerState,
        t: int,
        active_proj: Optional[torch.Tensor] = None,
        disable_comm: bool = False,
        override_slow: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, ControllerState]:
        """Advance fast/slow/projection states and return (router_logits, stop_logit, next_state)."""
        bsz = obs.shape[0]
        comm_in = self._compute_comm(state.p_mem, active_proj, disable_comm=disable_comm)
        s_in = override_slow if override_slow is not None else state.s

        h_next = self.fast_gru(torch.cat((obs, s_in, comm_in), dim=-1), state.h)
        fast_cnt = state.fast_updates + 1

        event_trig = (obs[:, EVENT_STALE_IDX] > 0.5) | (obs[:, EVENT_HOP_ADV_IDX] > 0.5)
        if override_slow is not None:
            s_next = override_slow
            slow_cnt = state.slow_updates
        elif t % self.cfg.slow_period == 0:
            s_next = self.slow_gru(h_next, state.s)
            slow_cnt = state.slow_updates + 1
        elif bool(event_trig.any()):
            s_cand = self.slow_gru(h_next, state.s)
            mask = event_trig.unsqueeze(-1)
            s_next = torch.where(mask, s_cand, state.s)
            slow_cnt = state.slow_updates + 1
        else:
            s_next = state.s
            slow_cnt = state.slow_updates

        p_mem_next = state.p_mem
        if active_proj is not None:
            clamped = active_proj.clamp(0, self.cfg.num_projs - 1)
            batch_idx = torch.arange(bsz, device=obs.device)
            cur_slot = state.p_mem[batch_idx, clamped]
            gate_cand = self.proj_gate(torch.cat((h_next, cur_slot), dim=-1))
            z_gate, cand = gate_cand.chunk(2, dim=-1)
            z = torch.sigmoid(z_gate)
            updated_slot = (1.0 - z) * cur_slot + z * torch.tanh(cand)
            p_mem_next = state.p_mem.clone()
            p_mem_next[batch_idx, clamped] = updated_slot

        comm_out = self._compute_comm(p_mem_next, active_proj, disable_comm=disable_comm)
        geom = obs[:, GEOM_SLICE]
        router_logits = self.router_head(torch.cat((s_next, h_next, comm_out, geom), dim=-1))

        stop_ctx = torch.stack((obs[:, 28], obs[:, 29], obs[:, 30], obs[:, 33]), dim=-1)
        stop_logit = self.stop_head(torch.cat((h_next, s_next, stop_ctx), dim=-1)).squeeze(-1)

        return (
            router_logits,
            stop_logit,
            ControllerState(
                h=h_next,
                s=s_next,
                p_mem=p_mem_next,
                fast_updates=fast_cnt,
                slow_updates=slow_cnt,
            ),
        )

    def predict_stride(
        self,
        state: ControllerState,
        obs: torch.Tensor,
        chosen_proj: torch.Tensor,
        cand_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute factored (combined_stride_logits[B, 13], res_logits[B, 2])."""
        proj_oh = F.one_hot(chosen_proj.clamp(0, self.cfg.num_projs - 1), self.cfg.num_projs).to(obs.dtype)
        direct_ctx = obs[:, 14:26]  # [B, 12]
        ctx_h = self.stride_ctx(torch.cat((state.h, state.s, proj_oh, direct_ctx), dim=-1))  # [B, H]
        cand_h = self.stride_cand(cand_feats)  # [B, 13, H]
        joint = F.relu(ctx_h.unsqueeze(1) + cand_h + ctx_h.unsqueeze(1) * torch.tanh(cand_h))  # [B, 13, H]
        stride_raw = self.stride_out(joint).squeeze(-1)  # [B, 13]
        res_logits = self.res_head(F.relu(ctx_h))  # [B, 2]
        res_bias = (
            res_logits[:, 0:1] * self.fine_mask[None, :]
            + res_logits[:, 1:2] * self.coarse_mask[None, :]
        )
        return stride_raw + res_bias, res_logits


def optimal_action_set_loss(
    logits: torch.Tensor,
    optimal_mask: torch.Tensor,
    legal_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multi-positive log-sum-exp marginal likelihood loss over optimal action set A*."""
    if legal_mask is not None:
        masked_all = logits.masked_fill(~legal_mask, -1e9)
        pos_mask = optimal_mask & legal_mask
    else:
        masked_all = logits
        pos_mask = optimal_mask

    log_denom = torch.logsumexp(masked_all, dim=-1)
    masked_pos = masked_all.masked_fill(~pos_mask, -1e9)
    log_num = torch.logsumexp(masked_pos, dim=-1)
    return (log_denom - log_num).clamp_min(0.0)
