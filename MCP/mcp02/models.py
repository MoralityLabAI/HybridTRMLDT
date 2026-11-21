"""Learned router architectures for MCP-02 (requires torch).

All architectures share the embedding of ``features.py`` and return one score
per candidate from the same input. They differ only in how they process it:

* ``dense``: dual encoder, mean-pooled embeddings and separate MLP towers.
* ``transformer``: ``layers`` distinct pre-LN blocks (cross-encoder).
* ``looped``: one shared block applied ``passes`` times with a pass embedding.
* ``trm``: recursive latent refinement. A shared ``net`` updates latent ``z``
  ``inner`` times from (x, y, z), then the answer state ``y`` from (y, z);
  repeated for ``cycles``.
* ``trm_ldt``: ``trm`` plus an exclusion head over index children. Its
  exclusions are proposals; the environment certifies each one.
* ``hrm``: two-timescale recurrence. A fast block ``f_L`` runs ``low_steps``
  updates per slow update of ``f_H``, for ``high_cycles`` slow cycles.
"""
from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import features as FE
from .agent import ITEM_TYPES, Policy
from .textutil import digest

ARCHES = ("dense", "transformer", "looped", "trm", "trm_ldt", "hrm")


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ffn_mult: int = 4):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ff1, self.ff2 = nn.Linear(d, ffn_mult * d), nn.Linear(ffn_mult * d, d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).view(b, s, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d // self.heads)
        att = att.masked_fill(~mask[:, None, None, :], -1e9).softmax(-1)
        y = torch.matmul(att, v).transpose(1, 2).reshape(b, s, d)
        x = x + self.out(y)
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))


class Embed(nn.Module):
    def __init__(self, d: int, segments: int = 2):
        super().__init__()
        self.tok = nn.Embedding(FE.OFFSET + FE.VOCAB_BUCKETS, d)
        self.pos = nn.Embedding(FE.SEQ_LEN, d)
        self.seg = nn.Embedding(segments, d)
        self.struct = nn.Linear(FE.STRUCT_DIM, d)

    def forward(self, ids, segs, structs):
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.tok(ids) + self.pos(pos)[None] + self.seg(segs)
        return torch.cat([x[:, :1] + self.struct(structs)[:, None], x[:, 1:]], dim=1)


class Router(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        arch, d, heads = config["arch"], config["d"], config.get("heads", 4)
        if arch not in ARCHES:
            raise ValueError("unknown architecture")
        self.arch = arch
        if config.get("utility_heads") not in (None, "expand_read"):
            raise ValueError("unknown utility_heads mode")
        self.utility_heads = config.get("utility_heads") == "expand_read"
        self.single_utility = bool(config.get("single_utility", False))
        if self.utility_heads and self.single_utility:
            raise ValueError("use either dual utility heads or the single-utility ablation")
        self.uses_horizon = self.utility_heads or self.single_utility or bool(config.get("use_horizon", False))
        if self.utility_heads and arch == "dense":
            raise ValueError("dual utility heads require a pooled recurrent/transformer encoder")
        self.encoder = config.get("encoder", "tokens")
        if self.encoder not in FE.ENCODERS:
            raise ValueError("unknown encoder")
        self.embed = Embed(d, 4 if self.encoder == "match" else 2)
        if arch == "dense":
            self.need_tower = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
            self.cand_tower = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        elif arch == "transformer":
            self.blocks = nn.ModuleList(Block(d, heads) for _ in range(config["layers"]))
        elif arch == "looped":
            self.block = Block(d, heads)
            self.pass_embed = nn.Embedding(32, d)
        elif arch in ("trm", "trm_ldt"):
            self.net = nn.ModuleList(Block(d, heads) for _ in range(config.get("net_layers", 1)))
            self.y0, self.z0 = nn.Parameter(torch.zeros(d)), nn.Parameter(torch.zeros(d))
        elif arch == "hrm":
            self.f_low, self.f_high = Block(d, heads), Block(d, heads)
            self.h0, self.l0 = nn.Parameter(torch.zeros(d)), nn.Parameter(torch.zeros(d))
        self.norm = nn.LayerNorm(d)
        self.score_head = None if self.utility_heads else nn.Linear(d, 1)
        self.expand_head = nn.Linear(d, 1) if self.utility_heads else None
        self.read_head = nn.Linear(d, 1) if self.utility_heads else None
        self.exclusion_head = nn.Linear(d, len(FE.EXCLUSION_LABELS)) if arch == "trm_ldt" else None
        self.depth_override: int | None = None

    # -- recurrent depth --------------------------------------------------------------
    def depth(self) -> int:
        c = self.config
        if self.arch == "transformer":
            return c["layers"]
        if self.arch == "looped":
            return self.depth_override or c["passes"]
        if self.arch in ("trm", "trm_ldt"):
            return (self.depth_override or c["cycles"]) * (c["inner"] + 1) * c.get("net_layers", 1)
        if self.arch == "hrm":
            return (self.depth_override or c["high_cycles"]) * (c["low_steps"] + 1)
        return 0

    def _net(self, x, mask):
        for block in self.net:
            x = block(x, mask)
        return x

    def forward(self, ids, segs, structs):
        mask = ids != FE.SPECIALS["pad"]
        x = self.embed(ids, segs, structs)
        c = self.config
        if self.arch == "dense":
            m = mask.float()
            need_m = m * ((segs == 0) | (segs == 2)).float()
            cand_m = m * ((segs == 1) | (segs == 3)).float()
            tok = x
            need = (tok * need_m[..., None]).sum(1) / need_m.sum(1, keepdim=True).clamp(min=1)
            cand = (tok * cand_m[..., None]).sum(1) / cand_m.sum(1, keepdim=True).clamp(min=1)
            cand = cand + x[:, 0]
            a, b = self.need_tower(need), self.cand_tower(cand)
            return (a * b).sum(-1) / math.sqrt(a.shape[-1]), None
        if self.arch == "transformer":
            h = x
            for block in self.blocks:
                h = block(h, mask)
        elif self.arch == "looped":
            h = x
            for t in range(self.depth_override or c["passes"]):
                h = self.block(h + self.pass_embed.weight[t][None, None], mask)
        elif self.arch in ("trm", "trm_ldt"):
            y = self.y0.expand_as(x)
            z = self.z0.expand_as(x)
            for _ in range(self.depth_override or c["cycles"]):
                for _ in range(c["inner"]):
                    z = self._net(x + y + z, mask)
                y = self._net(y + z, mask)
            h = y
        else:  # hrm
            zh, zl = self.h0.expand_as(x), self.l0.expand_as(x)
            for _ in range(self.depth_override or c["high_cycles"]):
                for _ in range(c["low_steps"]):
                    zl = self.f_low(zl + zh + x, mask)
                zh = self.f_high(zh + zl, mask)
            h = zh
        pooled = self.norm(h[:, 0])
        excl = self.exclusion_head(pooled) if self.exclusion_head is not None else None
        if self.utility_heads:
            is_read = structs[:, ITEM_TYPES.index("doc")] > 0.5
            scores = torch.where(is_read, self.read_head(pooled).squeeze(-1),
                                 self.expand_head(pooled).squeeze(-1))
        else:
            scores = self.score_head(pooled).squeeze(-1)
        return scores, excl


# -- measurement ---------------------------------------------------------------------------
def parameter_counts(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    embedding = sum(p.numel() for n, p in model.named_parameters() if n.startswith("embed.tok"))
    return {"total": total, "non_embedding": total - embedding, "embedding": embedding,
            "bytes_fp32": 4 * total}


def flops_per_candidate(model: Router) -> dict:
    """Measured with torch's FlopCounterMode on one candidate at the fixed sequence length."""
    from torch.utils.flop_counter import FlopCounterMode
    device = next(model.parameters()).device
    ids = torch.full((1, FE.SEQ_LEN), FE.OFFSET + 1, dtype=torch.long, device=device)
    segs = torch.zeros_like(ids)
    segs[:, FE.NEED_TOKENS + 2:] = 1
    structs = torch.zeros(1, FE.STRUCT_DIM, device=device)
    was = model.training
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(ids, segs, structs)
    model.train(was)
    return {"flops": int(counter.get_total_flops()), "method": "torch.utils.flop_counter.FlopCounterMode",
            "scope": "one candidate, fixed sequence length; padded positions included",
            "sequence_length": FE.SEQ_LEN, "recurrent_depth": model.depth()}


def build(config: dict) -> Router:
    return Router(config)


def load(path: str, device: str = "cpu") -> Router:
    blob = torch.load(path, map_location=device, weights_only=False)
    model = Router(blob["config"])
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval()


def file_sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RouterPolicy(Policy):
    """Adapter from a trained ``Router`` to the shared policy interface."""

    def __init__(self, name: str, model: Router, *, device: str = "cpu", checkpoint: str | None = None,
                 exclusion_threshold: float = 0.5, proposer: str | None = None, receipt: dict | None = None):
        self.name = name
        self.family = model.arch
        self.model = model.to(device).eval()
        self.device = device
        self.checkpoint = checkpoint
        self.threshold = exclusion_threshold
        self.proposer = proposer or ("learned" if model.exclusion_head is not None else "symbolic")
        self.receipt = receipt or {}
        self._flops = flops_per_candidate(self.model)
        self._cost = {"flops": 0, "router_tokens": 0, "lexical_ops": 0}

    def set_depth(self, depth: int | None) -> None:
        self.model.depth_override = depth
        self._flops = flops_per_candidate(self.model)

    def _tensors(self, ids, segs, structs):
        return (torch.tensor(ids, dtype=torch.long, device=self.device),
                torch.tensor(segs, dtype=torch.long, device=self.device),
                torch.tensor(structs, dtype=torch.float32, device=self.device))

    @torch.inference_mode()
    def score(self, obs: dict) -> list[float]:
        if self.model.uses_horizon:
            obs = dict(obs, needs=FE.utility_needs(obs))
        ids, segs, structs = FE.encode_observation(obs, self.model.encoder)
        scores, _ = self.model(*self._tensors(ids, segs, structs))
        self._cost = {"flops": self._flops["flops"] * len(ids), "router_tokens": FE.router_tokens(ids),
                      "lexical_ops": 0}
        return scores.float().cpu().tolist()

    @torch.inference_mode()
    def propose_exclusions(self, obs: dict, rows: list[dict]) -> list[tuple[str, str]]:
        if self.proposer == "symbolic":
            return super().propose_exclusions(obs, rows)
        if self.proposer != "learned" or self.model.exclusion_head is None:
            return []
        cands = FE.rows_as_candidates(rows, 1)
        ids, segs, structs = FE.encode_observation(dict(obs, candidates=cands), self.model.encoder)
        _, logits = self.model(*self._tensors(ids, segs, structs))
        probs = logits.float().softmax(-1).cpu().tolist()
        self._cost["flops"] += self._flops["flops"] * len(ids)
        out = []
        for row, p in zip(rows, probs):
            label = max(range(len(p)), key=lambda i: p[i])
            if label != 0 and p[label] >= self.threshold:
                out.append((row.get("node_id") or row["uri"], FE.EXCLUSION_LABELS[label]))
        return out

    def last_cost(self) -> dict:
        cost, self._cost = self._cost, {"flops": 0, "router_tokens": 0, "lexical_ops": 0}
        return cost

    def describe(self) -> dict:
        return {"name": self.name, "family": self.family, "config": self.model.config,
                "parameters": parameter_counts(self.model), "recurrent_depth": self.model.depth(),
                "depth_override": self.model.depth_override, "flops_per_candidate": self._flops,
                "proposer": self.proposer, "checkpoint_sha256": self.checkpoint,
                "encoder": f"{FE.ENCODER_VERSION}/{self.model.encoder}", "uses_private_oracle": False, "training": self.receipt}
