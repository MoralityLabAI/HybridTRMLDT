"""MOLT-GOBLIN-02 modular Mixture-of-Loops architecture factory.

One small PyTorch implementation compiles a restricted configuration into a recurrent
model. Two axes stay separate:

* memory: how loop states are retained, retrieved, combined and discarded
  (tied, prev, prefix, full, topk, lowrank, retention, plus confidence bias,
  window and loop-index keys);
* routing: heterogeneous loop modules and the manager that allocates them
  (single, cycle, mix, hrm), plus halting and answer-state update rules.

With memory="prefix" or "prev", one head, no relative bias, no input injection and
direct updates, the block is numerically the MOLT-GOBLIN-01 PrefixBlock; see
test_factory.test_pilot_parity.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F

MEMORY = ("tied", "prev", "prefix", "full", "topk", "lowrank", "retention")
ROUTING = ("single", "cycle", "mix", "hrm")
UPDATE = ("direct", "damped", "rlt")
LOSS = ("last", "all", "ramp")
HALT = ("none", "utility")
TYPE_NAMES = ("deduce", "construct", "repair", "verify")
INTERVENTIONS = ("none", "reset_history", "shuffle_history", "stale_swap", "noise")
HISTORY_MODES = ("full", "topk", "lowrank", "retention")
MAX_REL = 16
MAX_LOOPS = 16


@dataclass(frozen=True)
class MoltConfig:
    width: int = 32
    heads: int = 2
    loops: int = 4
    memory: str = "prefix"
    topk: int = 8
    rank: int = 8
    window: int = 0            # 0 keeps every stored loop; w>0 keeps the last w loops
    conf_bias: bool = False    # confidence-weighted memory attention
    loop_keys: bool = True     # loop-index embedding on stored keys (history modes)
    rel_bias: bool = True
    inject: bool = True        # add input embedding to the query each loop
    error_feedback: bool = False
    routing: str = "single"
    n_types: int = 1
    route_hard_eval: bool = False
    slow_period: int = 2       # HRM slow-state update period, in loops
    update: str = "direct"
    update_rank: int = 0       # 0 = full-rank loop update; r>0 = update in a learned rank-r subspace
    halt: str = "none"
    loss: str = "last"
    ldt_mask: bool = False     # apply sound family admissibility masks to logits
    lr: float = 3e-3
    steps: int = 400
    batch: int = 32

    def validate(self) -> "MoltConfig":
        def need(ok, msg):
            if not ok:
                raise ValueError(msg)
        need(8 <= self.width <= 96 and self.width % self.heads == 0, "width 8..96 divisible by heads")
        need(1 <= self.heads <= 4, "heads 1..4")
        need(1 <= self.loops <= 12, "loops 1..12")
        need(self.memory in MEMORY, f"memory in {MEMORY}")
        need(1 <= self.topk <= 64 and 1 <= self.rank <= self.width, "topk 1..64, rank 1..width")
        need(0 <= self.window <= 12, "window 0..12")
        need(self.routing in ROUTING, f"routing in {ROUTING}")
        need(1 <= self.n_types <= 4, "n_types 1..4")
        need((self.routing == "single") == (self.n_types == 1), "single routing iff n_types == 1")
        need(1 <= self.slow_period <= 4, "slow_period 1..4")
        need(self.update in UPDATE and self.halt in HALT and self.loss in LOSS, "update/halt/loss")
        need(0 <= self.update_rank <= self.width, "update_rank 0..width")
        need(not (self.memory == "prefix" and self.error_feedback), "prefix keeps a strict map; no error feedback")
        need(1e-4 <= self.lr <= 1e-2 and 16 <= self.steps <= 4000 and 8 <= self.batch <= 64, "lr/steps/batch")
        return self

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

    @staticmethod
    def from_dict(d: dict) -> "MoltConfig":
        names = {f.name for f in dataclasses.fields(MoltConfig)}
        extra = set(d) - names
        if extra:
            raise ValueError(f"unknown config keys {sorted(extra)}")
        return MoltConfig(**d).validate()

    def replace(self, **kw) -> "MoltConfig":
        return MoltConfig.from_dict({**self.to_dict(), **kw})


@dataclass
class Memory:
    """Retrieval bank: source states (B,S,d), positions (S,), loop index (S,), bias (B,S)."""
    states: torch.Tensor
    pos: torch.Tensor
    loop: torch.Tensor
    bias: torch.Tensor | None = None


class LoopBlock(nn.Module):
    """Attention over a memory bank followed by an MLP; one recurrent computation."""

    def __init__(self, cfg: MoltConfig):
        super().__init__()
        d, self.h = cfg.width, cfg.heads
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.rel = nn.Parameter(torch.zeros(self.h, MAX_REL)) if cfg.rel_bias else None
        self.loop_key = nn.Embedding(MAX_LOOPS + 1, d) if (cfg.loop_keys and cfg.memory in HISTORY_MODES) else None
        if self.loop_key is not None:
            nn.init.zeros_(self.loop_key.weight)
        self.topk = cfg.topk if cfg.memory == "topk" else 0

    def forward(self, query, mem: Memory, q_pos, strict: bool):
        b, t, d = query.shape
        hd = d // self.h
        src = mem.states
        keys_src = src if self.loop_key is None else src + self.loop_key(mem.loop.clamp(max=MAX_LOOPS))[None]
        q = self.q(query).view(b, t, self.h, hd).transpose(1, 2)
        k = self.k(keys_src).view(b, -1, self.h, hd).transpose(1, 2)
        v = self.v(src).view(b, -1, self.h, hd).transpose(1, 2)
        null = q.new_zeros((b, self.h, 1, hd))
        k = torch.cat((null, k), dim=2)
        v = torch.cat((null, v), dim=2)
        diff = q_pos[:, None] - mem.pos[None, :]                      # (t,S)
        allowed = diff > 0 if strict else diff >= 0
        allowed = torch.cat((torch.ones((t, 1), dtype=torch.bool, device=query.device), allowed), dim=1)
        scores = (q @ k.transpose(-1, -2)) * hd ** -0.5               # (b,h,t,S+1)
        if self.rel is not None:
            rel = self.rel[:, diff.clamp(0, MAX_REL - 1)]             # (h,t,S)
            scores = scores + F.pad(rel, (1, 0))[None]
        if mem.bias is not None:
            scores = scores + F.pad(mem.bias, (1, 0))[:, None, None, :]
        scores = scores.masked_fill(~allowed, -torch.inf)
        if self.topk and scores.shape[-1] - 1 > self.topk:
            body = scores[..., 1:]
            kth = body.topk(self.topk, dim=-1).values[..., -1:]
            body = body.masked_fill(body < kth, -torch.inf)
            scores = torch.cat((scores[..., :1], body), dim=-1)
        a = scores.softmax(dim=-1)
        z = self.norm(query + self.o((a @ v).transpose(1, 2).reshape(b, t, d)))
        return z + self.ff(z)


class MoltNet(nn.Module):
    def __init__(self, cfg: MoltConfig, vocab: int, n_out: int):
        super().__init__()
        self.cfg = cfg.validate()
        d = cfg.width
        self.embedding = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(LoopBlock(cfg) for _ in range(cfg.n_types))
        self.type_emb = nn.Parameter(torch.zeros(cfg.n_types, d)) if cfg.n_types > 1 else None
        self.head = nn.Linear(d, n_out)
        self.n_out = n_out
        if cfg.error_feedback:
            self.answer_emb = nn.Linear(n_out, d, bias=False)
        if cfg.memory == "lowrank":
            self.down = nn.Linear(d, cfg.rank, bias=False)
            self.up = nn.Linear(cfg.rank, d, bias=False)
        if cfg.memory == "retention":
            self.keep = nn.Linear(d, 1)
            nn.init.constant_(self.keep.bias, 2.0)
        if cfg.conf_bias:
            self.conf_scale = nn.Parameter(torch.tensor(1.0))
        if cfg.update == "damped":
            self.damp = nn.Parameter(torch.zeros(d))
        if cfg.update == "rlt":
            self.answer_in = nn.Linear(d, d)
            self.answer_norm = nn.LayerNorm(d)
        if cfg.update_rank:
            self.u_down = nn.Linear(d, cfg.update_rank, bias=False)
            self.u_up = nn.Linear(cfg.update_rank, d, bias=False)
        if cfg.routing == "hrm":
            self.fast = nn.GRUCell(2 * d, d)
            self.slow = nn.GRUCell(d, d)
            self.route = nn.Linear(d, cfg.n_types)
        if cfg.halt == "utility":
            self.halt_head = nn.Linear(d + 1, 1)

    @staticmethod
    def _prefix_mean(z):
        """Causal running mean over positions: (B,T,C) -> (B,T,C)."""
        n = torch.arange(1, z.shape[1] + 1, device=z.device, dtype=z.dtype)[None, :, None]
        return z.cumsum(1) / n

    # ---------------------------------------------------------------- memory
    def _store(self, h, r):
        """Representation stored for loop r and its retained-bytes cost per example."""
        cfg = self.cfg
        if cfg.memory == "lowrank":
            c = self.down(h)
            return self.up(c), c.shape[1] * c.shape[2] * 4
        return h, h.shape[1] * h.shape[2] * 4

    def _bank(self, history, bias_hist, r, t, device, intervene, gen, hard):
        """Assemble the retrieval bank for loop r from stored loop states."""
        cfg = self.cfg
        idx = list(range(len(history)))
        if cfg.window:
            idx = idx[-cfg.window:]
        states = [history[i] for i in idx]
        biases = [bias_hist[i] for i in idx]
        loops = list(idx)
        if intervene != "none" and len(states) > 1:
            older, cur = states[:-1], states[-1]
            if intervene == "reset_history":
                states, biases, loops = [cur], [biases[-1]], [loops[-1]]
            elif intervene == "shuffle_history":
                perm = torch.randperm(len(older), generator=gen).tolist()
                states = [older[p] for p in perm] + [cur]
                biases = [biases[p] for p in perm] + [biases[-1]]
            elif intervene == "stale_swap":
                states = [s.roll(1, dims=0) for s in older] + [cur]
                biases = [None if bb is None else bb.roll(1, dims=0) for bb in biases[:-1]] + [biases[-1]]
            elif intervene == "noise":
                states = [s + s.std() * torch.randn(s.shape, generator=gen) for s in older] + [cur]
        pos = torch.arange(t, device=device).repeat(len(states))
        loop = torch.tensor(loops, device=device).repeat_interleave(t)
        bias = torch.cat(biases, dim=1) if biases[0] is not None else None
        st = torch.cat(states, dim=1)
        kept = None
        if cfg.memory == "retention" and hard:
            kept = bias > math.log(0.5)
            bias = bias.masked_fill(~kept, -torch.inf)
        return Memory(st, pos, loop, bias), kept

    def _entry_bias(self, h, logits):
        cfg = self.cfg
        bias = None
        if cfg.memory == "retention":
            bias = F.logsigmoid(self.keep(h)).squeeze(-1)
        if cfg.conf_bias:
            conf = logits.detach().softmax(-1).max(-1).values
            bias = conf * self.conf_scale if bias is None else bias + conf * self.conf_scale
        return bias

    # ---------------------------------------------------------------- forward
    def forward(self, x, loops: int | None = None, *, mask=None, intervene: str = "none",
                hard_eval: bool | None = None, halt_eval: bool = False, seed: int = 0,
                project=None, trace: bool = False):
        """Run the recurrence; return per-loop logits and accounting.

        mask: optional (B,T,n_out) admissible mask used when cfg.ldt_mask.
        project: optional callable(delta, r) -> delta applied to loop updates (BPD-style
        subspace interventions).
        """
        cfg = self.cfg
        if intervene not in INTERVENTIONS:
            raise ValueError("unknown intervention")
        R = cfg.loops if loops is None else loops
        b, t = x.shape
        device = x.device
        hard = (not self.training) and (cfg.route_hard_eval if hard_eval is None else hard_eval)
        gen = torch.Generator().manual_seed(seed)
        base = self.embedding(x)
        q_pos = torch.arange(t, device=device)
        h = base
        y_state = base if cfg.update == "rlt" else None
        if cfg.routing == "hrm":
            u = base.new_zeros(b * t, cfg.width)
            g = base.new_zeros(b * t, cfg.width)

        def readout(state):
            lg = self.head(state)
            if cfg.ldt_mask and mask is not None:
                lg = lg.masked_fill(~mask, -1e4)
            return lg

        logits0 = readout(h)
        history, bias_hist = [], []
        if cfg.memory in HISTORY_MODES:
            st, nbytes = self._store(h, 0)
            history.append(st)
            bias_hist.append(self._entry_bias(h, logits0))
        out_logits, halt_p, route_w = [], [], []
        retained_bytes = 0
        block_visits = torch.zeros(b, device=device)
        active = torch.ones((b, t), dtype=torch.bool, device=device)
        prev_logits = logits0
        kept_frac = []
        states = []
        for r in range(R):
            # ----- query and memory for this loop
            if cfg.memory == "prefix":
                query = base
            else:
                query = h + base if (cfg.inject and r > 0) else h
            if cfg.error_feedback and r > 0:
                query = query + self.answer_emb(prev_logits.softmax(-1))
            if cfg.memory in HISTORY_MODES:
                mem, kept = self._bank(history, bias_hist, r, t, device, intervene, gen, hard)
                strict = False
                if kept is not None:
                    kept_frac.append(kept.float().mean().item())
            else:
                src = h
                if intervene in ("stale_swap", "noise", "reset_history") and r > 0:
                    if intervene == "stale_swap":
                        src = h.roll(1, dims=0)
                    elif intervene == "noise":
                        src = h + h.std() * torch.randn(h.shape, generator=gen)
                    else:
                        src = base
                bias = self._entry_bias(src, prev_logits) if cfg.conf_bias else None
                mem = Memory(src, q_pos, torch.zeros(t, dtype=torch.long, device=device), bias)
                strict = cfg.memory in ("prev", "prefix")
                kept = None
            # ----- routing among loop modules
            if cfg.routing == "single":
                w = None
                outs = [self.blocks[0](query, mem, q_pos, strict)]
                visits = 1.0
            elif cfg.routing == "cycle":
                w = None
                i = r % cfg.n_types
                outs = [self.blocks[i](query + self.type_emb[i], mem, q_pos, strict)]
                visits = 1.0
            else:
                if cfg.routing == "mix":
                    w = base.new_full((b, t, cfg.n_types), 1.0 / cfg.n_types)
                else:
                    pooled = self._prefix_mean(h).reshape(b * t, -1)   # causal per-position summary
                    if r % cfg.slow_period == 0:
                        g = self.slow(pooled, g)
                    u = self.fast(torch.cat((pooled, g), -1), u)
                    w = self.route(u).softmax(-1).view(b, t, cfg.n_types)
                outs = [blk(query + self.type_emb[i], mem, q_pos, strict) for i, blk in enumerate(self.blocks)]
                if hard:
                    # Logical cost of top-1 routing is one module visit; this reference
                    # implementation still evaluates every module and selects.
                    w = F.one_hot(w.argmax(-1), cfg.n_types).to(base.dtype)
                    visits = 1.0
                else:
                    visits = float(cfg.n_types)
                route_w.append(w)
            out = outs[0] if w is None else sum(w[..., i, None] * o for i, o in enumerate(outs))
            # ----- answer-state update rule
            if cfg.update_rank:
                out = h + self.u_up(self.u_down(out - h))
            if project is not None:
                out = h + project(out - h, r)
            if cfg.update == "direct":
                new_h = out
            elif cfg.update == "damped":
                new_h = h + torch.sigmoid(self.damp) * (out - h)
            else:
                new_h = out
                y_state = self.answer_norm(y_state + self.answer_in(new_h))
            if halt_eval and cfg.halt == "utility":
                new_h = torch.where(active[..., None], new_h, h)
            block_visits = block_visits + visits * active.float().mean(1)
            h = new_h
            lg = readout(y_state if cfg.update == "rlt" else h)
            if halt_eval and cfg.halt == "utility":
                lg = torch.where(active[..., None], lg, prev_logits)
            out_logits.append(lg)
            if cfg.halt == "utility":
                conf = self._prefix_mean(lg.detach().softmax(-1).max(-1).values[..., None])
                p = torch.sigmoid(self.halt_head(torch.cat((self._prefix_mean(h), conf), -1))).squeeze(-1)
                halt_p.append(p)                                   # (B,T) per-position continue prob
                if halt_eval:
                    active = active & (p >= 0.5)
            if cfg.memory in HISTORY_MODES:
                st, nbytes = self._store(h, r + 1)
                history.append(st)
                bias_hist.append(self._entry_bias(h, lg))
            if trace:
                states.append(h.detach())
            prev_logits = lg
        if cfg.memory in HISTORY_MODES:
            n_kept = len(history) if not cfg.window else min(cfg.window, len(history))
            per_entry = nbytes
            retained_bytes = n_kept * per_entry
            if kept_frac:
                retained_bytes *= kept_frac[-1]
        else:
            retained_bytes = 2 * t * cfg.width * 4  # read bank and write bank
        return {"logits": out_logits, "logits0": logits0, "halt_p": halt_p, "route_w": route_w,
                "block_visits": block_visits, "retained_bytes_per_example": float(retained_bytes),
                "states": states, "base": base.detach() if trace else None}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def loss_fn(cfg: MoltConfig, out: dict, y, mask):
    """Supervision schedule plus the halting head's future-utility target."""
    logits = out["logits"]
    R = len(logits)
    per_loop, ce_maps = [], []
    for lg in logits:
        ce = F.cross_entropy(lg.flatten(0, 1), y.flatten(), reduction="none", ignore_index=-100).view(y.shape)
        ce_maps.append(ce.detach())
        per_loop.append((ce * mask).sum(1) / mask.sum(1).clamp(min=1))   # (B,)
    if cfg.loss == "last":
        main = per_loop[-1].mean()
    else:
        w = torch.ones(R) if cfg.loss == "all" else torch.arange(1, R + 1, dtype=torch.float)
        w = w / w.sum()
        main = sum(wi * pl.mean() for wi, pl in zip(w.tolist(), per_loop))
    total = main
    if cfg.halt == "utility" and out["halt_p"]:
        hl = []
        for r, p in enumerate(out["halt_p"]):
            if r + 1 < R:
                target = (ce_maps[r + 1] < ce_maps[r] - 0.01).float()
            else:
                target = torch.zeros_like(p)
            bce = F.binary_cross_entropy(p.clamp(1e-5, 1 - 1e-5), target, reduction="none")
            hl.append((bce * mask).sum() / mask.sum().clamp(min=1))
        total = total + 0.1 * torch.stack(hl).mean()
    return total, [pl.detach() for pl in per_loop]


def describe_schema() -> str:
    """Public schema text shown to the proposer."""
    return json.dumps({
        "width": "8..96 (default 32)", "heads": "1..4", "loops": "1..12 (train depth)",
        "memory": {"tied": "plain looped self-attention over the current state (causal, j<=i)",
                   "prev": "previous-loop-only recurrence: query=current state, memory=previous state at j<i",
                   "prefix": "existing MoLT-Goblin prefix memory: query=input embedding, memory=previous state at j<i",
                   "full": "full loop memory: attend over all stored loop states h0..hr at j<=i",
                   "topk": "full memory with sparse top-k retrieval per query (topk)",
                   "lowrank": "full memory stored through a rank-`rank` compression",
                   "retention": "full memory with a learned per-entry retention gate; hard eviction at eval"},
        "topk": "1..64", "rank": "1..width", "window": "0 (all loops) or keep last w loops",
        "conf_bias": "bool: bias memory attention by readout confidence of each stored entry",
        "loop_keys": "bool: loop-index embedding on stored keys (history modes)",
        "rel_bias": "bool: learned relative position bias", "inject": "bool: input injection each loop",
        "error_feedback": "bool: feed previous-loop answer distribution into the next loop (not with prefix)",
        "routing": {"single": "one tied block", "cycle": "typed modules used in fixed order (n_types=loops is untied depth)",
                    "mix": "uniform mixture of typed modules", "hrm": "slow/fast GRU manager routes among typed modules"},
        "n_types": "1..4 (deduce, construct, repair, verify); 1 iff routing=single",
        "route_hard_eval": "bool: argmax routing at evaluation (1 module visit per loop)",
        "slow_period": "1..4", "update": {"direct": "h=out", "damped": "learned per-dim step toward out",
                                            "rlt": "separate answer state accumulated from the latent state"},
        "update_rank": "0 or 1..width: constrain loop updates to a learned low-rank subspace",
        "halt": {"none": "fixed depth", "utility": "halting head trained to predict next-loop loss improvement"},
        "loss": {"last": "final-loop supervision", "all": "uniform loop forcing", "ramp": "loop forcing weighted toward later loops"},
        "ldt_mask": "bool: sound certified pruning of answer candidates from input-only rules",
        "lr": "1e-4..1e-2", "steps": "16..4000", "batch": "8..64",
    }, indent=1)
