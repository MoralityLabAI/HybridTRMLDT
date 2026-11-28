"""MOLT-GOBLIN-03 factory: typed-module recurrence with native sparse routing.

Extends the MOLT-GOBLIN-02 block (imported unchanged) with:
* routing: single, cycle, mix, hrm (soft), sparse_st (straight-through top-1),
  anneal (soft-to-hard temperature schedule), sequential (modules composed in order);
* tie_modules: one shared block behind distinct type embeddings;
* auxiliary objectives: load balance, routing entropy, module diversity, functional
  module objectives (deduce / construct / repair / verify);
* distillation targets (handled in train3) and intervention hooks for mechanistic
  measurement: routing override, module permutation, module masks, output collection,
  and a hard-dispatch path that computes each module only on its routed positions.

memory is restricted to tied / prev / prefix: MOLT-GOBLIN-02 found no gain from
history memories at this scale.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from research_gym.molt_goblin_02.factory import LoopBlock, Memory, MoltConfig as V2Config

MEMORY = ("tied", "prev", "prefix")
ROUTING = ("single", "cycle", "mix", "hrm", "sparse_st", "anneal", "sequential")
SOFT_PARALLEL = ("mix", "hrm")
HARD_PARALLEL = ("sparse_st", "anneal")
LOSS = ("last", "all", "ramp")
DISTILL = ("none", "logits_final", "logits_all", "hidden")
INTERVENTIONS = ("none", "reset_history", "stale_swap", "noise")
TYPE_NAMES = ("deduce", "construct", "repair", "verify")


@dataclass(frozen=True)
class Config:
    width: int = 32
    heads: int = 2
    loops: int = 4
    memory: str = "prefix"
    rel_bias: bool = True
    inject: bool = True
    routing: str = "single"
    n_types: int = 1
    tie_modules: bool = False
    slow_period: int = 2
    st_tau: float = 1.0
    anneal_frac: float = 0.7
    anneal_end_temp: float = 0.05
    update: str = "direct"
    loss: str = "all"
    load_balance: float = 0.0
    route_entropy: float = 0.0
    diversity: float = 0.0
    functional: bool = False
    func_coef: float = 0.0
    distill: str = "none"
    distill_coef: float = 0.0
    distill_temp: float = 2.0
    lr: float = 3e-3
    steps: int = 300
    batch: int = 32

    def validate(self) -> "Config":
        def need(ok, msg):
            if not ok:
                raise ValueError(msg)
        need(8 <= self.width <= 128 and self.width % self.heads == 0, "width 8..128 divisible by heads")
        need(1 <= self.heads <= 4 and 1 <= self.loops <= 16, "heads 1..4, loops 1..16")
        need(self.memory in MEMORY and self.routing in ROUTING, "memory/routing")
        need(1 <= self.n_types <= 4, "n_types 1..4")
        need((self.routing == "single") == (self.n_types == 1), "single routing iff n_types == 1")
        need(not (self.tie_modules and self.n_types == 1), "tie_modules needs n_types > 1")
        need(self.update in ("direct", "damped") and self.loss in LOSS and self.distill in DISTILL, "update/loss/distill")
        need(0.05 <= self.st_tau <= 5 and 0.1 <= self.anneal_frac <= 1 and 0.01 <= self.anneal_end_temp <= 1, "sparse schedule")
        need(1 <= self.slow_period <= 4, "slow_period")
        for k in ("load_balance", "diversity", "func_coef", "distill_coef"):
            need(0 <= getattr(self, k) <= 2, f"{k} 0..2")
        need(-1 <= self.route_entropy <= 1, "route_entropy -1..1")
        need(not self.functional or (self.n_types == 4 and self.routing != "single"), "functional objectives need 4 modules")
        need((self.distill == "none") == (self.distill_coef == 0), "distill mode iff distill_coef > 0")
        need(0.5 <= self.distill_temp <= 8, "distill_temp")
        need(1e-4 <= self.lr <= 1e-2 and 16 <= self.steps <= 6000 and 8 <= self.batch <= 64, "lr/steps/batch")
        return self

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def canonical(self) -> dict:
        """Drop fields that cannot affect the computation, for deduplication."""
        d = self.to_dict()
        r = d["routing"]
        if r not in ("hrm", "sparse_st", "anneal"):
            d.pop("slow_period")
        if r != "sparse_st":
            d.pop("st_tau")
        if r != "anneal":
            d.pop("anneal_frac"), d.pop("anneal_end_temp")
        if r not in SOFT_PARALLEL + HARD_PARALLEL:
            d.pop("load_balance"), d.pop("route_entropy")
        if r == "single":
            d.pop("tie_modules"), d.pop("diversity")
        if not d["functional"]:
            d.pop("func_coef")
        if d["distill"] == "none":
            d.pop("distill_temp")
        return d

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.canonical(), sort_keys=True).encode()).hexdigest()[:16]

    @staticmethod
    def from_dict(d: dict) -> "Config":
        names = {f.name for f in dataclasses.fields(Config)}
        extra = set(d) - names
        if extra:
            raise ValueError(f"unknown config keys {sorted(extra)}")
        return Config(**d).validate()

    def replace(self, **kw) -> "Config":
        return Config.from_dict({**self.to_dict(), **kw})

    def block_config(self) -> V2Config:
        return V2Config(width=self.width, heads=self.heads, loops=self.loops, memory=self.memory,
                        rel_bias=self.rel_bias, inject=self.inject, loop_keys=False)


def logical_visits(cfg: Config, hard_eval: bool = False) -> int:
    """Module visits per position per forward pass at evaluation."""
    if cfg.routing in SOFT_PARALLEL and not hard_eval:
        return cfg.loops * cfg.n_types
    if cfg.routing == "sequential":
        return cfg.loops * cfg.n_types
    return cfg.loops


class Net(nn.Module):
    def __init__(self, cfg: Config, vocab: int, n_out: int):
        super().__init__()
        self.cfg = cfg.validate()
        d = cfg.width
        self.embedding = nn.Embedding(vocab, d)
        n_blocks = 1 if (cfg.tie_modules or cfg.n_types == 1) else cfg.n_types
        self.blocks = nn.ModuleList(LoopBlock(cfg.block_config()) for _ in range(n_blocks))
        self.type_emb = nn.Parameter(torch.zeros(cfg.n_types, d)) if cfg.n_types > 1 else None
        if cfg.tie_modules:
            nn.init.normal_(self.type_emb, std=0.5)   # distinct modules need distinct inputs
        self.head = nn.Linear(d, n_out)
        self.n_out = n_out
        if cfg.update == "damped":
            self.damp = nn.Parameter(torch.zeros(d))
        if cfg.routing in ("hrm",) + HARD_PARALLEL:
            self.fast = nn.GRUCell(2 * d, d)
            self.slow = nn.GRUCell(d, d)
            self.route = nn.Linear(d, cfg.n_types)
        if cfg.functional:
            self.verify_head = nn.Linear(d, 1)
        self.progress = 1.0          # training progress in [0,1], set by the trainer
        self.aux: dict = {}

    def block(self, i: int) -> LoopBlock:
        return self.blocks[0 if len(self.blocks) == 1 else i]

    @staticmethod
    def _prefix_mean(z):
        n = torch.arange(1, z.shape[1] + 1, device=z.device, dtype=z.dtype)[None, :, None]
        return z.cumsum(1) / n

    def anneal_temp(self) -> float:
        c = self.cfg
        p = min(1.0, self.progress / c.anneal_frac)
        return math.exp((1 - p) * math.log(1.0) + p * math.log(c.anneal_end_temp))

    def _weights(self, logits, hard: bool, gen):
        """Routing weights (B,T,K) for parallel routings and the hard selection used."""
        c = self.cfg
        if c.routing == "hrm":
            w = logits.softmax(-1)
        elif c.routing == "sparse_st":
            if self.training:
                u = torch.rand(logits.shape, generator=gen).clamp(1e-6, 1 - 1e-6)
                soft = ((logits - torch.log(-torch.log(u))) / c.st_tau).softmax(-1)
                hard_w = F.one_hot(soft.argmax(-1), c.n_types).to(soft.dtype)
                return hard_w + soft - soft.detach(), logits.softmax(-1)
            w = logits.softmax(-1)
        else:  # anneal
            w = (logits / (self.anneal_temp() if self.training else c.anneal_end_temp)).softmax(-1)
        probs = logits.softmax(-1)
        if hard or (c.routing in HARD_PARALLEL and not self.training):
            w = F.one_hot(w.argmax(-1), c.n_types).to(w.dtype)
        return w, probs

    def forward(self, x, loops: int | None = None, *, mask=None, intervene: str = "none", hard_eval=None,
                seed: int = 0, route_override=None, module_perm=None, module_mask=None, collect=None,
                dispatch: bool = False):
        c = self.cfg
        R = c.loops if loops is None else loops
        b, t = x.shape
        dev = x.device
        gen = torch.Generator().manual_seed(seed if not self.training else int(self.progress * 1e6) + seed)
        hard = (not self.training) and bool(hard_eval)
        base = self.embedding(x)
        q_pos = torch.arange(t, device=dev)
        h = base
        if c.routing in ("hrm",) + HARD_PARALLEL:
            u = base.new_zeros(b * t, c.width)
            g = base.new_zeros(b * t, c.width)
        perm = list(range(c.n_types)) if module_perm is None else list(module_perm)
        active = None if module_mask is None else torch.tensor(module_mask, dtype=base.dtype)

        def readout(state):
            lg = self.head(state)
            return lg if mask is None else lg.masked_fill(~mask, -1e4)

        out_logits, route_w, route_p, visits = [], [], [], 0.0
        aux = {"lb": [], "ent": [], "div": [], "func": []}
        prev_logits = readout(h)
        for r in range(R):
            src = h
            if intervene != "none" and r > 0:
                src = {"stale_swap": h.roll(1, dims=0), "noise": h + h.std() * torch.randn(h.shape, generator=gen),
                       "reset_history": base}[intervene]
            mem = Memory(src, q_pos, torch.zeros(t, dtype=torch.long, device=dev), None)
            strict = c.memory in ("prev", "prefix")
            q0 = base if c.memory == "prefix" else (h + base if (c.inject and r > 0) else h)
            outs = None
            if c.routing == "single":
                new = self.block(0)(q0, mem, q_pos, strict)
                visits += 1
            elif c.routing == "cycle":
                i = r % c.n_types
                new = self.block(perm[i])(q0 + self.type_emb[perm[i]], mem, q_pos, strict)
                visits += 1
            elif c.routing == "sequential":
                z = h
                outs = []
                for i in range(c.n_types):
                    qi = (q0 if i == 0 else z) + self.type_emb[perm[i]]
                    mi = mem if i == 0 else Memory(z, q_pos, mem.loop, None)
                    z = self.block(perm[i])(qi, mi, q_pos, strict)
                    outs.append(z)
                new = z
                visits += c.n_types
            else:
                if c.routing == "mix":
                    logits = base.new_zeros(b, t, c.n_types)
                else:
                    pooled = self._prefix_mean(h).reshape(b * t, -1)
                    if r % c.slow_period == 0:
                        g = self.slow(pooled, g)
                    u = self.fast(torch.cat((pooled, g), -1), u)
                    logits = self.route(u).view(b, t, c.n_types)
                if c.routing == "mix":
                    w = logits.softmax(-1)
                    probs = w
                    if hard:
                        w = F.one_hot(w.argmax(-1), c.n_types).to(w.dtype)
                else:
                    w, probs = self._weights(logits, hard, gen)
                if route_override is not None:
                    w = route_override(w, r)
                if active is not None:
                    w = w * active
                    w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
                route_w.append(w.detach())
                route_p.append(probs)
                one_hot_eval = (not self.training) and bool((w.max(-1).values > 0.999).all())
                if dispatch and one_hot_eval:
                    new = self._dispatch(w.argmax(-1), q0, mem, q_pos, strict, perm)
                    visits += 1
                else:
                    outs = [self.block(perm[i])(q0 + self.type_emb[perm[i]], mem, q_pos, strict)
                            for i in range(c.n_types)]
                    new = sum(w[..., i, None] * o for i, o in enumerate(outs))
                    visits += 1 if one_hot_eval or c.routing in HARD_PARALLEL else c.n_types
                if self.training and c.routing in HARD_PARALLEL + ("hrm",):
                    frac = F.one_hot(probs.argmax(-1), c.n_types).float().mean((0, 1))
                    aux["lb"].append(c.n_types * (frac * probs.mean((0, 1))).sum())
                    aux["ent"].append(-(probs.clamp_min(1e-9).log() * probs).sum(-1).mean())
            if outs is not None and self.training and c.diversity > 0:
                flat = [F.normalize(o.reshape(-1, c.width), dim=-1) for o in outs]
                sims = [(flat[i] * flat[j]).sum(-1).mean() for i in range(len(flat)) for j in range(i + 1, len(flat))]
                aux["div"].append(torch.stack(sims).mean())
            if outs is not None and self.training and c.functional:
                aux["func"].append((outs, prev_logits.detach()))
            if collect is not None and outs is not None:
                collect.setdefault("module_outputs", []).append([o.detach() for o in outs])
            new_h = new if c.update == "direct" else h + torch.sigmoid(self.damp) * (new - h)
            h = new_h
            lg = readout(h)
            out_logits.append(lg)
            prev_logits = lg
            if collect is not None:
                collect.setdefault("states", []).append(h.detach())
        self.aux = aux
        if collect is not None:
            collect["base"] = base.detach()
        return {"logits": out_logits, "route_w": route_w, "route_p": [p.detach() for p in route_p],
                "visits": visits, "h": h}

    def _dispatch(self, sel, query, mem, q_pos, strict, perm):
        """Hard routing: run module i only on positions routed to it (K/V still per module)."""
        b, t, d = query.shape
        out = torch.zeros_like(query)
        for i in range(self.cfg.n_types):
            rows = (sel == i).nonzero(as_tuple=False)
            if rows.numel() == 0:
                continue
            blk = self.block(perm[i])
            bi, ti = rows[:, 0], rows[:, 1]
            hd = d // blk.h
            q = blk.q(query[bi, ti] + self.type_emb[perm[i]]).view(-1, blk.h, 1, hd)
            k = blk.k(mem.states).view(b, -1, blk.h, hd).transpose(1, 2)[bi]
            v = blk.v(mem.states).view(b, -1, blk.h, hd).transpose(1, 2)[bi]
            null = q.new_zeros((len(bi), blk.h, 1, hd))
            k, v = torch.cat((null, k), 2), torch.cat((null, v), 2)
            diff = q_pos[ti][:, None] - mem.pos[None, :]
            allowed = torch.cat((torch.ones((len(bi), 1), dtype=torch.bool), diff > 0 if strict else diff >= 0), 1)
            sc = (q @ k.transpose(-1, -2)).squeeze(2) * hd ** -0.5            # (n,h,S+1)
            if blk.rel is not None:
                sc = sc + F.pad(blk.rel[:, diff.clamp(0, blk.rel.shape[1] - 1)].permute(1, 0, 2), (1, 0))
            a = sc.masked_fill(~allowed[:, None, :], -torch.inf).softmax(-1)
            att = (a[:, :, None, :] @ v).reshape(len(bi), d)
            z = blk.norm(query[bi, ti] + self.type_emb[perm[i]] + blk.o(att))
            out[bi, ti] = z + blk.ff(z)
        return out


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def loss_fn(model: Net, out: dict, y, mask):
    """Task loss under the supervision schedule plus registered auxiliary objectives."""
    c = model.cfg
    logits = out["logits"]
    R = len(logits)
    per_loop = []
    for lg in logits:
        ce = F.cross_entropy(lg.flatten(0, 1), y.flatten(), reduction="none", ignore_index=-100).view(y.shape)
        # Per-example mean then batch mean, exactly as MOLT-GOBLIN-02, so shared arms train identically.
        per_loop.append(((ce * mask).sum(1) / mask.sum(1).clamp(min=1)).mean())
    if c.loss == "last":
        task = per_loop[-1]
    else:
        w = torch.ones(R) if c.loss == "all" else torch.arange(1, R + 1, dtype=torch.float)
        w = w / w.sum()
        task = sum(wi * pl for wi, pl in zip(w.tolist(), per_loop))
    total = task
    a = model.aux
    parts = {"task": task.item()}
    if a.get("lb") and c.load_balance:
        v = torch.stack(a["lb"]).mean(); total = total + c.load_balance * v; parts["lb"] = v.item()
    if a.get("ent") and c.route_entropy:
        v = torch.stack(a["ent"]).mean(); total = total + c.route_entropy * v; parts["ent"] = v.item()
    if a.get("div") and c.diversity:
        v = torch.stack(a["div"]).mean(); total = total + c.diversity * v; parts["div"] = v.item()
    if a.get("func") and c.functional and c.func_coef:
        fl = []
        for outs, prev in a["func"]:
            wrong = (prev.argmax(-1) != y) & mask.bool()
            ce0 = F.cross_entropy(model.head(outs[0]).flatten(0, 1), y.flatten(), reduction="none", ignore_index=-100).view(y.shape)
            ce2 = F.cross_entropy(model.head(outs[2]).flatten(0, 1), y.flatten(), reduction="none", ignore_index=-100).view(y.shape)
            ver = model.verify_head(outs[3]).squeeze(-1)
            bce = F.binary_cross_entropy_with_logits(ver, (~wrong).float(), reduction="none")
            m = mask.float()
            fl.append((ce0 * m).sum() / m.sum().clamp(min=1)
                      + (ce2 * wrong).sum() / wrong.sum().clamp(min=1)
                      + (bce * m).sum() / m.sum().clamp(min=1))
        v = torch.stack(fl).mean(); total = total + c.func_coef * v; parts["func"] = v.item()
    return total, parts


def distill_loss(cfg: Config, s_out: dict, t_out: dict, mask):
    """Student-teacher term on scored positions."""
    T = cfg.distill_temp
    m = mask.float()

    def kl(s, t):
        k = F.kl_div(F.log_softmax(s / T, -1), F.log_softmax(t / T, -1), log_target=True, reduction="none").sum(-1)
        return (k * m).sum() / m.sum().clamp(min=1) * T * T

    if cfg.distill == "logits_final":
        return kl(s_out["logits"][-1], t_out["logits"][-1])
    if cfg.distill == "logits_all":
        tl = t_out["logits"]
        sl = s_out["logits"]
        idx = [round((i + 1) * len(tl) / len(sl)) - 1 for i in range(len(sl))]
        return sum(kl(s, tl[j]) for s, j in zip(sl, idx)) / len(sl)
    return ((s_out["h"] - t_out["h"]) ** 2).mean(-1).mul(m).sum() / m.sum().clamp(min=1)


def describe_grammar() -> str:
    return json.dumps({
        "width": "8..128", "heads": "1..4", "loops": "1..16",
        "memory": "tied | prev | prefix (prefix = MOLT-GOBLIN-01 strict prefix memory)",
        "routing": {"single": "one block", "cycle": "typed modules in fixed loop order (untied depth)",
                    "mix": "uniform soft mixture of n_types modules", "hrm": "slow/fast GRU manager, soft mixture",
                    "sparse_st": "HRM manager, Gumbel straight-through top-1 (1 module visit per loop)",
                    "anneal": "HRM manager, softmax temperature annealed to anneal_end_temp then hard top-1 at eval",
                    "sequential": "modules applied in order within each loop (n_types visits per loop)"},
        "n_types": "1..4 (1 iff routing=single)", "tie_modules": "bool: modules share one block; only type embeddings differ",
        "slow_period": "1..4", "st_tau": "0.05..5", "anneal_frac": "0.1..1 of training", "anneal_end_temp": "0.01..1",
        "update": "direct | damped", "loss": "last | all (loop forcing) | ramp",
        "load_balance": "0..2 Switch-style balance loss", "route_entropy": "-1..1 (+ sharpens, - spreads)",
        "diversity": "0..2 penalty on cosine similarity between module outputs",
        "functional": "bool: module objectives deduce=solve alone, repair=fix wrong positions, verify=predict correctness",
        "func_coef": "0..2", "distill": "none | logits_final | logits_all | hidden (student of the frozen B teacher)",
        "distill_coef": "0..2 (>0 iff distill != none)", "distill_temp": "0.5..8",
        "lr": "1e-4..1e-2", "steps": "16..6000", "batch": "8..64"}, indent=1)
