"""Reusable skill-module contract and its compiled implementations.

Contract (one *visit* of a module, see ``SkillModule.visit``):

* input: typed skill observation (``cand`` / ``rel`` / ``glob`` tensors from
  ``tasks``) including the host-written verifier feedback channel
* persistent latent workspace ``z`` and recurrent answer/candidate state ``y``
  carried across visits (detached between visits, as in TRM deep supervision)
* output: candidate action scores (logits), confidence that the current decision
  is exact, predicted value of another recurrence, and the updated state
* optional LDT projection: sound host certificates overwrite certified logits
  and are exposed to the latent through a reserved observation column
* independent verifier receipts are produced by the host (``tasks.verify``) and
  never by the module
* explicit termination: ``episode`` halts at the first visit whose confidence
  reaches ``halt_tau`` or whose predicted value of recurrence is <= 0

Cores:

* ``trm``  native TRM update rule (Jolicoeur-Martineau 2025): n latent updates
  ``z <- net(x + y + z)`` then an answer update ``y <- net(y + z)``, T cycles per
  visit with gradient only through the last ``grad_cycles`` cycles, deep
  supervision across visits.  The backbone ``net`` is a small set/relation mixer,
  not the paper's 2-layer attention stack over grid tokens.
* ``gru``  recurrent proxy: a GRU cell over the same mixer, same visit count and
  step count.  Reported as ``recurrent_proxy_gru``.
* ``ff``   feedforward head: same mixer stack applied once per visit, no carried
  state; it only sees verifier feedback through the observation channel.

Answer-state feedback (TRM cores) uses DataGoblin's validated schedules
(``datagoblin.training_gym.rlt_feedback.feedback_weight``): ``direct`` (eta=1),
``rlt`` (eta = 1/(D - u)), ``matched_damping`` (constant eta with RLT's
coefficient sum); the terminal update always fully replaces.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tasks as T

CERT_COL = T.FS - 1  # reserved observation column for LDT certificates
IN_DIM = T.FC + T.G

_FLOPS = [0, False]


def flops_reset() -> None:
    _FLOPS[0] = 0
    _FLOPS[1] = True


def flops_read() -> int:
    _FLOPS[1] = False
    return int(_FLOPS[0])


class Lin(nn.Linear):
    def forward(self, x):
        if _FLOPS[1]:
            _FLOPS[0] += 2 * self.in_features * self.out_features * (x.numel() // x.shape[-1])
        return super().forward(x)


def feedback_weight(mode: str, visit: int, depth: int) -> float:
    """Byte-for-byte port of DataGoblin rlt_feedback.feedback_weight."""
    if mode not in ("direct", "rlt", "matched_damping"):
        raise ValueError(f"unknown feedback {mode!r}")
    if depth < 1 or not 0 <= visit < depth:
        raise ValueError("bad visit/depth")
    if mode == "direct" or visit == depth - 1:
        return 1.0
    if mode == "rlt":
        return 1 / (depth - visit)
    return math.fsum(1 / j for j in range(2, depth + 1)) / (depth - 1)


class Mixer(nn.Module):
    """Candidate-set mixer: per-candidate MLP over [u, aggregate(u)]."""

    def __init__(self, H: int, aggregation: str, layers: int, norm: str):
        super().__init__()
        self.H, self.aggregation = H, aggregation
        extra = {"pointwise": 0, "set_mean": H, "relation": 2 * H, "attention": H, "relation+mean": 3 * H}[aggregation]
        if aggregation == "attention":
            self.qkv = Lin(H, 3 * H, bias=False)
            self.rel_bias = nn.Parameter(torch.tensor(1.0))
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.ModuleDict({"a": Lin(H + extra, 2 * H), "b": Lin(2 * H, H)}))
        self.norm = nn.LayerNorm(H) if norm == "layer" else nn.Identity()

    def aggregate(self, u, rel, mask):
        B, N, H = u.shape
        m = mask[..., None]
        parts = []
        if self.aggregation in ("set_mean", "relation+mean"):
            mean = (u * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
            parts.append(mean.expand(B, N, H))
        if self.aggregation in ("relation", "relation+mean"):
            fwd = torch.bmm(rel, u) / (rel.sum(-1, keepdim=True) + 1)
            bwd = torch.bmm(rel.transpose(1, 2), u) / (rel.sum(1)[..., None] + 1)
            parts += [fwd, bwd]
            if _FLOPS[1]:
                _FLOPS[0] += 4 * B * N * N * H
        if self.aggregation == "attention":
            q, k, v = self.qkv(u).chunk(3, -1)
            att = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(H) + self.rel_bias * rel
            att = att.masked_fill(mask[:, None, :] == 0, -1e9).softmax(-1)
            parts.append(torch.bmm(att, v))
            if _FLOPS[1]:
                _FLOPS[0] += 4 * B * N * N * H
        return parts

    def forward(self, u, rel, mask):
        for layer in self.layers:
            h = torch.cat([u] + self.aggregate(u, rel, mask), -1)
            u = u + layer["b"](F.gelu(layer["a"](h)))
        return self.norm(u) * mask[..., None]


class SkillModule(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = dict(cfg)
        H = cfg["hidden"]
        self.H = H
        self.embed = Lin(IN_DIM, H)
        core = cfg["core"]
        self.net_z = Mixer(H, cfg["aggregation"], cfg["mixer_layers"], cfg["norm"])
        if core == "trm":
            self.net_y = self.net_z if cfg["state_partition"] == "shared" else Mixer(H, cfg["aggregation"], cfg["mixer_layers"], cfg["norm"])
            self.z0 = nn.Parameter(torch.randn(H) * 0.1)
            self.y0 = nn.Parameter(torch.randn(H) * 0.1)
            if cfg["residual"] == "gated":
                self.gate = nn.Parameter(torch.zeros(H))
        elif core == "gru":
            self.cell = nn.GRUCell(H, H)
            self.h0 = nn.Parameter(torch.randn(H) * 0.1)
        self.head = Lin(H, 1)
        self.conf_head = Lin(2 * H, 1)
        self.value_head = Lin(2 * H, 1)

    # -------------------------------------------------------------- helpers
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def answer_updates_per_visit(self) -> int:
        c = self.cfg
        if c["core"] != "trm":
            return 1
        return c["cycles"] * (c["latent_steps"] if c["update_schedule"] == "interleaved" else 1)

    def init_state(self, batch) -> Dict[str, torch.Tensor]:
        B = batch["cand"].shape[0]
        shape = (B, T.N, self.H)
        if self.cfg["core"] == "trm":
            return {"z": self.z0.expand(shape).clone(), "y": self.y0.expand(shape).clone(), "u": 0}
        if self.cfg["core"] == "gru":
            return {"h": self.h0.expand(shape).clone(), "u": 0}
        return {"u": 0}

    def _pool(self, a, b, mask):
        m = mask[..., None]
        pa = (a * m).sum(1) / m.sum(1).clamp(min=1)
        pb = (b * m).sum(1) / m.sum(1).clamp(min=1)
        return torch.cat([pa, pb], -1)

    def _latent(self, x, y, z, rel, mask):
        new = self.net_z(x + y + z, rel, mask)
        r = self.cfg["residual"]
        if r == "none":
            return new
        if r == "add":
            return z + new
        return z + torch.sigmoid(self.gate) * (new - z)

    def _answer(self, y, z, rel, mask, u, depth):
        e = self.net_y(y + z, rel, mask)
        eta = feedback_weight(self.cfg["feedback"], min(u, depth - 1), depth)
        return y + eta * (e - y) if eta != 1.0 else e

    def _trm_cycle(self, x, st, rel, mask, depth):
        c = self.cfg
        y, z, u = st["y"], st["z"], st["u"]
        n = c["latent_steps"]
        if c["update_schedule"] == "latent_then_answer":
            for _ in range(n):
                z = self._latent(x, y, z, rel, mask)
            y = self._answer(y, z, rel, mask, u, depth)
            u += 1
        elif c["update_schedule"] == "answer_first":
            y = self._answer(y, z, rel, mask, u, depth)
            u += 1
            for _ in range(n):
                z = self._latent(x, y, z, rel, mask)
        else:  # interleaved
            for _ in range(n):
                z = self._latent(x, y, z, rel, mask)
                y = self._answer(y, z, rel, mask, u, depth)
                u += 1
        return {"y": y, "z": z, "u": u}

    # -------------------------------------------------------------- one visit
    def visit(self, cand, batch, st, planned_visits: int):
        c = self.cfg
        rel, mask = batch["rel"], batch["mask"]
        obs = torch.cat([cand, batch["glob"][:, None, :].expand(-1, T.N, -1)], -1)
        x = self.embed(obs) * mask[..., None]
        if c["core"] == "trm":
            depth = planned_visits * self.answer_updates_per_visit()
            if c["latent_reset"]:
                st = dict(st, z=self.z0.expand_as(st["z"]).clone())
            st = {"y": st["y"].detach(), "z": st["z"].detach(), "u": st["u"]}
            free = c["cycles"] - c["grad_cycles"]
            if free > 0:
                with torch.no_grad():
                    for _ in range(free):
                        st = self._trm_cycle(x, st, rel, mask, depth)
                st = {"y": st["y"].detach(), "z": st["z"].detach(), "u": st["u"]}
            for _ in range(c["grad_cycles"]):
                st = self._trm_cycle(x, st, rel, mask, depth)
            ans, lat = st["y"], st["z"]
        elif c["core"] == "gru":
            h = st["h"].detach()
            steps = c["cycles"] * (c["latent_steps"] + 1)
            B, N, H = h.shape
            for _ in range(steps):
                m = self.net_z(h + x, rel, mask)
                h = self.cell(m.reshape(B * N, H), h.reshape(B * N, H)).reshape(B, N, H) * mask[..., None]
                if _FLOPS[1]:
                    _FLOPS[0] += 2 * 3 * 2 * H * H * B * N
            st = {"h": h, "u": st["u"] + 1}
            ans = lat = h
        else:
            ans = lat = self.net_z(x, rel, mask)
            st = {"u": st["u"] + 1}
        logits = self.head(ans).squeeze(-1)
        pooled = self._pool(ans, lat, mask)
        conf = self.conf_head(pooled).squeeze(-1)
        value = torch.tanh(self.value_head(pooled).squeeze(-1))
        return logits, conf, value, st


def default_verifier_visits(n_calls: int, visits: int) -> List[int]:
    """In-loop verifier calls go to the visits just before the final one."""
    n_calls = min(n_calls, max(visits - 1, 0))
    return list(range(visits - 1 - n_calls, visits - 1))


def project(logits: torch.Tensor, cert: torch.Tensor) -> torch.Tensor:
    out = torch.where(cert > 0, torch.clamp(logits, min=6.0), logits)
    return torch.where(cert < 0, torch.clamp(out, max=-6.0), out)


def episode(model: Optional[SkillModule], skill: str, batch: Dict[str, torch.Tensor], cfg: Dict,
            visits: Optional[int] = None, heuristic: bool = False) -> Dict:
    """Run the matched host protocol: R visits, verifier diagnostics written into the
    feedback channel after scheduled visits, LDT projection when configured.
    ``heuristic=True`` runs arm A (existing skill rule + its local repair step)."""
    R = visits or cfg["visits"]
    mask = batch["mask"]
    cand = batch["cand"].clone()
    cand[..., T.FS:] = 0
    cert = None
    if cfg.get("ldt_projection"):
        cert = T.certify(skill, batch)
        cand[..., CERT_COL] = cert
    sched = default_verifier_visits(cfg.get("verifier_visits", 1), R)
    st = model.init_state(batch) if model is not None else None
    out = {"logits": [], "conf": [], "value": [], "pred": [], "exact": [], "vcalls": 0}
    pred = None
    for r in range(R):
        if heuristic:
            if pred is None:
                logits = T.heuristic_scores(skill, batch)
                if cert is not None:
                    logits = project(logits, cert)
                pred = T.decide(skill, logits, mask)
            elif out["_diag"] is not None:
                pred = T.heuristic_repair_step(skill, batch, pred, out["_diag"])
                logits = pred * 8 - 4
            conf = torch.zeros(mask.shape[0])
            value = torch.zeros(mask.shape[0])
        else:
            logits, conf, value, st = model.visit(cand, batch, st, R)
            if cert is not None:
                logits = project(logits, cert)
            pred = T.decide(skill, logits.detach(), mask)
        out["_diag"] = None
        out["logits"].append(logits)
        out["conf"].append(conf)
        out["value"].append(value)
        out["pred"].append(pred)
        out["exact"].append(T.exact(skill, batch, pred))
        if r in sched:
            ok, diag = T.verify(skill, batch, pred)
            cand = cand.clone()
            cand[..., T.FS] = mask
            cand[..., T.FS + 1] = ok.float()[:, None] * mask
            cand[..., T.FS + 2 : T.FS + 4] = diag
            out["vcalls"] += 1
            out["_diag"] = diag
    out.pop("_diag", None)
    out["sched"] = sched
    return out


def task_loss(skill: str, logits: torch.Tensor, batch: Dict[str, torch.Tensor], label: torch.Tensor) -> torch.Tensor:
    mask = batch["mask"]
    if T.LABEL_KIND[skill] == "single":
        masked = logits.masked_fill(mask == 0, -1e9)
        return F.cross_entropy(masked, label.argmax(-1))
    bce = F.binary_cross_entropy_with_logits(logits, label, reduction="none") * mask
    return (bce.sum(-1) / mask.sum(-1).clamp(min=1)).mean()


def episode_loss(skill: str, out: Dict, batch: Dict[str, torch.Tensor], cfg: Dict, label: torch.Tensor) -> torch.Tensor:
    R = len(out["logits"])
    if cfg["supervision"] == "final":
        w = [0.0] * (R - 1) + [1.0]
    elif cfg["supervision"] == "deep":
        w = [1.0 / R] * R
    else:  # deep_ramp
        s = R * (R + 1) / 2
        w = [(r + 1) / s for r in range(R)]
    loss = 0.0
    for r in range(R):
        if w[r] > 0:
            loss = loss + w[r] * task_loss(skill, out["logits"][r], batch, label)
        loss = loss + (0.5 / R) * F.binary_cross_entropy_with_logits(out["conf"][r], out["exact"][r])
        if r + 1 < R:
            target = out["exact"][r + 1] - out["exact"][r]
            loss = loss + (0.25 / R) * F.mse_loss(out["value"][r], target)
    return loss


def halting(out: Dict, tau: float = 0.85) -> Dict[str, torch.Tensor]:
    """Explicit termination: halt at the first visit with sigmoid(conf) >= tau or
    predicted value of another recurrence <= 0 (after at least one visit)."""
    R = len(out["exact"])
    B = out["exact"][0].shape[0]
    halted = torch.full((B,), R - 1, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    for r in range(R):
        stop = (torch.sigmoid(out["conf"][r]) >= tau) | (out["value"][r] <= 0)
        if r == R - 1:
            stop = torch.ones(B, dtype=torch.bool)
        new = stop & ~done
        halted[new] = r
        done |= stop
    ex = torch.stack(out["exact"], 1).gather(1, halted[:, None]).squeeze(1)
    sched = out["sched"]
    vcalls = torch.tensor([sum(1 for s in sched if s < h) for h in halted.tolist()], dtype=torch.float)
    return {"halted_visit": halted, "exact": ex, "visits_used": (halted + 1).float(), "vcalls": vcalls}


def match_hidden(cfg: Dict, target_params: int) -> Dict:
    """Return cfg with ``hidden`` chosen so the parameter count is closest to target."""
    best, best_gap = cfg, None
    for H in range(16, 161, 2):
        c = dict(cfg, hidden=H)
        n = SkillModule(c).n_params()
        gap = abs(n - target_params)
        if best_gap is None or gap < best_gap:
            best, best_gap = c, gap
    return best
