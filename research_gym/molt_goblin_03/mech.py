"""Module specialization measurements for typed-module recurrent models.

All measurements run on a trained model and a batch of evaluation roots:
activation similarity (linear CKA and cosine between module outputs), gradient
similarity between modules' parameter gradients, module permutation with the router
fixed, single-module ablation, routing interventions (uniform, shuffled across examples,
forced one-hot), residual effective rank, and task-conditional routing from a model
trained on several families at once.
"""
from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F

from .factory3 import Config, Net, loss_fn
from .gym3 import FAMILIES3, TaskSpec


def linear_cka(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    num = (a.T @ b).pow(2).sum()
    den = (a.T @ a).pow(2).sum().sqrt() * (b.T @ b).pow(2).sum().sqrt()
    return float(num / den.clamp_min(1e-12))


def effective_rank(m: torch.Tensor) -> float:
    s = torch.linalg.svdvals(m - m.mean(0, keepdim=True))
    p = s / s.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))


def _exact(model, x, batch, **kw) -> float:
    out = model(x, **kw)
    return float((out["logits"][-1].argmax(-1).eq(batch.y) | ~batch.mask).all(1).float().mean())


@torch.no_grad()
def specialization(model: Net, spec: TaskSpec, n: int, seed: int) -> dict:
    model.eval()
    c = model.cfg
    batch = FAMILIES3[spec.family].generate(spec, n, seed)
    x = batch.x
    sel = batch.mask.reshape(-1)
    col: dict = {}
    clean = _exact(model, x, batch, collect=col)
    states = col["states"]
    res = {"spec": spec.key(), "clean": clean,
           "effective_rank_final": effective_rank(states[-1].reshape(-1, c.width)[sel]),
           "effective_rank_updates": effective_rank(torch.cat([(states[i] - (states[i - 1] if i else col["base"]))
                                                               .reshape(-1, c.width)[sel] for i in range(len(states))]))}
    if c.n_types == 1:
        return res
    mods = col.get("module_outputs")
    if mods:
        cka, cos = [], []
        for loop_outs in mods:
            flat = [o.reshape(-1, c.width)[sel] for o in loop_outs]
            for i, j in itertools.combinations(range(len(flat)), 2):
                cka.append(linear_cka(flat[i], flat[j]))
                cos.append(float(F.cosine_similarity(flat[i], flat[j], dim=-1).mean()))
        res["module_cka_mean"] = sum(cka) / len(cka)
        res["module_cos_mean"] = sum(cos) / len(cos)
    K = c.n_types
    res["ablate_module"] = [_exact(model, x, batch, module_mask=[float(j != i) for j in range(K)]) for i in range(K)]
    shift = [(i + 1) % K for i in range(K)]
    res["permute_modules"] = _exact(model, x, batch, module_perm=shift)
    if c.routing in ("mix", "hrm", "sparse_st", "anneal"):
        res["route_uniform"] = _exact(model, x, batch, route_override=lambda w, r: torch.full_like(w, 1.0 / K))
        res["route_shuffled"] = _exact(model, x, batch, route_override=lambda w, r: w.roll(1, dims=0))
        res["route_forced"] = [_exact(model, x, batch, route_override=lambda w, r, i=i:
                                      F.one_hot(torch.full(w.shape[:-1], i, dtype=torch.long), K).to(w.dtype))
                               for i in range(K)]
    return res


def gradient_similarity(model: Net, spec: TaskSpec, n: int, seed: int) -> dict:
    """Cosine between modules' task-loss gradients on one batch (untied modules only)."""
    if len(model.blocks) < 2:
        return {}
    model.train()
    batch = FAMILIES3[spec.family].generate(spec, n, seed)
    model.zero_grad(set_to_none=True)
    loss, _ = loss_fn(model, model(batch.x, seed=seed), batch.y, batch.mask.float())
    loss.backward()
    grads = [torch.cat([p.grad.reshape(-1) for p in blk.parameters() if p.grad is not None]) for blk in model.blocks]
    sims = [float(F.cosine_similarity(grads[i], grads[j], dim=0)) for i, j in itertools.combinations(range(len(grads)), 2)]
    norms = [float(g.norm()) for g in grads]
    model.zero_grad(set_to_none=True)
    model.eval()
    return {"grad_cos_mean": sum(sims) / len(sims), "grad_cos_pairs": sims, "grad_norms": norms}


# ------------------------------------------------------------------ task-conditional routing
def _offsets(families):
    voff, ooff, v, o = {}, {}, 0, 0
    for f in families:
        voff[f], ooff[f] = v, o
        v += FAMILIES3[f].vocab
        o += FAMILIES3[f].n_out
    return voff, ooff, v, o


def _union(batch, f, voff, ooff, n_out_total):
    x = batch.x + voff[f]
    y = torch.where(batch.mask, batch.y + ooff[f], torch.full_like(batch.y, -100))
    m = torch.zeros(batch.x.shape + (n_out_total,), dtype=torch.bool)
    m[..., ooff[f]:ooff[f] + FAMILIES3[f].n_out] = True
    return x, y, m


def multitask_routing(cfg: Config, panels: dict, seed: int, steps_per_family: int, base_seed: int,
                      n_eval: int = 256) -> dict:
    """Train one model on several families (shared modules and router); report routing by family."""
    fams = list(panels)
    voff, ooff, V, O = _offsets(fams)
    torch.manual_seed(seed)
    model = Net(cfg, V, O)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    steps = steps_per_family * len(fams)
    data = {f: FAMILIES3[f].generate(panels[f]["train"], steps_per_family * cfg.batch, base_seed + 1000 * i + seed)
            for i, f in enumerate(fams)}
    for s in range(steps):
        f = fams[s % len(fams)]
        k = s // len(fams)
        d = data[f]
        sl = slice(k * cfg.batch, (k + 1) * cfg.batch)
        sub = type(d)(d.x[sl], d.y[sl], d.mask[sl])
        x, y, m = _union(sub, f, voff, ooff, O)
        model.train()
        model.progress = s / max(1, steps - 1)
        out = model(x, mask=m, seed=seed)
        loss, _ = loss_fn(model, out, y, sub.mask.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    per = {}
    usage = []
    with torch.no_grad():
        for i, f in enumerate(fams):
            b = FAMILIES3[f].generate(panels[f]["eval"]["dev"], n_eval, base_seed + 500_000 + 1000 * i + seed)
            x, y, m = _union(b, f, voff, ooff, O)
            out = model(x, mask=m)
            ok = (out["logits"][-1].argmax(-1).eq(y) | ~b.mask).all(1).float().mean().item()
            if out["route_p"]:
                p = torch.stack(out["route_p"])                     # (R,B,T,K)
                mm = b.mask[None, :, :, None].float()
                pu = ((p * mm).sum((0, 1, 2)) / mm.sum((0, 1, 2)))
                hard = F.one_hot(torch.stack(out["route_w"]).argmax(-1), cfg.n_types).float()
                hu = (hard * mm).sum((0, 1, 2)) / mm.sum((0, 1, 2))
                usage.append(pu)
                per[f] = {"exact": ok, "router_usage": pu.tolist(), "hard_usage": hu.tolist()}
            else:
                per[f] = {"exact": ok}
    res = {"families": per}
    if usage:
        U = torch.stack(usage)                                   # (F,K), equal family weights
        pk = U.mean(0)
        res["family_module_mi_nats"] = float((U * (U.clamp_min(1e-9).log() - pk.clamp_min(1e-9).log())).sum(-1).mean())
        res["max_mi_nats"] = math.log(min(len(fams), cfg.n_types))
    return res
