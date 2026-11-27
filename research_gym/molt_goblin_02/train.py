"""Train one compiled configuration on one task family and evaluate it with controls."""
from __future__ import annotations

import math
import time

import torch

from . import resources
from .factory import MoltConfig, MoltNet, count_params, loss_fn
from .gym import FAMILIES, TaskSpec, score

# Root ranges. Development evaluation and training never touch HIDDEN_BASE roots.
TRAIN_BASE = 1_000_000
DEV_BASE = 5_000_000
HIDDEN_BASE = 9_000_000
INTERVENTION_SET = ("reset_history", "shuffle_history", "stale_swap", "noise")


def root_seed(base: int, family: str, seed: int, salt: int = 0) -> int:
    fam = sorted(FAMILIES).index(family)
    return base + fam * 100_000 + seed * 1_000 + salt


def _exact_rows(logits, batch):
    pred = logits.argmax(-1)
    ok = (pred.eq(batch.y) | ~batch.mask).all(1)
    return ok


@torch.no_grad()
def evaluate(model: MoltNet, spec: TaskSpec, n: int, seed: int, loops: int | None = None,
             intervene: str = "none", halt_eval: bool = False, hard_eval: bool | None = None,
             project=None) -> dict:
    model.eval()
    fam = FAMILIES[spec.family]
    batch = fam.generate(spec, n, seed)
    mask = fam.admissible(batch.x) if model.cfg.ldt_mask else None
    out = model(batch.x, loops, mask=mask, intervene=intervene, halt_eval=halt_eval,
                hard_eval=hard_eval, seed=seed, project=project)
    per_loop = [_exact_rows(lg, batch) for lg in out["logits"]]
    final = per_loop[-1]
    s = score(batch, out["logits"][-1])
    res = {"spec": spec.key(), "n": n, "loops": len(per_loop), "exact": s["exact"], "token_acc": s["token_acc"],
           "per_example": final.int().tolist(),
           "exact_by_loop": [p.float().mean().item() for p in per_loop],
           "mean_block_visits": out["block_visits"].mean().item(),
           "retained_bytes_per_example": out["retained_bytes_per_example"]}
    # Useful versus redundant recurrence: transitions of whole-example correctness across loops.
    if len(per_loop) > 1:
        st = torch.stack(per_loop)              # (R,B)
        fixed = (~st[:-1] & st[1:]).sum().item()
        broke = (st[:-1] & ~st[1:]).sum().item()
        stay_ok = (st[:-1] & st[1:]).sum().item()
        first_ok = torch.where(st.any(0), st.float().argmax(0), torch.full((n,), len(per_loop))).float()
        res.update({"loop_fixes": fixed, "loop_breaks": broke, "redundant_after_correct": stay_ok,
                    "mean_first_correct_loop": first_ok.mean().item() + 1})
    if out["route_w"]:
        w = torch.stack(out["route_w"])         # (R,B,K)
        ent = -(w.clamp_min(1e-9).log() * w).sum(-1).mean().item()
        pk = w.mean((0, 1))
        pkr = w.mean(1)                          # (R,K)
        mi = (pkr * (pkr.clamp_min(1e-9).log() - pk.clamp_min(1e-9).log())).sum(-1).mean().item()
        res.update({"routing_entropy": ent, "routing_loop_mi": mi, "type_usage": pk.tolist()})
    if out["halt_p"]:
        hp = torch.stack(out["halt_p"])          # (R,B,T) p(continue) per position
        tok = torch.stack([lg.argmax(-1).eq(batch.y) for lg in out["logits"]]).float()   # (R,B,T)
        # realised utility: the position's answer goes from wrong to right on the next loop
        if len(per_loop) > 1:
            sel_m = batch.mask[None].expand(len(per_loop) - 1, -1, -1)
            target = (tok[1:] > tok[:-1]).float()[sel_m]
            p = hp[:-1][sel_m]
            brier = ((p - target) ** 2).mean().item()
            bins = torch.clamp((p * 10).long(), max=9)
            ece = 0.0
            for bi in range(10):
                sel = bins == bi
                if sel.any():
                    ece += sel.float().mean().item() * abs(p[sel].mean().item() - target[sel].mean().item())
            res.update({"halt_brier": brier, "halt_ece": ece, "halt_mean_p": hp.mean().item()})
    return res


def train_and_eval(cfg: MoltConfig, family: str, train_spec: TaskSpec, eval_specs: dict,
                   seed: int, n_eval: int = 256, base: int = DEV_BASE, interventions: bool = True,
                   extrap_loops: int | None = None, return_model: bool = False) -> dict:
    cfg = cfg.validate()
    fam = FAMILIES[family]
    torch.manual_seed(seed)
    model = MoltNet(cfg, fam.vocab, fam.n_out)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    data = fam.generate(train_spec, cfg.steps * cfg.batch, root_seed(TRAIN_BASE, family, seed))
    adm = fam.admissible(data.x) if cfg.ldt_mask else None
    wall, cpu = time.perf_counter(), time.process_time()
    losses = []
    finite = True
    for step in range(cfg.steps):
        model.train()
        sl = slice(step * cfg.batch, (step + 1) * cfg.batch)
        x, y, m = data.x[sl], data.y[sl], data.mask[sl]
        out = model(x, mask=None if adm is None else adm[sl])
        loss, _ = loss_fn(cfg, out, y, m.float())
        if not torch.isfinite(loss):
            finite = False
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == cfg.steps - 1:
            losses.append(round(loss.item(), 5))
    train_wall = time.perf_counter() - wall
    train_cpu = time.process_time() - cpu
    ev = {}
    t_eval = time.process_time()
    for name, spec in eval_specs.items():
        es = root_seed(base, spec.family, seed, salt=hash_name(name))
        ev[name] = evaluate(model, spec, n_eval, es)
        if extrap_loops and name.startswith("ood"):
            ev[name + "@deep"] = evaluate(model, spec, n_eval, es, loops=extrap_loops)
    main = next(iter(eval_specs))
    ms = root_seed(base, eval_specs[main].family, seed, salt=hash_name(main))
    if interventions and cfg.loops > 1:
        for iv in INTERVENTION_SET:
            ev[f"{main}|{iv}"] = evaluate(model, eval_specs[main], n_eval, ms, intervene=iv)
    if cfg.halt == "utility":
        ev[f"{main}|halt"] = evaluate(model, eval_specs[main], n_eval, ms, halt_eval=True)
    if cfg.routing in ("mix", "hrm"):
        ev[f"{main}|hard_route"] = evaluate(model, eval_specs[main], n_eval, ms, hard_eval=True)
    eval_cpu = time.process_time() - t_eval
    res = {"config": cfg.to_dict(), "config_digest": cfg.digest(), "family": family,
           "train_spec": train_spec.key(), "seed": seed, "params": count_params(model),
           "finite": finite, "loss_trace": losses, "train_wall_s": train_wall, "train_cpu_s": train_cpu,
           "eval_cpu_s": eval_cpu, "peak_rss_bytes": resources.peak_rss(),
           "train_tokens": cfg.steps * cfg.batch * data.x.shape[1],
           "train_block_visits_per_example": cfg.loops * (cfg.n_types if cfg.routing in ("mix", "hrm") else 1),
           "evals": ev}
    m = ev[main]
    res["summary"] = {"exact": m["exact"], "success_per_visit": m["exact"] / max(m["mean_block_visits"], 1e-9),
                      "success_per_cpu_s": m["exact"] / max(train_cpu, 1e-9),
                      "retained_bytes": m["retained_bytes_per_example"]}
    if return_model:
        res["_model"] = model
    return res


def hash_name(name: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(name)) % 997
