"""Training, evaluation and cost accounting for MOLT-GOBLIN-03 configurations."""
from __future__ import annotations

import hashlib
import json
import statistics
import threading
import time
from pathlib import Path

import psutil
import torch
from torch.utils.flop_counter import FlopCounterMode

from .factory3 import Config, Net, count_params, distill_loss, logical_visits, loss_fn
from .gym3 import FAMILIES3, TaskSpec, score

# Fresh root ranges, disjoint from MOLT-GOBLIN-02 (1M / 5M / 9M bases).
TRAIN3_BASE = 21_000_000
DEV3_BASE = 25_000_000
HIDDEN3_BASE = 29_000_000
TEACHER_CACHE = Path(__file__).resolve().parent / "receipts" / "teacher_cache"


# Frozen family order for root derivation; new families are appended, never re-sorted.
ROOT_ORDER = ("bounded_walk", "cex_repair", "cumsum", "cumsum3", "experiment_portfolio", "experiment_select",
              "local_map", "pointer_chase", "proof_deps", "proof_resolve", "repair", "research_plan", "revision",
              "revision_retract", "shallow_copy", "local_map3")


def root_seed(base: int, family: str, seed: int, salt: int = 0) -> int:
    return base + ROOT_ORDER.index(family) * 100_000 + seed * 1_000 + salt


def salt_of(name: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(name)) % 997


class PeakRSS:
    """Samples process RSS every 20 ms; reports peak and increment over the start value."""

    def __init__(self):
        self.p = psutil.Process()
        self.start = self.peak = self.p.memory_info().rss
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.p.memory_info().rss)
            self._stop.wait(0.02)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self.p.memory_info().rss)


def train(cfg: Config, family: str, spec: TaskSpec, seed: int, base: int = TRAIN3_BASE, teacher: Net | None = None):
    cfg = cfg.validate()
    fam = FAMILIES3[family]
    torch.manual_seed(seed)
    model = Net(cfg, fam.vocab, fam.n_out)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    data = fam.generate(spec, cfg.steps * cfg.batch, root_seed(base, family, seed))
    if teacher is not None:
        teacher.eval()
    trace, finite = [], True
    wall, cpu = time.perf_counter(), time.process_time()
    for step in range(cfg.steps):
        model.train()
        model.progress = step / max(1, cfg.steps - 1)
        sl = slice(step * cfg.batch, (step + 1) * cfg.batch)
        x, y, m = data.x[sl], data.y[sl], data.mask[sl].float()
        out = model(x, seed=seed)
        loss, parts = loss_fn(model, out, y, m)
        if teacher is not None and cfg.distill != "none":
            with torch.no_grad():
                t_out = teacher(x)
            dl = distill_loss(cfg, out, t_out, m)
            loss = loss + cfg.distill_coef * dl
            parts["distill"] = dl.item()
        if not torch.isfinite(loss):
            finite = False
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == cfg.steps - 1:
            trace.append({"step": step, **{k: round(v, 5) for k, v in parts.items()}})
    model.progress = 1.0
    return model, {"finite": finite, "train_wall_s": time.perf_counter() - wall,
                   "train_cpu_s": time.process_time() - cpu, "loss_trace": trace,
                   "train_tokens": cfg.steps * cfg.batch * data.x.shape[1]}


def _exact_rows(logits, batch):
    return (logits.argmax(-1).eq(batch.y) | ~batch.mask).all(1)


@torch.no_grad()
def evaluate(model: Net, spec: TaskSpec, n: int, seed: int, **kw) -> dict:
    model.eval()
    batch = FAMILIES3[spec.family].generate(spec, n, seed)
    out = model(batch.x, seed=seed, **kw)
    per_loop = [_exact_rows(lg, batch) for lg in out["logits"]]
    s = score(batch, out["logits"][-1])
    res = {"spec": spec.key(), "n": n, "exact": s["exact"], "token_acc": s["token_acc"],
           "per_example": per_loop[-1].int().tolist(), "exact_by_loop": [p.float().mean().item() for p in per_loop],
           "visits": out["visits"]}
    if len(per_loop) > 1:
        st = torch.stack(per_loop)
        res.update({"loop_fixes": int((~st[:-1] & st[1:]).sum()), "loop_breaks": int((st[:-1] & ~st[1:]).sum()),
                    "redundant_after_correct": int((st[:-1] & st[1:]).sum())})
    if out["route_w"]:
        w = torch.stack(out["route_w"])                 # (R,B,T,K) applied weights
        p = torch.stack(out["route_p"])                 # router probabilities
        m = batch.mask[None, :, :, None].float()
        pk = (p * m).sum((0, 1, 2)) / m.sum((0, 1, 2))
        pkr = (p * m).sum((1, 2)) / m.sum((1, 2))
        res["routing_entropy"] = float((-(p.clamp_min(1e-9).log() * p).sum(-1) * m[..., 0]).sum() / m.sum())
        res["routing_loop_mi"] = float((pkr * (pkr.clamp_min(1e-9).log() - pk.clamp_min(1e-9).log())).sum(-1).mean())
        res["type_usage"] = pk.tolist()
        res["hard_usage"] = (F_one_hot_mean(w, m)).tolist()
    return res


def F_one_hot_mean(w, m):
    oh = torch.nn.functional.one_hot(w.argmax(-1), w.shape[-1]).float()
    return (oh * m).sum((0, 1, 2)) / m.sum((0, 1, 2))


@torch.no_grad()
def measure_cost(model: Net, spec: TaskSpec, seed: int, n: int = 64, reps: int = 10, dispatch: bool = False) -> dict:
    """Measured forward FLOPs per example and median latency per batch of n."""
    model.eval()
    batch = FAMILIES3[spec.family].generate(spec, n, seed)
    with FlopCounterMode(display=False) as fc:
        model(batch.x, dispatch=dispatch)
    flops = fc.get_total_flops() / n
    for _ in range(2):
        model(batch.x, dispatch=dispatch)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(batch.x, dispatch=dispatch)
        times.append(time.perf_counter() - t0)
    return {"flops_per_example": flops, "latency_ms_per_batch": 1000 * statistics.median(times), "batch": n,
            "dispatch": dispatch}


def teacher_path(teacher_cfg: Config, family: str, spec: TaskSpec, seed: int) -> Path:
    key = hashlib.sha256(json.dumps([teacher_cfg.canonical(), family, spec.key(), seed]).encode()).hexdigest()[:20]
    return TEACHER_CACHE / f"{key}.pt"


def save_teacher(model: Net, family: str, spec: TaskSpec, seed: int) -> None:
    TEACHER_CACHE.mkdir(parents=True, exist_ok=True)
    path = teacher_path(model.cfg, family, spec, seed)
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        torch.save(model.state_dict(), tmp)
        tmp.replace(path)


def teacher_for(teacher_cfg: Config, family: str, spec: TaskSpec, seed: int) -> Net:
    """Train (or load) the frozen teacher for a family and seed; cached on disk by digest."""
    TEACHER_CACHE.mkdir(parents=True, exist_ok=True)
    path = teacher_path(teacher_cfg, family, spec, seed)
    fam = FAMILIES3[family]
    if path.exists():
        m = Net(teacher_cfg, fam.vocab, fam.n_out)
        m.load_state_dict(torch.load(path, weights_only=True))
        return m.eval()
    m, _ = train(teacher_cfg, family, spec, seed)
    tmp = path.with_suffix(".tmp")
    torch.save(m.state_dict(), tmp)
    tmp.replace(path)
    return m.eval()


INTERVENTIONS = ("reset_history", "stale_swap", "noise")


def run_cell(cfg: Config, family: str, panel: dict, seed: int, base: int, teacher_cfg: Config | None = None,
             n_eval: int = 256, extrap_loops: int | None = None, interventions: bool = True,
             return_model: bool = False, cache_teacher: bool = False) -> dict:
    """Train one arm on one family and seed; evaluate every registered spec and cost."""
    teacher = teacher_for(teacher_cfg, family, panel["train"], seed) if cfg.distill != "none" else None
    with PeakRSS() as mem:
        model, st = train(cfg, family, panel["train"], seed, teacher=teacher)
        if cache_teacher:
            save_teacher(model, family, panel["train"], seed)
        ev = {}
        for name, spec in panel["eval"].items():
            es = root_seed(base, spec.family, seed, salt_of(name))
            ev[name] = evaluate(model, spec, n_eval, es)
            if extrap_loops and name.startswith("ood"):
                ev[name + "@deep"] = evaluate(model, spec, n_eval, es, loops=extrap_loops)
        dev = panel["eval"]["dev"]
        ds = root_seed(base, family, seed, salt_of("dev"))
        if interventions and cfg.loops > 1:
            for iv in INTERVENTIONS:
                ev[f"dev|{iv}"] = evaluate(model, dev, n_eval, ds, intervene=iv)
        if cfg.routing in ("mix", "hrm"):
            ev["dev|hard_route"] = evaluate(model, dev, n_eval, ds, hard_eval=True)
        cost = measure_cost(model, dev, ds)
        if cfg.routing in ("sparse_st", "anneal", "mix", "hrm"):
            cost["dispatch_hard"] = _dispatch_cost(model, dev, ds)
    fam = FAMILIES3[family]
    res = {"config": cfg.to_dict(), "config_digest": cfg.digest(), "family": family, "seed": seed,
           "params": count_params(model), "logical_visits": logical_visits(cfg), **st, "evals": ev, "cost": cost,
           "peak_rss_bytes": mem.peak, "rss_increment_bytes": mem.peak - mem.start,
           "summary": {"exact": ev["dev"]["exact"]}}
    if return_model:
        res["_model"] = model
    return res


@torch.no_grad()
def _dispatch_cost(model: Net, spec: TaskSpec, seed: int, n: int = 64, reps: int = 10) -> dict:
    """Hard top-1 routing with per-module dispatch: the inference path of sparse arms."""
    model.eval()
    batch = FAMILIES3[spec.family].generate(spec, n, seed)
    kw = {"hard_eval": True, "dispatch": True}
    with FlopCounterMode(display=False) as fc:
        model(batch.x, **kw)
    times = []
    for i in range(reps + 2):
        t0 = time.perf_counter()
        model(batch.x, **kw)
        if i >= 2:
            times.append(time.perf_counter() - t0)
    return {"flops_per_example": fc.get_total_flops() / n, "latency_ms_per_batch": 1000 * statistics.median(times)}
