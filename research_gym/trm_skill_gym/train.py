"""CPU training and deterministic evaluation for compiled skill modules."""
from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
import torch

from . import modules as M
from . import tasks as T

RAM_CAP_BYTES = 3 * 1024**3  # hard incremental RAM cap
BATCH = 32
POOL_TRAIN = 8192
POOL_EVAL = 1024

_POOLS: Dict[Tuple, Dict[str, torch.Tensor]] = {}


class RamCapExceeded(RuntimeError):
    pass


class RamGuard:
    """Samples process RSS; trips when RSS exceeds the baseline by the cap.  The
    baseline is fixed when the campaign process starts, so the cap is incremental."""

    baseline: Optional[int] = None

    def __init__(self, cap: int = RAM_CAP_BYTES, period: float = 0.25):
        proc = psutil.Process(os.getpid())
        if RamGuard.baseline is None:
            RamGuard.baseline = proc.memory_info().rss
        self.proc, self.cap, self.period = proc, cap, period
        self.peak = 0
        self.tripped = False
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            d = self.proc.memory_info().rss - RamGuard.baseline
            self.peak = max(self.peak, d)
            if d > self.cap:
                self.tripped = True
            self._stop.wait(self.period)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def check(self):
        if self.tripped:
            raise RamCapExceeded(f"incremental RSS above {self.cap} bytes")


def pool(skill: str, families: List[str], split: str, count: int, cache_dir: Optional[Path] = None, offset: int = 0) -> Dict[str, torch.Tensor]:
    key = (skill, tuple(families), split, count, offset)
    if key in _POOLS:
        return _POOLS[key]
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"{skill}__{'+'.join(families)}__{split}__{offset}_{count}.pt"
        if path.exists():
            _POOLS[key] = torch.load(path)
            return _POOLS[key]
    b = T.make_batch(skill, families, split, offset, count)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(b, path)
    _POOLS[key] = b
    return b


def take(b: Dict[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {k: v[idx] for k, v in b.items()}


def train_module(skill: str, cfg: Dict, steps: int, seed: int, train_pool: Dict[str, torch.Tensor],
                 shuffled: bool = False, init_state: Optional[Dict] = None,
                 freeze_core: bool = False, deadline: Optional[float] = None) -> Tuple[M.SkillModule, Dict]:
    torch.manual_seed(seed)
    model = M.SkillModule(cfg)
    if init_state is not None:
        own = model.state_dict()
        compatible = {k: v for k, v in init_state.items() if k in own and own[k].shape == v.shape}
        own.update(compatible)
        model.load_state_dict(own)
    params = list(model.parameters())
    if freeze_core:
        frozen = {n for n, _ in model.named_parameters() if n.startswith(("net_z", "net_y", "cell", "z0", "y0", "h0", "gate"))}
        for n, p in model.named_parameters():
            p.requires_grad_(n not in frozen)
        params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / max(steps, 1))))
    )
    g = torch.Generator().manual_seed(seed)
    n = train_pool["cand"].shape[0]
    label_perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 17)) if shuffled else None
    shuffle_gen = torch.Generator().manual_seed(seed + 29)
    t0 = time.time()
    losses = []
    stopped = None
    with RamGuard() as guard:
        for step in range(steps):
            idx = torch.randint(0, n, (BATCH,), generator=g)
            batch = take(train_pool, idx)
            label = batch["label"]
            if shuffled:  # shuffled-supervision ablation: labels from other instances
                label = train_pool["label"][label_perm[idx]] * batch["mask"]
                if T.LABEL_KIND[skill] == "single":  # keep a valid one-hot target without using the true label
                    bad = label.sum(-1) == 0
                    if bad.any():
                        noise = torch.rand(int(bad.sum()), T.N, generator=shuffle_gen) * batch["mask"][bad]
                        label[bad] = torch.nn.functional.one_hot(noise.argmax(-1), T.N).float()
            model.train()
            out = M.episode(model, skill, batch, cfg)
            if shuffled:
                out["exact"] = [(((p - label).abs() * batch["mask"]).sum(-1) == 0).float() for p in out["pred"]]
            loss = M.episode_loss(skill, out, batch, cfg, label)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
            guard.check()
            if deadline is not None and time.time() > deadline:
                stopped = f"deadline at step {step + 1}"
                break
        peak = guard.peak
    info = {
        "steps": len(losses),
        "train_seconds": time.time() - t0,
        "examples_seen": len(losses) * BATCH,
        "final_loss": sum(losses[-20:]) / max(len(losses[-20:]), 1),
        "peak_rss_delta_bytes": int(peak),
        "stopped": stopped,
    }
    return model, info


def ece(conf: torch.Tensor, correct: torch.Tensor, bins: int = 10) -> float:
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (conf >= edges[i]) & ((conf < edges[i + 1]) if i < bins - 1 else (conf <= 1))
        if m.any():
            total += m.float().mean().item() * abs(conf[m].mean().item() - correct[m].mean().item())
    return total


@torch.no_grad()
def evaluate(model: Optional[M.SkillModule], skill: str, cfg: Dict, eval_pool: Dict[str, torch.Tensor],
             visits: Optional[int] = None, heuristic: bool = False, chunk: int = 256) -> Dict:
    n = eval_pool["cand"].shape[0]
    rows = {"exact": [], "ok": [], "conf": [], "halt_exact": [], "halt_visits": [], "halt_vcalls": [], "exact_by_visit": []}
    flops = 0
    vcalls = 0
    t0 = time.time()
    if model is not None:
        model.eval()
    for s in range(0, n, chunk):
        b = {k: v[s : s + chunk] for k, v in eval_pool.items()}
        M.flops_reset()
        out = M.episode(model, skill, b, cfg, visits=visits, heuristic=heuristic)
        flops += M.flops_read()
        vcalls += out["vcalls"] * b["cand"].shape[0]
        final = out["pred"][-1]
        rows["exact"].append(out["exact"][-1])
        rows["ok"].append(T.verify(skill, b, final)[0].float())
        rows["conf"].append(torch.sigmoid(out["conf"][-1]))
        rows["exact_by_visit"].append(torch.stack(out["exact"], 1))
        if not heuristic:
            h = M.halting(out)
            rows["halt_exact"].append(h["exact"])
            rows["halt_visits"].append(h["visits_used"])
            rows["halt_vcalls"].append(h["vcalls"])
    cat = {k: torch.cat(v) for k, v in rows.items() if v}
    R = cat["exact_by_visit"].shape[1]
    res = {
        "n": n,
        "exact": cat["exact"].mean().item(),
        "repair_success": cat["ok"].mean().item(),
        "exact_by_visit": cat["exact_by_visit"].mean(0).tolist(),
        "visits": R,
        "verifier_calls_per_instance": vcalls / n + 1.0,  # in-loop calls + the final receipt
        "flops_per_instance": flops / n,
        "eval_seconds": time.time() - t0,
    }
    if not heuristic:
        res["ece"] = ece(cat["conf"], cat["exact"])
        res["mean_confidence"] = cat["conf"].mean().item()
        res["adaptive_exact"] = cat["halt_exact"].mean().item()
        res["adaptive_visits"] = cat["halt_visits"].mean().item()
        res["adaptive_verifier_calls"] = cat["halt_vcalls"].mean().item() + 1.0
    return res
