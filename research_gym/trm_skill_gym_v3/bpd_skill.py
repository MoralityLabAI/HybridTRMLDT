"""Behavioral Parameter Decomposition (BPD) of the recurrent TRM skill modules.

Unlike the GYM-02 controls (``trm_skill_gym_v2.subspace``: latent orthogonal projector and an exact SVD of
one matrix; ``runner.ResetModule``: coordinate reset) this module trains a real BPD decomposition with the
public BPD library (MoralityLabAI/BPD): one shared factor pair (V, U) per physical recurrent matrix
(``TiedComponentLinear``, exact residual, pass index tau = invocation count), pass-tied causal importance
(``TiedCINetwork``), faithfulness + trajectory reconstruction + minimality losses, pass-conditional
mean-write ablation, and the pass-shared persistent PGD adversary, all driven by ``BPDTrainer``.

Interventions compared on identical held-out roots (fit 2,300,000+, calibration 2,400,000+, test
2,500,000+; selection never sees test):

  A unmodified                     B full latent reset          C selected coordinate reset
  D behavioral orthogonal projector (subspace.fit_basis)        E exact-SVD component ablation
  F learned BPD component ablation G rank/energy matched random H shuffled-importance BPD control

Run:  python -m research_gym.trm_skill_gym_v3.bpd_skill --out DIR --skills repair curation research --seeds 0 1 --steps S
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from research_gym.trm_skill_gym import modules as M, tasks as T
from research_gym.trm_skill_gym_v2 import subspace as SS
from research_gym.trm_skill_gym_v2.runner import evaluate, pools

REPO_ROOT = Path(__file__).resolve().parents[2]
BPD_COMMIT_EXPECTED = "ab93e4da986beef87897f8d1244c7fe7564b5bde"
FIT, CAL, TEST = 2_300_000, 2_400_000, 2_500_000
ROOT_RANGES = {"fit": [FIT, 2_400_000], "calibration": [CAL, 2_500_000], "test": [TEST, 2_600_000]}
DEFAULT_CKPT_DIR = Path("experiments/trm_skill_gym_01/run/checkpoints")
SKILLS_DEFAULT = ("repair", "curation", "research")


# ----------------------------------------------------------------------------- BPD import
def bpd_root() -> Path:
    env = os.environ.get("BPD_ROOT")
    return Path(env) if env else REPO_ROOT.parent / "BPD"


def _resolve_git_dir(root: Path) -> Optional[Path]:
    g = root / ".git"
    if g.is_dir():
        return g
    if g.is_file():
        text = g.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            p = Path(text.split(":", 1)[1].strip())
            return p if p.is_absolute() else (root / p).resolve()
    return None


def bpd_commit(root: Optional[Path] = None) -> Optional[str]:
    """Commit sha of the BPD checkout, read from .git without invoking git."""
    root = root or bpd_root()
    gd = _resolve_git_dir(root)
    if gd is None:
        return None
    head = (gd / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref:"):
        return head
    ref = head.split(":", 1)[1].strip()
    for base in (gd, gd.parent if gd.name == "worktrees" else gd):
        rp = base / ref
        if rp.is_file():
            return rp.read_text(encoding="utf-8").strip()
    packed = gd / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0]
    common = gd / "commondir"
    if common.is_file():
        cd = (gd / common.read_text(encoding="utf-8").strip()).resolve()
        rp = cd / ref
        if rp.is_file():
            return rp.read_text(encoding="utf-8").strip()
    return None


BPD = None
BPD_ERROR: Optional[str] = None


def _import_bpd():
    global BPD, BPD_ERROR
    if BPD is not None:
        return BPD
    root = bpd_root()
    if not (root / "mechinterp" / "bpd").is_dir():
        BPD_ERROR = f"BPD checkout not found at {root} (set BPD_ROOT)"
        return None
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import mechinterp.bpd as bpd_pkg
        from mechinterp.bpd import analysis, ci, components, config, losses, trainer, adversary
    except Exception as exc:  # pragma: no cover - environment dependent
        BPD_ERROR = f"BPD import failed: {type(exc).__name__}: {exc}"
        return None

    class NS:
        pass

    ns = NS()
    ns.pkg, ns.analysis, ns.ci, ns.components = bpd_pkg, analysis, ci, components
    ns.config, ns.losses, ns.trainer, ns.adversary = config, losses, trainer, adversary
    BPD = ns
    return BPD


_import_bpd()


def require_bpd():
    if BPD is None:
        raise RuntimeError(BPD_ERROR or "BPD unavailable")
    return BPD


# ----------------------------------------------------------------------------- hashing / io
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_tree_digest(tree) -> str:
    h = hashlib.sha256()

    def walk(x, prefix):
        if isinstance(x, dict):
            for k in sorted(x, key=str):
                walk(x[k], f"{prefix}/{k}")
        elif isinstance(x, torch.Tensor):
            h.update(prefix.encode())
            h.update(str(tuple(x.shape)).encode())
            h.update(x.detach().cpu().contiguous().to(torch.float32).numpy().tobytes())

    walk(tree, "")
    return h.hexdigest()


def write_json(path: Path, obj) -> str:
    """LF-only canonical JSON; returns sha256 of the bytes written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(data)
    return sha256_bytes(data)


def acc_digest(a: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(a, dtype=np.float32).tobytes())


# ----------------------------------------------------------------------------- skill model with BPD hooks
class BPDSkill(SS.Probe):
    """SkillModule + latent projector probe (subspace.Probe) + trace recording for BPD rollouts."""

    rec = None
    tau_ref = None

    def visit(self, cand, batch, st, planned_visits):
        rec = self.rec
        if rec is not None:
            rec["cands"].append(cand.detach().clone())
        out = super().visit(cand, batch, st, planned_visits)
        if rec is not None and self.tau_ref is not None:
            rec["visit_end"].append(self.tau_ref.calls - 1)
        return out

    def _latent(self, x, y, z, rel, mask):
        z2 = super()._latent(x, y, z, rel, mask)
        if self.rec is not None and self.tau_ref is not None:
            self.rec["latents"].append((self.tau_ref.calls - 1, z2))
        return z2


def load_model(ck: Dict) -> BPDSkill:
    m = BPDSkill(ck["config"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def recurrent_linears(model: M.SkillModule) -> Dict[str, torch.nn.Module]:
    """All distinct ``Lin`` weights inside net_z / net_y (shared partition aliases are listed once)."""
    out, seen = {}, set()
    roots = [("net_z", model.net_z)]
    if getattr(model, "net_y", None) is not None and model.net_y is not model.net_z:
        roots.append(("net_y", model.net_y))
    for prefix, root in roots:
        for name, mod in root.named_modules():
            if isinstance(mod, torch.nn.Linear) and id(mod.weight) not in seen:
                seen.add(id(mod.weight))
                out[f"{prefix}.{name}"] = mod
    return out


def wrap_model(model: BPDSkill, c_mult: float, svd: bool, ablation):
    """Install TiedComponentLinear on every recurrent matrix. ``svd=True``: full-rank exact-SVD factors."""
    b = require_bpd()
    if model.cfg["core"] != "trm":
        raise NotImplementedError("BPD adapter supports the TRM core only")
    if model.net_y is not model.net_z:
        raise NotImplementedError("non-shared state_partition gives net_z/net_y different pass clocks; only 'shared' is supported")
    lins = recurrent_linears(model)
    c_per = {}
    for name, lin in lins.items():
        rank = min(lin.weight.shape)
        c_per[name] = rank if svd else max(rank, int(math.ceil(c_mult * rank)))
    bm = b.components.BPDModel(model, c_per, max_passes=64, ablation=ablation, init_svd=True)
    object.__setattr__(model, "tau_ref", bm.wrappers[bm.module_order[0]])  # not a registered submodule
    return bm


# ----------------------------------------------------------------------------- batches and adapter
def cat_pool(pool: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = next(iter(pool.values())).keys()
    return {k: torch.cat([b[k] for b in pool.values()], 0) for k in keys}


def take(batch: Dict[str, torch.Tensor], idx) -> Dict[str, torch.Tensor]:
    idx = torch.as_tensor(np.asarray(idx), dtype=torch.long)
    return {k: v[idx] for k, v in batch.items()}


class SkillBatch:
    """A held batch with the clean episode's per-visit observations cached (teacher forcing)."""

    def __init__(self, batch, visits: Optional[int] = None):
        self.batch, self.visits, self.cands = batch, visits, None


def readout_tensor(logits: torch.Tensor, mask: torch.Tensor, kind: str) -> torch.Tensor:
    valid = mask > 0
    if kind == "single":
        return logits.masked_fill(~valid, -1e4)  # softmax over candidates -> categorical KL
    l = logits[valid]
    return torch.stack([torch.zeros_like(l), l], -1)  # two-class logits -> Bernoulli KL per candidate


class SkillAdapter:
    """BPDTrainer rollout adapter: (batch, n_passes, pass_offset, init_state) -> (readouts, boundaries, vis, prompt).

    The first execution of a batch is the real closed-loop episode (verifier feedback written by the host)
    and caches the observation fed to every visit; later executions (masked / intervened) replay those
    observations so that the clean and perturbed trajectories differ only through the weights.
    Readout pass = invocation index of the answer update closing a visit; boundary passes = the invocation of
    every latent update ``z <- net_z(x+y+z)`` (read-free inner steps)."""

    def __init__(self, model: BPDSkill, skill: str):
        self.model, self.skill, self.cfg = model, skill, model.cfg

    def __call__(self, sb: SkillBatch, n_passes: int, pass_offset: int = 0, init_state=None):
        if pass_offset or init_state is not None:
            raise NotImplementedError("closed-loop rollouts only")
        m, batch = self.model, sb.batch
        R = sb.visits or self.cfg["visits"]
        m.rec = {"cands": [], "latents": [], "visit_end": []}
        try:
            if sb.cands is None:
                out = M.episode(m, self.skill, batch, self.cfg, visits=R)
                sb.cands = list(m.rec["cands"])
                logits = out["logits"]
            else:
                st, logits = m.init_state(batch), []
                cert = T.certify(self.skill, batch) if self.cfg.get("ldt_projection") else None
                for r in range(R):
                    lg, _, _, st = m.visit(sb.cands[r], batch, st, R)
                    logits.append(M.project(lg, cert) if cert is not None else lg)
            rec = m.rec
        finally:
            m.rec = None
        calls = m.tau_ref.calls
        if calls != n_passes:
            raise RuntimeError(f"rollout made {calls} invocations, expected n_passes={n_passes}")
        mask, kind = batch["mask"], T.LABEL_KIND[self.skill]
        readouts = {tau: readout_tensor(lg, mask, kind) for tau, lg in zip(rec["visit_end"], logits)}
        valid = mask > 0
        boundaries = {tau: z[valid] for tau, z in rec["latents"] if tau not in readouts}
        return readouts, boundaries, None, None


# ----------------------------------------------------------------------------- hyper-parameters
@dataclass
class HP:
    steps: int = 600
    batch: int = 32
    lr: float = 2e-3
    c_mult: float = 1.5
    warmup: int = 100
    coeff_faith: float = 1e4
    coeff_stoch: float = 0.5
    coeff_adv: float = 0.5
    adv_inner_steps: int = 2
    ci_d_model: int = 64
    fit_count: int = 96
    cal_count: int = 96
    test_count: int = 128
    n_random: int = 5
    n_draw: int = 8
    classify_batch: int = 48
    classify_comps: int = 24
    classify_rotations: int = 2
    active_frac: float = 0.5
    depths: tuple = (1, 2, 4)
    base_seed: int = 7000
    log_every: int = 25

    @classmethod
    def smoke(cls) -> "HP":
        return cls(steps=4, batch=16, warmup=5, fit_count=10, cal_count=10, test_count=10, n_random=2, n_draw=2,
                   classify_batch=12, classify_comps=2, classify_rotations=1, log_every=2)


class JobIncomplete(RuntimeError):
    pass


# ----------------------------------------------------------------------------- energy meter
class WriteMeter:
    """Removed write energy: sum over valid candidates of |W x + b (intervened) - (clean)|^2, per instance."""

    def __init__(self, bm):
        self.total, self.mask, self.handles = 0.0, None, []
        for w in bm.wrappers.values():
            self.handles.append(w.register_forward_hook(self._hook))

    def _hook(self, mod, inputs, out):
        if mod.intervention is None or self.mask is None:
            return
        x = inputs[0]
        clean = F.linear(x, mod.W_target.to(x.dtype), mod.bias.to(x.dtype) if mod.bias is not None else None)
        d = (out - clean) * self.mask[..., None]
        self.total += float(d.square().sum())

    def close(self):
        for h in self.handles:
            h.remove()


# ----------------------------------------------------------------------------- the rig
def flat(accs: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([accs[k] for k in accs])


def macro(accs) -> float:
    return float(np.mean([a.mean() for a in accs.values()]))


def spec_pairs(spec: Dict[str, Sequence[int]]) -> List[tuple]:
    return [(m, int(c)) for m in sorted(spec) for c in sorted(spec[m])]


def pairs_spec(pairs) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for m, c in pairs:
        out.setdefault(m, []).append(int(c))
    return {m: sorted(v) for m, v in out.items()}


class Rig:
    """One skill checkpoint with plain / BPD-wrapped / exact-SVD-wrapped copies sharing the same weights."""

    def __init__(self, skill: str, ck: Dict, hp: HP):
        b = require_bpd()
        self.skill, self.ck, self.hp, self.cfg = skill, ck, hp, ck["config"]
        self.R = self.cfg["visits"]
        self.abl = b.config.AblationConfig(mode="pass_mean", sphere_rescale=False, ema_momentum=0.1)
        self.plain = load_model(ck)
        self.bmodel = load_model(ck)
        self.bm = wrap_model(self.bmodel, hp.c_mult, False, self.abl)
        self.smodel = load_model(ck)
        self.sm = wrap_model(self.smodel, hp.c_mult, True, self.abl)
        self.adapter = SkillAdapter(self.bmodel, skill)
        self.means = {"bpd": {}, "svd": {}}
        self.meters = {"bpd": WriteMeter(self.bm), "svd": WriteMeter(self.sm)}

    # -- plumbing
    def parts(self, kind):
        return (self.bmodel, self.bm) if kind == "bpd" else (self.smodel, self.sm)

    def calls_per_visit(self) -> int:
        batch = T.stack([T.make_instance(self.skill, T.FAMILIES[self.skill]["dev"][0], FIT + i) for i in range(2)])
        with torch.no_grad(), self.bm.target_capture():
            M.episode(self.bmodel, self.skill, batch, self.cfg)
        counts = {n: w.calls for n, w in self.bm.wrappers.items()}
        if len(set(counts.values())) != 1:
            raise NotImplementedError(f"recurrent matrices have different pass clocks {counts}")
        n = next(iter(counts.values()))
        assert n % self.R == 0
        return n // self.R

    # -- pass-conditional means over valid candidates only
    @torch.no_grad()
    def calibrate(self, kind: str, pool, depths) -> None:
        model, bm = self.parts(kind)
        for v in depths:
            rows = {n: {} for n in bm.module_order}
            for b in pool.values():
                with bm.target_capture():
                    M.episode(model, self.skill, b, self.cfg, visits=v)
                    inputs = bm.collect_recorded_inputs()
                valid = b["mask"] > 0
                for n, x in inputs.items():
                    for tau in range(x.shape[1]):
                        rows[n].setdefault(tau, []).append(x[:, tau][valid])
            store = {}
            for n, w in bm.wrappers.items():
                n_tau = len(rows[n])
                w.calibrated_passes.zero_()
                w._ensure_pass_capacity(n_tau - 1)
                for tau in range(n_tau):
                    w.observe_clean_pass(torch.cat(rows[n][tau], 0), tau)
                store[n] = {"comp": w.mean_comp_acts[:n_tau].clone(), "delta": w.mean_delta_out[:n_tau].clone(),
                            "rms": w.mean_write_rms[:n_tau].clone()}
            self.means[kind][v] = store

    def use_means(self, kind: str, v: int) -> None:
        _, bm = self.parts(kind)
        store = self.means[kind][v]
        for n, w in bm.wrappers.items():
            d = store[n]
            n_tau = d["comp"].shape[0]
            w._ensure_pass_capacity(n_tau - 1)
            w.calibrated_passes.zero_()
            w.mean_comp_acts[:n_tau] = d["comp"]
            w.mean_delta_out[:n_tau] = d["delta"]
            w.mean_write_rms[:n_tau] = d["rms"]
            w.calibrated_passes[:n_tau] = True

    # -- evaluation
    def eval_latent(self, pool, basis: Optional[torch.Tensor], visits: Optional[int] = None):
        m = self.plain
        m.basis, m.energy = basis, []
        n = 0
        accs = {}
        for fam, b in pool.items():
            a, _ = evaluate(self.skill, m, self.cfg, b, visits)
            accs[fam] = a
            n += len(a)
        energy = float(sum(m.energy)) / n if basis is not None else 0.0
        m.basis, m.energy = None, None
        return accs, energy

    def eval_weights(self, kind: str, pool, spec=None, passes=None, visits: Optional[int] = None):
        model, bm = self.parts(kind)
        v = visits or self.R
        self.use_means(kind, v)
        meter = self.meters[kind]
        spec = {m: list(c) for m, c in (spec or {}).items() if len(c)}
        bm.clear_schedules()
        comp = require_bpd().components
        for name, comps in spec.items():
            bm.wrappers[name].intervention = comp.ComponentIntervention(
                components=tuple(int(c) for c in comps), passes=None if passes is None else frozenset(int(p) for p in passes),
                mode="ablate", scale=0.0)
        meter.total, n, accs = 0.0, 0, {}
        try:
            for fam, b in pool.items():
                bm.reset_wrappers()
                meter.mask = b["mask"] > 0
                a, _ = evaluate(self.skill, model, self.cfg, b, v)
                accs[fam] = a
                n += len(a)
        finally:
            meter.mask = None
            for w in bm.wrappers.values():
                w.intervention = None
        return accs, meter.total / n

    def eval_method(self, method: str, pool, spec=None, passes=None, visits=None, basis=None):
        if method == "latent":
            return self.eval_latent(pool, basis, visits)
        return self.eval_weights(method, pool, spec, passes, visits)


# ----------------------------------------------------------------------------- training
def make_trainer(rig: Rig, hp: HP, seed: int, n_passes: int):
    b = require_bpd()
    cfgm = b.config
    cfg = cfgm.BPDConfig(
        c_per_module=dict(rig.bm.c_per_module), n_passes=n_passes, n_steps=hp.steps, batch_size=hp.batch,
        seq_len=T.N, seed=seed, init_svd=True, faithfulness_warmup_steps=hp.warmup,
        faithfulness_warmup_lr=1e-3, main_lr=hp.lr, main_lr_final_frac=0.1, grad_clip_components=1.0,
        coeff_faith=hp.coeff_faith, coeff_stoch=hp.coeff_stoch, coeff_adv=hp.coeff_adv, state_boundary_weight=1.0,
        readout_policy="readout_or_boundary", rollout_mode="closed_loop",
        ci=cfgm.CIConfig(variant="state_only", d_model=hp.ci_d_model, n_blocks=1, n_heads=4,
                         mlp_hidden=2 * hp.ci_d_model, max_passes=64, max_seq_len=T.N),
        ablation=rig.abl,
        minimality=cfgm.MinimalityConfig(pool_mode="max", coeff_level=2e-3, coeff_switch=2e-3),
        adversary=cfgm.AdversaryConfig(scope="shared_batch_and_pass", lr=0.05, inner_steps=hp.adv_inner_steps),
        null_calibration=cfgm.NullCalibrationConfig(use_tolerance_weights=False),
    )

    class SkillTrainer(b.trainer.BPDTrainer):
        def run_clean_rollout(self, batch, *, n_passes=None, pass_offset=0, init_state=None, update_means=False):
            traj = super().run_clean_rollout(batch, n_passes=n_passes, pass_offset=pass_offset,
                                             init_state=init_state, update_means=False)
            if update_means:  # BPD's own EMA update, restricted to valid (non-padded) candidates
                valid = batch.batch["mask"] > 0
                with torch.no_grad():
                    for name, x in traj.module_inputs.items():
                        w = self.bpd_model.wrappers[name]
                        for tau in range(x.shape[1]):
                            w.observe_clean_pass(x[:, tau][valid], tau)
            return traj

    return SkillTrainer(rig.bm, cfg, rig.adapter), cfg


def train_bpd(rig: Rig, fit_pool, hp: HP, seed: int, n_passes: int, deadline: Optional[float]):
    torch.manual_seed(hp.base_seed + seed)
    trainer, cfg = make_trainer(rig, hp, seed, n_passes)
    faith0 = trainer.run_faithfulness_warmup(hp.warmup)
    flat_fit = cat_pool(fit_pool)
    g = np.random.default_rng(hp.base_seed + seed)
    log = []
    for step in range(hp.steps):
        if deadline is not None and time.monotonic() > deadline:
            raise JobIncomplete(f"deadline during BPD training at step {step}")
        idx = g.choice(len(flat_fit["mask"]), min(hp.batch, len(flat_fit["mask"])), replace=False)
        sm = trainer.train_step(SkillBatch(take(flat_fit, idx)), step)
        if step % hp.log_every == 0 or step == hp.steps - 1:
            log.append({k: float(v) if not isinstance(v, str) else v for k, v in asdict(sm).items()})
    return trainer, cfg, {"faith_after_warmup": faith0, "log": log}


@torch.no_grad()
def importance_reports(rig: Rig, trainer, pool, hp: HP, n_per_visit: int) -> Dict[str, Dict]:
    """Pass-specific CI schedules at the trained and extended depths (CI is pass-tied, so it extrapolates)."""
    b = require_bpd()
    flat_pool = cat_pool(pool)
    sample = take(flat_pool, np.arange(min(64, len(flat_pool["mask"]))))
    valid = (sample["mask"] > 0)[:, None, :, None].float()
    out = {}
    for mult in hp.depths:
        v = rig.R * mult
        rig.use_means("bpd", v)
        sb = SkillBatch(sample, visits=v)
        traj = trainer.run_clean_rollout(sb, n_passes=v * n_per_visit)
        ci = trainer.ci_net.forward_trajectory(traj)
        adj = {}
        for name, g in ci.lower_leaky.items():
            vm = (g * valid).sum(2, keepdim=True) / valid.sum(2, keepdim=True).clamp_min(1.0)
            adj[name] = torch.where(valid > 0, g, vm.expand_as(g))
        rep = b.analysis.extract_schedule(b.ci.CIOutputs(adj, adj, adj), position_pool="mean")
        out[str(mult)] = {
            "visits": v,
            "keys": [[k.module, k.component] for k in rep.keys],
            "mean_schedule": rep.mean_schedule.tolist(),
            "duty_cycle": rep.duty_cycle.tolist(),
            "center_of_mass_pass": rep.center_of_mass_pass.tolist(),
            "temporal_entropy": rep.temporal_entropy.tolist(),
            "switch_rate": rep.switch_rate.tolist(),
            "is_stationary": rep.is_stationary.tolist(),
        }
    rig.use_means("bpd", rig.R)
    return out


# ----------------------------------------------------------------------------- selection helpers
def ladder(k_max: int, n: int) -> List[int]:
    return sorted({k for k in (1, 2, 4, 8, 16, k_max) if 1 <= k <= min(k_max, n)})


def rank_by_gain(rig: Rig, kind: str, cal, cands, base_macro):
    gains = []
    for (m, c) in cands:
        accs, _ = rig.eval_weights(kind, cal, {m: [c]})
        gains.append(macro(accs) - base_macro)
    order = sorted(range(len(cands)), key=lambda i: (-gains[i], i))
    return [cands[i] for i in order], [gains[i] for i in order], gains


def select_prefix(rig: Rig, kind: str, cal, ranked, k_max):
    """Largest-gain-first prefix whose joint ablation maximizes calibration accuracy (ties: smaller)."""
    best, best_k, trace = -1.0, 1, {}
    for k in ladder(k_max, len(ranked)):
        accs, e = rig.eval_weights(kind, cal, pairs_spec(ranked[:k]))
        s = macro(accs)
        trace[str(k)] = {"macro": s, "energy": e}
        if s > best + 1e-12:
            best, best_k = s, k
    return ranked[:best_k], trace


def summarize(accs, base_accs, energy: float, kind: str) -> Dict:
    base_flat = flat(base_accs)
    out = {
        "exact": {f: float(a.mean()) for f, a in accs.items()},
        "macro": macro(accs),
        "pooled": float(flat(accs).mean()),
        "paired_vs_A": {"pooled": SS.paired_delta(base_flat, flat(accs))},
        "removed_energy_per_instance": float(energy),
        "energy_kind": kind,
        "acc_sha256": acc_digest(flat(accs)),
    }
    for f in accs:
        out["paired_vs_A"][f] = SS.paired_delta(base_accs[f], accs[f])
    return out


# ----------------------------------------------------------------------------- the job
def run_job(skill: str, arm: str, seed: int, out_dir: Path, hp: HP, ckpt_dir: Path,
            deadline: Optional[float] = None) -> Dict:
    t_start = time.time()
    require_bpd()
    ck_path = ckpt_dir / f"{skill}__{arm}__s{seed}.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=True)
    rig = Rig(skill, ck, hp)
    R, H = rig.R, rig.plain.H
    n_per_visit = rig.calls_per_visit()
    n_passes = R * n_per_visit
    fit = pools(skill, FIT, hp.fit_count)
    cal = pools(skill, CAL, hp.cal_count)
    test = pools(skill, TEST, hp.test_count, True)
    depths = [R * m for m in hp.depths]

    # 1. train BPD
    trainer, bcfg, train_info = train_bpd(rig, fit, hp, seed, n_passes, deadline)
    t_train = time.time() - t_start
    rig.calibrate("bpd", fit, depths)
    rig.calibrate("svd", fit, depths)
    rig.use_means("bpd", R)
    cal_flat = cat_pool(cal)
    ev_batch = take(cal_flat, np.arange(min(64, len(cal_flat["mask"]))))
    recon = {}
    for dv in (0.0, 1.0):
        rig.use_means("bpd", R)
        met, _, _ = trainer.evaluate(SkillBatch(ev_batch), n_passes=n_passes, delta_value=dv)
        recon[f"delta_mask_{dv}"] = {k: float(v) for k, v in met.items()}
    schedules = importance_reports(rig, trainer, fit, hp, n_per_visit)
    faithfulness = {n: float(w.weight_delta().detach().pow(2).mean()) for n, w in rig.bm.wrappers.items()}

    # 2. exactness of wrapped target mode (reported, also unit-tested)
    exact_check = {}
    chk = take(cat_pool(cal), np.arange(min(32, len(cal_flat["mask"]))))
    with torch.no_grad():
        o0 = M.episode(rig.plain, skill, chk, rig.cfg)
        for kind in ("bpd", "svd"):
            model, bm = rig.parts(kind)
            bm.clear_schedules()
            bm.reset_wrappers()
            o1 = M.episode(model, skill, chk, rig.cfg)
            exact_check[kind] = max(float((a - c).abs().max()) for a, c in zip(o0["logits"], o1["logits"]))

    def check_deadline(stage):
        if deadline is not None and time.monotonic() > deadline:
            raise JobIncomplete(f"deadline during {stage}")

    # 3. calibration-side selection (never touches test)
    rank = max(1, H // 4)
    bases = {}
    base_cal, _ = rig.eval_latent(cal, None)
    base_cal_macro = macro(base_cal)
    coord_ids, coord_scores = SS.select_directions(rig.plain, skill, rig.cfg, cal, torch.eye(H), rank)
    basis, _pca, fit_info = SS.fit_basis(rig.plain, skill, rig.cfg, fit)
    d_ids, d_scores = SS.select_directions(rig.plain, skill, rig.cfg, cal, basis, rank)
    bases["B"] = torch.eye(H)
    bases["C"] = torch.eye(H)[:, coord_ids]
    bases["D"] = basis[:, d_ids].contiguous()
    check_deadline("latent selection")

    svd_total = sum(w.C for w in rig.sm.wrappers.values())
    k_max = max(1, svd_total // 4)
    base_w = macro(rig.eval_weights("svd", cal)[0])
    assert abs(base_w - base_cal_macro) < 1e-9, "wrapped target mode differs from plain model on calibration"
    svd_cands = [(n, c) for n in rig.sm.module_order for c in range(rig.sm.wrappers[n].C)]
    svd_rank, svd_gain_sorted, svd_gain = rank_by_gain(rig, "svd", cal, svd_cands, base_w)
    E_sel, E_trace = select_prefix(rig, "svd", cal, svd_rank, k_max)
    check_deadline("SVD selection")

    sched_R = schedules["1"]
    keys = [tuple(k) for k in sched_R["keys"]]
    score = np.max(np.asarray(sched_R["mean_schedule"]), axis=0)          # peak pass-mean importance per component
    n_active = max(1, int(round(hp.active_frac * len(keys))))
    bpd_cands_all = keys

    def active_set(s):
        order = sorted(range(len(keys)), key=lambda i: (-s[i], i))
        return [keys[i] for i in order[:n_active]]

    active_F = active_set(score)
    perm = np.random.default_rng(hp.base_seed + 31 * seed + 5).permutation(len(keys))
    active_H = active_set(score[perm])                                    # importance permuted over components
    F_rank, F_gain_sorted, _ = rank_by_gain(rig, "bpd", cal, active_F, base_w)
    F_sel, F_trace = select_prefix(rig, "bpd", cal, F_rank, k_max)
    check_deadline("BPD selection")
    H_rank, H_gain_sorted, _ = rank_by_gain(rig, "bpd", cal, active_H, base_w)
    H_sel = H_rank[:len(F_sel)]
    counts = {m: len(c) for m, c in pairs_spec(F_sel).items()}
    _, F_cal_energy = rig.eval_weights("bpd", cal, pairs_spec(F_sel))
    rng = np.random.default_rng(hp.base_seed + 97 * seed + 13)
    rand_sets = []
    for s in range(hp.n_random):
        draws = []
        for _ in range(hp.n_draw):
            d = {m: sorted(rng.choice(rig.bm.wrappers[m].C, n, replace=False).tolist()) for m, n in counts.items()}
            _, e = rig.eval_weights("bpd", cal, d)
            draws.append((abs(e - F_cal_energy), d, e))
        rank_only = draws[0][1]
        best = min(draws, key=lambda t: t[0])
        rand_sets.append({"rank": rank_only, "energy": best[1], "cal_energy": best[2]})
    check_deadline("random control selection")

    # 4. held-out evaluation, one function so every row is replayable
    evals: List[Dict] = []
    stats: Dict[str, Dict] = {}
    base_acc: Dict[int, Dict] = {}

    def run_eval(name, method, visits_mult=1, spec=None, passes=None, basis_key=None, family="", kind=None):
        v = R * visits_mult
        bs = bases.get(basis_key) if basis_key else None
        accs, energy = rig.eval_method(method, test, spec, passes, v, bs)
        if method == "latent" and basis_key is None:
            base_acc[visits_mult] = accs
        stats[name] = {"accs": accs, "energy": energy}
        evals.append({"name": name, "method": method, "visits": v, "spec": spec, "passes": passes,
                      "basis_key": basis_key, "family": family,
                      "energy_kind": kind or ("latent_squared_norm" if method == "latent" else "pre_norm_write_squared_norm"),
                      "exact": {f: float(a.mean()) for f, a in accs.items()},
                      "acc_sha256": acc_digest(flat(accs)), "energy": float(energy)})
        return accs, energy

    run_eval("A", "latent")
    run_eval("B", "latent", basis_key="B", family="B_full_latent_reset")
    run_eval("C", "latent", basis_key="C", family="C_selected_coordinate_reset")
    run_eval("D", "latent", basis_key="D", family="D_behavioral_orthogonal_projector")
    run_eval("E", "svd", spec=pairs_spec(E_sel), family="E_svd_components")
    run_eval("F", "bpd", spec=pairs_spec(F_sel), family="F_learned_bpd_components")
    run_eval("H", "bpd", spec=pairs_spec(H_sel), family="H_shuffled_importance_bpd")
    for i, rs in enumerate(rand_sets):
        run_eval(f"G_rank_{i}", "bpd", spec=rs["rank"], family="G_random_rank_matched")
        run_eval(f"G_energy_{i}", "bpd", spec=rs["energy"], family="G_random_rank_energy_matched")
    check_deadline("test evaluation")

    # 5. per-pass vs whole-trajectory
    F_spec = pairs_spec(F_sel)
    for tau in range(n_passes):
        run_eval(f"F_pass{tau}", "bpd", spec=F_spec, passes=[tau], family="F_single_pass")
    check_deadline("per-pass evaluation")

    # transient / sustaining classification on a calibration batch (BPD analysis, teacher-forced rollouts)
    cls_batch = SkillBatch(take(cal_flat, np.arange(min(hp.classify_batch, len(cal_flat["mask"])))))
    b = require_bpd()
    rig.use_means("bpd", R)
    comps_by_module = {m: [] for m in rig.bm.module_order}
    comps_by_module.update(pairs_spec(F_sel[:hp.classify_comps]))
    with torch.no_grad():
        rig.bm.clear_schedules()
        with rig.bm.target_capture():
            ro, bo, _, _ = rig.adapter(cls_batch, n_passes, 0, None)
            mod_in = rig.bm.collect_recorded_inputs(expected_passes=n_passes)
        clean_traj = b.components.RolloutTrajectory(ro, bo, mod_in)

        def rollout():
            r_, b_, _, _ = rig.adapter(cls_batch, n_passes, 0, None)
            return b.components.RolloutTrajectory(r_, b_, {})

        roles = b.analysis.classify_transient_sustaining(
            rig.bm, clean_traj, rollout, components_by_module=comps_by_module,
            n_rotations=hp.classify_rotations, seed=seed)
    classification = [asdict(e) for e in roles]

    # 6. depth generalisation (components selected at the trained visit count, calibration means per depth)
    for mult in hp.depths:
        if mult == 1:
            continue
        run_eval(f"A@{mult}x", "latent", visits_mult=mult)
        run_eval(f"B@{mult}x", "latent", visits_mult=mult, basis_key="B", family="B_full_latent_reset")
        run_eval(f"C@{mult}x", "latent", visits_mult=mult, basis_key="C", family="C_selected_coordinate_reset")
        run_eval(f"D@{mult}x", "latent", visits_mult=mult, basis_key="D", family="D_behavioral_orthogonal_projector")
        run_eval(f"E@{mult}x", "svd", visits_mult=mult, spec=pairs_spec(E_sel), family="E_svd_components")
        run_eval(f"F@{mult}x", "bpd", visits_mult=mult, spec=F_spec, family="F_learned_bpd_components")
        run_eval(f"H@{mult}x", "bpd", visits_mult=mult, spec=pairs_spec(H_sel), family="H_shuffled_importance_bpd")
        for i, rs in enumerate(rand_sets):
            run_eval(f"G_energy_{i}@{mult}x", "bpd", visits_mult=mult, spec=rs["energy"], family="G_random_rank_energy_matched")
        check_deadline(f"depth {mult}x")

    # 7. summaries with paired deltas against A at the same depth
    summaries = {}
    for e in evals:
        name = e["name"]
        mult = int(name.rsplit("@", 1)[1][:-1]) if "@" in name else 1
        summaries[name] = summarize(stats[name]["accs"], base_acc[mult], stats[name]["energy"], e["energy_kind"])
    F_all = summaries["F"]["paired_vs_A"]["pooled"]["delta"]
    per_pass = {
        "all_pass_delta": F_all,
        "single_pass_deltas": [summaries[f"F_pass{t}"]["paired_vs_A"]["pooled"]["delta"] for t in range(n_passes)],
        "single_pass_energy": [summaries[f"F_pass{t}"]["removed_energy_per_instance"] for t in range(n_passes)],
        "all_pass_energy": summaries["F"]["removed_energy_per_instance"],
    }
    role_counts: Dict[str, int] = {}
    for c in classification:
        role_counts[c["role"]] = role_counts.get(c["role"], 0) + 1

    def mean_of(prefix, mult=1):
        suffix = "" if mult == 1 else f"@{mult}x"
        names = [n for n in summaries if n.startswith(prefix) and n.endswith(suffix) and ("@" in n) == (mult != 1)]
        return {"n": len(names), "macro": float(np.mean([summaries[n]["macro"] for n in names])) if names else None,
                "delta": float(np.mean([summaries[n]["paired_vs_A"]["pooled"]["delta"] for n in names])) if names else None,
                "energy": float(np.mean([summaries[n]["removed_energy_per_instance"] for n in names])) if names else None}

    depth_table = {}
    for mult in hp.depths:
        sfx = "" if mult == 1 else f"@{mult}x"
        row = {}
        for m in ("A", "B", "C", "D", "E", "F", "H"):
            row[m] = {"macro": summaries[m + sfx]["macro"],
                      "delta_vs_A": summaries[m + sfx]["paired_vs_A"]["pooled"]["delta"],
                      "ci95": summaries[m + sfx]["paired_vs_A"]["pooled"]["root_bootstrap_95"]}
        row["G_energy_mean"] = mean_of("G_energy_", mult)
        depth_table[f"{mult}x"] = row

    # 8. exports
    job_key = f"{skill}__{arm}__s{seed}"
    jd = out_dir / "jobs" / job_key
    jd.mkdir(parents=True, exist_ok=True)
    factors = {
        "c_per_module": dict(rig.bm.c_per_module),
        "bpd": {n: {"V": w.V.detach().clone(), "U": w.U.detach().clone()} for n, w in rig.bm.wrappers.items()},
        "means": {"bpd": rig.means["bpd"], "svd": rig.means["svd"]},
        "latent_bases": {k: v.clone() for k, v in bases.items()},
        "ci_state_dict": {k: v.clone() for k, v in trainer.ci_net.state_dict().items()},
    }
    torch.save(factors, jd / "factors.pt")
    imp_sha = write_json(jd / "importance.json", {"schedules": schedules, "active_F": [list(k) for k in active_F],
                                                  "active_H": [list(k) for k in active_H]})
    selection = {
        "rank_latent": rank, "k_max_components": k_max,
        "C_coordinates": coord_ids, "C_cal_effects": coord_scores,
        "D_directions": d_ids, "D_cal_effects": d_scores, "D_fit": fit_info,
        "E_components": [list(p) for p in E_sel], "E_prefix_cal": E_trace,
        "F_components": [list(p) for p in F_sel], "F_prefix_cal": F_trace,
        "F_top_gains": F_gain_sorted[:10], "E_top_gains": svd_gain_sorted[:10],
        "H_components": [list(p) for p in H_sel], "H_top_gains": H_gain_sorted[:10],
        "G_sets": rand_sets, "cal_energy_F": F_cal_energy, "n_active": n_active,
        "calibration_macro_A": base_cal_macro,
    }
    result = {
        "job": job_key, "skill": skill, "arm": arm, "seed": seed, "complete": True,
        "visits": R, "calls_per_visit": n_per_visit, "n_passes": n_passes, "components": dict(rig.bm.c_per_module),
        "svd_components": dict(rig.sm.c_per_module), "train": train_info, "recon_eval": recon,
        "faithfulness_mse_per_matrix": faithfulness, "target_mode_max_logit_error": exact_check,
        "selection": selection, "methods": summaries, "per_pass": per_pass,
        "classification": classification, "role_counts": role_counts, "depth": depth_table,
        "adversary": {"used": True, "scope": "shared_batch_and_pass", "inner_steps": hp.adv_inner_steps},
        "seconds": {"train": t_train, "total": time.time() - t_start},
    }
    res_sha = write_json(jd / "result.json", result)
    sample_specs = [{k: e[k] for k in ("name", "method", "visits", "spec", "passes", "basis_key", "energy_kind",
                                       "exact", "acc_sha256", "energy")} for e in evals]
    receipt = {
        "kind": "trm_skill_gym_v3_bpd_receipt", "version": 1,
        "inputs": {
            "skill": skill, "arm": arm, "seed": seed, "checkpoint": str(ck_path).replace("\\", "/"),
            "checkpoint_sha256": sha256_file(ck_path), "bpd_commit": bpd_commit(), "bpd_commit_expected": BPD_COMMIT_EXPECTED,
            "bpd_root": str(bpd_root()).replace("\\", "/"), "roots": ROOT_RANGES,
            "counts": {"fit": hp.fit_count, "calibration": hp.cal_count, "test": hp.test_count},
            "hyperparameters": asdict(hp), "model_config": rig.cfg,
        },
        "outputs": {"result_sha256": res_sha, "importance_sha256": imp_sha,
                    "factors_file_sha256": sha256_file(jd / "factors.pt"),
                    "factors_tensor_sha256": tensor_tree_digest({"bpd": factors["bpd"], "means": factors["means"],
                                                                 "bases": factors["latent_bases"]})},
        "replay_samples": sample_specs,
    }
    write_json(jd / "receipt.json", receipt)
    return result


# ----------------------------------------------------------------------------- replay
def replay(receipt_path, sample: int = 4, seed: int = 0, ckpt_root: Optional[Path] = None) -> Dict:
    """Recompute a deterministic sample of intervention results from the exported factors and assert equality."""
    require_bpd()
    receipt_path = Path(receipt_path)
    rc = json.loads(receipt_path.read_text(encoding="utf-8"))
    jd = receipt_path.parent
    inp, outp = rc["inputs"], rc["outputs"]
    ck_path = Path(inp["checkpoint"])
    if not ck_path.is_absolute() and ckpt_root is not None:
        ck_path = Path(ckpt_root) / ck_path
    assert sha256_file(ck_path) == inp["checkpoint_sha256"], "checkpoint sha256 mismatch"
    assert sha256_file(jd / "factors.pt") == outp["factors_file_sha256"], "factors file digest mismatch"
    assert sha256_file(jd / "result.json") == outp["result_sha256"], "result digest mismatch"
    assert sha256_file(jd / "importance.json") == outp["importance_sha256"], "importance digest mismatch"
    assert bpd_commit() == inp["bpd_commit"], "BPD checkout differs from the receipt"
    factors = torch.load(jd / "factors.pt", map_location="cpu", weights_only=True)
    assert tensor_tree_digest({"bpd": factors["bpd"], "means": factors["means"], "bases": factors["latent_bases"]}) \
        == outp["factors_tensor_sha256"], "factor tensors differ"
    hp = HP(**{k: (tuple(v) if k == "depths" else v) for k, v in inp["hyperparameters"].items()})
    skill = inp["skill"]
    ck = torch.load(ck_path, map_location="cpu", weights_only=True)
    torch.set_num_threads(1)
    rig = Rig(skill, ck, hp)
    with torch.no_grad():
        for n, w in rig.bm.wrappers.items():
            w.V.copy_(factors["bpd"][n]["V"])
            w.U.copy_(factors["bpd"][n]["U"])
    rig.means = {k: {int(v): d for v, d in factors["means"][k].items()} for k in ("bpd", "svd")}
    stored = rig.means
    rig.means = {"bpd": {}, "svd": {}}
    rig.calibrate("bpd", pools(skill, FIT, inp["counts"]["fit"]), [rig.R])
    for n, d in rig.means["bpd"][rig.R].items():
        for k, t in d.items():
            assert torch.allclose(t, stored["bpd"][rig.R][n][k], atol=1e-5), f"pass-conditional mean {n}/{k} not reproducible"
    rig.means = stored
    test = pools(skill, TEST, inp["counts"]["test"], True)
    rows = rc["replay_samples"]
    rng = np.random.default_rng(seed)
    pick = {"A", "F"} | {rows[i]["name"] for i in rng.permutation(len(rows))[:sample]}
    done = []
    for row in rows:
        if row["name"] not in pick:
            continue
        basis = factors["latent_bases"][row["basis_key"]] if row["basis_key"] else None
        accs, energy = rig.eval_method(row["method"], test, row["spec"], row["passes"], row["visits"], basis)
        got = {f: float(a.mean()) for f, a in accs.items()}
        assert got == row["exact"], f"{row['name']}: exact success differs {got} != {row['exact']}"
        assert acc_digest(flat(accs)) == row["acc_sha256"], f"{row['name']}: per-root outcome digest differs"
        assert math.isclose(energy, row["energy"], rel_tol=1e-6, abs_tol=1e-9), f"{row['name']}: energy differs"
        done.append(row["name"])
    return {"replayed": done, "n_rows": len(rows)}


# ----------------------------------------------------------------------------- campaign
def aggregate(jobs: Dict[str, Dict]) -> Dict:
    out: Dict[str, Dict] = {}
    skills = sorted({j["skill"] for j in jobs.values()})
    for skill in skills:
        rows = [j for j in jobs.values() if j["skill"] == skill]
        agg = {"seeds": sorted(j["seed"] for j in rows), "methods": {}}
        names = sorted(set.intersection(*[set(j["methods"]) for j in rows]))
        for n in names:
            ms = [j["methods"][n] for j in rows]
            agg["methods"][n] = {
                "macro_mean": float(np.mean([m["macro"] for m in ms])),
                "delta_vs_A_mean": float(np.mean([m["paired_vs_A"]["pooled"]["delta"] for m in ms])),
                "delta_vs_A_by_seed": [m["paired_vs_A"]["pooled"]["delta"] for m in ms],
                "ci95_by_seed": [m["paired_vs_A"]["pooled"]["root_bootstrap_95"] for m in ms],
                "energy_mean": float(np.mean([m["removed_energy_per_instance"] for m in ms])),
            }
        agg["role_counts"] = {j["job"]: j["role_counts"] for j in rows}
        out[skill] = agg
    return out


def campaign(out_dir: Path, skills, seeds, hp: HP, arm: str, ckpt_dir: Path, hours: float) -> Dict:
    deadline = time.monotonic() + 3600.0 * hours
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs, incomplete = {}, {}
    for skill in skills:
        for seed in seeds:
            key = f"{skill}__{arm}__s{seed}"
            rpath = out_dir / "jobs" / key / "result.json"
            if rpath.is_file():
                prev = json.loads(rpath.read_text(encoding="utf-8"))
                if prev.get("complete"):
                    jobs[key] = prev
                    continue
            if time.monotonic() >= deadline:
                incomplete[key] = "deadline reached before job start"
                continue
            try:
                jobs[key] = run_job(skill, arm, seed, out_dir, hp, ckpt_dir, deadline)
            except JobIncomplete as exc:
                incomplete[key] = str(exc)
            except Exception as exc:  # recorded, excluded, campaign continues
                incomplete[key] = f"{type(exc).__name__}: {exc}"
            results = _results(out_dir, jobs, incomplete, hp, skills, seeds, arm, hours)
            write_json(out_dir / "results.json", results)
    results = _results(out_dir, jobs, incomplete, hp, skills, seeds, arm, hours)
    write_json(out_dir / "results.json", results)
    return results


def _results(out_dir, jobs, incomplete, hp, skills, seeds, arm, hours) -> Dict:
    slim = {k: {kk: vv for kk, vv in v.items() if kk not in ("classification",)} for k, v in jobs.items()}
    return {
        "protocol": "TRM-SKILL-GYM-02 BPD of recurrent skill modules",
        "bpd_commit": bpd_commit(), "bpd_commit_expected": BPD_COMMIT_EXPECTED, "bpd_root": str(bpd_root()).replace("\\", "/"),
        "roots": ROOT_RANGES, "arm": arm, "skills": list(skills), "seeds": list(seeds), "hours": hours,
        "hyperparameters": asdict(hp), "complete": not incomplete and len(jobs) == len(skills) * len(seeds),
        "incomplete_jobs": incomplete, "jobs": slim, "aggregate": aggregate(jobs) if jobs else {},
        "receipts": {k: f"jobs/{k}/receipt.json" for k in jobs},
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--skills", nargs="+", default=list(SKILLS_DEFAULT))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--arm", default="F_trm_rlt")
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--hours", type=float, default=8.0)
    p.add_argument("--smoke", action="store_true")
    for name in ("batch", "fit_count", "cal_count", "test_count", "n_random", "n_draw", "classify_batch",
                 "classify_comps", "warmup"):
        p.add_argument("--" + name.replace("_", "-"), type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--c-mult", type=float, default=None)
    args = p.parse_args(argv)
    if not 0 < args.hours <= 8:
        p.error("--hours must be in (0, 8]")
    if BPD is None:
        p.error(BPD_ERROR or "BPD unavailable")
    torch.set_num_threads(1)
    hp = HP.smoke() if args.smoke else HP()
    for name in ("steps", "batch", "fit_count", "cal_count", "test_count", "n_random", "n_draw", "classify_batch",
                 "classify_comps", "warmup", "lr", "c_mult"):
        v = getattr(args, name)
        if v is not None:
            setattr(hp, name, v)
    res = campaign(args.out, args.skills, args.seeds, hp, args.arm, args.ckpt_dir, args.hours)
    print(json.dumps({"complete": res["complete"], "jobs": sorted(res["jobs"]), "incomplete": res["incomplete_jobs"],
                      "bpd_commit": res["bpd_commit"]}, indent=1))
    return res


if __name__ == "__main__":
    main()
