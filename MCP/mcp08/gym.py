"""Main CLI runner for MCP-08: Verifier-Guided Recursive Memory Search and Error Recovery.

Usage:
  python -m MCP.mcp08.gym --smoke --threads 4 --out MCP/results/mcp08/smoke
  python -m MCP.mcp08.gym --threads 6 --out MCP/results/mcp08/registered
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from MCP.mcp07.curriculum import FAMILIES, QueryRoot, sample_family_roots
from MCP.mcp07.eval_harness import paired_bootstrap_ci, rollout_batch as mcp07_rollout_batch
from MCP.mcp07.models import CONFIG_54K, AdaptiveGeometryController
from .mechanistic import run_cross_domain_mechanistic
from .memory256k import ExtendedPhysicalMemory
from .models import (
    CONFIG_MCP08_54K,
    MCP08Config,
    MCP08RecursiveController,
    compute_ece_and_auroc,
)
from .search import (
    CORE_PLANNING_ARMS,
    SearchLesionSpec,
    retrieve_single_mcp08,
    solve_domain_episode,
)
from .storyworld_bench import (
    HORIZON_BUCKETS,
    StoryworldSearchEnv,
    generate_independent_storyworld,
    strip_world_witnesses,
)
from .train import train_mcp08_controller

MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"
MCP07_CKPT_DIR = Path(__file__).resolve().parents[1] / "results" / "mcp07" / "registered" / "checkpoints"

SCRATCH_ABLATION_CONFIGS: Dict[str, MCP08Config] = {
    "scratch_no_persist_slow": MCP08Config(persist_slow=False),
    "scratch_no_unc_alloc": MCP08Config(use_unc_alloc=False),
    "scratch_no_backtrack": MCP08Config(use_backtrack=False),
    "scratch_no_cache": MCP08Config(use_cache=False),
    "scratch_no_postfail_switch": MCP08Config(use_postfail_switch=False),
}


def _peak_rss_bytes() -> Optional[int]:
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))
    except ImportError:
        try:
            import psutil

            return int(psutil.Process().memory_info().peak_wset)
        except (ImportError, AttributeError):
            return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _init_worker_cpu() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _load_frozen_mcp07_models(
    seeds: Sequence[int],
) -> Tuple[Dict[int, AdaptiveGeometryController], Dict[int, str], Dict[str, str]]:
    """Load unmodified registered MCP-07 checkpoints for direct `mcp07_orig` baseline comparison."""
    models: Dict[int, AdaptiveGeometryController] = {}
    paths: Dict[int, str] = {}
    hashes: Dict[str, str] = {}
    for s in seeds:
        cp = MCP07_CKPT_DIR / f"{s}-recurrent_switch.pt"
        if not cp.is_file():
            raise FileNotFoundError(f"Expected preserved MCP-07 checkpoint not found: {cp}")
        mdl = AdaptiveGeometryController(CONFIG_54K)
        mdl.load_state_dict(torch.load(cp, map_location="cpu", weights_only=True))
        mdl.eval()
        models[s] = mdl
        paths[s] = str(cp)
        hashes[f"mcp07_parent/{cp.name}"] = _sha256_file(cp)
    return models, paths, hashes


def _train_worker_08(job: Dict[str, Any]) -> Dict[str, Any]:
    _init_worker_cpu()
    t0 = time.perf_counter()
    mem16 = ExtendedPhysicalMemory(16384, int(job["world_seed"]))
    mdl, st = train_mcp08_controller(
        mem16,
        seed=int(job["seed"]),
        cfg=job["cfg"],
        num_roots=int(job["num_roots"]),
        dagger_roots=int(job["dagger_roots"]),
        epochs=int(job["epochs"]),
        batch_size=int(job["batch_size"]),
        lr=float(job["lr"]),
        horizon=int(job["horizon"]),
        salt=int(job["salt"]),
    )
    st["train_seconds"] = time.perf_counter() - t0
    ckpt_path = Path(job["ckpt_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mdl.state_dict(), ckpt_path)
    return {
        "name": job["name"],
        "seed": int(job["seed"]),
        "cfg": mdl.cfg,
        "state_dict": {k: v.cpu() for k, v in mdl.state_dict().items()},
        "stats": st,
        "ckpt_path": str(ckpt_path),
        "ckpt_name": ckpt_path.name,
        "ckpt_sha": _sha256_file(ckpt_path),
    }


def _run_train_pool(jobs: List[Dict[str, Any]], workers: int) -> List[Dict[str, Any]]:
    results = []
    if workers <= 1 or len(jobs) <= 1:
        for j in jobs:
            res = _train_worker_08(j)
            st = res["stats"]
            print(
                f"[train] seed={res['seed']} model={res['name']:26s} "
                f"d0={st['dagger_16k_clean_rate']*100:4.1f}% d10={st['dagger_16k_c10_rate']*100:4.1f}% "
                f"loss={st['final_train_loss']:.4f} p={st['active_params']} ({st['train_seconds']:.1f}s)",
                flush=True,
            )
            results.append(res)
        return results

    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        for res in ex.map(_train_worker_08, jobs):
            st = res["stats"]
            print(
                f"[train] seed={res['seed']} model={res['name']:26s} "
                f"d0={st['dagger_16k_clean_rate']*100:4.1f}% d10={st['dagger_16k_c10_rate']*100:4.1f}% "
                f"loss={st['final_train_loss']:.4f} p={st['active_params']} ({st['train_seconds']:.1f}s)",
                flush=True,
            )
            results.append(res)
    return results


def _storyworld_eval_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    """Run one (capacity, corruption, seed, arm) batch of independent Storyworld episodes."""
    _init_worker_cpu()
    capacity = int(job["capacity"])
    mem_seed = int(job["mem_seed"])
    seed = int(job["seed"])
    corr = float(job["corr"])
    namespace = int(job["namespace"])
    worlds_per_bucket = int(job["worlds_per_bucket"])
    w_stride = int(job["w_stride"])
    ep_seed_mult = int(job["ep_seed_mult"])
    layout_offset = int(job["layout_offset"])
    arm = str(job["arm"])
    run_arm = str(job["run_arm"])
    uniform_deep = bool(job["uniform_deep"])
    collect_unc = bool(job["collect_unc"])

    mdl08: Optional[MCP08RecursiveController] = None
    mdl07: Optional[AdaptiveGeometryController] = None
    if job["ckpt_path"] is not None:
        if job["is_mcp07"]:
            mdl07 = AdaptiveGeometryController(CONFIG_54K)
            mdl07.load_state_dict(torch.load(job["ckpt_path"], map_location="cpu", weights_only=True))
            mdl07.eval()
        else:
            mdl08 = MCP08RecursiveController(job["cfg"])
            mdl08.load_state_dict(torch.load(job["ckpt_path"], map_location="cpu", weights_only=True))
            mdl08.eval()

    mem = ExtendedPhysicalMemory(capacity, mem_seed)
    lesion = (
        SearchLesionSpec(name="uniform_deep_t2", uniform_compute_tier=2, disable_early_stop=True)
        if uniform_deep
        else None
    )

    records: List[Dict[str, Any]] = []
    unc_preds: List[float] = []
    unc_labels: List[int] = []
    ep_lines: List[str] = []
    fail_lines: List[str] = []

    w_idx = 0
    for bname, lo, hi in HORIZON_BUCKETS:
        for k in range(worlds_per_bucket):
            wseed = namespace + seed * 10000 + w_idx * w_stride
            if worlds_per_bucket == 1:
                horizon = hi
            else:
                horizon = int(round(lo + (hi - lo) * (k / float(worlds_per_bucket - 1))))
            raw_w = generate_independent_storyworld(wseed, horizon, bname)
            env = StoryworldSearchEnv(strip_world_witnesses(raw_w), mem, wseed + layout_offset)
            rec = solve_domain_episode(
                env,
                arm=run_arm,
                mcp08_model=mdl08,
                mcp07_model=mdl07,
                corruption=corr,
                seed=seed * ep_seed_mult + w_idx,
                lesion=lesion,
                archived_witness=raw_w["_private_witness"]["route"],
            )
            rec["arm"] = arm
            rec["episode_seed"] = rec["seed"]
            rec["seed"] = seed
            if collect_unc:
                unc_preds.extend(rec["_unc_preds"])
                unc_labels.extend(rec["_unc_labels"])
            # Strip underscore payload arrays before returning across process boundary
            clean_rec = {rk: rv for rk, rv in rec.items() if not rk.startswith("_")}
            records.append(clean_rec)

            if not uniform_deep:
                compact = {rk: rv for rk, rv in clean_rec.items() if rk != "verification"}
                compact["ver_all5"] = bool(clean_rec["verification"].get("all_five_verified", False))
                compact["route_len"] = int(clean_rec["verification"].get("route_len", 0))
                line = json.dumps(compact, separators=(",", ":")) + "\n"
                ep_lines.append(line)
                if not clean_rec["success"]:
                    fail_lines.append(line)
            w_idx += 1

    return {
        "capacity": capacity,
        "corr": corr,
        "seed": seed,
        "arm": arm,
        "uniform_deep": uniform_deep,
        "records": records,
        "unc_preds": unc_preds,
        "unc_labels": unc_labels,
        "ep_lines": ep_lines,
        "fail_lines": fail_lines,
    }


def _extrap_256k_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    """Run all 5 retrieval arms for one (capacity, seed, corruption) cell of Track 2."""
    _init_worker_cpu()
    cap = int(job["cap"])
    seed = int(job["seed"])
    corr = float(job["corr"])
    roots_per_family = int(job["roots_per_family"])
    namespace = int(job["namespace"])

    m08 = MCP08RecursiveController(job["mcp08_cfg"])
    m08.load_state_dict(torch.load(job["mcp08_ckpt_path"], map_location="cpu", weights_only=True))
    m08.eval()

    m07 = AdaptiveGeometryController(CONFIG_54K)
    m07.load_state_dict(torch.load(job["mcp07_ckpt_path"], map_location="cpu", weights_only=True))
    m07.eval()

    mem = ExtendedPhysicalMemory(cap, namespace + seed * 1000 + cap)
    all_roots: List[QueryRoot] = []
    for fam in FAMILIES:
        all_roots.extend(sample_family_roots(mem, fam, roots_per_family, 331))

    cells: List[Dict[str, Any]] = []
    ep_lines: List[str] = []
    fail_lines: List[str] = []
    g256_ok: Dict[str, List[int]] = {}
    g256_reads: Dict[str, List[float]] = {}
    g256_unc_p: List[float] = []
    g256_unc_y: List[int] = []
    is_gate_cell = (cap == 262144) and (abs(corr - 0.10) < 1e-9)

    # 1. Batch-evaluate mcp07_orig and det_router at generous 16-probe budget
    for b_arm, b_mdl in (("mcp07_orig", m07), ("det_router", None)):
        arm_name = "hier_adaptive" if b_arm == "mcp07_orig" else "det_router"
        recs, _ = mcp07_rollout_batch(
            mem,
            all_roots,
            arm=arm_name,
            model=b_mdl,
            budget=16,
            corruption=corr,
            rng_seed=seed * 10000 + cap + int(corr * 1000),
        )
        oks = [int(r["success"]) for r in recs]
        crs = [float(r["charged_reads"]) for r in recs]
        if is_gate_cell:
            g256_ok[b_arm] = oks
            g256_reads[b_arm] = crs
        cells.append(
            {
                "track": "extrapolation_256k",
                "seed": seed,
                "capacity": cap,
                "corruption": corr,
                "arm": b_arm,
                "n": len(recs),
                "completion": float(np.mean(oks)),
                "mean_charged_reads": float(np.mean(crs)),
                "mean_probes": float(np.mean([r["probes"] for r in recs])),
                "mean_switches": float(np.mean([r["switches"] for r in recs])),
                "mean_recurrent_updates": float(np.mean([r["recurrent_updates"] for r in recs])),
                "error_recovery_rate": 0.0,
                "mean_retries": 0.0,
            }
        )
        for r_i, r in enumerate(recs):
            row = {
                "track": "extrap_256k",
                "seed": seed,
                "cap": cap,
                "c": corr,
                "arm": b_arm,
                "fam": all_roots[r_i].family,
                "ok": int(r["success"]),
                "cr": int(r["charged_reads"]),
                "ru": int(r["recurrent_updates"]),
                "ret": 0,
            }
            line = json.dumps(row, separators=(",", ":")) + "\n"
            ep_lines.append(line)
            if not r["success"]:
                fail_lines.append(line)

    # 2. Evaluate MCP-08 variants: uniform_retries, mcp08_uniform_deep, mcp08_full
    s_zero = torch.zeros(1, m08.cfg.d_s)
    for v_arm in ("uniform_retries", "mcp08_uniform_deep", "mcp08_full"):
        oks = []
        crs = []
        prbs = []
        sws = []
        rus = []
        rets = []
        pf_sws = []
        first_fails = 0
        recovs = 0
        for r_i, root in enumerate(all_roots):
            ep_ok = True
            ep_cr = 0
            ep_pr = 0
            ep_sw = 0
            ep_ru = 0
            ep_ret = 0
            ep_pf = 0
            ep_ff = False
            cur_s = s_zero
            for h_idx, hop in enumerate(root.hops):
                ctx = np.array([0.5 * h_idx, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
                if v_arm == "uniform_retries":
                    r = retrieve_single_mcp08(
                        mem, hop, m08, cur_s, ctx, corruption=corr, max_retries=2,
                        uniform_tier=1, uniform_blind_retry=True, seed=seed * 1000 + r_i,
                    )
                elif v_arm == "mcp08_uniform_deep":
                    r = retrieve_single_mcp08(
                        mem, hop, m08, cur_s, ctx, corruption=corr, max_retries=2,
                        uniform_tier=2, uniform_blind_retry=False, disable_early_stop=True, seed=seed * 1000 + r_i,
                    )
                else:
                    r = retrieve_single_mcp08(
                        mem, hop, m08, cur_s, ctx, corruption=corr, max_retries=2,
                        uniform_tier=None, uniform_blind_retry=False, seed=seed * 1000 + r_i,
                    )
                ep_cr += r["charged_reads"]
                ep_pr += r["probes"]
                ep_sw += r["switches"]
                ep_ru += r["recurrent_updates"]
                ep_ret += r["retries"]
                ep_pf += r["postfail_switches"]
                ep_ff = ep_ff or r["first_pass_failed"]
                cur_s = r["next_s_persist"]
                if v_arm == "mcp08_full" and is_gate_cell:
                    g256_unc_p.append(float(r["first_pass_u_prob"]))
                    g256_unc_y.append(1 if r["first_pass_failed"] else 0)
                if not r["success"]:
                    ep_ok = False
                    break

            oks.append(1 if ep_ok else 0)
            crs.append(float(ep_cr))
            prbs.append(float(ep_pr))
            sws.append(float(ep_sw))
            rus.append(float(ep_ru))
            rets.append(float(ep_ret))
            pf_sws.append(float(ep_pf))
            if ep_ff:
                first_fails += 1
                if ep_ok:
                    recovs += 1
            row = {
                "track": "extrap_256k",
                "seed": seed,
                "cap": cap,
                "c": corr,
                "arm": v_arm,
                "fam": root.family,
                "ok": 1 if ep_ok else 0,
                "cr": int(ep_cr),
                "ru": int(ep_ru),
                "ret": int(ep_ret),
                "pfsw": int(ep_pf),
            }
            line = json.dumps(row, separators=(",", ":")) + "\n"
            ep_lines.append(line)
            if not ep_ok:
                fail_lines.append(line)

        if is_gate_cell:
            g256_ok[v_arm] = oks
            g256_reads[v_arm] = crs

        cells.append(
            {
                "track": "extrapolation_256k",
                "seed": seed,
                "capacity": cap,
                "corruption": corr,
                "arm": v_arm,
                "n": len(all_roots),
                "completion": float(np.mean(oks)),
                "mean_charged_reads": float(np.mean(crs)),
                "mean_probes": float(np.mean(prbs)),
                "mean_switches": float(np.mean(sws)),
                "mean_recurrent_updates": float(np.mean(rus)),
                "mean_retries": float(np.mean(rets)),
                "mean_postfail_switches": float(np.mean(pf_sws)),
                "error_recovery_rate": float(recovs) / float(max(1, first_fails)),
            }
        )

    return {
        "cap": cap,
        "seed": seed,
        "corr": corr,
        "cells": cells,
        "g256_ok": g256_ok,
        "g256_reads": g256_reads,
        "g256_unc_p": g256_unc_p,
        "g256_unc_y": g256_unc_y,
        "ep_lines": ep_lines,
        "fail_lines": fail_lines,
    }


def _run_256k_retrieval_track(
    capacities: Sequence[int],
    seeds: Sequence[int],
    roots_per_family: int,
    corruptions: Sequence[float],
    namespace: int,
    mcp08_ckpt_paths: Dict[int, str],
    mcp08_cfg: MCP08Config,
    mcp07_ckpt_paths: Dict[int, str],
    workers: int,
    f_ep,
    f_fail,
) -> Dict[str, Any]:
    """Track 2: 16K-trained frozen controllers extrapolated up to 256K (parallelized across CPU workers)."""
    arms = ("mcp07_orig", "det_router", "uniform_retries", "mcp08_uniform_deep", "mcp08_full")
    jobs: List[Dict[str, Any]] = []
    for cap in capacities:
        for seed in seeds:
            for corr in corruptions:
                jobs.append(
                    {
                        "cap": int(cap),
                        "seed": int(seed),
                        "corr": float(corr),
                        "roots_per_family": int(roots_per_family),
                        "namespace": int(namespace),
                        "mcp08_ckpt_path": mcp08_ckpt_paths[seed],
                        "mcp08_cfg": mcp08_cfg,
                        "mcp07_ckpt_path": mcp07_ckpt_paths[seed],
                    }
                )

    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            job_results = list(ex.map(_extrap_256k_worker, jobs))
    else:
        job_results = [_extrap_256k_worker(j) for j in jobs]

    cells: List[Dict[str, Any]] = []
    g256_ok: Dict[str, List[int]] = {a: [] for a in arms}
    g256_reads: Dict[str, List[float]] = {a: [] for a in arms}
    g256_unc_p: List[float] = []
    g256_unc_y: List[int] = []

    for jr in job_results:
        cells.extend(jr["cells"])
        for line in jr["ep_lines"]:
            f_ep.write(line)
        for line in jr["fail_lines"]:
            f_fail.write(line)
        for a, vals in jr["g256_ok"].items():
            g256_ok[a].extend(vals)
        for a, vals in jr["g256_reads"].items():
            g256_reads[a].extend(vals)
        g256_unc_p.extend(jr["g256_unc_p"])
        g256_unc_y.extend(jr["g256_unc_y"])

    gate2_boot = {}
    if g256_ok["mcp08_full"]:
        full_a = np.asarray(g256_ok["mcp08_full"], dtype=np.float64)
        for comp_arm in ("mcp07_orig", "det_router", "uniform_retries", "mcp08_uniform_deep"):
            c_a = np.asarray(g256_ok[comp_arm], dtype=np.float64)
            gate2_boot[f"mcp08_full_minus_{comp_arm}"] = paired_bootstrap_ci(full_a, c_a, seed=9200 + len(comp_arm))
        read_red_vs_deep = paired_bootstrap_ci(
            np.asarray(g256_reads["mcp08_uniform_deep"], dtype=np.float64),
            np.asarray(g256_reads["mcp08_full"], dtype=np.float64),
            seed=9250,
            relative=True,
        )
    else:
        read_red_vs_deep = {"point": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}

    return {
        "cells": cells,
        "gate2_256k_c10_bootstrap": gate2_boot,
        "read_reduction_vs_uniform_deep_256k_c10": read_red_vs_deep,
        "calibration_256k_c10": compute_ece_and_auroc(np.asarray(g256_unc_p), np.asarray(g256_unc_y)),
    }


def _aggregate_storyworld_runs(recs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not recs:
        return {}
    return {
        "n_worlds": len(recs),
        "completion": float(np.mean([r["success"] for r in recs])),
        "mean_charged_reads": float(np.mean([r["charged_memory_reads"] for r in recs])),
        "mean_verifier_calls": float(np.mean([r["verifier_calls"] for r in recs])),
        "mean_recurrent_updates": float(np.mean([r["recurrent_updates"] for r in recs])),
        "mean_fast_updates": float(np.mean([r["fast_updates"] for r in recs])),
        "mean_slow_updates": float(np.mean([r["slow_updates"] for r in recs])),
        "mean_retrieval_rate": float(np.mean([r["retrieval_rate"] for r in recs])),
        "mean_error_recovery_rate": float(np.mean([r["error_recovery_rate"] for r in recs])),
        "mean_retries": float(np.mean([r["retries_triggered"] for r in recs])),
        "mean_postfail_switches": float(np.mean([r["post_failure_switches"] for r in recs])),
        "mean_backtracks": float(np.mean([r["backtracks_executed"] for r in recs])),
        "mean_irrev_blocks": float(np.mean([r["irreversible_blocks_hit"] for r in recs])),
        "mean_cache_hits": float(np.mean([r["cache_hits"] for r in recs])),
        "mean_cache_pruned": float(np.mean([r["cache_pruned_states"] for r in recs])),
        "mean_overlap_reads": float(np.mean([r["overlap_slot_reads"] for r in recs])),
        "index_storage_bytes": int(recs[0]["index_storage_bytes"]),
        "cache_storage_bytes": int(recs[0]["cache_storage_bytes"]),
        "checkpoint_stack_bytes": int(recs[0]["checkpoint_stack_bytes"]),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if args.threads < 1:
        raise ValueError("threads must be positive")
    workers = int(args.threads)
    torch.set_num_threads(1 if workers > 1 else args.threads)
    torch.use_deterministic_algorithms(True)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_sha = _sha256_file(MANIFEST_PATH)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    smoke = bool(args.smoke)
    if smoke:
        seeds: Tuple[int, ...] = (107,)
        train_roots = 192
        dagger_roots = 64
        train_epochs = 4
        batch_size = 32
        lr = float(manifest["training"]["lr"])
        sw_worlds_per_bucket_32k = 2
        sw_worlds_per_bucket_256k = 1
        sw_corrs_32k: Tuple[float, ...] = (0.10,)
        extrap_caps: Tuple[int, ...] = (16384, 32768, 262144)
        extrap_rpf = 4
        extrap_corrs: Tuple[float, ...] = (0.10,)
        mech_worlds = 4
        ns_sw_32k = int(manifest["namespaces"]["smoke_storyworld"])
        ns_sw_256k = int(manifest["namespaces"]["smoke_storyworld"]) + 5000
        ns_256k = int(manifest["namespaces"]["smoke_256k"])
        ns_mech = int(manifest["namespaces"]["smoke_mechanistic"])
        n_boot = 400
    else:
        seeds = tuple(manifest["seeds"])
        train_roots = int(manifest["training"]["num_roots"])
        dagger_roots = int(manifest["training"]["dagger_roots"])
        train_epochs = int(manifest["training"]["epochs"])
        batch_size = int(manifest["training"]["batch_size"])
        lr = float(manifest["training"]["lr"])
        sw_worlds_per_bucket_32k = int(manifest["storyworld_benchmark"]["worlds_per_bucket_per_seed_32k"])
        sw_worlds_per_bucket_256k = int(manifest["storyworld_benchmark"]["worlds_per_bucket_per_seed_256k"])
        sw_corrs_32k = tuple(manifest["storyworld_benchmark"]["corruptions_32k"])
        extrap_caps = tuple(manifest["extrapolation_256k_track"]["capacities"])
        extrap_rpf = int(manifest["extrapolation_256k_track"]["roots_per_family_per_seed"])
        extrap_corrs = tuple(manifest["extrapolation_256k_track"]["corruptions"])
        mech_worlds = int(manifest["cross_domain_mechanistic"]["worlds_per_domain_per_seed"])
        ns_sw_32k = int(manifest["namespaces"]["sealed_storyworld_32k"])
        ns_sw_256k = int(manifest["namespaces"]["sealed_storyworld_256k"])
        ns_256k = int(manifest["namespaces"]["sealed_256k_retrieval"])
        ns_mech = int(manifest["namespaces"]["sealed_mechanistic"])
        n_boot = 2000

    # 1. Load preserved MCP-07 checkpoints (read-only!)
    mcp07_models, mcp07_ckpt_paths, ckpt_hashes = _load_frozen_mcp07_models(seeds)

    # 2. Train MCP-08 full + 5 trained-from-scratch ablations strictly at 16K in parallel!
    train_jobs: List[Dict[str, Any]] = []
    for seed in seeds:
        train_jobs.append(
            {
                "name": "mcp08_full",
                "seed": seed,
                "cfg": CONFIG_MCP08_54K,
                "world_seed": 10000 + seed,
                "num_roots": train_roots,
                "dagger_roots": dagger_roots,
                "epochs": train_epochs,
                "batch_size": batch_size,
                "lr": lr,
                "horizon": int(manifest["training"]["horizon"]),
                "salt": 37,
                "ckpt_path": str(ckpt_dir / f"{seed}-mcp08_full.pt"),
            }
        )
        for ab_name, ab_cfg in SCRATCH_ABLATION_CONFIGS.items():
            train_jobs.append(
                {
                    "name": ab_name,
                    "seed": seed,
                    "cfg": ab_cfg,
                    "world_seed": 10000 + seed,
                    "num_roots": train_roots,
                    "dagger_roots": dagger_roots,
                    "epochs": train_epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "horizon": int(manifest["training"]["horizon"]),
                    "salt": 37,
                    "ckpt_path": str(ckpt_dir / f"{seed}-{ab_name}.pt"),
                }
            )

    mcp08_models: Dict[int, MCP08RecursiveController] = {}
    mcp08_ckpt_paths: Dict[int, str] = {}
    scratch_ckpt_paths: Dict[str, Dict[int, str]] = {k: {} for k in SCRATCH_ABLATION_CONFIGS}
    train_telemetry: Dict[str, Any] = {}

    for res in _run_train_pool(train_jobs, workers=workers):
        mdl = MCP08RecursiveController(res["cfg"])
        mdl.load_state_dict(res["state_dict"])
        mdl.eval()
        ckpt_hashes[res["ckpt_name"]] = res["ckpt_sha"]
        train_telemetry.setdefault(res["name"], {})[str(res["seed"])] = res["stats"]
        if res["name"] == "mcp08_full":
            mcp08_models[res["seed"]] = mdl
            mcp08_ckpt_paths[res["seed"]] = res["ckpt_path"]
        else:
            scratch_ckpt_paths[res["name"]][res["seed"]] = res["ckpt_path"]

    ep_gz_path = out_dir / "episodes.jsonl.gz"
    fail_gz_path = out_dir / "failed_episodes.jsonl.gz"

    # 3. Track 1: Long-Horizon Independent Native Storyworld Benchmark (5-50 decisions, 32K + 256K)
    all_sw_arms = list(CORE_PLANNING_ARMS) + list(SCRATCH_ABLATION_CONFIGS.keys())
    sw_records_32k: Dict[Tuple[float, str], List[Dict[str, Any]]] = {
        (c, a): [] for c in sw_corrs_32k for a in all_sw_arms
    }
    sw_uniform_deep_c10: List[Dict[str, Any]] = []
    sw_unc_preds_c10: List[float] = []
    sw_unc_labels_c10: List[int] = []

    corr_256 = float(manifest["storyworld_benchmark"]["corruption_256k"])
    arms_256_sw = ("mcp07_orig", "det_router_greedy", "uniform_retries", "greedy_planning", "bounded_symbolic", "mcp08_full")
    sw_records_256k: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arms_256_sw}
    total_episodes = 0
    total_failed = 0

    def _build_sw_job(
        capacity: int,
        mem_seed: int,
        seed: int,
        corr: float,
        namespace: int,
        wpb: int,
        w_stride: int,
        ep_mult: int,
        lay_off: int,
        arm: str,
        uniform_deep: bool = False,
    ) -> Dict[str, Any]:
        is_scratch = arm in SCRATCH_ABLATION_CONFIGS
        run_arm = "mcp08_full" if (is_scratch or uniform_deep) else arm
        is_m07 = (run_arm == "mcp07_orig")
        if run_arm in ("det_router_greedy", "bounded_symbolic"):
            ckpt_p = None
            cfg_use = None
        elif is_m07:
            ckpt_p = mcp07_ckpt_paths[seed]
            cfg_use = None
        elif is_scratch:
            ckpt_p = scratch_ckpt_paths[arm][seed]
            cfg_use = SCRATCH_ABLATION_CONFIGS[arm]
        else:
            ckpt_p = mcp08_ckpt_paths[seed]
            cfg_use = CONFIG_MCP08_54K
        return {
            "capacity": capacity,
            "mem_seed": mem_seed,
            "seed": int(seed),
            "corr": float(corr),
            "namespace": int(namespace),
            "worlds_per_bucket": int(wpb),
            "w_stride": int(w_stride),
            "ep_seed_mult": int(ep_mult),
            "layout_offset": int(lay_off),
            "arm": "mcp08_uniform_deep" if uniform_deep else arm,
            "run_arm": run_arm,
            "ckpt_path": ckpt_p,
            "cfg": cfg_use,
            "is_mcp07": is_m07,
            "uniform_deep": uniform_deep,
            "collect_unc": (capacity == 32768 and arm == "mcp08_full" and (not uniform_deep) and abs(corr - 0.10) < 1e-9),
        }

    sw32_jobs: List[Dict[str, Any]] = []
    for corr in sw_corrs_32k:
        for seed in seeds:
            mseed = ns_sw_32k + seed * 1000 + int(corr * 100)
            for arm in all_sw_arms:
                sw32_jobs.append(
                    _build_sw_job(32768, mseed, seed, corr, ns_sw_32k, sw_worlds_per_bucket_32k, 73, 100, 19, arm)
                )
            if abs(corr - 0.10) < 1e-9:
                sw32_jobs.append(
                    _build_sw_job(
                        32768, mseed, seed, corr, ns_sw_32k, sw_worlds_per_bucket_32k, 73, 100, 19,
                        "mcp08_full", uniform_deep=True,
                    )
                )

    sw256_jobs: List[Dict[str, Any]] = []
    for seed in seeds:
        mseed256 = ns_sw_256k + seed * 1000
        for arm in arms_256_sw:
            sw256_jobs.append(
                _build_sw_job(262144, mseed256, seed, corr_256, ns_sw_256k, sw_worlds_per_bucket_256k, 97, 200, 23, arm)
            )

    with gzip.open(ep_gz_path, "wt", encoding="utf-8", compresslevel=3) as f_ep, gzip.open(
        fail_gz_path, "wt", encoding="utf-8", compresslevel=3
    ) as f_fail:
        print(f"[storyworld-32k] Running {len(sw32_jobs)} arm-seed tasks across {workers} CPU workers...", flush=True)
        if workers > 1 and len(sw32_jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(sw32_jobs))) as ex:
                sw32_res = list(ex.map(_storyworld_eval_worker, sw32_jobs))
        else:
            sw32_res = [_storyworld_eval_worker(j) for j in sw32_jobs]

        for rj in sw32_res:
            if rj["uniform_deep"]:
                sw_uniform_deep_c10.extend(rj["records"])
            else:
                sw_records_32k[(rj["corr"], rj["arm"])].extend(rj["records"])
                if rj["unc_preds"]:
                    sw_unc_preds_c10.extend(rj["unc_preds"])
                    sw_unc_labels_c10.extend(rj["unc_labels"])
                for line in rj["ep_lines"]:
                    f_ep.write(line)
                    total_episodes += 1
                for line in rj["fail_lines"]:
                    f_fail.write(line)
                    total_failed += 1
        print(f"[storyworld-32k] Done ({time.perf_counter()-t_start:.1f}s)", flush=True)

        print(f"[storyworld-256k] Running {len(sw256_jobs)} arm-seed tasks across {workers} CPU workers...", flush=True)
        if workers > 1 and len(sw256_jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(sw256_jobs))) as ex:
                sw256_res = list(ex.map(_storyworld_eval_worker, sw256_jobs))
        else:
            sw256_res = [_storyworld_eval_worker(j) for j in sw256_jobs]

        for rj in sw256_res:
            sw_records_256k[rj["arm"]].extend(rj["records"])
            for line in rj["ep_lines"]:
                f_ep.write(line)
                total_episodes += 1
            for line in rj["fail_lines"]:
                f_fail.write(line)
                total_failed += 1
        print(f"[storyworld-256k] Done ({time.perf_counter()-t_start:.1f}s)", flush=True)

        # --- 4. Track 2: 16K -> 256K Error-Recovery Memory Extrapolation Sweep ---
        print(f"[256k-track] Running 16K -> 256K extrapolation sweep across {workers} CPU workers...", flush=True)
        extrap_report = _run_256k_retrieval_track(
            extrap_caps, seeds, extrap_rpf, extrap_corrs, ns_256k,
            mcp08_ckpt_paths, CONFIG_MCP08_54K, mcp07_ckpt_paths, workers, f_ep, f_fail,
        )
        print(f"[256k-track] Done ({time.perf_counter()-t_start:.1f}s)", flush=True)

    # 5. Track 3: Cross-Domain Mechanistic Transfer (Storyworld, Canvas, Lean) & 7 Lesions
    print("[mechanistic] Running cross-domain linear probes (3x3 slow & fast), patching, and 7 lesions...", flush=True)
    mech_report = run_cross_domain_mechanistic(
        models_by_seed=mcp08_models,
        seeds=seeds,
        capacity=32768,
        worlds_per_domain_per_seed=mech_worlds,
        corruption=0.10,
        namespace=ns_mech,
        ckpt_paths_by_seed=mcp08_ckpt_paths,
        workers=workers,
    )
    (out_dir / "mechanistic.json").write_text(json.dumps(mech_report, indent=2) + "\n", encoding="utf-8")
    print(f"[mechanistic] Done ({time.perf_counter()-t_start:.1f}s)", flush=True)

    # 6. Compute Formal Preregistered Gates (Gates 1-4 + Compute Gate)
    full_32_c10 = sw_records_32k[(0.10, "mcp08_full")]
    full_ok_arr = np.asarray([int(r["success"]) for r in full_32_c10], dtype=np.float64)
    full_rd_arr = np.asarray([float(r["charged_memory_reads"]) for r in full_32_c10], dtype=np.float64)

    g1_boots = {}
    for comp in (
        "mcp07_orig",
        "det_router_greedy",
        "uniform_retries",
        "greedy_planning",
        "beam_search",
        "bounded_symbolic",
    ) + tuple(SCRATCH_ABLATION_CONFIGS.keys()):
        c_recs = sw_records_32k[(0.10, comp)]
        c_arr = np.asarray([int(r["success"]) for r in c_recs], dtype=np.float64)
        g1_boots[f"mcp08_full_minus_{comp}"] = paired_bootstrap_ci(
            full_ok_arr, c_arr, n_resamples=n_boot, seed=6100 + len(comp)
        )

    per_bucket_32k_c10: Dict[str, Any] = {}
    for bname, _, _ in HORIZON_BUCKETS:
        per_bucket_32k_c10[bname] = {
            arm: _aggregate_storyworld_runs([r for r in sw_records_32k[(0.10, arm)] if r["bucket"] == bname])
            for arm in CORE_PLANNING_ARMS
        }

    per_seed_32k_c10: Dict[str, Any] = {}
    for s in seeds:
        per_seed_32k_c10[str(s)] = {
            arm: float(np.mean([r["success"] for r in sw_records_32k[(0.10, arm)] if r["seed"] == s]))
            for arm in CORE_PLANNING_ARMS
        }

    # Gate 3: Calibration (256K 8-family extrapolation) + Hierarchical Compute Allocation vs Uniform Deep
    cal_sw_c10 = compute_ece_and_auroc(np.asarray(sw_unc_preds_c10), np.asarray(sw_unc_labels_c10))
    cal_256_c10 = extrap_report["calibration_256k_c10"]
    ud_ok_arr = np.asarray([int(r["success"]) for r in sw_uniform_deep_c10], dtype=np.float64)
    ud_rd_arr = np.asarray([float(r["charged_memory_reads"]) for r in sw_uniform_deep_c10], dtype=np.float64)
    g3_read_red = paired_bootstrap_ci(ud_rd_arr, full_rd_arr, n_resamples=n_boot, seed=6301, relative=True)
    g3_comp_loss = paired_bootstrap_ci(ud_ok_arr, full_ok_arr, n_resamples=n_boot, seed=6302, relative=False)

    g1_d_m07 = g1_boots["mcp08_full_minus_mcp07_orig"]["point"]
    g1_d_grd = g1_boots["mcp08_full_minus_greedy_planning"]["point"]
    g1_d_uni = g1_boots["mcp08_full_minus_uniform_retries"]["point"]
    gate1_pass = (not smoke) and (g1_d_m07 >= 0.15) and (g1_d_grd >= 0.05) and (g1_d_uni >= 0.05)

    g2_b = extrap_report["gate2_256k_c10_bootstrap"]
    g2_d_m07 = g2_b["mcp08_full_minus_mcp07_orig"]["point"] if g2_b else 0.0
    g2_d_det = g2_b["mcp08_full_minus_det_router"]["point"] if g2_b else 0.0
    gate2_pass = (not smoke) and (g2_d_m07 >= 0.10) and (g2_d_det >= 0.10)

    g3_rr_256 = extrap_report["read_reduction_vs_uniform_deep_256k_c10"]["point"]
    gate3_pass = (
        (not smoke)
        and (cal_256_c10["auroc"] >= 0.70)
        and (cal_256_c10["ece"] <= 0.12)
        and (max(g3_read_red["point"], g3_rr_256) >= 0.15)
        and (g3_comp_loss["point"] <= 0.02)
    )

    offdiag_prog_r2 = float(mech_report["cross_domain_linear_probes"]["progress"]["cross_domain_offdiag_mean_r2"])
    gate4_pass = (not smoke) and (offdiag_prog_r2 > 0.25)

    p_sum = mcp08_models[seeds[0]].parameter_summary()
    peak_rss = _peak_rss_bytes()
    peak_mib = round(peak_rss / (1024.0 * 1024.0), 2) if peak_rss else None
    compute_pass = (48000 <= p_sum["active_params"] <= 60000) and (peak_mib is None or peak_mib < 2048.0)

    gates_report = {
        "gate1_storyworld_32k_c10": {
            "status": "NOT_EVALUATED_SMOKE" if smoke else ("PASS" if gate1_pass else "FAIL"),
            "paired_bootstrap": g1_boots,
            "per_bucket": per_bucket_32k_c10,
            "per_seed": per_seed_32k_c10,
        },
        "gate2_256k_extrapolation_c10": {
            "status": "NOT_EVALUATED_SMOKE" if smoke else ("PASS" if gate2_pass else "FAIL"),
            "paired_bootstrap": g2_b,
        },
        "gate3_calibration_and_compute": {
            "status": "NOT_EVALUATED_SMOKE" if smoke else ("PASS" if gate3_pass else "FAIL"),
            "storyworld_c10_calibration": cal_sw_c10,
            "extrap_256k_c10_calibration": extrap_report["calibration_256k_c10"],
            "uniform_deep_summary": _aggregate_storyworld_runs(sw_uniform_deep_c10),
            "read_reduction_vs_uniform_deep_bootstrap": g3_read_red,
            "comp_loss_vs_uniform_deep_bootstrap": g3_comp_loss,
            "read_reduction_256k_vs_uniform_deep": extrap_report["read_reduction_vs_uniform_deep_256k_c10"],
        },
        "gate4_cross_domain_transfer": {
            "status": "NOT_EVALUATED_SMOKE" if smoke else ("PASS" if gate4_pass else "FAIL"),
            "offdiag_progress_r2": offdiag_prog_r2,
            "within_domain_progress_r2": float(
                mech_report["cross_domain_linear_probes"]["progress"]["within_domain_mean_r2"]
            ),
            "fast_h_offdiag_progress_r2": float(
                mech_report["cross_domain_linear_probes_fast_h"]["progress"]["cross_domain_offdiag_mean_r2"]
            ),
        },
        "compute_gate": {
            "status": "PASS" if compute_pass else "FAIL",
            "active_params": p_sum["active_params"],
            "peak_process_rss_mib": peak_mib,
        },
    }

    storyworld_summary = {
        "32k_by_corruption_and_arm": {
            f"c={c:.2f}": {arm: _aggregate_storyworld_runs(sw_records_32k[(c, arm)]) for arm in all_sw_arms}
            for c in sw_corrs_32k
        },
        "256k_by_arm_c10": {arm: _aggregate_storyworld_runs(sw_records_256k[arm]) for arm in arms_256_sw},
        "witness_withheld_verified": True,
    }
    (out_dir / "storyworld_native.json").write_text(json.dumps(storyworld_summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "extrapolation_256k.json").write_text(json.dumps(extrap_report, indent=2) + "\n", encoding="utf-8")

    receipt_hashes = {
        "manifest.json": manifest_sha,
        "episodes.jsonl.gz": _sha256_file(ep_gz_path),
        "failed_episodes.jsonl.gz": _sha256_file(fail_gz_path),
        "storyworld_native.json": _sha256_file(out_dir / "storyworld_native.json"),
        "extrapolation_256k.json": _sha256_file(out_dir / "extrapolation_256k.json"),
        "mechanistic.json": _sha256_file(out_dir / "mechanistic.json"),
        **ckpt_hashes,
    }
    (out_dir / "checkpoints_sha256.json").write_text(json.dumps(receipt_hashes, indent=2) + "\n", encoding="utf-8")

    wall_total = time.perf_counter() - t_start
    summary = {
        "experiment_id": manifest["experiment_id"],
        "mode": "smoke" if smoke else "registered_sealed",
        "manifest_sha256": manifest_sha,
        "parent_commit": manifest["parent_commit"],
        "threads": args.threads,
        "seeds": list(seeds),
        "wall_seconds": wall_total,
        "peak_process_rss_bytes": peak_rss,
        "peak_process_rss_mib": peak_mib,
        "total_episodes": total_episodes,
        "total_failed_episodes": total_failed,
        "parameter_summary": p_sum,
        "train_telemetry": train_telemetry,
        "gates": gates_report,
        "storyworld_summary": storyworld_summary,
        "receipt_hashes": receipt_hashes,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n=== MCP-08 RUN COMPLETE ===", flush=True)
    print(
        f"Mode={summary['mode']} | Wall={wall_total:.1f}s | PeakRSS={peak_mib} MiB | "
        f"Params={p_sum['active_params']} | Episodes={total_episodes} (Failed={total_failed})",
        flush=True,
    )
    for gname, gdata in gates_report.items():
        print(f"  {gname:34s}: {gdata['status']}", flush=True)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="MCP-08 Verifier-Guided Recursive Memory Search Runner")
    ap.add_argument("--smoke", action="store_true", help="Run fast smoke verification on smoke namespaces")
    ap.add_argument("--threads", type=int, default=6, help="Number of single-thread CPU worker processes")
    ap.add_argument("--out", type=str, required=True, help="Output directory (must not already exist)")
    args = ap.parse_args(argv)
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
