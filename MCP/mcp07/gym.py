"""Main CLI runner for MCP-07: Geometry-Adaptive Recurrent Memory Routing.

Usage:
  python -m MCP.mcp07.gym --smoke --threads 4 --out MCP/results/mcp07/smoke
  python -m MCP.mcp07.gym --threads 6 --out MCP/results/mcp07/registered
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

from .curriculum import (
    FAMILIES,
    SHIFT_EVAL_WEIGHTS,
    SHIFT_TRAIN_WEIGHTS,
    QueryRoot,
    sample_family_roots,
    sample_mixed_roots,
)
from .eval_harness import (
    ALL_REGISTERED_ARMS,
    DETERMINISTIC_FIXED_ARMS,
    LEARNED_FIXED_ARMS,
    extract_matched_rand_profile,
    paired_bootstrap_ci,
    rollout_batch,
)
from .mechanistic import run_mechanistic_analysis
from .memory import STRUCTURED_PROJECTIONS, SharedPhysicalMemory
from .models import (
    CONFIG_54K,
    CONFIG_54K_FIXED,
    CONFIG_54K_NOCOMM,
    CONFIG_116K,
    AdaptiveGeometryController,
    ModelConfig,
)
from .native_storyworld import (
    DEFAULT_NATIVE_WORLD,
    run_archived_witness_replay,
    run_independent_route_discovery,
)
from .train import train_controller

MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"
TRAINED_ARMS: Tuple[str, ...] = LEARNED_FIXED_ARMS + ("router_single", "recurrent_switch")


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


def _arm_projections(arm: str) -> Tuple[str, ...]:
    if arm.startswith("det_") and arm != "det_router":
        return (arm[len("det_") :],)
    if arm.startswith("fixed_"):
        return (arm[len("fixed_") :],)
    return STRUCTURED_PROJECTIONS


def _train_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level picklable worker executing 16K training on 1 deterministic CPU thread."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    t0 = time.perf_counter()
    mem16 = SharedPhysicalMemory(16384, int(job["world_seed"]))
    mdl, st = train_controller(
        mem16,
        arm_kind=job["arm_kind"],
        seed=int(job["model_seed"]),
        num_roots=int(job["num_roots"]),
        epochs=int(job["epochs"]),
        batch_size=int(job["batch_size"]),
        lr=float(job["lr"]),
        horizon=int(job["horizon"]),
        supervision=str(job["supervision"]),
        cfg=job["cfg"],
        mixture_weights=job["mixture_weights"],
        salt=int(job["salt"]),
    )
    st["train_seconds"] = time.perf_counter() - t0
    ckpt_path = Path(job["ckpt_path"]) if job.get("ckpt_path") else None
    ckpt_sha = None
    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mdl.state_dict(), ckpt_path)
        ckpt_sha = _sha256_file(ckpt_path)

    return {
        "tag": job["tag"],
        "seed": int(job["seed"]),
        "arm": job["arm_label"],
        "cfg": mdl.cfg,
        "state_dict": {k: v.cpu() for k, v in mdl.state_dict().items()},
        "stats": st,
        "ckpt_name": ckpt_path.name if ckpt_path is not None else None,
        "ckpt_sha": ckpt_sha,
    }


def _run_train_jobs(jobs: List[Dict[str, Any]], workers: int) -> List[Dict[str, Any]]:
    """Run training jobs either sequentially (workers<=1) or across `workers` 1-thread CPU workers."""
    if workers <= 1 or len(jobs) <= 1:
        results = []
        for j in jobs:
            res = _train_worker(j)
            st = res["stats"]
            print(
                f"[train] {res['tag']} seed={res['seed']} arm={res['arm']:20s} "
                f"teacher={st['teacher_verified_rate']*100:5.1f}% dagger={st['dagger_16k_onpolicy_rate']*100:4.1f}% "
                f"loss={st['final_train_loss']:.4f} p={st['active_params']} ({st['train_seconds']:.1f}s)",
                flush=True,
            )
            results.append(res)
        return results

    results = []
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        for res in ex.map(_train_worker, jobs):
            st = res["stats"]
            print(
                f"[train] {res['tag']} seed={res['seed']} arm={res['arm']:20s} "
                f"teacher={st['teacher_verified_rate']*100:5.1f}% dagger={st['dagger_16k_onpolicy_rate']*100:4.1f}% "
                f"loss={st['final_train_loss']:.4f} p={st['active_params']} ({st['train_seconds']:.1f}s)",
                flush=True,
            )
            results.append(res)
    return results


def _aggregate_cell(
    seed: int,
    size: int,
    arm: str,
    budget: int,
    corruption: float,
    fam: str,
    recs: Sequence[Dict[str, Any]],
    index_cost: Dict[str, float],
    wall_s: float,
) -> Dict[str, Any]:
    n = len(recs)
    solved = sum(1 for r in recs if r["success"])
    raw_tot = sum(r["raw_slot_reads"] for r in recs)
    matched_tot = sum(r["matched_good"] for r in recs)
    return {
        "seed": seed,
        "size": size,
        "family": fam,
        "arm": arm,
        "budget": budget,
        "corruption": corruption,
        "n": n,
        "solved": solved,
        "completion": float(solved) / float(max(1, n)),
        "mean_probes": float(np.mean([r["probes"] for r in recs])),
        "mean_charged_reads": float(np.mean([r["charged_reads"] for r in recs])),
        "mean_raw_slot_reads": float(np.mean([r["raw_slot_reads"] for r in recs])),
        "mean_unique_physical_slots": float(np.mean([r["unique_physical_slots"] for r in recs])),
        "mean_overlap_slot_reads": float(np.mean([r["overlap_slot_reads"] for r in recs])),
        "mean_unique_logical_records": float(np.mean([r["unique_logical_records"] for r in recs])),
        "mean_switches": float(np.mean([r["switches"] for r in recs])),
        "mean_recurrent_updates": float(np.mean([r["recurrent_updates"] for r in recs])),
        "mean_fast_updates": float(np.mean([r["fast_updates"] for r in recs])),
        "mean_slow_updates": float(np.mean([r["slow_updates"] for r in recs])),
        "retrieval_precision": float(matched_tot) / float(max(1, raw_tot)),
        "stale_seen": sum(1 for r in recs if r["stale_seen"]),
        "recovered": sum(1 for r in recs if r["recovered"]),
        "premature_stops": sum(1 for r in recs if r["premature_stop"]),
        "index_storage_bytes": int(index_cost["index_storage_bytes"]),
        "preprocessing_ops": int(index_cost["preprocessing_ops"]),
        "wall_seconds": wall_s,
    }


def _run_ablations(
    manifest: Dict[str, Any],
    seeds: Sequence[int],
    models_base: Dict[int, Dict[str, AdaptiveGeometryController]],
    out_dir: Path,
    train_roots: int,
    train_epochs: int,
    batch_size: int,
    lr: float,
    eval_roots_per_family: int,
    eval_namespace: int,
    smoke: bool,
    workers: int,
) -> Dict[str, Any]:
    """Run registered ablations: single_best vs action_set, comm_off, 116K scale, and distribution shift."""
    ablation_ckpts: Dict[str, str] = {}
    eval_cap = 32768
    eval_budget = 16

    variants = (
        ("action_set_54k", None),
        ("single_best_54k", dict(supervision="single_best", cfg=CONFIG_54K)),
        ("no_comm_54k", dict(supervision="action_set", cfg=CONFIG_54K_NOCOMM)),
        ("scaled_116k", dict(supervision="action_set", cfg=CONFIG_116K)),
    )

    variant_models: Dict[str, Dict[int, AdaptiveGeometryController]] = {v[0]: {} for v in variants}
    variant_train_stats: Dict[str, Dict[str, Any]] = {v[0]: {} for v in variants}

    ablation_jobs: List[Dict[str, Any]] = []
    for seed in seeds:
        variant_models["action_set_54k"][seed] = models_base[seed]["recurrent_switch"]
        variant_train_stats["action_set_54k"][str(seed)] = models_base[seed]["recurrent_switch"].parameter_summary(True)
        for v_name, kw in variants[1:]:
            ablation_jobs.append(
                {
                    "tag": "ablation",
                    "seed": seed,
                    "arm_label": v_name,
                    "arm_kind": "recurrent_switch",
                    "world_seed": 10000 + seed,
                    "model_seed": seed,
                    "num_roots": train_roots,
                    "epochs": train_epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "horizon": int(manifest["training"]["teacher_horizon"]),
                    "supervision": kw["supervision"],
                    "cfg": kw["cfg"],
                    "mixture_weights": None,
                    "salt": 37,
                    "ckpt_path": str(out_dir / "checkpoints" / f"{seed}-ablation-{v_name}.pt"),
                }
            )

    # Also build distribution-shift training jobs so all ablation + shift models train in parallel!
    shift_arms = ("fixed_row", "fixed_col", "fixed_hilbert", "fixed_morton", "router_single", "recurrent_switch")
    shift_train_ns = int(manifest["namespaces"]["distribution_shift_train"])
    shift_eval_ns = int(manifest["namespaces"]["distribution_shift_eval"])
    shift_models: Dict[str, Dict[int, AdaptiveGeometryController]] = {a: {} for a in shift_arms}

    for seed in seeds:
        for arm in shift_arms:
            ablation_jobs.append(
                {
                    "tag": "shift",
                    "seed": seed,
                    "arm_label": arm,
                    "arm_kind": arm,
                    "world_seed": shift_train_ns + seed,
                    "model_seed": seed + 500,
                    "num_roots": train_roots,
                    "epochs": train_epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "horizon": int(manifest["training"]["teacher_horizon"]),
                    "supervision": "action_set",
                    "cfg": None,
                    "mixture_weights": SHIFT_TRAIN_WEIGHTS,
                    "salt": 89,
                    "ckpt_path": (
                        str(out_dir / "checkpoints" / f"{seed}-shift-recurrent_switch.pt")
                        if arm == "recurrent_switch"
                        else None
                    ),
                }
            )

    for res in _run_train_jobs(ablation_jobs, workers=workers):
        mdl = AdaptiveGeometryController(res["cfg"])
        mdl.load_state_dict(res["state_dict"])
        mdl.eval()
        if res["ckpt_name"] and res["ckpt_sha"]:
            ablation_ckpts[res["ckpt_name"]] = res["ckpt_sha"]
        if res["tag"] == "ablation":
            variant_models[res["arm"]][res["seed"]] = mdl
            variant_train_stats[res["arm"]][str(res["seed"])] = res["stats"]
        else:
            shift_models[res["arm"]][res["seed"]] = mdl

    # Evaluate architecture & supervision variants at 32K, 16 probes, clean + 15% corruption
    arch_results: Dict[str, Any] = {}
    pooled_ok: Dict[str, List[int]] = {v[0]: [] for v in variants}
    pooled_reads: Dict[str, List[int]] = {v[0]: [] for v in variants}

    for v_name, _ in variants:
        per_seed_comp = {}
        per_fam_ok: Dict[str, List[int]] = {f: [] for f in FAMILIES}
        c15_ok = []
        adapt_ok = []
        adapt_reads = []
        for seed in seeds:
            mem32 = SharedPhysicalMemory(eval_cap, eval_namespace + 100 * seed + eval_cap)
            roots_all: List[QueryRoot] = []
            for fam in FAMILIES:
                roots_all.extend(sample_family_roots(mem32, fam, eval_roots_per_family, 331))
            recs_clean, _ = rollout_batch(
                mem32, roots_all, arm="recurrent_switch", model=variant_models[v_name][seed],
                budget=eval_budget, corruption=0.0, rng_seed=seed * 100 + 3,
            )
            recs_c15, _ = rollout_batch(
                mem32, roots_all, arm="recurrent_switch", model=variant_models[v_name][seed],
                budget=eval_budget, corruption=0.15, rng_seed=seed * 100 + 4,
            )
            recs_ad, _ = rollout_batch(
                mem32, roots_all, arm="hier_adaptive", model=variant_models[v_name][seed],
                budget=eval_budget, corruption=0.0, rng_seed=seed * 100 + 3,
            )
            ok_c = [int(r["success"]) for r in recs_clean]
            pooled_ok[v_name].extend(ok_c)
            pooled_reads[v_name].extend([int(r["charged_reads"]) for r in recs_clean])
            c15_ok.extend([int(r["success"]) for r in recs_c15])
            adapt_ok.extend([int(r["success"]) for r in recs_ad])
            adapt_reads.extend([int(r["charged_reads"]) for r in recs_ad])
            per_seed_comp[str(seed)] = float(np.mean(ok_c))
            for r in recs_clean:
                per_fam_ok[r["family"]].append(int(r["success"]))

        arch_results[v_name] = {
            "params": variant_models[v_name][seeds[0]].parameter_summary(True),
            "clean_32k_b16_completion": float(np.mean(pooled_ok[v_name])),
            "c15_32k_b16_completion": float(np.mean(c15_ok)),
            "adaptive_32k_b16_completion": float(np.mean(adapt_ok)),
            "adaptive_mean_charged_reads": float(np.mean(adapt_reads)),
            "fixed_mean_charged_reads": float(np.mean(pooled_reads[v_name])),
            "per_seed_clean": per_seed_comp,
            "per_family_clean": {f: float(np.mean(v)) for f, v in per_fam_ok.items()},
            "train_stats": variant_train_stats[v_name],
        }

    base_arr = np.asarray(pooled_ok["action_set_54k"], dtype=np.float64)
    arch_deltas = {}
    for v_name in ("single_best_54k", "no_comm_54k", "scaled_116k"):
        v_arr = np.asarray(pooled_ok[v_name], dtype=np.float64)
        arch_deltas[f"{v_name}_minus_action_set_54k"] = paired_bootstrap_ci(v_arr, base_arr, seed=4400 + len(v_name))

    # Distribution-shift evaluation on inverted 20/80 mixture at 32K
    shift_eval_arms = shift_arms + ("hier_adaptive", "det_router")
    shift_pooled: Dict[str, List[int]] = {a: [] for a in shift_eval_arms}
    shift_reads: Dict[str, List[int]] = {a: [] for a in shift_eval_arms}
    shift_per_fam: Dict[str, Dict[str, List[int]]] = {a: {f: [] for f in FAMILIES} for a in shift_eval_arms}

    shift_eval_n = 32 if smoke else 1024
    for seed in seeds:
        mem32_sh = SharedPhysicalMemory(eval_cap, shift_eval_ns + 100 * seed + eval_cap)
        sh_roots = sample_mixed_roots(mem32_sh, shift_eval_n, salt=509, weights=SHIFT_EVAL_WEIGHTS)
        for arm in shift_eval_arms:
            if arm == "det_router":
                mdl = None
            elif arm == "hier_adaptive":
                mdl = shift_models["recurrent_switch"][seed]
            else:
                mdl = shift_models[arm][seed]
            recs, _ = rollout_batch(
                mem32_sh, sh_roots, arm=arm, model=mdl, budget=eval_budget, corruption=0.0, rng_seed=seed * 77
            )
            shift_pooled[arm].extend([int(r["success"]) for r in recs])
            shift_reads[arm].extend([int(r["charged_reads"]) for r in recs])
            for r in recs:
                shift_per_fam[arm][r["family"]].append(int(r["success"]))

    best_sh_fixed = max(
        ("fixed_row", "fixed_col", "fixed_hilbert", "fixed_morton"),
        key=lambda a: float(np.mean(shift_pooled[a])),
    )
    shift_summary = {
        "train_mixture": SHIFT_TRAIN_WEIGHTS,
        "eval_mixture": SHIFT_EVAL_WEIGHTS,
        "eval_capacity": eval_cap,
        "eval_budget": eval_budget,
        "n_eval_roots_total": len(shift_pooled["recurrent_switch"]),
        "by_arm": {
            a: {
                "completion": float(np.mean(shift_pooled[a])),
                "mean_charged_reads": float(np.mean(shift_reads[a])),
                "per_family": {f: float(np.mean(v)) for f, v in shift_per_fam[a].items() if v},
            }
            for a in shift_eval_arms
        },
        "best_fixed_under_shift": best_sh_fixed,
        "switch_minus_best_fixed_bootstrap": paired_bootstrap_ci(
            np.asarray(shift_pooled["recurrent_switch"]),
            np.asarray(shift_pooled[best_sh_fixed]),
            seed=5555,
        ),
        "switch_minus_single_router_bootstrap": paired_bootstrap_ci(
            np.asarray(shift_pooled["recurrent_switch"]),
            np.asarray(shift_pooled["router_single"]),
            seed=5556,
        ),
    }

    return {
        "architecture_and_supervision": arch_results,
        "architecture_paired_deltas": arch_deltas,
        "distribution_shift": shift_summary,
        "ablation_checkpoints": ablation_ckpts,
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if args.threads < 1:
        raise ValueError("threads must be positive")
    workers = int(args.threads)
    torch.set_num_threads(1 if workers > 1 else args.threads)
    torch.use_deterministic_algorithms(True)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_sha256 = _sha256_file(MANIFEST_PATH)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    smoke = bool(args.smoke)
    if smoke:
        seeds: Tuple[int, ...] = (107,)
        capacities: Tuple[int, ...] = (16384, 32768)
        budgets: Tuple[int, ...] = (8, 16)
        corruptions: Tuple[float, ...] = (0.0, 0.15)
        train_roots = 128
        train_epochs = 4
        batch_size = 32
        lr = float(manifest["training"]["lr"])
        roots_per_family = 8
        eval_namespace = manifest["namespaces"]["smoke_eval"]
        n_boot = 500
    else:
        seeds = tuple(manifest["seeds"])
        capacities = tuple(manifest["eval_capacities"])
        if args.skip_128k:
            capacities = tuple(c for c in capacities if c <= 65536)
        budgets = tuple(manifest["budgets"])
        corruptions = tuple(manifest["corruptions"])
        train_roots = int(manifest["training"]["train_roots"])
        train_epochs = int(manifest["training"]["epochs"])
        batch_size = int(manifest["training"]["batch_size"])
        lr = float(manifest["training"]["lr"])
        roots_per_family = int(manifest["roots_per_seed_capacity_family"])
        eval_namespace = int(manifest["namespaces"]["registered_eval"])
        n_boot = int(manifest["gates"]["primary"]["bootstrap_resamples"])

    # 1. Train all learned models exclusively at 16K (parallelized across `workers` 1-thread CPU workers)
    models_by_seed: Dict[int, Dict[str, AdaptiveGeometryController]] = {s: {} for s in seeds}
    checkpoints_sha: Dict[str, str] = {}
    train_telemetry: Dict[str, Any] = {str(s): {} for s in seeds}

    core_jobs: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in TRAINED_ARMS:
            core_jobs.append(
                {
                    "tag": "core",
                    "seed": seed,
                    "arm_label": arm,
                    "arm_kind": arm,
                    "world_seed": 10000 + seed,
                    "model_seed": seed,
                    "num_roots": train_roots,
                    "epochs": train_epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "horizon": int(manifest["training"]["teacher_horizon"]),
                    "supervision": str(manifest["training"]["supervision"]),
                    "cfg": None,
                    "mixture_weights": None,
                    "salt": 37,
                    "ckpt_path": str(ckpt_dir / f"{seed}-{arm}.pt"),
                }
            )

    for res in _run_train_jobs(core_jobs, workers=workers):
        mdl = AdaptiveGeometryController(res["cfg"])
        mdl.load_state_dict(res["state_dict"])
        mdl.eval()
        models_by_seed[res["seed"]][res["arm"]] = mdl
        train_telemetry[str(res["seed"])][res["arm"]] = res["stats"]
        if res["ckpt_name"] and res["ckpt_sha"]:
            checkpoints_sha[res["ckpt_name"]] = res["ckpt_sha"]

    # 2. Registered evaluation sweep across capacities, budgets, corruptions, families, arms A-H
    summary_rows: List[Dict[str, Any]] = []
    gate_ok: Dict[Tuple[int, str], np.ndarray] = {}
    gate_reads: Dict[Tuple[int, str], np.ndarray] = {}
    gate_fams: Dict[int, List[str]] = {}
    matched_profiles_32k: Dict[int, Dict[str, Any]] = {}

    roots_gz_path = out_dir / "roots.jsonl.gz"
    failed_gz_path = out_dir / "failed_episodes.jsonl.gz"
    total_episodes = 0
    total_failed = 0
    eval_arm_order = tuple(ALL_REGISTERED_ARMS)

    with gzip.open(roots_gz_path, "wt", encoding="utf-8", compresslevel=3) as f_roots, gzip.open(
        failed_gz_path, "wt", encoding="utf-8", compresslevel=3
    ) as f_fail:
        for seed in seeds:
            seed_models = models_by_seed[seed]
            for cap in capacities:
                mem = SharedPhysicalMemory(cap, eval_namespace + 100 * seed + cap)
                all_roots: List[QueryRoot] = []
                fam_slices: Dict[str, slice] = {}
                off = 0
                for fam in FAMILIES:
                    fr = sample_family_roots(mem, fam, roots_per_family, 331)
                    all_roots.extend(fr)
                    fam_slices[fam] = slice(off, off + len(fr))
                    off += len(fr)
                if cap == 32768:
                    gate_fams[seed] = [r.family for r in all_roots]

                for corr in corruptions:
                    for b in budgets:
                        cell_rand_profile: Optional[Dict[str, Any]] = None
                        for arm in eval_arm_order:
                            t_c = time.perf_counter()
                            if arm in DETERMINISTIC_FIXED_ARMS or arm == "det_router":
                                mdl = None
                            elif arm in ("hier_adaptive", "rand_routing", "oracle_proj"):
                                mdl = seed_models["recurrent_switch"]
                            else:
                                mdl = seed_models[arm]

                            recs, _ = rollout_batch(
                                mem,
                                all_roots,
                                arm=arm,
                                model=mdl,
                                budget=b,
                                corruption=corr,
                                matched_rand_profile=cell_rand_profile if arm == "rand_routing" else None,
                                rng_seed=seed * 100000 + cap + b * 31 + int(corr * 1000),
                            )
                            wall_c = time.perf_counter() - t_c

                            if arm == "recurrent_switch":
                                cell_rand_profile = extract_matched_rand_profile(recs)
                                if cap == 32768 and b == 16 and corr == 0.0:
                                    matched_profiles_32k[seed] = cell_rand_profile

                            idx_cost = mem.index_cost_for_projections(_arm_projections(arm))
                            for fam in FAMILIES:
                                sl = fam_slices[fam]
                                sub = recs[sl]
                                summary_rows.append(
                                    _aggregate_cell(
                                        seed, cap, arm, b, corr, fam, sub, idx_cost, wall_c / len(FAMILIES)
                                    )
                                )

                            if cap == 32768 and b == 16 and corr == 0.0:
                                gate_ok[(seed, arm)] = np.array([int(r["success"]) for r in recs], dtype=np.float64)
                                gate_reads[(seed, arm)] = np.array([float(r["charged_reads"]) for r in recs], dtype=np.float64)

                            for r_idx, r in enumerate(recs):
                                total_episodes += 1
                                row_dict = {
                                    "seed": seed,
                                    "size": cap,
                                    "fam": r["family"],
                                    "root": r_idx % roots_per_family,
                                    "arm": arm,
                                    "b": b,
                                    "c": corr,
                                    "ok": int(r["success"]),
                                    "probes": r["probes"],
                                    "cr": r["charged_reads"],
                                    "raw": r["raw_slot_reads"],
                                    "us": r["unique_physical_slots"],
                                    "ov": r["overlap_slot_reads"],
                                    "ur": r["unique_logical_records"],
                                    "sw": r["switches"],
                                    "ru": r["recurrent_updates"],
                                    "st": int(r["stale_seen"]),
                                    "rec": int(r["recovered"]),
                                    "ps": int(r["premature_stop"]),
                                }
                                line = json.dumps(row_dict, separators=(",", ":")) + "\n"
                                f_roots.write(line)
                                if not r["success"]:
                                    total_failed += 1
                                    f_fail.write(line)
                print(
                    f"[eval] seed={seed} size={cap:6d} done ({time.perf_counter()-t_start:.1f}s elapsed)",
                    flush=True,
                )

    # 3. Compute formal Primary & Secondary Gates at 32K, budget=16, corruption=0.0
    fixed_non_oracle_arms = DETERMINISTIC_FIXED_ARMS + LEARNED_FIXED_ARMS
    fixed_scores = {
        a: float(np.mean(np.concatenate([gate_ok[(s, a)] for s in seeds])))
        for a in fixed_non_oracle_arms
    }
    best_fixed_all = max(fixed_non_oracle_arms, key=lambda a: fixed_scores[a])
    best_fixed_det = max(DETERMINISTIC_FIXED_ARMS, key=lambda a: fixed_scores[a])
    best_fixed_learned = max(LEARNED_FIXED_ARMS, key=lambda a: fixed_scores[a])

    arm_e_ok = np.concatenate([gate_ok[(s, "recurrent_switch")] for s in seeds])
    arm_f_ok = np.concatenate([gate_ok[(s, "hier_adaptive")] for s in seeds])
    arm_d_ok = np.concatenate([gate_ok[(s, "router_single")] for s in seeds])
    arm_b_ok = np.concatenate([gate_ok[(s, "det_router")] for s in seeds])
    arm_g_ok = np.concatenate([gate_ok[(s, "rand_routing")] for s in seeds])
    arm_h_ok = np.concatenate([gate_ok[(s, "oracle_proj")] for s in seeds])
    best_fix_all_ok = np.concatenate([gate_ok[(s, best_fixed_all)] for s in seeds])
    best_fix_lrn_ok = np.concatenate([gate_ok[(s, best_fixed_learned)] for s in seeds])

    arm_e_reads = np.concatenate([gate_reads[(s, "recurrent_switch")] for s in seeds])
    arm_f_reads = np.concatenate([gate_reads[(s, "hier_adaptive")] for s in seeds])

    prim_vs_all_fixed = paired_bootstrap_ci(arm_e_ok, best_fix_all_ok, n_resamples=n_boot, seed=7001)
    prim_vs_lrn_fixed = paired_bootstrap_ci(arm_e_ok, best_fix_lrn_ok, n_resamples=n_boot, seed=7002)
    prim_adapt_vs_all = paired_bootstrap_ci(arm_f_ok, best_fix_all_ok, n_resamples=n_boot, seed=7003)
    switch_vs_single = paired_bootstrap_ci(arm_e_ok, arm_d_ok, n_resamples=n_boot, seed=7004)
    switch_vs_det_router = paired_bootstrap_ci(arm_e_ok, arm_b_ok, n_resamples=n_boot, seed=7005)
    switch_vs_rand_routing = paired_bootstrap_ci(arm_e_ok, arm_g_ok, n_resamples=n_boot, seed=7006)
    oracle_vs_switch = paired_bootstrap_ci(arm_h_ok, arm_e_ok, n_resamples=n_boot, seed=7007)

    sec_read_red = paired_bootstrap_ci(arm_e_reads, arm_f_reads, n_resamples=n_boot, seed=7008, relative=True)
    sec_comp_loss = paired_bootstrap_ci(arm_e_ok, arm_f_ok, n_resamples=n_boot, seed=7009, relative=False)

    per_seed_primary = {
        str(s): {
            "recurrent_switch": float(gate_ok[(s, "recurrent_switch")].mean()),
            "hier_adaptive": float(gate_ok[(s, "hier_adaptive")].mean()),
            "best_fixed_all": float(gate_ok[(s, best_fixed_all)].mean()),
            "best_fixed_learned": float(gate_ok[(s, best_fixed_learned)].mean()),
            "delta_vs_best_fixed_all": float(
                gate_ok[(s, "recurrent_switch")].mean() - gate_ok[(s, best_fixed_all)].mean()
            ),
            "delta_vs_best_fixed_learned": float(
                gate_ok[(s, "recurrent_switch")].mean() - gate_ok[(s, best_fixed_learned)].mean()
            ),
            "sec_read_reduction": float(
                (gate_reads[(s, "recurrent_switch")].mean() - gate_reads[(s, "hier_adaptive")].mean())
                / gate_reads[(s, "recurrent_switch")].mean()
            ),
            "sec_comp_loss": float(
                gate_ok[(s, "recurrent_switch")].mean() - gate_ok[(s, "hier_adaptive")].mean()
            ),
        }
        for s in seeds
    }

    all_fams_concat = [f for s in seeds for f in gate_fams[s]]
    per_family_32k_b16: Dict[str, Any] = {}
    for fam in FAMILIES:
        fm = np.array([x == fam for x in all_fams_concat], dtype=bool)
        per_family_32k_b16[fam] = {
            arm: float(np.concatenate([gate_ok[(s, arm)] for s in seeds])[fm].mean())
            for arm in eval_arm_order
        }
        per_family_32k_b16[fam]["delta_switch_minus_best_fixed_all"] = float(
            arm_e_ok[fm].mean() - best_fix_all_ok[fm].mean()
        )

    primary_pass = (not smoke) and (prim_vs_all_fixed["point"] >= 0.05) and (prim_vs_lrn_fixed["point"] >= 0.05)
    secondary_pass = (not smoke) and (sec_read_red["point"] >= 0.15) and (sec_comp_loss["point"] <= 0.01)

    gates_report = {
        "primary_gate": {
            "status": "NOT_EVALUATED_SMOKE" if smoke else ("PASS" if primary_pass else "FAIL"),
            "threshold_pp": 5.0,
            "best_fixed_non_oracle_all": best_fixed_all,
            "best_fixed_deterministic": best_fixed_det,
            "best_fixed_learned": best_fixed_learned,
            "fixed_arm_completions_32k_b16_c0": fixed_scores,
            "recurrent_switch_completion": float(arm_e_ok.mean()),
            "hier_adaptive_completion": float(arm_f_ok.mean()),
            "delta_vs_best_fixed_all_bootstrap": prim_vs_all_fixed,
            "delta_vs_best_fixed_learned_bootstrap": prim_vs_lrn_fixed,
            "delta_adaptive_vs_best_fixed_all_bootstrap": prim_adapt_vs_all,
            "delta_vs_single_router_bootstrap": switch_vs_single,
            "delta_vs_det_router_bootstrap": switch_vs_det_router,
            "delta_vs_rand_routing_bootstrap": switch_vs_rand_routing,
            "oracle_minus_switch_bootstrap": oracle_vs_switch,
        },
        "secondary_gate": {
            "status": "NOT_EVALUATED_SMOKE" if smoke else ("PASS" if secondary_pass else "FAIL"),
            "min_read_reduction_pct": 15.0,
            "max_completion_loss_pp": 1.0,
            "fixed_budget_mean_charged_reads": float(arm_e_reads.mean()),
            "adaptive_mean_charged_reads": float(arm_f_reads.mean()),
            "read_reduction_bootstrap": sec_read_red,
            "completion_loss_bootstrap": sec_comp_loss,
        },
        "per_seed_32k_b16_c0": per_seed_primary,
        "per_family_32k_b16_c0": per_family_32k_b16,
    }

    # 4. Ablations & Distribution Shift
    ablations_report: Optional[Dict[str, Any]] = None
    if not args.skip_ablations:
        print("[ablations] Running architecture, supervision, and distribution-shift suite...", flush=True)
        ablations_report = _run_ablations(
            manifest,
            seeds,
            models_by_seed,
            out_dir,
            train_roots=train_roots,
            train_epochs=train_epochs,
            batch_size=batch_size,
            lr=lr,
            eval_roots_per_family=roots_per_family,
            eval_namespace=eval_namespace,
            smoke=smoke,
            workers=workers,
        )
        checkpoints_sha.update(ablations_report["ablation_checkpoints"])
        (out_dir / "ablations.json").write_text(json.dumps(ablations_report, indent=2) + "\n", encoding="utf-8")

    # 5. Native Storyworld Integration (Separate Track!)
    native_report: Optional[Dict[str, Any]] = None
    world_path = Path(args.native_world) if args.native_world else DEFAULT_NATIVE_WORLD
    if world_path.is_file():
        print("[native] Running MCP-05 witness replay & witness-withheld independent route discovery...", flush=True)
        replay_arms = (
            "det_hilbert",
            "det_router",
            "fixed_row",
            "fixed_hilbert",
            "router_single",
            "recurrent_switch",
            "hier_adaptive",
            "rand_routing",
        )
        replay_runs = []
        for seed in seeds:
            sm = models_by_seed[seed]
            for arm in replay_arms:
                mdl = (
                    None
                    if arm in ("det_hilbert", "det_router")
                    else sm["recurrent_switch"]
                    if arm in ("hier_adaptive", "rand_routing")
                    else sm[arm]
                )
                replay_runs.append(
                    run_archived_witness_replay(
                        world_path,
                        arm=arm,
                        model=mdl,
                        seed=seed,
                        budget=int(manifest["native_storyworld"]["witness_replay_budget"]),
                        corruption=0.0,
                        matched_rand_profile=matched_profiles_32k.get(seed),
                    )
                )

        disc_arms_models = {
            "det_hilbert": None,
            "det_router": None,
            "fixed_row": {s: models_by_seed[s]["fixed_row"] for s in seeds},
            "fixed_col": {s: models_by_seed[s]["fixed_col"] for s in seeds},
            "fixed_hilbert": {s: models_by_seed[s]["fixed_hilbert"] for s in seeds},
            "fixed_morton": {s: models_by_seed[s]["fixed_morton"] for s in seeds},
            "router_single": {s: models_by_seed[s]["router_single"] for s in seeds},
            "recurrent_switch": {s: models_by_seed[s]["recurrent_switch"] for s in seeds},
            "hier_adaptive": {s: models_by_seed[s]["recurrent_switch"] for s in seeds},
            "rand_routing": {s: models_by_seed[s]["recurrent_switch"] for s in seeds},
        }
        discovery_report = run_independent_route_discovery(
            world_path,
            arms_models=disc_arms_models,
            seeds=seeds,
            retrieval_budget=int(manifest["native_storyworld"]["route_discovery_retrieval_budget"]),
            corruptions=(0.0,) if smoke else tuple(manifest["native_storyworld"]["route_discovery_corruptions"]),
            matched_rand_profiles=matched_profiles_32k,
        )
        native_report = {
            "status": "completed_separate_from_synthetic",
            "world_sha256": _sha256_file(world_path),
            "archived_witness_replay": replay_runs,
            "independent_route_discovery": discovery_report,
        }
        (out_dir / "native_storyworld.json").write_text(
            json.dumps(native_report, indent=2) + "\n", encoding="utf-8"
        )
    else:
        native_report = {"status": "world_file_missing", "path": str(world_path)}

    # 6. Mechanistic Interpretation & Causal Interventions
    mech_report: Optional[Dict[str, Any]] = None
    if not args.skip_mechanistic:
        print("[mechanistic] Extracting trajectories, linear probes, and 5 causal interventions...", flush=True)
        mech_report = run_mechanistic_analysis(
            models_by_seed={s: models_by_seed[s]["recurrent_switch"] for s in seeds},
            seeds=seeds,
            capacity=32768,
            budget=16,
            roots_per_family=roots_per_family,
            namespace=eval_namespace,
        )
        (out_dir / "mechanistic.json").write_text(json.dumps(mech_report, indent=2) + "\n", encoding="utf-8")

    # 7. Save checkpoint & receipt SHA-256 manifest + summary.json
    receipt_hashes = {
        "manifest.json": manifest_sha256,
        "roots.jsonl.gz": _sha256_file(roots_gz_path),
        "failed_episodes.jsonl.gz": _sha256_file(failed_gz_path),
        **checkpoints_sha,
    }
    (out_dir / "checkpoints_sha256.json").write_text(
        json.dumps(receipt_hashes, indent=2) + "\n", encoding="utf-8"
    )

    wall_total = time.perf_counter() - t_start
    peak_rss = _peak_rss_bytes()
    summary = {
        "experiment_id": manifest["experiment_id"],
        "mode": "smoke" if smoke else "registered_full",
        "manifest_sha256": manifest_sha256,
        "threads": args.threads,
        "seeds": list(seeds),
        "capacities": list(capacities),
        "budgets": list(budgets),
        "corruptions": list(corruptions),
        "roots_per_family_per_seed": roots_per_family,
        "total_episodes_evaluated": total_episodes,
        "total_failed_episodes": total_failed,
        "wall_seconds": wall_total,
        "peak_process_rss_bytes": peak_rss,
        "peak_process_rss_mib": round(peak_rss / (1024 * 1024), 2) if peak_rss else None,
        "parameter_counts": {
            "54K_switch": AdaptiveGeometryController(CONFIG_54K).parameter_summary(True),
            "54K_fixed": AdaptiveGeometryController(CONFIG_54K_FIXED).parameter_summary(False),
            "54K_nocomm": AdaptiveGeometryController(CONFIG_54K_NOCOMM).parameter_summary(True),
            "116K_switch": AdaptiveGeometryController(CONFIG_116K).parameter_summary(True),
        },
        "train_telemetry": train_telemetry,
        "gates": gates_report,
        "native_status": native_report["status"] if native_report else "not_run",
        "receipt_hashes": receipt_hashes,
        "rows": summary_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    brief = {
        "mode": summary["mode"],
        "wall_seconds": round(wall_total, 2),
        "peak_process_rss_mib": summary["peak_process_rss_mib"],
        "total_episodes": total_episodes,
        "primary_gate": gates_report["primary_gate"]["status"],
        "primary_delta_vs_best_fixed_all_pp": round(
            100.0 * gates_report["primary_gate"]["delta_vs_best_fixed_all_bootstrap"]["point"], 2
        ),
        "secondary_gate": gates_report["secondary_gate"]["status"],
        "secondary_read_reduction_pct": round(
            100.0 * gates_report["secondary_gate"]["read_reduction_bootstrap"]["point"], 2
        ),
        "secondary_comp_loss_pp": round(
            100.0 * gates_report["secondary_gate"]["completion_loss_bootstrap"]["point"], 2
        ),
        "native_status": summary["native_status"],
    }
    print(json.dumps(brief, indent=2), flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="MCP-07 Geometry-Adaptive Recurrent Memory Routing")
    p.add_argument("--smoke", action="store_true", help="Run fast smoke verification sweep")
    p.add_argument("--threads", type=int, default=6, help="CPU worker/thread count (4-8 recommended)")
    p.add_argument("--out", required=True, help="Fresh output directory for run receipts")
    p.add_argument("--native-world", default=str(DEFAULT_NATIVE_WORLD))
    p.add_argument("--skip-128k", action="store_true", help="Skip optional 128K capacity")
    p.add_argument("--skip-ablations", action="store_true")
    p.add_argument("--skip-mechanistic", action="store_true")
    run_experiment(p.parse_args())


if __name__ == "__main__":
    main()
