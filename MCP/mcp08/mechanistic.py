"""Cross-domain mechanistic probes, activation patching, and inference-time lesions for MCP-08.

Investigates whether the persistent slow recurrent state `s_t` encodes transferable search
progress across three distinct verifier-backed domains:
  1. Native Storyworld (two-qubit Hilbert algebra, 5-50 decisions)
  2. Canvas Search (2D layout/collision/aesthetic verification)
  3. Lean-Style Proof Search (symbolic subgoal-stack tactic kernel)

Produces:
  - 3 x 3 zero-shot cross-domain linear probe transfer matrices for BOTH slow state (`s_t`)
    and fast state (`h_t`) across `progress`, `deficit`, `verify_fail`, and `u_prob`
  - 6-pair cross-domain slow-state activation patching (early vs late vs post-failure prototypes)
    measuring direct causal steering of uncertainty, compute tier, meta-action, and live episodes
  - All 7 paired inference-time lesions on frozen `mcp08_full` checkpoints for side-by-side
    comparison against trained-from-scratch ablations.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from MCP.mcp07.eval_harness import paired_bootstrap_ci
from .domains import (
    CanvasSearchEnv,
    LeanSearchEnv,
    generate_canvas_world,
    generate_lean_world,
)
from .memory256k import ExtendedPhysicalMemory
from .models import MCP08Config, MCP08RecursiveController
from .search import SearchLesionSpec, solve_domain_episode
from .storyworld_bench import (
    HORIZON_BUCKETS,
    StoryworldSearchEnv,
    generate_independent_storyworld,
    strip_world_witnesses,
)

DOMAINS: Tuple[str, ...] = ("storyworld", "canvas", "lean")
RIDGE_ALPHA: float = 25.0
LESION_NAMES: Tuple[str, ...] = (
    "lesion_reset_slow",
    "lesion_shuffle_slow",
    "lesion_uniform_compute",
    "lesion_no_postfail_switch",
    "lesion_no_backtrack",
    "lesion_no_cache",
    "lesion_corrupt_uncertainty",
)
PROBE_TARGETS: Tuple[str, ...] = ("progress", "deficit", "verify_fail", "u_prob")


def _fit_ridge(x_tr: np.ndarray, y_tr: np.ndarray, alpha: float = RIDGE_ALPHA) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x_tr.mean(axis=0, keepdims=True)
    sd = x_tr.std(axis=0, keepdims=True) + 1e-5
    xn = np.hstack([(x_tr - mu) / sd, np.ones((len(x_tr), 1))])
    reg = alpha * np.eye(xn.shape[1])
    reg[-1, -1] = 0.0
    w = np.linalg.solve(xn.T @ xn + reg, xn.T @ y_tr)
    return w, mu, sd


def _eval_ridge_r2(w: np.ndarray, mu: np.ndarray, sd: np.ndarray, x_te: np.ndarray, y_te: np.ndarray) -> float:
    if len(x_te) < 3 or float(np.std(y_te)) < 1e-8:
        return 0.0
    xn = np.hstack([(x_te - mu) / sd, np.ones((len(x_te), 1))])
    pred = xn @ w
    ss_res = float(np.sum((y_te - pred) ** 2))
    ss_tot = float(np.sum((y_te - y_te.mean()) ** 2)) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def _eval_pearson_r(w: np.ndarray, mu: np.ndarray, sd: np.ndarray, x_te: np.ndarray, y_te: np.ndarray) -> float:
    if len(x_te) < 3 or float(np.std(y_te)) < 1e-8:
        return 0.0
    xn = np.hstack([(x_te - mu) / sd, np.ones((len(x_te), 1))])
    pred = xn @ w
    if float(np.std(pred)) < 1e-9:
        return 0.0
    return float(np.corrcoef(pred, y_te)[0, 1])


def _build_domain_envs(
    domain: str,
    mem: ExtendedPhysicalMemory,
    n_worlds: int,
    seed: int,
    namespace: int,
) -> List[Any]:
    envs: List[Any] = []
    for k in range(n_worlds):
        wseed = namespace + seed * 1000 + k * 37
        if domain == "storyworld":
            bname, lo, hi = HORIZON_BUCKETS[k % len(HORIZON_BUCKETS)]
            horizon = min(40, lo + ((k * 5) % max(1, hi - lo + 1)))
            raw = generate_independent_storyworld(wseed, horizon, bname)
            envs.append(StoryworldSearchEnv(strip_world_witnesses(raw), mem, wseed + 7))
        elif domain == "canvas":
            horizon = 8 + (k % 4) * 4  # 8, 12, 16, 20
            raw = generate_canvas_world(wseed, horizon)
            envs.append(CanvasSearchEnv(strip_world_witnesses(raw), mem, wseed + 11))
        elif domain == "lean":
            horizon = 8 + (k % 4) * 4  # 8, 12, 16, 20
            raw = generate_lean_world(wseed, horizon)
            envs.append(LeanSearchEnv(strip_world_witnesses(raw), mem, wseed + 17))
        else:
            raise ValueError(f"Unknown domain: {domain}")
    return envs


@torch.inference_mode()
def _measure_prototype_steering(
    model: MCP08RecursiveController,
    donor_early: np.ndarray,
    donor_late: np.ndarray,
    donor_fail: np.ndarray,
    recip_samples: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Measure direct causal steering on recipient controller heads when patching cross-domain slow vectors."""
    if not recip_samples:
        return {
            "exp_tier_early": 1.0,
            "exp_tier_late": 1.0,
            "exp_tier_fail": 1.0,
            "delta_tier_fail_minus_late": 0.0,
            "delta_tier2_prob_fail_minus_late": 0.0,
            "delta_u_prob_fail_minus_late": 0.0,
            "delta_u_prob_early_minus_late": 0.0,
            "delta_retry_bt_prob_fail_minus_late": 0.0,
            "delta_halt_prob_late_minus_early": 0.0,
        }

    bsz = min(64, len(recip_samples))
    init_obs = torch.zeros(bsz, model.cfg.d_obs)
    search_ctx = torch.zeros(bsz, 8)
    for i in range(bsz):
        sp = recip_samples[i]
        init_obs[i, 20] = 0.5  # neutral sx
        init_obs[i, 21] = 0.5  # neutral sy
        init_obs[i, 34] = float(sp["progress"])
        init_obs[i, 35] = float(sp["deficit"])
        init_obs[i, 36] = float(sp["verify_fail"])
        search_ctx[i] = init_obs[i, 34:42]

    def eval_with_s(s_vec: np.ndarray) -> Tuple[float, float, float, float, float]:
        s_t = torch.from_numpy(s_vec.astype(np.float32))[None, :].expand(bsz, -1).clone()
        t_logits = model.allocate_compute_tier(s_t, init_obs)
        t_probs = torch.softmax(t_logits, dim=-1)
        exp_tier = float((t_probs * torch.tensor([0.0, 1.0, 2.0])).sum(dim=-1).mean().item())
        p_tier2 = float(t_probs[:, 2].mean().item())

        st0 = model.initial_state(bsz, persist_s=s_t)
        _, _, u_logit, st1 = model.step_recurrent(init_obs, st0, t=0, active_proj=None, override_slow=s_t)
        u_prob = torch.sigmoid(u_logit)
        meta_probs = torch.softmax(model.predict_meta_action(st1, u_prob, search_ctx), dim=-1)
        p_retry_bt = float((meta_probs[:, 1] + meta_probs[:, 2]).mean().item())
        p_halt = float(meta_probs[:, 3].mean().item())
        return exp_tier, p_tier2, float(u_prob.mean().item()), p_retry_bt, p_halt

    t_early, p2_early, u_early, rb_early, h_early = eval_with_s(donor_early)
    t_late, p2_late, u_late, rb_late, h_late = eval_with_s(donor_late)
    t_fail, p2_fail, u_fail, rb_fail, h_fail = eval_with_s(donor_fail)

    return {
        "exp_tier_early": t_early,
        "exp_tier_late": t_late,
        "exp_tier_fail": t_fail,
        "delta_tier_fail_minus_late": t_fail - t_late,
        "delta_tier2_prob_fail_minus_late": p2_fail - p2_late,
        "delta_u_prob_fail_minus_late": u_fail - u_late,
        "delta_u_prob_early_minus_late": u_early - u_late,
        "delta_retry_bt_prob_fail_minus_late": rb_fail - rb_late,
        "delta_halt_prob_late_minus_early": h_late - h_early,
    }


def _run_mechanistic_single_seed(job: Dict[str, Any]) -> Dict[str, Any]:
    """Worker-safe per-seed mechanistic evaluation (Storyworld, Canvas, Lean + 7 lesions)."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    seed = int(job["seed"])
    capacity = int(job["capacity"])
    worlds_per_domain = int(job["worlds_per_domain"])
    corruption = float(job["corruption"])
    namespace = int(job["namespace"])

    if "model" in job and job["model"] is not None:
        model = job["model"]
    else:
        model = MCP08RecursiveController(job["cfg"])
        model.load_state_dict(torch.load(job["ckpt_path"], map_location="cpu", weights_only=True))
    model.eval()
    d_s = model.cfg.d_s

    mem = ExtendedPhysicalMemory(capacity, namespace + seed * 100 + capacity)
    domain_envs: Dict[str, List[Any]] = {}
    domain_traces_tr: Dict[str, List[Dict[str, Any]]] = {d: [] for d in DOMAINS}
    domain_traces_te: Dict[str, List[Dict[str, Any]]] = {d: [] for d in DOMAINS}
    domain_ok: Dict[str, List[int]] = {d: [] for d in DOMAINS}
    domain_reads: Dict[str, List[float]] = {d: [] for d in DOMAINS}
    sw_base_recs: List[Dict[str, Any]] = []

    half = max(1, worlds_per_domain // 2)
    for dom in DOMAINS:
        envs = _build_domain_envs(dom, mem, worlds_per_domain, seed, namespace + 50000 * DOMAINS.index(dom))
        domain_envs[dom] = envs
        for k, env in enumerate(envs):
            rec = solve_domain_episode(
                env,
                arm="mcp08_full",
                mcp08_model=model,
                corruption=corruption,
                seed=seed + k * 13,
                record_slow_trace=True,
            )
            domain_ok[dom].append(int(rec["success"]))
            domain_reads[dom].append(float(rec["charged_memory_reads"]))
            if dom == "storyworld":
                sw_base_recs.append(rec)
            tr_list = domain_traces_tr[dom] if k < half else domain_traces_te[dom]
            tr_list.extend(rec["_slow_trace"])

    # Fit 3 x 3 probes for BOTH slow state ("s") and fast state ("h")!
    rng_null = np.random.default_rng(seed + 888)
    probe_mats: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
        state_key: {
            tgt: {
                "r2": np.zeros((3, 3), dtype=np.float64),
                "corr": np.zeros((3, 3), dtype=np.float64),
                "null_r2": np.zeros((3, 3), dtype=np.float64),
            }
            for tgt in PROBE_TARGETS
        }
        for state_key in ("s", "h")
    }

    for state_key in ("s", "h"):
        for tgt in PROBE_TARGETS:
            for i_src, d_src in enumerate(DOMAINS):
                tr_s = domain_traces_tr[d_src]
                if len(tr_s) < 6:
                    continue
                X_tr = np.asarray([row[state_key] for row in tr_s], dtype=np.float64)
                y_tr = np.asarray([row[tgt] for row in tr_s], dtype=np.float64)
                if float(np.std(y_tr)) < 1e-8:
                    continue
                w, mu, sd = _fit_ridge(X_tr, y_tr, alpha=RIDGE_ALPHA)
                y_tr_perm = rng_null.permutation(y_tr)
                w_null, mu_n, sd_n = _fit_ridge(X_tr, y_tr_perm, alpha=RIDGE_ALPHA)

                for j_tgt, d_tgt in enumerate(DOMAINS):
                    te_s = domain_traces_te[d_tgt]
                    if len(te_s) < 6:
                        continue
                    X_te = np.asarray([row[state_key] for row in te_s], dtype=np.float64)
                    y_te = np.asarray([row[tgt] for row in te_s], dtype=np.float64)
                    probe_mats[state_key][tgt]["r2"][i_src, j_tgt] = _eval_ridge_r2(w, mu, sd, X_te, y_te)
                    probe_mats[state_key][tgt]["corr"][i_src, j_tgt] = _eval_pearson_r(w, mu, sd, X_te, y_te)
                    probe_mats[state_key][tgt]["null_r2"][i_src, j_tgt] = _eval_ridge_r2(w_null, mu_n, sd_n, X_te, y_te)

    # Extract early/late/fail slow-state prototypes per domain (safely guarded against empty slices)
    protos: Dict[str, Dict[str, np.ndarray]] = {}
    for dom in DOMAINS:
        all_tr = domain_traces_tr[dom] + domain_traces_te[dom]
        if not all_tr:
            z = np.zeros(d_s, dtype=np.float32)
            protos[dom] = {"early": z, "late": z, "fail": z, "mean": z}
            continue
        S_all = np.asarray([r["s"] for r in all_tr], dtype=np.float32)
        P_all = np.asarray([r["progress"] for r in all_tr], dtype=np.float32)
        F_all = np.asarray([r["verify_fail"] for r in all_tr], dtype=np.float32)
        s_mean = S_all.mean(axis=0)
        early_m = P_all <= float(np.quantile(P_all, 0.30))
        late_m = P_all >= float(np.quantile(P_all, 0.70))
        fail_m = F_all > 0.5
        s_early = S_all[early_m].mean(axis=0) if bool(early_m.any()) else s_mean
        s_late = S_all[late_m].mean(axis=0) if bool(late_m.any()) else s_mean
        s_fail = S_all[fail_m].mean(axis=0) if bool(fail_m.any()) else s_early
        protos[dom] = {"early": s_early, "late": s_late, "fail": s_fail, "mean": s_mean}

    # 6-pair cross-domain activation patching
    pair_steering: Dict[str, Dict[str, float]] = {}
    pair_live_patch: Dict[str, Dict[str, List[float]]] = {}
    for d_src in DOMAINS:
        for d_tgt in DOMAINS:
            if d_src == d_tgt:
                continue
            pkey = f"{d_src}_to_{d_tgt}"
            pair_steering[pkey] = _measure_prototype_steering(
                model,
                protos[d_src]["early"],
                protos[d_src]["late"],
                protos[d_src]["fail"],
                domain_traces_te[d_tgt],
            )
            lp = {"base_ok": [], "patch_late_ok": [], "patch_fail_ok": [], "patch_late_reads": [], "patch_fail_reads": []}
            test_envs = domain_envs[d_tgt][half : min(len(domain_envs[d_tgt]), half + 4)]
            for k_e, env in enumerate(test_envs):
                r_b = solve_domain_episode(env, "mcp08_full", model, corruption=corruption, seed=seed + 500 + k_e)
                r_l = solve_domain_episode(
                    env,
                    "mcp08_full",
                    model,
                    corruption=corruption,
                    seed=seed + 500 + k_e,
                    lesion=SearchLesionSpec(name=f"{pkey}_late", patch_slow_vec=protos[d_src]["late"], patch_alpha=0.8),
                )
                r_f = solve_domain_episode(
                    env,
                    "mcp08_full",
                    model,
                    corruption=corruption,
                    seed=seed + 500 + k_e,
                    lesion=SearchLesionSpec(name=f"{pkey}_fail", patch_slow_vec=protos[d_src]["fail"], patch_alpha=0.8),
                )
                lp["base_ok"].append(float(r_b["success"]))
                lp["patch_late_ok"].append(float(r_l["success"]))
                lp["patch_fail_ok"].append(float(r_f["success"]))
                lp["patch_late_reads"].append(float(r_l["charged_memory_reads"]))
                lp["patch_fail_reads"].append(float(r_f["charged_memory_reads"]))
            pair_live_patch[pkey] = lp

    # 7 paired inference-time lesions on Storyworld envs
    shuf_vec = protos["lean"]["fail"]
    lesion_specs = [
        ("lesion_reset_slow", SearchLesionSpec(name="lesion_reset_slow", reset_slow_per_step=True)),
        ("lesion_shuffle_slow", SearchLesionSpec(name="lesion_shuffle_slow", patch_slow_vec=shuf_vec, patch_alpha=1.0)),
        ("lesion_uniform_compute", SearchLesionSpec(name="lesion_uniform_compute", uniform_compute_tier=1)),
        ("lesion_no_postfail_switch", SearchLesionSpec(name="lesion_no_postfail_switch", disable_postfail_switch=True)),
        ("lesion_no_backtrack", SearchLesionSpec(name="lesion_no_backtrack", disable_backtrack=True)),
        ("lesion_no_cache", SearchLesionSpec(name="lesion_no_cache", disable_cache=True)),
        ("lesion_corrupt_uncertainty", SearchLesionSpec(name="lesion_corrupt_uncertainty", corrupt_uncertainty=True)),
    ]
    lesion_ok: Dict[str, List[int]] = {ln: [] for ln in LESION_NAMES}
    lesion_reads: Dict[str, List[float]] = {ln: [] for ln in LESION_NAMES}
    for ln, lspec in lesion_specs:
        for k, env in enumerate(domain_envs["storyworld"]):
            r_iv = solve_domain_episode(
                env, "mcp08_full", model, corruption=corruption, seed=seed + k * 13, lesion=lspec
            )
            lesion_ok[ln].append(int(r_iv["success"]))
            lesion_reads[ln].append(float(r_iv["charged_memory_reads"]))

    return {
        "seed": seed,
        "domain_ok": domain_ok,
        "domain_reads": domain_reads,
        "trace_counts": {d: {"train": len(domain_traces_tr[d]), "test": len(domain_traces_te[d])} for d in DOMAINS},
        "probe_mats": probe_mats,
        "pair_steering": pair_steering,
        "pair_live_patch": pair_live_patch,
        "sw_base_ok": [int(r["success"]) for r in sw_base_recs],
        "sw_base_reads": [float(r["charged_memory_reads"]) for r in sw_base_recs],
        "lesion_ok": lesion_ok,
        "lesion_reads": lesion_reads,
    }


def _summarize_probe_mats(
    seed_results: Sequence[Dict[str, Any]], state_key: str
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for tgt in PROBE_TARGETS:
        r2_stack = np.stack([sr["probe_mats"][state_key][tgt]["r2"] for sr in seed_results], axis=0)
        corr_stack = np.stack([sr["probe_mats"][state_key][tgt]["corr"] for sr in seed_results], axis=0)
        null_stack = np.stack([sr["probe_mats"][state_key][tgt]["null_r2"] for sr in seed_results], axis=0)
        mean_r2 = r2_stack.mean(axis=0)
        mean_corr = corr_stack.mean(axis=0)
        mean_null = null_stack.mean(axis=0)
        diag_r2 = float(np.mean([mean_r2[i, i] for i in range(3)]))
        off_r2 = float(np.mean([mean_r2[i, j] for i in range(3) for j in range(3) if i != j]))
        off_corr = float(np.mean([mean_corr[i, j] for i in range(3) for j in range(3) if i != j]))
        off_null = float(np.mean([mean_null[i, j] for i in range(3) for j in range(3) if i != j]))
        out[tgt] = {
            "domains": list(DOMAINS),
            "r2_matrix": [[round(float(x), 4) for x in row] for row in mean_r2],
            "pearson_r_matrix": [[round(float(x), 4) for x in row] for row in mean_corr],
            "null_r2_matrix": [[round(float(x), 4) for x in row] for row in mean_null],
            "within_domain_mean_r2": diag_r2,
            "cross_domain_offdiag_mean_r2": off_r2,
            "cross_domain_offdiag_mean_r": off_corr,
            "cross_domain_offdiag_null_r2": off_null,
        }
    return out


def run_cross_domain_mechanistic(
    models_by_seed: Dict[int, MCP08RecursiveController],
    seeds: Sequence[int] = (107, 109, 113),
    capacity: int = 32768,
    worlds_per_domain_per_seed: int = 12,
    corruption: float = 0.10,
    namespace: int = 4400000,
    ckpt_paths_by_seed: Optional[Dict[int, str]] = None,
    workers: int = 1,
) -> Dict[str, Any]:
    """Run full cross-domain linear probing (3x3 slow & fast), activation patching (6 pairs), and 7 lesions."""
    jobs: List[Dict[str, Any]] = []
    for seed in seeds:
        use_pool = (workers > 1) and (len(seeds) > 1) and (ckpt_paths_by_seed is not None)
        jobs.append(
            {
                "seed": int(seed),
                "capacity": int(capacity),
                "worlds_per_domain": int(worlds_per_domain_per_seed),
                "corruption": float(corruption),
                "namespace": int(namespace),
                "model": None if use_pool else models_by_seed[seed],
                "cfg": models_by_seed[seed].cfg,
                "ckpt_path": ckpt_paths_by_seed[seed] if ckpt_paths_by_seed is not None else None,
            }
        )

    if workers > 1 and len(jobs) > 1 and ckpt_paths_by_seed is not None:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            seed_results = list(ex.map(_run_mechanistic_single_seed, jobs))
    else:
        seed_results = [_run_mechanistic_single_seed(j) for j in jobs]

    domain_completion: Dict[str, List[int]] = {d: [] for d in DOMAINS}
    domain_reads: Dict[str, List[float]] = {d: [] for d in DOMAINS}
    trace_counts: Dict[str, Dict[str, int]] = {d: {"train": 0, "test": 0} for d in DOMAINS}
    pair_steering: Dict[str, List[Dict[str, float]]] = {}
    pair_live_patch: Dict[str, Dict[str, List[float]]] = {}
    pooled_sw_base_ok: List[int] = []
    pooled_sw_base_reads: List[float] = []
    pooled_lesion_ok: Dict[str, List[int]] = {ln: [] for ln in LESION_NAMES}
    pooled_lesion_reads: Dict[str, List[float]] = {ln: [] for ln in LESION_NAMES}

    for sr in seed_results:
        for d in DOMAINS:
            domain_completion[d].extend(sr["domain_ok"][d])
            domain_reads[d].extend(sr["domain_reads"][d])
            trace_counts[d]["train"] += sr["trace_counts"][d]["train"]
            trace_counts[d]["test"] += sr["trace_counts"][d]["test"]
        for pkey, st_d in sr["pair_steering"].items():
            pair_steering.setdefault(pkey, []).append(st_d)
        for pkey, lp_d in sr["pair_live_patch"].items():
            acc = pair_live_patch.setdefault(
                pkey, {"base_ok": [], "patch_late_ok": [], "patch_fail_ok": [], "patch_late_reads": [], "patch_fail_reads": []}
            )
            for k, vals in lp_d.items():
                acc[k].extend(vals)
        pooled_sw_base_ok.extend(sr["sw_base_ok"])
        pooled_sw_base_reads.extend(sr["sw_base_reads"])
        for ln in LESION_NAMES:
            pooled_lesion_ok[ln].extend(sr["lesion_ok"][ln])
            pooled_lesion_reads[ln].extend(sr["lesion_reads"][ln])

    slow_transfer = _summarize_probe_mats(seed_results, "s")
    fast_transfer = _summarize_probe_mats(seed_results, "h")

    patching_summary: Dict[str, Any] = {}
    for pkey, st_list in pair_steering.items():
        lp = pair_live_patch[pkey]
        patching_summary[pkey] = {
            "steering": {k: float(np.mean([x[k] for x in st_list])) for k in st_list[0].keys()},
            "live_episodes": {
                "n": len(lp["base_ok"]),
                "baseline_completion": float(np.mean(lp["base_ok"])),
                "patch_late_completion": float(np.mean(lp["patch_late_ok"])),
                "patch_fail_completion": float(np.mean(lp["patch_fail_ok"])),
                "patch_late_mean_reads": float(np.mean(lp["patch_late_reads"])),
                "patch_fail_mean_reads": float(np.mean(lp["patch_fail_reads"])),
                "delta_reads_fail_minus_late": float(np.mean(lp["patch_fail_reads"]) - np.mean(lp["patch_late_reads"])),
            },
        }

    base_arr = np.asarray(pooled_sw_base_ok, dtype=np.float64)
    base_rd = np.asarray(pooled_sw_base_reads, dtype=np.float64)
    lesions_summary: Dict[str, Any] = {}
    for ln in LESION_NAMES:
        iv_arr = np.asarray(pooled_lesion_ok[ln], dtype=np.float64)
        iv_rd = np.asarray(pooled_lesion_reads[ln], dtype=np.float64)
        lesions_summary[ln] = {
            "baseline_completion": float(base_arr.mean()),
            "lesioned_completion": float(iv_arr.mean()),
            "delta_completion_bootstrap": paired_bootstrap_ci(iv_arr, base_arr, seed=8100 + len(ln)),
            "baseline_mean_reads": float(base_rd.mean()),
            "lesioned_mean_reads": float(iv_rd.mean()),
            "delta_reads_mean": float(iv_rd.mean() - base_rd.mean()),
        }

    return {
        "capacity": capacity,
        "corruption": corruption,
        "ridge_alpha": RIDGE_ALPHA,
        "seeds": list(seeds),
        "worlds_per_domain_per_seed": worlds_per_domain_per_seed,
        "trace_sample_counts": trace_counts,
        "domain_baseline_performance": {
            d: {
                "n_worlds": len(domain_completion[d]),
                "completion": float(np.mean(domain_completion[d])),
                "mean_charged_reads": float(np.mean(domain_reads[d])),
            }
            for d in DOMAINS
        },
        "cross_domain_linear_probes": slow_transfer,
        "cross_domain_linear_probes_fast_h": fast_transfer,
        "cross_domain_activation_patching": patching_summary,
        "inference_time_lesions": lesions_summary,
    }
