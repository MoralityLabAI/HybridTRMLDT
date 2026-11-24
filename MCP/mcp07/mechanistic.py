"""Mechanistic interpretation, linear probing, and causal interventions for MCP-07.

Executes Section 7:
1. Extract router probabilities, slow/fast recurrent trajectories, and switch events at 32K.
2. Linear probes testing whether the slow state (vs fast state) encodes:
   - Query geometry (8-way family classification & dominant projection)
   - Search progress (normalized step, solved status, hop progress)
   - Uncertainty (router entropy)
3. Five paired causal interventions:
   - Freeze the projection router
   - Shuffle slow recurrent states across episodes
   - Remove cross-projection communication
   - Force incorrect projections (argmin router probability)
   - Patch slow states between matched (same-family) vs mismatched (cross-family) query families
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .curriculum import FAMILIES, FAMILY_TO_ID, QueryRoot, sample_family_roots
from .eval_harness import InterventionSpec, paired_bootstrap_ci, rollout_batch
from .memory import STRUCTURED_PROJECTIONS, SharedPhysicalMemory
from .models import AdaptiveGeometryController


def _ridge_regression_r2(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 1.0,
) -> float:
    """Closed-form L2-regularized linear probe returning held-out R^2."""
    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True) + 1e-6
    xtr = np.hstack([(x_train - mu) / sd, np.ones((len(x_train), 1))])
    xte = np.hstack([(x_test - mu) / sd, np.ones((len(x_test), 1))])
    dim = xtr.shape[1]
    reg = alpha * np.eye(dim)
    reg[-1, -1] = 0.0
    w = np.linalg.solve(xtr.T @ xtr + reg, xtr.T @ y_train)
    pred = xte @ w
    ss_res = float(np.sum((y_test - pred) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2)) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def _ridge_classifier_acc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    alpha: float = 1.0,
) -> float:
    """One-vs-rest L2-regularized linear probe returning held-out classification accuracy."""
    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True) + 1e-6
    xtr = np.hstack([(x_train - mu) / sd, np.ones((len(x_train), 1))])
    xte = np.hstack([(x_test - mu) / sd, np.ones((len(x_test), 1))])
    y_oh = np.eye(n_classes, dtype=np.float64)[y_train]
    dim = xtr.shape[1]
    reg = alpha * np.eye(dim)
    reg[-1, -1] = 0.0
    w = np.linalg.solve(xtr.T @ xtr + reg, xtr.T @ y_oh)
    preds = np.argmax(xte @ w, axis=1)
    return float(np.mean(preds == y_test))


def _router_entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-8, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def run_mechanistic_analysis(
    models_by_seed: Dict[int, AdaptiveGeometryController],
    seeds: Sequence[int] = (107, 109, 113),
    capacity: int = 32768,
    budget: int = 16,
    roots_per_family: int = 128,
    namespace: int = 2000000,
) -> Dict[str, Any]:
    """Execute full Section 7 mechanistic probing and causal intervention suite."""
    per_seed_results: Dict[str, Any] = {}

    # Cross-family pairing for slow-state patching (contrasting geometries!)
    cross_pair_map = {
        "horizontal": "vertical",
        "vertical": "horizontal",
        "diagonal": "isotropic",
        "isotropic": "diagonal",
        "sparse_long_range": "horizontal",
        "multihop": "vertical",
        "regime_shift": "diagonal",
        "corrupted_replica": "isotropic",
    }

    pooled_baseline_ok: List[int] = []
    pooled_iv_ok: Dict[str, List[int]] = {
        "freeze_router": [],
        "shuffle_slow_state": [],
        "remove_cross_proj_comm": [],
        "force_wrong_projection": [],
        "patch_slow_same_family": [],
        "patch_slow_cross_family": [],
    }
    pooled_fam_ids: List[int] = []
    pooled_steering_same: List[float] = []
    pooled_steering_cross: List[float] = []

    for seed in seeds:
        model = models_by_seed[seed]
        mem = SharedPhysicalMemory(capacity, namespace + 100 * seed + capacity)
        roots: List[QueryRoot] = []
        fam_slices: Dict[str, slice] = {}
        offset = 0
        for fam in FAMILIES:
            f_roots = sample_family_roots(mem, fam, roots_per_family, 331)
            roots.extend(f_roots)
            fam_slices[fam] = slice(offset, offset + len(f_roots))
            offset += len(f_roots)

        # 1. Baseline rollout with full trajectory extraction
        base_recs, base_traj = rollout_batch(
            mem,
            roots,
            arm="recurrent_switch",
            model=model,
            budget=budget,
            corruption=0.0,
            record_trajectories=True,
            rng_seed=seed * 100 + 1,
        )
        assert base_traj is not None
        base_ok = np.array([int(r["success"]) for r in base_recs], dtype=np.int32)
        fam_ids = np.array([r.family_id for r in roots], dtype=np.int32)
        pooled_baseline_ok.extend(base_ok.tolist())
        pooled_fam_ids.extend(fam_ids.tolist())

        # Router probability & switch summaries per family
        r_probs = base_traj["router_probs"]   # [T+1, B, 4]
        chosen_p = base_traj["chosen_proj"]   # [T+1, B]
        slow_all = base_traj["slow"]          # [T+1, B, d_s]
        fast_all = base_traj["fast"]          # [T+1, B, d_h]
        valid_all = base_traj["valid"]        # [T+1, B]
        switch_all = base_traj["switch"]      # [T+1, B]
        solved_all = base_traj["solved"]      # [T+1, B]
        hop_all = base_traj["hop"]            # [T+1, B]

        fam_telemetry: Dict[str, Any] = {}
        switch_matrix = np.zeros((4, 4), dtype=np.int64)
        for t in range(1, budget + 1):
            sw = switch_all[t]
            if sw.any():
                prev_p = chosen_p[t - 1, sw]
                curr_p = chosen_p[t, sw]
                for a_p, b_p in zip(prev_p, curr_p):
                    if 0 <= a_p < 4 and 0 <= b_p < 4:
                        switch_matrix[int(a_p), int(b_p)] += 1

        for fam in FAMILIES:
            sl = fam_slices[fam]
            sub_recs = base_recs[sl]
            sub_valid = valid_all[:, sl]
            sub_probs = r_probs[:, sl, :][sub_valid]
            sub_ent = _router_entropy(sub_probs)
            fam_telemetry[fam] = {
                "completion": float(np.mean([r["success"] for r in sub_recs])),
                "mean_switches": float(np.mean([r["switches"] for r in sub_recs])),
                "mean_router_probs": [float(x) for x in sub_probs.mean(axis=0)],
                "mean_router_entropy": float(sub_ent.mean()),
                "init_router_probs": [float(x) for r_p in [r_probs[0, sl, :].mean(axis=0)] for x in r_p],
            }

        # 2. Linear probes on slow state s_t vs fast state h_t (strict root-level 50/50 split)
        n_total = len(roots)
        split_mask = np.zeros(n_total, dtype=bool)
        for fam in FAMILIES:
            sl = fam_slices[fam]
            half = (sl.stop - sl.start) // 2
            split_mask[sl.start : sl.start + half] = True  # train half; False = held-out test half

        # Collect active step samples for train & test roots
        tr_s, te_s, tr_h, te_h = [], [], [], []
        tr_fam, te_fam = [], []
        tr_prog, te_prog = [], []
        tr_ent, te_ent = [], []
        tr_sol, te_sol = [], []
        tr_proj, te_proj = [], []

        for t in range(1, budget + 1):
            ent_t = _router_entropy(r_probs[t])
            step_frac = float(t) / float(budget)
            for i in range(n_total):
                if not valid_all[t, i]:
                    continue
                if split_mask[i]:
                    tr_s.append(slow_all[t, i])
                    tr_h.append(fast_all[t, i])
                    tr_fam.append(fam_ids[i])
                    tr_prog.append(step_frac)
                    tr_ent.append(ent_t[i])
                    tr_sol.append(int(solved_all[t, i]))
                    tr_proj.append(int(chosen_p[t, i]))
                else:
                    te_s.append(slow_all[t, i])
                    te_h.append(fast_all[t, i])
                    te_fam.append(fam_ids[i])
                    te_prog.append(step_frac)
                    te_ent.append(ent_t[i])
                    te_sol.append(int(solved_all[t, i]))
                    te_proj.append(int(chosen_p[t, i]))

        Xtr_s, Xte_s = np.asarray(tr_s, np.float64), np.asarray(te_s, np.float64)
        Xtr_h, Xte_h = np.asarray(tr_h, np.float64), np.asarray(te_h, np.float64)
        ytr_fam, yte_fam = np.asarray(tr_fam, np.int64), np.asarray(te_fam, np.int64)
        ytr_prog, yte_prog = np.asarray(tr_prog, np.float64), np.asarray(te_prog, np.float64)
        ytr_ent, yte_ent = np.asarray(tr_ent, np.float64), np.asarray(te_ent, np.float64)
        ytr_sol, yte_sol = np.asarray(tr_sol, np.int64), np.asarray(te_sol, np.int64)
        ytr_proj, yte_proj = np.asarray(tr_proj, np.int64), np.asarray(te_proj, np.int64)

        probes_report = {
            "slow_state": {
                "query_geometry_8way_acc": _ridge_classifier_acc(Xtr_s, ytr_fam, Xte_s, yte_fam, len(FAMILIES)),
                "active_projection_4way_acc": _ridge_classifier_acc(Xtr_s, ytr_proj, Xte_s, yte_proj, 4),
                "search_progress_step_r2": _ridge_regression_r2(Xtr_s, ytr_prog, Xte_s, yte_prog),
                "solved_state_acc": _ridge_classifier_acc(Xtr_s, ytr_sol, Xte_s, yte_sol, 2),
                "uncertainty_entropy_r2": _ridge_regression_r2(Xtr_s, ytr_ent, Xte_s, yte_ent),
            },
            "fast_state": {
                "query_geometry_8way_acc": _ridge_classifier_acc(Xtr_h, ytr_fam, Xte_h, yte_fam, len(FAMILIES)),
                "active_projection_4way_acc": _ridge_classifier_acc(Xtr_h, ytr_proj, Xte_h, yte_proj, 4),
                "search_progress_step_r2": _ridge_regression_r2(Xtr_h, ytr_prog, Xte_h, yte_prog),
                "solved_state_acc": _ridge_classifier_acc(Xtr_h, ytr_sol, Xte_h, yte_sol, 2),
                "uncertainty_entropy_r2": _ridge_regression_r2(Xtr_h, ytr_ent, Xte_h, yte_ent),
            },
            "chance_baselines": {
                "query_geometry_8way_acc": 1.0 / float(len(FAMILIES)),
                "active_projection_4way_acc": 0.25,
                "solved_state_majority_acc": float(max(yte_sol.mean(), 1.0 - yte_sol.mean())),
            },
        }

        # 3. Five causal interventions
        # Build donor slow-state tensors for same-family & cross-family patching
        donor_same = np.empty_like(slow_all)
        donor_cross = np.empty_like(slow_all)
        donor_cross_proj = np.empty_like(chosen_p)
        for fam in FAMILIES:
            sl = fam_slices[fam]
            # Roll by 1 within the same family so every recipient gets a DIFFERENT episode's slow state
            donor_same[:, sl, :] = np.roll(slow_all[:, sl, :], shift=1, axis=1)
            c_fam = cross_pair_map[fam]
            c_sl = fam_slices[c_fam]
            donor_cross[:, sl, :] = slow_all[:, c_sl, :]
            donor_cross_proj[:, sl] = chosen_p[:, c_sl]

        iv_specs = [
            ("freeze_router", InterventionSpec(name="freeze_router", freeze_router=True, seed=seed)),
            ("shuffle_slow_state", InterventionSpec(name="shuffle_slow_state", shuffle_slow_state=True, seed=seed + 7)),
            ("remove_cross_proj_comm", InterventionSpec(name="remove_cross_proj_comm", disable_comm=True, seed=seed)),
            ("force_wrong_projection", InterventionSpec(name="force_wrong_projection", force_wrong_proj=True, seed=seed)),
            ("patch_slow_same_family", InterventionSpec(name="patch_slow_same_family", patch_slow_donor=donor_same, seed=seed)),
            ("patch_slow_cross_family", InterventionSpec(name="patch_slow_cross_family", patch_slow_donor=donor_cross, seed=seed)),
        ]

        iv_summary: Dict[str, Any] = {}
        for iv_name, spec in iv_specs:
            iv_recs, iv_traj = rollout_batch(
                mem,
                roots,
                arm="recurrent_switch",
                model=model,
                budget=budget,
                corruption=0.0,
                intervention=spec,
                record_trajectories=iv_name.startswith("patch_slow_"),
                rng_seed=seed * 100 + 1,
            )
            iv_ok = np.array([int(r["success"]) for r in iv_recs], dtype=np.int32)
            pooled_iv_ok[iv_name].extend(iv_ok.tolist())

            fam_deltas = {}
            for fam in FAMILIES:
                sl = fam_slices[fam]
                fam_deltas[fam] = {
                    "baseline": float(base_ok[sl].mean()),
                    "intervened": float(iv_ok[sl].mean()),
                    "delta": float(iv_ok[sl].mean() - base_ok[sl].mean()),
                }

            entry: Dict[str, Any] = {
                "baseline_completion": float(base_ok.mean()),
                "intervened_completion": float(iv_ok.mean()),
                "delta_completion": float(iv_ok.mean() - base_ok.mean()),
                "mean_switches": float(np.mean([r["switches"] for r in iv_recs])),
                "per_family": fam_deltas,
            }
            if iv_name == "patch_slow_cross_family" and iv_traj is not None:
                # Measure causal steering: how often did recipient shift router probability toward donor's proj?
                iv_rp = iv_traj["router_probs"]  # [T+1, B, 4]
                d_prob_shifts = []
                for t in range(1, min(6, budget + 1)):
                    for i in range(n_total):
                        dp = int(donor_cross_proj[t, i])
                        if 0 <= dp < 4 and dp != int(chosen_p[t, i]):
                            d_prob_shifts.append(float(iv_rp[t, i, dp] - r_probs[t, i, dp]))
                steer = float(np.mean(d_prob_shifts)) if d_prob_shifts else 0.0
                entry["donor_proj_prob_shift"] = steer
                pooled_steering_cross.append(steer)
            elif iv_name == "patch_slow_same_family" and iv_traj is not None:
                # Recovery metric vs shuffled slow state
                iv_rp = iv_traj["router_probs"]
                l1_dev = float(np.mean(np.abs(iv_rp[1:6] - r_probs[1:6])))
                entry["router_prob_l1_deviation"] = l1_dev
                pooled_steering_same.append(l1_dev)

            iv_summary[iv_name] = entry

        per_seed_results[str(seed)] = {
            "family_telemetry": fam_telemetry,
            "switch_transition_matrix": switch_matrix.tolist(),
            "linear_probes": probes_report,
            "interventions": iv_summary,
        }

    # Pooled paired bootstrap intervals across all seeds
    base_arr = np.asarray(pooled_baseline_ok, dtype=np.float64)
    fam_arr = np.asarray(pooled_fam_ids, dtype=np.int32)
    pooled_interventions: Dict[str, Any] = {}
    for iv_name, vals in pooled_iv_ok.items():
        iv_arr = np.asarray(vals, dtype=np.float64)
        boot = paired_bootstrap_ci(iv_arr, base_arr, seed=9090 + len(iv_name))
        per_fam_pool = {}
        for fid, fam in enumerate(FAMILIES):
            m = fam_arr == fid
            per_fam_pool[fam] = {
                "baseline": float(base_arr[m].mean()),
                "intervened": float(iv_arr[m].mean()),
                "delta": float(iv_arr[m].mean() - base_arr[m].mean()),
            }
        pooled_interventions[iv_name] = {
            "baseline_completion": float(base_arr.mean()),
            "intervened_completion": float(iv_arr.mean()),
            "delta_bootstrap": boot,
            "per_family": per_fam_pool,
        }

    # Patch recovery ratio: how much of shuffle-slow drop is recovered by patching a SAME-family slow state
    shuf_drop = float(
        pooled_interventions["shuffle_slow_state"]["baseline_completion"]
        - pooled_interventions["shuffle_slow_state"]["intervened_completion"]
    )
    same_drop = float(
        pooled_interventions["patch_slow_same_family"]["baseline_completion"]
        - pooled_interventions["patch_slow_same_family"]["intervened_completion"]
    )
    cross_drop = float(
        pooled_interventions["patch_slow_cross_family"]["baseline_completion"]
        - pooled_interventions["patch_slow_cross_family"]["intervened_completion"]
    )
    recovery_vs_shuffle = (
        float((shuf_drop - same_drop) / shuf_drop) if abs(shuf_drop) > 1e-6 else None
    )
    recovery_vs_cross = (
        float((cross_drop - same_drop) / cross_drop) if abs(cross_drop) > 1e-6 else None
    )

    # Average linear probe metrics across seeds
    probe_means: Dict[str, Dict[str, float]] = {"slow_state": {}, "fast_state": {}}
    for kind in ("slow_state", "fast_state"):
        keys = per_seed_results[str(seeds[0])]["linear_probes"][kind].keys()
        for k in keys:
            probe_means[kind][k] = float(
                np.mean([per_seed_results[str(s)]["linear_probes"][kind][k] for s in seeds])
            )

    return {
        "capacity": capacity,
        "budget": budget,
        "roots_per_family_per_seed": roots_per_family,
        "seeds": list(seeds),
        "projections_order": list(STRUCTURED_PROJECTIONS),
        "pooled_linear_probes": probe_means,
        "pooled_interventions": pooled_interventions,
        "patch_recovery": {
            "shuffle_drop": shuf_drop,
            "cross_family_patch_drop": cross_drop,
            "same_family_patch_drop": same_drop,
            "same_vs_shuffle_recovery_fraction": recovery_vs_shuffle,
            "same_vs_cross_recovery_fraction": recovery_vs_cross,
            "mean_cross_donor_prob_shift": float(np.mean(pooled_steering_cross)) if pooled_steering_cross else 0.0,
            "mean_same_router_l1_dev": float(np.mean(pooled_steering_same)) if pooled_steering_same else 0.0,
        },
        "per_seed": per_seed_results,
    }
