"""Stage 5: mixture of loops.  A typed skill router over executable experts.

Episode (root r): five typed stages.
  R  retrieval on a *noisy* observed graph (``hybrid.make('noisy:<family>')``) -> typed
     handoff: the retrieved node set S (+ per-node confidence)
  D  deduction on a dominance DAG over the same 16 slots, restricted to the retrieved S;
     the label is the kernel over the TRUE S, so retrieval errors propagate
  P  repair of a rejected 2-colouring (complete observation)
  S  stopping decision (GYM-01 stopping task)
  Q  research experiment choice (GYM-01 research task)
Utility = 0.2 per correct stage + 0.2 bonus if all five are correct, minus metered cost:
1e-8 per model FLOP, 1e-5 per host edge examination / tensor MAC unit, 0.01 per verifier
call, 0.05 per Inky call.  Only experts that actually execute are metered.

Experts (each behind the same typed contract: observation tensors in, 0/1 decision,
confidence, metered cost out):
  symbolic  exact solver on the public observation (graph search on the thresholded
            noisy graph for R/D, parity BFS for P, finite-horizon DP for S, exact EIG for Q)
  cert      sound three-valued certificate + constant fill (R/D only; uncertainty = unresolved)
  neural    GYM-01 packaged modules: retrieval D-TRM, deduction C-GRU, repair C-GRU,
            stopping C-GRU; research FF checkpoint (B arm)
  heuristic GYM-01 existing-skill procedure
Dispatch modes: fixed_neural, fixed_symbolic, learned (ridge success predictors on public
features, fitted on calibration episodes where every expert is run offline to obtain
training labels), confidence_gated (cheapest expert first; escalate while its own
uncertainty is high), oracle (hindsight cheapest correct expert; an upper bound that is
not realizable).  Composition for R->D: isolated typed handoff (default) vs latent
transfer (the retrieval latent mapped by a ridge adapter into the deduction GRU's
initial state).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from research_gym.trm_skill_gym import campaign as C, modules as M, tasks as T
from . import graphs as G
from . import hybrid as H
from . import supervisor as SUP

CAL_ROOT, TEST_ROOT = 7_100_000, 7_200_000
PKG = Path("experiments/trm_skill_gym_01/packages")
CKPT = Path("experiments/trm_skill_gym_01/run/checkpoints")
EXPERT_FILES = {"retrieval": PKG / "retrieval_D_trm_direct" / "weights.pt", "deduction": PKG / "deduction_C_gru" / "weights.pt",
                "repair": PKG / "repair_C_gru" / "weights.pt", "stopping": PKG / "stopping_C_gru" / "weights.pt",
                "research": CKPT / "research__B_feedforward__s0.pt"}
STAGES = ("retrieval", "deduction", "repair", "stopping", "research")
EXPERTS = {"retrieval": ("symbolic", "cert", "neural", "heuristic"), "deduction": ("symbolic", "cert", "neural", "heuristic"),
           "repair": ("symbolic", "neural", "heuristic"), "stopping": ("symbolic", "neural", "heuristic"),
           "research": ("symbolic", "neural", "heuristic")}
COST = {"flop": 1e-8, "host": 1e-5, "verifier": 0.01, "inky": 0.05}
FAMS = H.BASE_FAMILIES


def load_expert(skill):
    ck = torch.load(EXPERT_FILES[skill], map_location="cpu", weights_only=True)
    m = M.SkillModule(ck["config"])
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck["config"]


# ------------------------------------------------------------------ episodes
def episode_instances(root: int) -> Dict[str, T.Instance]:
    fr = FAMS["retrieval"][root % len(FAMS["retrieval"])]
    ret = H.make("retrieval", f"noisy:{fr}", root)
    fd = FAMS["deduction"][root % len(FAMS["deduction"])]
    ded = H.base_instance("deduction", fd, root + 1)
    rep = T.make_instance("repair", T.FAMILIES["repair"]["dev"][root % 2], root + 2)
    stp = T.make_instance("stopping", T.FAMILIES["stopping"]["dev"][root % 2], root + 3)
    res = T.make_instance("research", T.FAMILIES["research"]["dev"][root % 2], root + 4)
    return {"retrieval": ret, "deduction": ded, "repair": rep, "stopping": stp, "research": res}


def restrict_deduction(ded: T.Instance, keep: np.ndarray) -> T.Instance:
    """Typed handoff: keep only the retrieved slots; label = kernel over the TRUE retrieved set
    is applied by the caller.  Here: kernel of the restricted DAG for an arbitrary set."""
    out = T.Instance(cand=ded.cand.copy(), rel=ded.rel.copy(), glob=ded.glob.copy(), mask=ded.mask * keep,
                     label=np.zeros_like(ded.label), aux=ded.aux.copy(), auxg=ded.auxg.copy())
    out.rel = out.rel * keep[:, None] * keep[None, :]
    adj = out.rel > 0
    idx = np.nonzero(keep > 0)[0]
    if len(idx):
        sol = H.exact_solve("deduction", adj[np.ix_(idx, idx)], None)
        out.label[idx] = sol
    c = (out.rel > 0).astype(np.float32)
    out.cand[:, 1], out.cand[:, 2] = c.sum(1) / 3, c.sum(0) / 3
    out.cand[:, 3] = (c.sum(1) == 0) * out.mask
    out.cand[keep == 0] = 0
    return out


# ------------------------------------------------------------------ experts
def _single(inst: T.Instance) -> Dict[str, torch.Tensor]:
    return T.stack([inst])


def parity_repair(inst: T.Instance) -> Tuple[np.ndarray, int]:
    n = int(inst.mask.sum())
    adj = inst.rel[:n, :n] > 0
    col = inst.cand[:n, 0].astype(int)
    t = -np.ones(n, int)
    work = 0
    for s in range(n):
        if t[s] >= 0:
            continue
        t[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in np.nonzero(adj[u])[0]:
                work += 1
                if t[v] < 0:
                    t[v] = 1 - t[u]
                    q.append(int(v))
    f1 = (col != t).astype(np.float32)
    f2 = 1 - f1
    out = np.zeros(T.N, np.float32)
    out[:n] = f1 if f1.sum() <= f2.sum() else f2
    return out, work


def stopping_exact(inst: T.Instance) -> Tuple[np.ndarray, int]:
    c = inst.cand
    p = {"R": 1.0, "g_r": float(c[1, 5]), "delta": float(c[0, 8]), "g_t": float(c[2, 5]), "k_r": float(c[1, 4]),
         "k_t": float(c[2, 4]), "k_v": float(c[3, 4]), "c_rej": float(c[0, 9])}
    b = int(round(float(c[0, 7]) * 6))
    _, vals = T._stop_value(float(c[0, 6]), b, 0, bool(c[0, 10] > 0.5), p)
    out = np.zeros(T.N, np.float32)
    out[int(np.argmax(vals))] = 1
    return out, 4 ** max(b, 1)


def research_exact(inst: T.Instance) -> Tuple[np.ndarray, int]:
    n = int(inst.mask.sum())
    prior, lam = inst.glob[:6].astype(float), float(inst.glob[6])
    sc = [T._eig(prior, inst.cand[i, :6].astype(float)) - lam * float(inst.cand[i, 6]) for i in range(n)]
    out = np.zeros(T.N, np.float32)
    out[int(np.argmax(sc))] = 1
    return out, n * 12


def run_expert(skill: str, expert: str, inst: T.Instance, models: Dict) -> Dict:
    """Execute one expert; returns decision, uncertainty in [0,1], metered cost."""
    b = _single(inst)
    t0 = time.process_time()
    flops, host, unc = 0, 0, 0.0
    if expert == "symbolic":
        if skill in ("retrieval", "deduction"):
            n = int(inst.mask.sum())
            adj = inst.rel[:n, :n] > 0.5
            q = int(np.argmax(inst.cand[:n, 0])) if skill == "retrieval" else None
            dec = np.zeros(T.N, np.float32)
            idx = np.nonzero(inst.mask > 0)[0] if skill == "deduction" else np.arange(n)
            if skill == "deduction":
                sol = H.exact_solve("deduction", (inst.rel > 0.5)[np.ix_(idx, idx)], None)
                dec[idx] = sol
            else:
                dec[:n] = H.exact_solve(skill, adj, q)
            host = int((inst.rel > 0).sum())
            unc = float(((inst.rel > 0) & (inst.rel < 1)).sum() > 0) * 0.5
        elif skill == "repair":
            dec, host = parity_repair(inst)
        elif skill == "stopping":
            dec, host = stopping_exact(inst)
        else:
            dec, host = research_exact(inst)
    elif expert == "cert":
        q = int(np.argmax(inst.cand[:, 0])) if skill == "retrieval" else None
        st, _, host = H.certify3(skill, inst.rel, inst.mask, q)
        dec = np.zeros(T.N, np.float32)
        dec[: len(st)] = (st > 0)
        unres = (st == 0) & (inst.mask[: len(st)] > 0)
        unc = float(unres.mean()) if len(st) else 0.0
    elif expert == "heuristic":
        out = M.episode(None, skill, b, dict(C.base_trm(), ldt_projection=False, verifier_visits=0), heuristic=True)
        dec = out["pred"][-1][0].numpy()
        host = T.N * T.N
        unc = 0.5
    else:
        model, cfg = models[skill]
        M.flops_reset()
        with torch.no_grad():
            cand = b["cand"].clone()
            cand[..., T.FS:] = 0
            st = model.init_state(b)
            if models.get("_init_h") is not None and skill == "deduction" and "h" in st:
                st = dict(st, h=models["_init_h"])
            for _ in range(cfg["visits"]):
                logits, conf, value, st = model.visit(cand, b, st, cfg["visits"])
            dec = T.decide(skill, logits, b["mask"])[0].numpy()
            unc = float(1 - torch.sigmoid(conf)[0])
            models["_last_state"] = st
        flops = M.flops_read()
    cpu = time.process_time() - t0
    correct = float(T.exact(skill, b, torch.from_numpy(dec)[None] * b["mask"])[0])
    cost = COST["flop"] * flops + COST["host"] * host
    return {"dec": dec, "unc": unc, "flops": flops, "host": host, "cost": cost, "cpu_s": cpu, "correct": correct}


def public_features(skill: str, inst: T.Instance) -> np.ndarray:
    m = inst.mask
    n = max(m.sum(), 1)
    unc_edges = float(((inst.rel > 0) & (inst.rel < 1)).sum())
    cand = inst.cand[:, : T.FS]
    mean = (cand * m[:, None]).sum(0) / n
    return np.concatenate([[1.0, n / T.N, float((inst.rel > 0).sum()) / T.N ** 2, unc_edges / T.N], inst.glob, mean])


# ------------------------------------------------------------------ dispatch
def run_episode(root: int, mode: str, models: Dict, predictors: Dict, tau: float, adapter=None) -> Dict:
    inst = episode_instances(root)
    rows = {}
    total_cost = 0.0
    executed = 0
    retrieved = None
    for skill in STAGES:
        x = inst[skill]
        if skill == "deduction":
            keep = retrieved if retrieved is not None else np.zeros(T.N, np.float32)
            true_keep = inst["retrieval"].label
            xs = restrict_deduction(x, keep)
            truth = restrict_deduction(x, true_keep)
            xs.label = truth.label * xs.mask  # the label is the kernel over the TRUE retrieved set
            if not np.array_equal(keep, true_keep):
                xs.label = truth.label  # mismatched masks cannot be exactly correct
            x = xs
            models["_init_h"] = None
            if adapter is not None and "_ret_state" in models:
                z = models["_ret_state"]["y"][0].numpy()
                models["_init_h"] = torch.from_numpy(z @ adapter).float()[None]
        if mode == "oracle":
            outs = {e: run_expert(skill, e, x, models) for e in EXPERTS[skill]}
            ok = [e for e in outs if outs[e]["correct"] > 0]
            pick = min(ok, key=lambda e: outs[e]["cost"]) if ok else min(outs, key=lambda e: outs[e]["cost"])
            res = outs[pick]
            res["expert"], res["executed"] = pick, 1
        elif mode in ("fixed_neural", "fixed_symbolic"):
            e = "neural" if mode == "fixed_neural" else "symbolic"
            res = run_expert(skill, e, x, models)
            res["expert"], res["executed"] = e, 1
        elif mode == "learned":
            f = public_features(skill, x)
            scores = {e: float(f @ predictors[skill][e]) - predictors[skill]["_cost"][e] * 5 for e in EXPERTS[skill]}
            e = max(scores, key=scores.get)
            res = run_expert(skill, e, x, models)
            res["expert"], res["executed"] = e, 1
        else:  # confidence_gated: cheapest first, escalate on own uncertainty
            order = [e for e in ("cert", "symbolic", "neural") if e in EXPERTS[skill]]
            acc_cost, n_exec = 0.0, 0
            for e in order:
                res = run_expert(skill, e, x, models)
                acc_cost += res["cost"]
                n_exec += 1
                if res["unc"] <= (0.0 if e == "cert" else tau):
                    break
            res["cost"], res["expert"], res["executed"] = acc_cost, e, n_exec
        if skill == "retrieval":
            retrieved = res["dec"].astype(np.float32)
            if res["expert"] == "neural" or adapter is not None:
                if res["expert"] != "neural":
                    run_expert("retrieval", "neural", x, models)  # latent transfer needs the retrieval latent
                models["_ret_state"] = models.get("_last_state")
        total_cost += res["cost"]
        executed += res["executed"]
        rows[skill] = {k: res[k] for k in ("expert", "correct", "cost", "flops", "host", "cpu_s", "executed")}
    n_ok = sum(r["correct"] for r in rows.values())
    util = 0.2 * n_ok + (0.2 if n_ok == len(STAGES) else 0.0) - total_cost
    return {"root": root, "utility": util, "all_correct": float(n_ok == len(STAGES)), "stages": rows,
            "cost": total_cost, "experts_executed": executed}


def fit_predictors(roots: List[int], models: Dict) -> Dict:
    X = {s: [] for s in STAGES}
    Y = {s: {e: [] for e in EXPERTS[s]} for s in STAGES}
    costs = {s: {e: [] for e in EXPERTS[s]} for s in STAGES}
    for r in roots:
        inst = episode_instances(r)
        for skill in STAGES:
            x = inst[skill]
            if skill == "deduction":
                x = restrict_deduction(x, inst["retrieval"].label)
            X[skill].append(public_features(skill, x))
            for e in EXPERTS[skill]:
                o = run_expert(skill, e, x, models)
                Y[skill][e].append(o["correct"])
                costs[skill][e].append(o["cost"])
    pred = {}
    for s in STAGES:
        x = np.asarray(X[s])
        pred[s] = {e: np.linalg.solve(x.T @ x + 1.0 * np.eye(x.shape[1]), x.T @ np.asarray(Y[s][e])) for e in EXPERTS[s]}
        pred[s]["_cost"] = {e: float(np.mean(costs[s][e])) for e in EXPERTS[s]}
        pred[s]["_cal_success"] = {e: float(np.mean(Y[s][e])) for e in EXPERTS[s]}
    return pred


def fit_adapter(roots: List[int], models: Dict) -> np.ndarray:
    """Ridge map from the retrieval TRM answer state to the deduction GRU's final hidden state
    on the same slots (both on calibration episodes)."""
    Zs, Hs = [], []
    for r in roots:
        inst = episode_instances(r)
        run_expert("retrieval", "neural", inst["retrieval"], models)
        z = models["_last_state"]["y"][0].numpy()
        x = restrict_deduction(inst["deduction"], inst["retrieval"].label)
        models["_init_h"] = None
        run_expert("deduction", "neural", x, models)
        h = models["_last_state"]["h"][0].numpy()
        Zs.append(z)
        Hs.append(h)
    Z, Hh = np.concatenate(Zs), np.concatenate(Hs)
    return np.linalg.solve(Z.T @ Z + 1.0 * np.eye(Z.shape[1]), Z.T @ Hh)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cal", type=int, default=192)
    p.add_argument("--test", type=int, default=384)
    p.add_argument("--hours", type=float, default=0.75)
    a = p.parse_args(argv)
    torch.set_num_threads(1)
    a.out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 3600 * a.hours
    models = {s: load_expert(s) for s in STAGES}
    cal_roots = [CAL_ROOT + 10 * i for i in range(a.cal)]
    test_roots = [TEST_ROOT + 10 * i for i in range(a.test)]
    pred = fit_predictors(cal_roots, models)
    taus = [0.05, 0.2, 0.4]
    tau_u = {t: float(np.mean([run_episode(r, "confidence_gated", models, pred, t)["utility"] for r in cal_roots[:96]])) for t in taus}
    tau = max(tau_u, key=tau_u.get)
    adapter = fit_adapter(cal_roots[:96], models)
    frozen = {"tau": tau, "tau_cal_utility": tau_u, "cal_success": {s: pred[s]["_cal_success"] for s in STAGES},
              "cal_cost": {s: pred[s]["_cost"] for s in STAGES}}
    SUP.write_json(a.out / "frozen.json", frozen)
    modes = ["fixed_neural", "fixed_symbolic", "learned", "confidence_gated", "oracle"]
    results = {}
    for mode in modes + ["fixed_neural+latent_transfer"]:
        if time.time() > deadline:
            results[mode] = {"status": "deadline_not_run"}
            continue
        t0 = time.process_time()
        eps = [run_episode(r, mode.split("+")[0], models, pred, tau, adapter if "latent" in mode else None) for r in test_roots]
        cpu = time.process_time() - t0
        results[mode] = {"status": "ok", "utility": float(np.mean([e["utility"] for e in eps])),
                         "all_correct": float(np.mean([e["all_correct"] for e in eps])),
                         "per_root_utility": [round(e["utility"], 6) for e in eps],
                         "stage_correct": {s: float(np.mean([e["stages"][s]["correct"] for e in eps])) for s in STAGES},
                         "stage_expert_counts": {s: {x: sum(e["stages"][s]["expert"] == x for e in eps) for x in EXPERTS[s]} for s in STAGES},
                         "metered_cost": float(np.mean([e["cost"] for e in eps])),
                         "experts_executed_per_episode": float(np.mean([e["experts_executed"] for e in eps])),
                         "model_flops_per_episode": float(np.mean([sum(e["stages"][s]["flops"] for s in STAGES) for e in eps])),
                         "host_ops_per_episode": float(np.mean([sum(e["stages"][s]["host"] for s in STAGES) for e in eps])),
                         "cpu_s_total": cpu, "realizable": mode != "oracle"}
        print(mode, results[mode]["utility"], results[mode]["all_correct"], flush=True)
    base = np.asarray(results["fixed_neural"]["per_root_utility"])
    rng = np.random.default_rng(5)
    for mode in results:
        if results[mode].get("status") == "ok" and mode != "fixed_neural":
            d = np.asarray(results[mode]["per_root_utility"]) - base
            boot = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
            results[mode]["delta_vs_fixed_neural"] = {"mean": float(d.mean()), "ci95": np.quantile(boot, [0.025, 0.975]).tolist()}
    src = {f.name: hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for f in sorted(Path(__file__).parent.glob("*.py"))}
    SUP.write_json(a.out / "results.json", {"frozen": frozen, "results": results, "roots": {"cal": CAL_ROOT, "test": TEST_ROOT},
                                            "cost_weights": COST, "expert_files": {k: str(v) for k, v in EXPERT_FILES.items()},
                                            "source_sha256": src})


if __name__ == "__main__":
    main()
