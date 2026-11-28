"""Freeze MOLT-GOBLIN-03 finalists, then evaluate on hidden roots (HIDDEN3_BASE).

Arms: registered A-K (E and G resolved to matched widths), a process-CPU-matched arm A_cpu,
the loop-forcing factorial X_last on the principal families, the top Inky finalists, and,
when the best development mixture is not B, a student of that mixture (K_win). The freeze
file and its digest are written before any hidden-root run; research_plan is trained and
evaluated only here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .resources3 import install_job_cap, peak_job_bytes
from .arms import ALL, ARMS, PRINCIPAL, SEEDS_FINAL, TEACHER, resolved_arms
from .factory3 import Config
from .mech import gradient_similarity, multitask_routing, specialization
from .train3 import HIDDEN3_BASE, root_seed, run_cell, salt_of

MIXTURES = ("mix", "hrm", "sparse_st", "anneal", "sequential", "cycle")
MECH_FAMILIES = PRINCIPAL + ("proof_deps",)


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def freeze(state_path: Path, out: Path, k: int = 2) -> dict:
    p = out / "freeze.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    st = json.loads(state_path.read_text(encoding="utf-8"))
    arch = st["archive"]
    arms = {name: {"config": cfg.to_dict(), "kind": "registered", "teacher": None} for name, cfg in resolved_arms().items()}
    arms["K_distill"]["teacher"] = "B_hrm"
    for name in [n for n in arms if n.endswith("_last")]:
        arms[name]["kind"] = "loop_forcing_factorial"
        arms[name]["families"] = list(PRINCIPAL)
    # Process-CPU-matched single block: scale A's updates by the measured B/A train-CPU ratio on development runs.
    cpu = {"A": [], "B": []}
    for key, c in st["cache"].items():
        cid = key.split("|")[0]
        if c.get("status") == "ok" and cid in (ARMS["A_single"].digest(), ARMS["B_hrm"].digest()):
            cpu["A" if cid == ARMS["A_single"].digest() else "B"].append(c["train_cpu_s"])
    ratio = (sum(cpu["B"]) / len(cpu["B"])) / (sum(cpu["A"]) / len(cpu["A"])) if cpu["A"] and cpu["B"] else 4.0
    arms["A_cpu"] = {"config": ARMS["A_single"].replace(steps=int(round(ARMS["A_single"].steps * ratio))).to_dict(),
                     "kind": "cpu_matched_control", "teacher": None, "cpu_ratio_B_over_A": ratio}
    registered = {Config.from_dict(v["config"]).digest() for v in arms.values()}
    cands = [(cid, a) for cid, a in arch.items() if a["origin"]["kind"] in ("candidate", "control") and a["scores"]
             and cid not in registered]
    cands.sort(key=lambda it: (it[1]["rung"], it[1]["scores"][str(it[1]["rung"])].get("score") or -9), reverse=True)
    for cid, a in cands[:k]:
        cfg = Config.from_dict(a["config"])
        arms[f"inky_{cid[:6]}"] = {"config": a["config"], "kind": "inky_finalist", "campaign_id": cid, "rung": a["rung"],
                                   "dev_score": a["scores"][str(a["rung"])]["score"],
                                   "dev_quality": a["scores"][str(a["rung"])]["quality"],
                                   "teacher": "B_hrm" if cfg.distill != "none" else None}
    # Winning development mixture (by quality, rung >= 1 or registered) as the distillation teacher.
    best = None
    for cid, a in arch.items():
        cfg = Config.from_dict(a["config"])
        if cfg.routing not in MIXTURES or cfg.distill != "none" or not a["scores"]:
            continue
        if a["rung"] < 1 and a["origin"]["kind"] != "registered_arm":
            continue
        q = a["scores"][str(a["rung"])].get("quality")
        if q is not None and (best is None or q > best[0]):
            best = (q, cid, a)
    if best and best[1] != TEACHER.digest():
        name = next((n for n, v in arms.items() if Config.from_dict(v["config"]).digest() == best[1]), None)
        if name is None:
            name = f"T_{best[1][:6]}"
            arms[name] = {"config": best[2]["config"], "kind": "winning_mixture_teacher", "teacher": None}
        arms["K_win"] = {"config": ARMS["K_distill"].to_dict(), "kind": "student_of_winning_mixture", "teacher": name}
    frozen = {"schema": "molt-goblin-03.freeze.v1", "frozen_at": time.time(),
              "campaign_state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
              "arms": arms, "seeds": list(SEEDS_FINAL), "families": list(ALL), "root_base": HIDDEN3_BASE,
              "winning_dev_mixture": None if best is None else {"campaign_id": best[1], "dev_quality": best[0]}}
    frozen["freeze_sha256"] = _sha(frozen)
    out.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    return frozen


def _order(arms: dict) -> list[str]:
    teachers = {v["teacher"] for v in arms.values() if v.get("teacher")}
    return [n for n in arms if n in teachers] + [n for n in arms if n not in teachers]


def run(out: Path, seeds, threads: int, deadline: float, shard=(0, 1), mech_seeds=(11, 23)):
    torch.set_num_threads(threads)
    frozen = json.loads((out / "freeze.json").read_text(encoding="utf-8"))
    arms = frozen["arms"]
    path = out / (f"final_runs.shard{shard[0]}.jsonl" if shard[1] > 1 else "final_runs.jsonl")
    done = {json.loads(l)["key"] for p in out.glob("final_runs*.jsonl") for l in p.read_text(encoding="utf-8").splitlines()}
    ck = out / "checkpoints"
    ck.mkdir(exist_ok=True)
    idx = -1
    for seed in seeds:
        for family in ALL:
            # A family's cells stay in one shard so teachers precede their students.
            idx += 1
            if idx % shard[1] != shard[0]:
                continue
            for name in _order(arms):
                arm = arms[name]
                if arm.get("families") and family not in arm["families"]:
                    continue
                key = f"{name}|{family}|{seed}"
                if key in done:
                    continue
                if time.time() > deadline:
                    return
                cfg = Config.from_dict(arm["config"])
                teacher_cfg = Config.from_dict(arms[arm["teacher"]]["config"]) if arm.get("teacher") else None
                is_teacher = any(v.get("teacher") == name for v in arms.values())
                want_mech = cfg.routing in MIXTURES and family in MECH_FAMILIES and seed in mech_seeds
                t0 = time.time()
                r = run_cell(cfg, family, ALL[family], seed, HIDDEN3_BASE, teacher_cfg=teacher_cfg, n_eval=256,
                             extrap_loops=min(16, 2 * cfg.loops), return_model=True, cache_teacher=is_teacher)
                model = r.pop("_model")
                r["wall_s"] = time.time() - t0
                if want_mech:
                    spec = ALL[family]["eval"]["dev"]
                    ms = root_seed(HIDDEN3_BASE, family, seed, salt_of("mech"))
                    r["mech"] = {**specialization(model, spec, 256, ms), **gradient_similarity(model, spec, 64, ms)}
                if seed == seeds[0]:
                    torch.save({"config": cfg.to_dict(), "family": family, "seed": seed, "state_dict": model.state_dict()},
                               ck / f"{name}__{family}__s{seed}.pt")
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "arm": name, "kind": arm["kind"], **r}) + "\n")
                done.add(key)
                print(json.dumps({"key": key, "exact": r["evals"]["dev"]["exact"], "wall": round(r["wall_s"], 1)}), flush=True)


def run_multitask(out: Path, arms_for_mt=("B_hrm", "H_sparse_st", "I_anneal"), seeds=(11, 23, 37), threads=2):
    """Task-conditional routing: one model per arm and seed trained on the principal families jointly."""
    torch.set_num_threads(threads)
    frozen = json.loads((out / "freeze.json").read_text(encoding="utf-8"))
    path = out / "multitask_routing.jsonl"
    done = {json.loads(l)["key"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
    panels = {f: ALL[f] for f in PRINCIPAL}
    for name in arms_for_mt:
        for seed in seeds:
            key = f"{name}|{seed}"
            if key in done:
                continue
            cfg = Config.from_dict(frozen["arms"][name]["config"])
            t0 = time.time()
            r = multitask_routing(cfg, panels, seed, cfg.steps, HIDDEN3_BASE + 3_000_000)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "arm": name, "seed": seed, "wall_s": time.time() - t0, **r}) + "\n")
            print(json.dumps({"key": key, "mi": r.get("family_module_mi_nats")}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_FINAL))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--ram-gib", type=float, default=3.0, help="hard total for the driver and all workers")
    ap.add_argument("--freeze-only", action="store_true")
    a = ap.parse_args()
    cap = install_job_cap(a.ram_gib)
    frozen = freeze(a.state, a.out, a.k)
    (a.out / "resource_cap.json").write_text(json.dumps(cap, indent=1), encoding="utf-8")
    print(json.dumps({"freeze_sha256": frozen["freeze_sha256"], "arms": list(frozen["arms"])}), flush=True)
    if a.freeze_only:
        return
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    deadline = time.time() + a.hours * 3600
    procs = [ctx.Process(target=run, args=(a.out, a.seeds, a.threads, deadline, (i, a.workers))) for i in range(a.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    if time.time() < deadline:
        run_multitask(a.out, threads=a.threads)
    peak = peak_job_bytes()
    (a.out / "resource_peak.json").write_text(json.dumps({"peak_job_memory_bytes": peak}), encoding="utf-8")


if __name__ == "__main__":
    main()
