"""SKILL-02 full fresh architecture race (four seeds, 800 steps, extended graph strata).

Roster per skill: A existing skill, B feedforward, C GRU, D direct TRM, E matched
damping, F RLT, plus distinct archived Inky-engineered rung-1 configurations from
GYM-01 (``--max-archive`` per skill, deterministic order).  Every trainable job runs
in its own process under ``supervisor.run`` (hard 3 GiB process-tree RSS limit).

Exposure control is matched gradient steps; a job that does not complete its
steps is recorded as ``incomplete`` and excluded from confirmatory comparisons.
Selection happens on calibration strata only (identical for all candidates),
ranking worst-stratum exact, then macro exact, then mean model FLOPs, within the
direct-TRM model-FLOP ceiling and at most two verifier calls (in-loop + receipt).
Test strata (fresh roots, plus withheld families and topologies) are evaluated
only after ``frozen.json`` is written.

Metered per job: training CPU seconds (process time), inference CPU seconds,
model FLOPs, host tensor operations of the verifier/heuristic/certificate calls
(``host_ops``: multiply-accumulates of the einsum/bmm calls actually executed),
verifier calls and recurrent visits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from research_gym.trm_skill_gym import campaign as C, dsl, modules as M, tasks as T, train as TR
from research_gym.trm_skill_gym_v2.core import paired_rank
from . import graphs as G
from . import supervisor as SUP

SOURCE = Path("experiments/trm_skill_gym_01/run")
TRAIN_ROOT, CAL_ROOT, TEST_ROOT = 4_000_000, 4_100_000, 4_200_000
FIXED = ("A_existing_skill", "B_feedforward", "C_gru", "D_trm_direct", "E_trm_matched_damping", "F_trm_rlt")
REQUIRED = FIXED
ORDER = ("retrieval", "deduction", "repair", "curation", "research", "stopping")

# ------------------------------------------------------------------ host-op metering
_HOST = [0, True]  # [ops, metering enabled]


def _meter(fn, cost):
    def wrapped(skill, batch, *a, **k):
        if _HOST[1]:
            _HOST[0] += cost(skill, batch)
        return fn(skill, batch, *a, **k)
    return wrapped


def _unmetered(fn):
    """Grading (exact success) is not inference work: switch metering off inside it."""
    def wrapped(*a, **k):
        prev, _HOST[1] = _HOST[1], False
        try:
            return fn(*a, **k)
        finally:
            _HOST[1] = prev
    return wrapped


def _verify_cost(skill, b):
    B, N = b["mask"].shape
    return B * N * N * (N if skill == "retrieval" else 2)  # reach propagation in retrieval is up to N bmm rounds


def _heur_cost(skill, b):
    B, N = b["mask"].shape
    return B * N * N


def install_meters():
    T.verify = _meter(T.verify, _verify_cost)
    T.heuristic_scores = _meter(T.heuristic_scores, _heur_cost)
    T.heuristic_repair_step = _meter(T.heuristic_repair_step, _heur_cost)
    T.certify = _meter(T.certify, _verify_cost)
    T.exact = _unmetered(T.exact)  # modules.episode reaches these through the same module object


# ------------------------------------------------------------------ data
def batch(skill: str, family: str, base: int, count: int):
    return T.stack([G.instance(skill, family, base + i) for i in range(count)])


def training_pool(skill: str, count: int):
    fams = G.strata(skill, "train")
    return T.stack([G.instance(skill, fams[i % len(fams)], TRAIN_ROOT + i) for i in range(count)])


def eval_pools(skill: str, stage: str, count: int) -> Dict[str, Dict[str, torch.Tensor]]:
    base = CAL_ROOT if stage == "cal" else TEST_ROOT
    return {f: batch(skill, f, base + 10_000 * k, count) for k, f in enumerate(G.strata(skill, stage))}


@torch.no_grad()
def evaluate(skill, model, cfg, b) -> Dict:
    M.flops_reset()
    _HOST[0] = 0
    t0 = time.process_time()
    out = M.episode(model, skill, b, cfg, heuristic=model is None)
    cpu = time.process_time() - t0
    flops = M.flops_read()
    acc = out["exact"][-1].numpy()
    n = len(acc)
    return {"exact": float(acc.mean()), "per_root": acc.astype(np.int8).tolist(), "flops": flops / n,
            "calls": out["vcalls"] + 1, "visits": len(out["exact"]), "inference_cpu_s": cpu,
            "inference_cpu_s_per_instance": cpu / n, "host_ops_per_instance": _HOST[0] / n}


# ------------------------------------------------------------------ worker
def _load(name, checkpoint):
    if name == "A_existing_skill":
        return None, C.base_trm()
    ck = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    model = M.SkillModule(ck["config"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck["config"]


def worker(spec_path: Path) -> None:
    """cal stage: train one config for each listed seed (sequentially, one process),
    evaluate on calibration strata.  test stage: evaluate every listed (name, seed)
    checkpoint of one skill on the test strata (pools generated once)."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    install_meters()
    skill = spec["skill"]
    pools = eval_pools(skill, spec["stage"], spec["count"])
    if spec["stage"] == "cal":
        name, cfg = spec["name"], spec["config"]
        train = None if name == "A_existing_skill" else training_pool(skill, spec["train_count"])
        for seed, out in zip(spec["seeds"], spec["outs"]):
            out_path = Path(out)
            if out_path.exists():
                continue
            if name == "A_existing_skill":
                model, info, ecfg = None, {"steps": 0, "train_cpu_s": 0.0, "train_wall_s": 0.0}, C.base_trm()
            else:
                t0, w0 = time.process_time(), time.time()
                model, tinfo = TR.train_module(skill, cfg, spec["steps"], seed, train,
                                               deadline=time.time() + spec["job_seconds"])
                info = dict(tinfo, train_cpu_s=time.process_time() - t0, train_wall_s=time.time() - w0)
                ecfg = cfg
                if tinfo["steps"] != spec["steps"]:
                    SUP.write_json(out_path, {"status": "incomplete", "train": info})
                    continue
                model.eval()
                torch.save({"config": cfg, "state_dict": model.state_dict()}, out_path.with_suffix(".pt"))
            res = {k: evaluate(skill, model, ecfg, b) for k, b in pools.items()}
            SUP.write_json(out_path, {"status": "ok", "train": info, "cal": res,
                                      "params": 0 if model is None else model.n_params()})
    else:
        for item in spec["items"]:
            out_path = Path(item["out"])
            if out_path.exists():
                continue
            model, ecfg = _load(item["name"], item["checkpoint"])
            res = {k: evaluate(skill, model, ecfg, b) for k, b in pools.items()}
            SUP.write_json(out_path, {"status": "ok", "test": res})


# ------------------------------------------------------------------ driver
def roster(archive, skill, max_archive):
    configs = C.arm_configs()
    out = {k: configs[k] for k in FIXED}
    seen = {dsl.config_id(v) for k, v in out.items() if k != "A_existing_skill"}
    added = 0
    for row in sorted(archive, key=lambda r: (r["arm"], r["round"], r["proposal_index"])):
        if row["skill"] != skill or (row.get("rung1") or {}).get("dev_exact") is None or row["arm"] == "random_mutation":
            continue
        cfg = dsl.validate(row["config"])
        key = dsl.config_id(cfg)
        if key in seen or not dsl.PARAM_MIN <= M.SkillModule(cfg).n_params() <= dsl.PARAM_MAX:
            continue
        if added >= max_archive:
            break
        out["archive_" + key] = cfg
        seen.add(key)
        added += 1
    return out


def aggregate(jobs):
    keys = jobs[0]["cal"].keys()
    return {k: {"exact": float(np.mean([j["cal"][k]["exact"] for j in jobs])),
                "flops": max(j["cal"][k]["flops"] for j in jobs),
                "calls": max(j["cal"][k]["calls"] for j in jobs)} for k in keys}


def bootstrap(delta_by_seed, seed=19):
    d = np.asarray(delta_by_seed, dtype=float).mean(0)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    return {"delta": float(d.mean()), "root_bootstrap_95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "seed_deltas": np.asarray(delta_by_seed, dtype=float).mean(1).tolist()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--skills", nargs="+", default=list(ORDER))
    p.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12, 13])
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--count", type=int, default=128)
    p.add_argument("--test-count", type=int, default=256)
    p.add_argument("--train-count", type=int, default=4096)
    p.add_argument("--max-archive", type=int, default=2)
    p.add_argument("--hours", type=float, default=3.0)
    p.add_argument("--job-seconds", type=float, default=900)
    p.add_argument("--mem-limit-gib", type=float, default=3.0)
    p.add_argument("--test-only", action="store_true",
                   help="evaluate already-frozen skills on the test strata; the calibration deadline does not apply")
    a = p.parse_args(argv)
    if a.worker:
        return worker(a.worker)
    if not 0 < a.hours <= 8:
        p.error("hours must be in (0, 8]")
    a.out.mkdir(parents=True, exist_ok=True)

    def log(m):
        print(m, flush=True)
        with open(a.out / "race.log", "a", encoding="utf-8", newline="\n") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}\n")

    archive_bytes = (SOURCE / "search" / "archive.jsonl").read_bytes()
    archive = [json.loads(l) for l in archive_bytes.splitlines() if l.strip()]
    configs = {s: roster(archive, s, a.max_archive) for s in a.skills}
    profile = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k not in ("worker", "out", "hours", "test_only")}
    profile.update(archive_sha256=hashlib.sha256(archive_bytes).hexdigest(), configs=configs,
                   roots={"train": TRAIN_ROOT, "cal": CAL_ROOT, "test": TEST_ROOT},
                   strata={s: {"cal": G.strata(s, "cal"), "test": G.strata(s, "test")} for s in a.skills},
                   source_sha256={f.name: hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                  for f in sorted(Path(__file__).parent.glob("*.py"))})
    man = a.out / "manifest.json"
    if man.exists():
        old = json.loads(man.read_text(encoding="utf-8"))
        if old["profile"]["configs"] != profile["configs"] or old["profile"]["seeds"] != profile["seeds"]:
            p.error("manifest mismatch; use a new output directory")
        deadline = old["deadline"]
    else:
        deadline = time.time() + 3600 * a.hours
        SUP.write_json(man, {"profile": profile, "deadline": deadline, "started": time.time()})
    limit = int(a.mem_limit_gib * 1024**3)

    def supervise(spec, tag, n_jobs):
        sp = a.out / "specs" / f"{tag}.json"
        SUP.write_json(sp, spec)
        return SUP.run(SUP.python_module("research_gym.trm_skill_gym_v3.race", ["--worker", str(sp)]),
                       wall_s=a.job_seconds * n_jobs + 600, limit_bytes=limit, log_path=a.out / "workers.log", log=log)

    def read(path: Path, sup=None):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"status": (sup or {}).get("status", "not_run"), "supervisor": sup}

    seeds_for = lambda name: a.seeds if name != "A_existing_skill" else a.seeds[:1]
    jobdir = lambda skill: a.out / "jobs" / skill
    frozen_path = a.out / "frozen.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.exists() else {}
    supervision = {}
    for skill in a.skills:
        if skill in frozen or a.test_only:
            continue
        metrics, incomplete = {}, {}
        for name, cfg in configs[skill].items():
            outs = [jobdir(skill) / f"{name}__s{s}.json" for s in seeds_for(name)]
            sup = None
            if not all(o.exists() for o in outs):
                if time.time() > deadline:
                    incomplete[name] = ["deadline_not_run"]
                    continue
                spec = {"skill": skill, "name": name, "config": cfg, "seeds": seeds_for(name), "outs": [str(o) for o in outs],
                        "stage": "cal", "steps": a.steps, "count": a.count, "train_count": a.train_count,
                        "job_seconds": a.job_seconds}
                sup = supervise(spec, f"cal__{skill}__{name}", len(outs))
                supervision[f"{skill}/{name}"] = sup
            jobs = [read(o, sup) for o in outs]
            log(f"{skill} {name}: " + ", ".join(j["status"] for j in jobs) +
                (f" worst-cal={min(min(v['exact'] for v in j['cal'].values()) for j in jobs):.3f}" if all(j["status"] == "ok" for j in jobs) else "") +
                (f" peak={sup['peak_tree_rss_bytes'] // 2**20}MiB wall={sup['wall_s']:.0f}s" if sup else ""))
            if all(j["status"] == "ok" for j in jobs):
                metrics[name] = aggregate(jobs)
            else:
                incomplete[name] = [j["status"] for j in jobs]
        if not all(k in metrics for k in REQUIRED):
            log(f"{skill}: missing fixed baseline(s) {[k for k in REQUIRED if k not in metrics]}; not frozen")
            SUP.write_json(a.out / f"partial_{skill}.json", {"incomplete": incomplete})
            continue
        cap = max(m["flops"] for m in metrics["D_trm_direct"].values())
        eligible = [k for k, m in metrics.items() if all(v["flops"] <= cap * 1.0001 and v["calls"] <= 2 for v in m.values())]
        cal_keys = G.strata(skill, "cal")
        rank = {k: paired_rank(metrics[k], cal_keys) for k in eligible}
        selected = max(eligible, key=lambda k: rank[k])
        baseline = max([k for k in eligible if k in ("A_existing_skill", "B_feedforward", "C_gru")], key=lambda k: rank[k])
        frozen[skill] = {"selected": selected, "baseline": baseline, "cap_model_flops": cap, "eligible": eligible,
                         "rank": rank, "incomplete": incomplete, "calibration": metrics, "frozen_at": time.time()}
        SUP.write_json(frozen_path, frozen)
        log(f"FROZEN {skill}: selected={selected} baseline={baseline} incomplete={list(incomplete)}")
    SUP.write_json(a.out / "supervision_cal.json", supervision)
    # held-out evaluation only after a skill is frozen; all complete candidates are evaluated
    # (descriptive); selected vs strongest eligible baseline is the confirmatory contrast
    held = {}
    hp = a.out / "heldout.json"
    for skill, row in frozen.items():
        names = [n for n in configs[skill] if n not in row["incomplete"]]
        items = [{"name": n, "seed": s, "checkpoint": str((jobdir(skill) / f"{n}__s{s}").with_suffix(".pt")),
                  "out": str(a.out / "test" / skill / f"{n}__s{s}.json")} for n in names for s in seeds_for(n)]
        sup = None
        if not all(Path(i["out"]).exists() for i in items):
            if time.time() > deadline and not a.test_only:
                held[skill] = {"status": "deadline_test_not_run"}
                continue
            sup = supervise({"skill": skill, "stage": "test", "count": a.test_count, "items": items}, f"test__{skill}", 1)
        per = {(i["name"], i["seed"]): read(Path(i["out"]), sup) for i in items}
        sel, base = row["selected"], row["baseline"]
        if not all(per[(n, s)].get("status") == "ok" for n in (sel, base) for s in seeds_for(n)):
            held[skill] = {"status": "test_incomplete", "supervisor": sup}
            continue
        strata_test = G.strata(skill, "test")
        contrast = {}
        for f in strata_test:
            bs = np.mean([per[(base, s)]["test"][f]["per_root"] for s in seeds_for(base)], 0)
            contrast[f] = bootstrap([np.asarray(per[(sel, s)]["test"][f]["per_root"]) - bs for s in seeds_for(sel)])
        summary = {}
        for (name, seed), j in per.items():
            if j.get("status") == "ok":
                summary.setdefault(name, []).append(j["test"])
        mean_exact = lambda n, f: float(np.mean([r[f]["exact"] for r in summary[n]]))
        held[skill] = {"status": "ok", "selected": sel, "baseline": base, "contrast": contrast,
                       "worst_stratum_selected": min(mean_exact(sel, f) for f in strata_test),
                       "worst_stratum_baseline": min(mean_exact(base, f) for f in strata_test),
                       "macro_selected": float(np.mean([mean_exact(sel, f) for f in strata_test])),
                       "macro_baseline": float(np.mean([mean_exact(base, f) for f in strata_test])),
                       "per_candidate": {n: {f: {"exact": mean_exact(n, f),
                                                 "exact_per_seed": [r[f]["exact"] for r in rows],
                                                 "flops": float(np.mean([r[f]["flops"] for r in rows])),
                                                 "inference_cpu_s_per_instance": float(np.mean([r[f]["inference_cpu_s_per_instance"] for r in rows])),
                                                 "host_ops_per_instance": float(np.mean([r[f]["host_ops_per_instance"] for r in rows])),
                                                 "calls": rows[0][f]["calls"], "visits": rows[0][f]["visits"], "n_seeds": len(rows)}
                                             for f in strata_test} for n, rows in summary.items()},
                       "supervisor": sup}
        SUP.write_json(hp, {"complete": len(held) == len(a.skills), "results": held})
        log(f"TEST {skill}: selected {sel} worst {held[skill]['worst_stratum_selected']:.3f} "
            f"vs baseline {base} {held[skill]['worst_stratum_baseline']:.3f}")
    SUP.write_json(hp, {"complete": len(held) == len(a.skills) and all(v.get("status") == "ok" for v in held.values()),
                        "results": held})


if __name__ == "__main__":
    main()
