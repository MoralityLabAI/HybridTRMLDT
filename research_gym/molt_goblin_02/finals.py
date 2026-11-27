"""Freeze finalists from the campaign, then evaluate on hidden roots and withheld families.

The freeze file is written (with its digest) before any hidden-root evaluation. Hidden
roots use HIDDEN_BASE, which neither training nor the discovery loop touches. Withheld
families (bounded_walk, experiment_select) were never shown to the proposer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch

from . import resources
from .factory import MoltConfig, MoltNet, count_params
from .gym import FAMILIES
from .interventions import subspace_panel
from .panel import BASELINES, CONTROL, FINAL_SEEDS, PANEL, WITHHELD
from .train import HIDDEN_BASE, hash_name, root_seed, train_and_eval

ALL = {**PANEL, **CONTROL, **WITHHELD}


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def param_matched(cfg: MoltConfig, target: int, family: str) -> MoltConfig:
    """Single-block config whose width brings parameter count closest to target."""
    fam = FAMILIES[family]
    best = None
    for w in range(16, 97, 2):
        heads = 2 if w % 2 == 0 else 1
        c = cfg.replace(routing="single", n_types=1, width=w, heads=heads,
                        rank=min(cfg.rank, w), update_rank=min(cfg.update_rank, w))
        n = count_params(MoltNet(c, fam.vocab, fam.n_out))
        if best is None or abs(n - target) < best[0]:
            best = (abs(n - target), c)
    return best[1]


def freeze(state_path: Path, out: Path, k: int = 3) -> dict:
    st = json.loads(state_path.read_text(encoding="utf-8"))
    arch = st["archive"]
    cands = [(cid, a) for cid, a in arch.items() if a["origin"]["kind"] in ("candidate", "control") and a["scores"]]
    def key(item):
        cid, a = item
        r = a["rung"]
        return (r, a["scores"][str(r)].get("score") or -9)
    cands.sort(key=key, reverse=True)
    finalists = {}
    for cid, a in cands[:k]:
        finalists[f"f{len(finalists)}_{cid[:6]}"] = {"config": a["config"], "campaign_id": cid, "rung": a["rung"],
                                                    "dev_score": a["scores"][str(a["rung"])]["score"], "diff": a["diff"]}
    arms = {name: {"config": cfg.to_dict(), "kind": "baseline"} for name, cfg in BASELINES.items()}
    for name, f in finalists.items():
        arms[name] = {"config": f["config"], "kind": "finalist", **{k2: f[k2] for k2 in ("campaign_id", "rung", "dev_score", "diff")}}
    top = next(iter(finalists), None)
    if top:
        cfg = MoltConfig.from_dict(finalists[top]["config"])
        fam = "pointer_chase"
        target = count_params(MoltNet(cfg, FAMILIES[fam].vocab, FAMILIES[fam].n_out))
        pm = param_matched(BASELINES["prefix_all"], target, fam)
        arms["prefix_all_param_matched"] = {"config": pm.to_dict(), "kind": "parameter_matched_control",
                                        "matched_to": top, "target_params": target}
        # Visit-matched control: soft mixtures run every module each loop, so match module visits
        # with a single tied block (loop forcing kept, as in the finalists' base).
        visits = cfg.loops * (cfg.n_types if cfg.routing in ("mix", "hrm") else 1)
        if visits > cfg.loops:
            vm = BASELINES["prefix_all"].replace(loops=min(visits, 16))
            arms["prefix_all_visit_matched"] = {"config": vm.to_dict(), "kind": "visit_matched_control",
                                                "matched_to": top, "target_visits": visits}
    frozen = {"schema": "molt-goblin-02.freeze.v1", "frozen_at": time.time(), "campaign_state_sha256":
              hashlib.sha256(state_path.read_bytes()).hexdigest(), "arms": arms, "seeds": list(FINAL_SEEDS),
              "families": list(ALL), "root_base": HIDDEN_BASE}
    frozen["freeze_sha256"] = _sha(frozen)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "freeze.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    p.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    return frozen


def run(out: Path, seeds, threads: int, deadline: float, shard: tuple[int, int] = (0, 1)) -> None:
    torch.set_num_threads(threads)
    frozen = json.loads((out / "freeze.json").read_text(encoding="utf-8"))
    runs_path = out / (f"final_runs.shard{shard[0]}.jsonl" if shard[1] > 1 else "final_runs.jsonl")
    done = set()
    for p in out.glob("final_runs*.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["key"])
    index = -1
    ck = out / "checkpoints"
    ck.mkdir(exist_ok=True)
    order = list(PANEL) + list(WITHHELD) + list(CONTROL)
    for seed in seeds:                      # seed-major so partial runs stay balanced across arms
        for family in order:
            for name, arm in frozen["arms"].items():
                key = f"{name}|{family}|{seed}"
                index += 1
                if key in done or index % shard[1] != shard[0]:
                    continue
                if time.time() > deadline:
                    return
                cfg = MoltConfig.from_dict(arm["config"])
                p = ALL[family]
                want_model = arm["kind"] == "finalist" or name == "prefix"
                t0 = time.time()
                r = train_and_eval(cfg, family, p["train"], p["eval"], seed, n_eval=256, base=HIDDEN_BASE,
                                   extrap_loops=min(12, 2 * cfg.loops), return_model=want_model)
                model = r.pop("_model", None)
                r["wall_s"] = time.time() - t0
                if model is not None and seed == seeds[0]:
                    path = ck / f"{name}__{family}__s{seed}.pt"
                    torch.save({"config": cfg.to_dict(), "family": family, "seed": seed,
                                "state_dict": model.state_dict()}, path)
                    r["checkpoint"] = path.name
                    if family in PANEL:
                        spec = p["eval"]["dev"]
                        r["subspace"] = subspace_panel(model, spec, fit_seed=root_seed(HIDDEN_BASE, family, seed, 7),
                                                       eval_seed=root_seed(HIDDEN_BASE, family, seed, hash_name("dev")))
                with runs_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "arm": name, "kind": arm["kind"], **r}) + "\n")
                done.add(key)
                print(json.dumps({"key": key, "exact": r["summary"]["exact"], "wall": round(r["wall_s"], 1)}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(FINAL_SEEDS[:3]))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--shard", type=str, default="0/1", help="i/n: run every n-th cell starting at i")
    ap.add_argument("--ram-gib", type=float, default=3.0)
    a = ap.parse_args()
    i, n = (int(v) for v in a.shard.split("/"))
    cap = resources.install_cap(a.ram_gib)
    frozen = freeze(a.state, a.out, a.k)
    (a.out / f"resource_cap.shard{i}.json").write_text(json.dumps(cap, indent=1), encoding="utf-8")
    print(json.dumps({"freeze_sha256": frozen["freeze_sha256"], "arms": list(frozen["arms"])}))
    run(a.out, a.seeds, a.threads, time.time() + a.hours * 3600, shard=(i, n))


if __name__ == "__main__":
    main()
