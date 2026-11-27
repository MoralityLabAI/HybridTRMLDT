"""Competence calibration on development roots before freezing the panel."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .gym import TaskSpec
from .panel import BASE
from .train import train_and_eval

CANDIDATE_SPECS = {
    "pointer_chase": [TaskSpec("pointer_chase", 6, depth=2, distract=2), TaskSpec("pointer_chase", 6, depth=1, distract=2)],
    "revision": [TaskSpec("revision", 12, depth=0, distract=0), TaskSpec("revision", 16, depth=1, distract=2)],
    "repair": [TaskSpec("repair", 6, depth=1, variant="unflagged"), TaskSpec("repair", 8, depth=2, variant="unflagged")],
    "proof_deps": [TaskSpec("proof_deps", 10, depth=3), TaskSpec("proof_deps", 14, depth=3)],
    "cumsum": [TaskSpec("cumsum", 8)],
    "shallow_copy": [TaskSpec("shallow_copy", 12)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--arms", nargs="+", default=["prefix", "tied", "full"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--panel", action="store_true", help="calibrate the registered panel train specs only")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    table = CANDIDATE_SPECS
    if a.panel:
        from .panel import PANEL
        table = {f: [p["train"]] for f, p in PANEL.items()}
    for fam, specs in table.items():
        for spec in specs:
            for arm in a.arms:
                cfg = BASE.replace(memory=arm, steps=a.steps)
                t0 = time.time()
                r = train_and_eval(cfg, fam, spec, {"dev": spec}, seed=11, n_eval=256, interventions=False)
                row = {"family": fam, "spec": spec.key(), "arm": arm, "steps": a.steps,
                       "exact": r["summary"]["exact"], "token_acc": r["evals"]["dev"]["token_acc"],
                       "exact_by_loop": r["evals"]["dev"]["exact_by_loop"], "train_cpu_s": r["train_cpu_s"],
                       "wall_s": time.time() - t0, "loss_trace": r["loss_trace"]}
                with a.out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps({k: row[k] for k in ("spec", "arm", "exact", "token_acc", "wall_s")}), flush=True)


if __name__ == "__main__":
    main()
