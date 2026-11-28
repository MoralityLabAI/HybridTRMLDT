"""Competence calibration of the MOLT-GOBLIN-03 panel and controls (development roots only)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .arms import ARMS, CONTROL, PANEL
from .train3 import DEV3_BASE, run_cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=["A_single", "B_hrm"])
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for fam, panel in {**PANEL, **CONTROL}.items():
        for arm in a.arms:
            t0 = time.time()
            r = run_cell(ARMS[arm], fam, panel, 11, DEV3_BASE, n_eval=256, interventions=False)
            row = {"family": fam, "arm": arm, "spec": panel["train"].key(), "exact": r["evals"]["dev"]["exact"],
                   "token_acc": r["evals"]["dev"]["token_acc"], "exact_by_loop": r["evals"]["dev"]["exact_by_loop"],
                   "ood": {k: v["exact"] for k, v in r["evals"].items() if k.startswith("ood")},
                   "train_cpu_s": r["train_cpu_s"], "wall_s": time.time() - t0}
            with a.out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("family", "arm", "exact", "token_acc", "ood", "train_cpu_s")}), flush=True)


if __name__ == "__main__":
    main()
