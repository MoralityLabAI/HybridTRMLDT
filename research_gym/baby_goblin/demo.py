"""CPU-only demonstration of one complete multi-skill Baby Goblin research episode.

    python -m research_gym.baby_goblin.demo --registry research_gym/baby_goblin/registry.json --out DIR --root 7300000

Runs the five typed stages of the GYM-02 router episode (noisy retrieval -> deduction
over the retrieved set -> repair -> stopping -> research) through the registry's
budget-aware router, verifies each stage with the host verifier, and writes the
trajectory JSONL, DataGoblin event rows and the PocketGoblin tool surface.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research_gym.trm_skill_gym import tasks as T
from research_gym.trm_skill_gym_v3 import router as R
from .core import Registry, Trajectory, datagoblin_events, pocket_goblin_tools, route


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=Path, default=Path(__file__).with_name("registry.json"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--root", type=int, default=7_300_000)
    p.add_argument("--budget-flops", type=int, default=20_000_000)
    a = p.parse_args(argv)
    torch.set_num_threads(1)
    a.out.mkdir(parents=True, exist_ok=True)
    reg = Registry.load(a.registry)
    traj = Trajectory(a.out / "trajectory.jsonl", run_id=f"baby-goblin-demo-{a.root}")
    inst = R.episode_instances(a.root)
    summary = {}
    retrieved = None
    for skill in R.STAGES:
        x = inst[skill]
        if skill == "deduction":
            x = R.restrict_deduction(x, retrieved)
            truth = R.restrict_deduction(inst["deduction"], inst["retrieval"].label)
            x.label = truth.label if not np.array_equal(retrieved, inst["retrieval"].label) else truth.label * x.mask
        res = route(reg, skill, x, a.budget_flops, traj)
        if skill == "retrieval":
            retrieved = res.decision.astype(np.float32)
        b = T.stack([x])
        ok = bool(T.verify(skill, b, torch.from_numpy(res.decision)[None] * b["mask"])[0][0])
        traj.log(skill=skill, event="verifier_receipt", ok=ok)
        summary[skill] = {"verified": ok, "uncertainty": res.uncertainty}
    (a.out / "datagoblin_events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in datagoblin_events(traj)),
                                                    encoding="utf-8", newline="\n")
    (a.out / "pocket_goblin_tools.json").write_text(json.dumps(pocket_goblin_tools(reg), indent=1) + "\n", encoding="utf-8", newline="\n")
    out = {"root": a.root, "stages": summary, "all_verified": all(v["verified"] for v in summary.values()),
           "trajectory_sha256": traj.digest(), "steps": len(traj.rows)}
    (a.out / "summary.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
