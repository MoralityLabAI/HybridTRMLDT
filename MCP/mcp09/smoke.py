"""Paired development smoke only. Never writes under registered result paths."""
import argparse
import json
from pathlib import Path
import torch
from MCP.mcp08.memory256k import ExtendedPhysicalMemory
from MCP.mcp08.models import MCP08RecursiveController, CONFIG_MCP08_54K
from MCP.mcp08.storyworld_bench import (
    HORIZON_BUCKETS, StoryworldSearchEnv, generate_independent_storyworld,
    strip_world_witnesses,
)
from .planner import Planner
from .storyworld import ChargedStoryworld, strict_route


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=[107, 109, 113], default=107)
    p.add_argument("--capacity", type=int, choices=[32768, 262144], default=32768)
    p.add_argument("--worlds-per-bucket", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--out", default="MCP/results/mcp09/smoke/paired.json")
    a = p.parse_args()
    out = Path(a.out)
    if "registered" in out.resolve().parts or "smoke" not in out.resolve().parts:
        p.error("development output must be in a smoke directory")
    torch.set_num_threads(1)
    checkpoint = Path(__file__).resolve().parents[1] / "results" / "mcp08" / "registered" / "checkpoints" / f"{a.seed}-mcp08_full.pt"
    model = MCP08RecursiveController(CONFIG_MCP08_54K)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    mem = ExtendedPhysicalMemory(a.capacity, a.seed + 920001)
    rows = []
    for idx, (bucket, lo, hi) in enumerate(HORIZON_BUCKETS):
        for k in range(a.worlds_per_bucket):
            wseed = 9100000 + a.seed * 100 + idx * 10 + k
            horizon = lo + (hi - lo) * k // max(1, a.worlds_per_bucket - 1)
            world = generate_independent_storyworld(wseed, horizon, bucket)
            stripped = strip_world_witnesses(world)
            audit_env = StoryworldSearchEnv(stripped, mem, wseed + 17)
            witness_valid = strict_route(audit_env, world["_private_witness"]["route"])
            # The offline witness result is logged separately and never fed to the policy.
            for depth in (0, 2):
                env = StoryworldSearchEnv(stripped, mem, wseed + 17)
                adapter = ChargedStoryworld(env, model, wseed + 41)
                rec = Planner(adapter, depth=depth, max_backtracks=max(16, horizon + 6)).run(env.initial_state())
                rows.append({"world_seed": wseed, "bucket": bucket, "horizon": horizon,
                             "seed": a.seed, "capacity": a.capacity, "depth": depth,
                             "offline_witness_valid": witness_valid, **rec, **adapter.receipt()})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"status": "SMOKE_ONLY", "rows": rows}, indent=2) + "\n")
    for d in (0, 2):
        rs = [r for r in rows if r["depth"] == d]
        print(d, sum(r["success"] for r in rs), "/", len(rs),
              "mean reads", sum(r["charged_memory_reads"] for r in rs) / len(rs),
              "valid budgets", all(r["budget_valid"] for r in rs))
    return 0 if all(r["budget_valid"] for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
