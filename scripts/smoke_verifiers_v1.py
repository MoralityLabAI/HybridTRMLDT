from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "environments" / "hybrid_sequencer_v1"
sys.path.insert(0, str(ENV_DIR))


async def main() -> None:
    vf = importlib.import_module("verifiers.v1")
    module = importlib.import_module("hybrid_sequencer_v1")
    config = vf.EnvConfig(
        taskset={"limit": 2},
        harness={"sequencer": "control_math", "max_turns": 1},
    )
    env = module.load_environment(config)
    task = next(iter(env.taskset))
    state = await env.harness.run(task)
    await env.harness.teardown()
    receipt = {
        "verifiers_version": importlib.import_module("verifiers").__version__,
        "taskset_type": type(env.taskset).__name__,
        "harness_type": type(env.harness).__name__,
        "example_id": task["example_id"],
        "sequencer": state["sequencer"],
        "selected_sequence": state["selected_sequence"],
        "reward": state["reward"],
        "trajectory_steps": len(state["trajectory"]),
        "stop_condition": state["stop_condition"],
    }
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
