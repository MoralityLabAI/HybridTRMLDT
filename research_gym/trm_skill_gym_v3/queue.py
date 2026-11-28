"""Serial stage queue for the GYM-02 full campaign (one training worker).

Re-reads ``stages.json`` before every stage, so stages implemented later can be
appended while earlier ones run.  Each stage runs under ``supervisor.run`` with the
3 GiB process-tree limit and a wall cap that never exceeds the campaign deadline.
A stage whose module is missing is recorded as ``skipped_missing_module`` and is
retried on the next pass.  Stages are resumable; a finished stage is not rerun.

    python -m research_gym.trm_skill_gym_v3.queue --out experiments/trm_skill_gym_02/full --hours 8
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

from . import supervisor as SUP


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hours", type=float, default=8.0)
    a = p.parse_args(argv)
    if not 0 < a.hours <= 8:
        p.error("the campaign cap is eight hours")
    a.out.mkdir(parents=True, exist_ok=True)
    state_path = a.out / "queue_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"t_start": time.time(), "stages": {}}
    deadline = state["t_start"] + 3600 * a.hours
    SUP.write_json(state_path, state)

    def log(m):
        print(m, flush=True)
        with open(a.out / "queue.log", "a", encoding="utf-8", newline="\n") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}\n")

    while True:
        stages = json.loads((a.out / "stages.json").read_text(encoding="utf-8"))
        pending = [s for s in stages if state["stages"].get(s["name"], {}).get("status") not in ("ok", "failed", "memory_killed", "wall_killed")]
        if not pending:
            log("all stages finished")
            break
        stage = None
        for s in pending:
            if importlib.util.find_spec(s["module"]) is not None:
                stage = s
                break
            state["stages"].setdefault(s["name"], {})["status"] = "waiting_for_module"
        if stage is None:
            if time.time() > deadline:
                break
            log("no runnable stage yet; waiting")
            time.sleep(120)
            continue
        remaining = deadline - time.time()
        if remaining < 300:
            log(f"deadline: {stage['name']} not started")
            for s in pending:
                state["stages"][s["name"]] = {"status": "deadline_not_run"}
            break
        wall = min(stage.get("hours", 1.0) * 3600 + 900, remaining)
        log(f"start {stage['name']} wall cap {wall / 3600:.2f} h")
        sup = SUP.run(SUP.python_module(stage["module"], stage["args"]), wall_s=wall,
                      log_path=a.out / f"{stage['name']}.log", log=log)
        state["stages"][stage["name"]] = dict(sup, started=time.time() - sup["wall_s"])
        SUP.write_json(state_path, state)
        log(f"end {stage['name']}: {sup['status']} wall {sup['wall_s'] / 60:.1f} min peak {sup['peak_tree_rss_bytes'] / 2**20:.0f} MiB")
    SUP.write_json(state_path, state)


if __name__ == "__main__":
    main()
