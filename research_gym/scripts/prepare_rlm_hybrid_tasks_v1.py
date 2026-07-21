"""Materialize the deterministic long-context control task suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.benchmarks.rlm_hybrid_neighborhood import materialize_task_suite
from research_gym.integrity import canonical_file_sha256


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "benchmarks" / "rlm_hybrid_long_context_tasks_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite task suite: {output}")
    payload = materialize_task_suite()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "materialized",
                "path": output.relative_to(ROOT).as_posix(),
                "sha256": canonical_file_sha256(output),
                "task_count": payload["task_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
