from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.architecture_discovery.tasks import build_task_bundle, write_task_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the sealed LSPG architecture task bundle.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/lsa/architecture_task_bundle_v1.json")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/loop_schedule_architecture_discovery_v1/datasets"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if int(config["vocabulary_size"]) != 2048:
        raise ValueError("v1 freezes a 2,048-token vocabulary")
    bundle = build_task_bundle(
        seed=int(config["seed"]),
        sequence_length=int(config["sequence_length"]),
        counts=config["counts"],
        routing_root=Path(config["routing"]["root"]),
        routing_max_per_env=int(config["routing"]["max_per_environment"]),
    )
    manifest = write_task_bundle(bundle, args.out)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
