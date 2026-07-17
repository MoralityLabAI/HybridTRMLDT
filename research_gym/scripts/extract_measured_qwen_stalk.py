from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.analysis.measured_stalk_bridge import extract_measured_stalk
from research_gym.scripts.bench_gaming_vs_improvement import _write_json


DEFAULT_CONFIG = Path("configs/qwen08_l23_measured_stalk_source_v1.json")
DEFAULT_OUTPUT = Path("data/bridge/qwen08_l23_measured_stalk_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seal a compact measured Qwen layer-23 stalk receipt."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rsi-topology-root", type=Path)
    parser.add_argument("--research-engine-root", type=Path)
    args = parser.parse_args()

    overrides = {}
    if args.rsi_topology_root is not None:
        overrides["rsi_topology"] = args.rsi_topology_root
    if args.research_engine_root is not None:
        overrides["research_engine"] = args.research_engine_root
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = extract_measured_stalk(config, root_overrides=overrides)
    _write_json(result, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
