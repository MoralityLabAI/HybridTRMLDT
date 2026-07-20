"""Generate and seal the frozen LSPG architecture proposal corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from research_gym.architecture_discovery.planner import (
    generate_architecture_proposals,
    read_proposals,
    write_proposals,
)


ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_registration(path: Path, inputs: dict[str, Path]) -> dict[str, Any]:
    registration = _load(path)
    if registration["status"] != "frozen_before_proposals_and_outcomes":
        raise ValueError("architecture protocol is not frozen")
    registered = registration["inputs"]
    for name, input_path in inputs.items():
        if name == "registration":
            continue
        receipt = registered.get(name)
        if receipt is None:
            raise ValueError(f"unregistered proposal input: {name}")
        if _sha256(input_path) != receipt["sha256"]:
            raise ValueError(f"registered input hash mismatch: {name}")
    return registration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registration",
        type=Path,
        default=ROOT / "configs/lsa/architecture_discovery_v1_registration.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments/loop_schedule_architecture_discovery_v1/proposals",
    )
    args = parser.parse_args()
    registration_path = args.registration.resolve()
    registration = _load(registration_path)
    inputs = {
        name: (ROOT / receipt["path"]).resolve()
        for name, receipt in registration["inputs"].items()
    }
    inputs["registration"] = registration_path
    _verify_registration(registration_path, inputs)
    task_manifest = _load(inputs["task_manifest"])
    resource = _load(inputs["resource_profile"])
    proposals = generate_architecture_proposals(
        code_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        task_manifest=task_manifest,
        scale_ladder=_load(inputs["scale_ladder"]),
        resource_profile={
            "sequence_length": int(resource["sequence_length"]),
            "effective_batch_size": int(resource["effective_batch_size"]),
            "microbatch_by_scale": resource["microbatch_by_scale"],
            "vram_bytes": int(resource["caps"]["vram_bytes"]),
        },
        search_space=_load(inputs["search_space"]),
    )
    manifest = write_proposals(
        proposals,
        args.out.resolve(),
        input_files=inputs,
        input_root=ROOT,
    )
    if len(read_proposals(args.out.resolve())) != 38:
        raise RuntimeError("sealed proposal bundle failed round-trip validation")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
