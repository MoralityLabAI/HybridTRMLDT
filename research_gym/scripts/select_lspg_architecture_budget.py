"""Select the frozen LSPG campaign profile from resource-only calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.architecture_discovery.calibration import (
    load_calibration_measurement,
    select_budget_profile,
)


ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        default=ROOT / "experiments/loop_schedule_architecture_discovery_v1/calibration_runs",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "configs/lsa/architecture_promotion_policy_v1.json",
    )
    parser.add_argument(
        "--resource-profile",
        type=Path,
        default=ROOT / "configs/lsa/architecture_resource_profile_v1.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments/loop_schedule_architecture_discovery_v1/calibration/profile_selection.json",
    )
    args = parser.parse_args()
    measurements = []
    for scale in ("S0", "S1", "S2"):
        cell_id = f"LSAD-C-K2L8-calibration-{scale}-s397"
        measurements.append(
            load_calibration_measurement(
                args.runs / cell_id / "result.json",
                args.runs / "resource_receipts" / f"{cell_id}.attempt-1.resource_receipt.json",
            )
        )
    profile = _load(args.resource_profile)
    selection = select_budget_profile(
        _load(args.policy),
        measurements,
        effective_batch_size=int(profile["effective_batch_size"]),
        sequence_length=int(profile["sequence_length"]),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(selection, indent=2, sort_keys=True))
    if selection["status"] != "selected":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
