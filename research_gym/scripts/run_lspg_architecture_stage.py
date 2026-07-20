"""Run one sealed LSPG stage through the hard-cap PowerShell wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from lsa.canonical import digest
from research_gym.architecture_discovery.campaign import read_stage_manifest


ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _elapsed_seconds(receipt_dir: Path) -> float:
    total = 0.0
    for path in receipt_dir.glob("*.resource_receipt.json"):
        value = _load(path)
        total += float(value.get("elapsed_seconds", 0.0))
    return total


def _completed_result(output: Path, cell_id: str) -> bool:
    result_path = output / cell_id / "result.json"
    receipt_path = output / cell_id / "result_receipt.json"
    if not result_path.exists() or not receipt_path.exists():
        return False
    result = _load(result_path)
    receipt = _load(receipt_path)
    return (
        result.get("status") == "completed"
        and result.get("integrity_passed")
        and result.get("cleanup_passed")
        and receipt.get("result_sha256")
        == hashlib.sha256(result_path.read_bytes()).hexdigest()
    )


def _write_stage_receipt(
    output: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    status: str,
    failure: str | None,
) -> None:
    cell_receipts: dict[str, Any] = {}
    for cell in manifest["cells"]:
        result_path = output / cell["cell_id"] / "result.json"
        if result_path.exists():
            result = _load(result_path)
            cell_receipts[cell["cell_id"]] = {
                "status": result.get("status"),
                "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            }
    receipt = {
        "schema_version": 1,
        "stage_id": manifest["stage_id"],
        "stage_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "stage_manifest_hash": manifest["manifest_hash"],
        "status": status,
        "failure": failure,
        "campaign_elapsed_seconds": _elapsed_seconds(output / "resource_receipts"),
        "completed_cells": len(
            [value for value in cell_receipts.values() if value["status"] == "completed"]
        ),
        "expected_cells": manifest["cell_count"],
        "cell_receipts": cell_receipts,
    }
    receipt["receipt_hash"] = digest(receipt)
    path = output / "stage_receipts" / f"{manifest['stage_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/loop_schedule_architecture_discovery_v1/runs",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = read_stage_manifest(args.manifest)
    profile = _load(ROOT / "configs/lsa/architecture_resource_profile_v1.json")
    receipt_dir = args.output / "resource_receipts"
    maximum_seconds = int(profile["caps"]["campaign_gpu_seconds"])
    maximum_attempts = 1 + int(profile["checkpoint"]["maximum_resumptions"])
    wrapper = ROOT / "scripts/run_lspg_architecture_cell.ps1"
    failure: str | None = None
    try:
        for cell in manifest["cells"]:
            if _completed_result(args.output, cell["cell_id"]):
                continue
            for attempt in range(1, maximum_attempts + 1):
                remaining = maximum_seconds - _elapsed_seconds(receipt_dir)
                if remaining <= 0:
                    raise RuntimeError("campaign_gpu_seconds_exhausted")
                timeout = min(
                    int(profile["caps"]["child_timeout_seconds"]),
                    max(1, int(remaining)),
                )
                command = [
                    "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(wrapper),
                    "-ProposalId", cell["proposal_id"],
                    "-Stage", cell["training_stage"],
                    "-Scale", cell["scale_rung"],
                    "-Seed", str(cell["seed"]),
                    "-TokenVisitBudget", str(cell["token_visit_budget"]),
                    "-Output", str(args.output),
                    "-Attempt", str(attempt),
                    "-TimeoutSecondsOverride", str(timeout),
                ]
                if cell["allow_locked_evaluation"]:
                    command.append("-AllowLockedEvaluation")
                if args.dry_run:
                    print(json.dumps({"cell_id": cell["cell_id"], "command": command}))
                    break
                completed = subprocess.run(command, cwd=ROOT, check=False)
                receipt_path = receipt_dir / (
                    f"{cell['cell_id']}.attempt-{attempt}.resource_receipt.json"
                )
                if not receipt_path.exists():
                    raise RuntimeError(f"missing resource receipt for {cell['cell_id']}")
                receipt = _load(receipt_path)
                if completed.returncode == 0 and _completed_result(args.output, cell["cell_id"]):
                    break
                if receipt.get("abort_reason") != "timeout" or attempt == maximum_attempts:
                    raise RuntimeError(
                        f"cell failed without resumable timeout: {cell['cell_id']} {receipt.get('abort_reason')}"
                    )
            else:
                raise RuntimeError(f"cell exhausted resumptions: {cell['cell_id']}")
    except Exception as error:
        failure = str(error)
        if not args.dry_run:
            _write_stage_receipt(
                args.output, args.manifest, manifest, status="failed", failure=failure
            )
        raise
    if not args.dry_run:
        _write_stage_receipt(
            args.output, args.manifest, manifest, status="completed", failure=None
        )


if __name__ == "__main__":
    main()
