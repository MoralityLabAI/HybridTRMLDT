"""Seal Stage-A posterior, report, and receipt after completion or abort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research_gym.prognostics.reporting import (
    canonical_bytes,
    load_partial_records,
    sha256,
    summarize_stage_a,
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    proposal_dir = args.proposal_dir.resolve()
    run_dir = args.run_dir.resolve()
    output = args.out.resolve()
    proposals = [
        json.loads(line)
        for line in (proposal_dir / "proposal_table.jsonl").read_text().splitlines()
    ]
    raw_partial = run_dir / "screening_records.partial.jsonl"
    records = load_partial_records(raw_partial)
    resource_path = run_dir / "stage_a.resource_receipt.json"
    cleanup_path = run_dir / "post_run_cleanup.json"
    resource = json.loads(resource_path.read_text(encoding="utf-8-sig"))
    cleanup = json.loads(cleanup_path.read_text(encoding="utf-8-sig"))
    summary = summarize_stage_a(proposals, records, resource)
    summary["resource_context"] = {
        "cleanup_passed": cleanup["cleanup_passed"],
        "lingering_owned_pids": len(cleanup["lingering_owned_pids"]),
        "concurrent_suspect_processes": len(cleanup["suspect_training_processes"]),
        "gpu_before": cleanup["gpu_before"]["gpu"],
        "gpu_after": cleanup["gpu_after"]["gpu"],
        "calibration_status": "confounded_by_concurrent_external_workload",
    }
    normalized_path = output / "receipts" / "stage_a_records.normalized.jsonl"
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    with normalized_path.open("wb") as handle:
        for record in sorted(records, key=lambda row: row["proposal_id"]):
            handle.write(canonical_bytes(record))
    posterior = {
        "protocol_id": summary["protocol_id"],
        "status": "partial_update_only",
        "observations": [
            {
                "proposal_id": row["proposal_id"],
                "status": row["status"],
                "stop_reason": row.get("stop_reason"),
                "maximum_gradient_norm": row.get("max_gradient_norm"),
                "loss_ratio": row.get("final_over_initial_loss"),
                "state_visit_exposures": row.get("state_visit_exposures"),
            }
            for row in records
        ],
        "boundary_prior": {
            "kind": "left_censored",
            "upper": 0.15,
            "cells": 2,
            "resolved": False,
        },
        "architecture_ranking_permitted": False,
        "reason": "nine proposals lack final Stage-A receipts",
        "resource_calibration": {
            "status": "confounded",
            "forecast_comparisons": summary["forecast_calibration"],
            "concurrent_suspect_processes": len(cleanup["suspect_training_processes"]),
        },
    }
    posterior_path = output / "posterior" / "stage_a_posterior.json"
    _write(posterior_path, posterior)
    report_lines = [
        "# LSPG-v0 Stage A Report",
        "",
        "## Outcome",
        "",
        "Stage A timed out at the registered 1,800-second wall limit. Three of twelve proposals produced receipts; nine remain pending. This is an incomplete cost-calibration result, not an architecture ranking.",
        "",
        "## Observed proposals",
        "",
        "| Proposal | Status | Stop | Exposures | Max gradient | Loss ratio | Mean step seconds |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in records:
        report_lines.append(
            "| {proposal_id} | {status} | {stop} | {exposures} | {gradient} | {ratio} | {seconds} |".format(
                proposal_id=row["proposal_id"],
                status=row["status"],
                stop=row.get("stop_reason") or "none",
                exposures=row.get("state_visit_exposures"),
                gradient=row.get("max_gradient_norm"),
                ratio=row.get("final_over_initial_loss"),
                seconds=row.get("mean_step_seconds"),
            )
        )
    report_lines.extend(
        [
            "",
            "The two lower-p grid extensions stopped on their first backward pass because gradient norms exceeded 100. The untied R=4 control completed 8,192 state-visit exposures. The next proposal reached a local 10% checkpoint but has no final receipt and is not included in the posterior.",
            "",
            "## Promotion",
            "",
            "No promotion is allowed. Stage A is incomplete, the original boundary remains left-censored, the run timed out, and measured throughput invalidates the nominal local compute calibration.",
            "",
            "The cleanup audit observed two unrelated Research_Engine processes and 100% GPU utilization both before and after LSPG cleanup. No LSPG-owned process lingered. The measured 27,034x forecast ratio is therefore retained as confounded calibration evidence, not treated as a clean device-throughput estimate.",
            "",
            "## Integrity",
            "",
            f"- Raw partial SHA-256: `{sha256(raw_partial)}`.",
            f"- Normalized records SHA-256: `{sha256(normalized_path)}`.",
            f"- Resource receipt SHA-256: `{sha256(resource_path)}`.",
            f"- Cleanup receipt SHA-256: `{sha256(cleanup_path)}`.",
            "- Non-finite values in immediate-stop raw rows are preserved in the raw trace and normalized to null in the strict JSON receipt.",
            "",
            "No prediction about task accuracy, sample efficiency, or reasoning quality is made. LSPG-v0 remains scoped to trainability boundaries, cost, and experiment selection.",
        ]
    )
    report_path = output / "reports" / "lspg_v0_stage_a.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    result_path = output / "receipts" / "stage_a_result.json"
    _write(result_path, summary)
    receipt = {
        "protocol_id": summary["protocol_id"],
        "status": "sealed_incomplete_stage_a",
        "proposal_receipt_sha256": sha256(proposal_dir / "proposal_receipt.json"),
        "raw_partial_sha256": sha256(raw_partial),
        "normalized_records_sha256": sha256(normalized_path),
        "resource_receipt_sha256": sha256(resource_path),
        "cleanup_receipt_sha256": sha256(cleanup_path),
        "posterior_sha256": sha256(posterior_path),
        "stage_a_result_sha256": sha256(result_path),
        "report_sha256": sha256(report_path),
        "promotion_allowed": False,
        "resource_calibration_status": "confounded_by_concurrent_external_workload",
        "outcome_counts": {
            "proposals": summary["proposal_count"],
            "receipts": summary["receipt_count"],
            "completed": summary["completed_count"],
            "stopped": summary["stopped_count"],
            "pending": len(summary["pending_proposal_ids"]),
        },
    }
    _write(output / "receipts" / "lspg_v0_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
