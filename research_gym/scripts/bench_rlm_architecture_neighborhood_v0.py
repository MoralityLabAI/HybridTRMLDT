"""Materialize, run, and seal the registered RLM architecture neighborhood."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from research_gym.benchmarks.rlm_architecture_neighborhood import (
    answer_matches,
    architecture_manifest,
    materialize_tasks,
    pareto_frontier,
    run_architecture,
    usage_totals,
)
from research_gym.integrity import canonical_file_sha256, verify_file_sha256


ROOT = Path(__file__).resolve().parents[2]


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for record in sorted(records, key=lambda row: row["record_id"]):
            handle.write(_canonical_bytes(record))


def _append_event(path: Path, event: dict[str, Any]) -> None:
    payload = dict(event)
    payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_canonical_bytes(payload))


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


DEFAULT_CONFIG = ROOT / "configs" / "rlm_architecture_neighborhood_v0.json"
REGISTRATION = ROOT / "configs" / "rlm_architecture_neighborhood_v0_registration.json"
TASKS_PATH = ROOT / "data" / "benchmarks" / "rlm_architecture_neighborhood_v0_tasks.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "rlm_architecture_neighborhood_v0"
FINAL_RECEIPT = ROOT / "data" / "benchmarks" / "rlm_architecture_neighborhood_v0_receipt.json"


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if path != ROOT.resolve() and ROOT.resolve() not in path.parents:
        raise RuntimeError(f"registered path escapes repository: {value}")
    return path


def _external_path(value: str) -> Path:
    return Path(value).resolve()


def materialize(path: Path) -> dict[str, Any]:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite task suite: {path}")
    payload = materialize_tasks()
    _write_json(path, payload)
    return {
        "status": "materialized",
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": canonical_file_sha256(path),
        "task_count": payload["task_count"],
    }


def load_registered_config(path: Path) -> tuple[dict[str, Any], str]:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    config_hash = canonical_file_sha256(path)
    if config_hash != registration["config_sha256"]:
        raise RuntimeError("RLM neighborhood config does not match frozen registration")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["protocol_id"] != registration["protocol_id"]:
        raise RuntimeError("RLM neighborhood protocol id does not match registration")
    if not verify_file_sha256(TASKS_PATH, config["task_suite"]["sha256"]):
        raise RuntimeError("RLM task suite hash mismatch")
    for source in config["official_rlm_source"]["files"]:
        if not verify_file_sha256(_external_path(source["path"]), source["sha256"]):
            raise RuntimeError(f"official RLM source hash mismatch: {source['path']}")
    return config, config_hash


def validate(config: Mapping[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    source = config["official_rlm_source"]
    checkout = _external_path(source["checkout_path"])
    head = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != source["commit"]:
        raise RuntimeError("official RLM checkout commit changed")
    interpreter = _external_path(config["runtime"]["python_executable"])
    if not interpreter.exists():
        raise RuntimeError("registered RLM Python interpreter is missing")
    environment = json.loads(
        subprocess.check_output(
            [
                str(interpreter),
                "-c",
                (
                    "import importlib.metadata,json,openai,platform;"
                    "print(json.dumps({'python':platform.python_version(),"
                    "'openai':openai.__version__,'rlms':importlib.metadata.version('rlms')}))"
                ),
            ],
            text=True,
        )
    )
    expected_environment = {
        "python": config["runtime"]["python_version"],
        "openai": config["runtime"]["openai_package_version"],
        "rlms": config["official_rlm_source"]["package"].split("==", 1)[1],
    }
    if environment != expected_environment:
        raise RuntimeError(
            f"registered RLM Python environment changed: {environment} != {expected_environment}"
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is absent")
    tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    if tasks["task_count"] != 4:
        raise RuntimeError("registered RLM task count changed")
    architecture_ids = [row["architecture_id"] for row in architecture_manifest()]
    if architecture_ids != config["architecture_order"]:
        raise RuntimeError("registered architecture order changed")
    outcomes = (
        output_dir / "rlm_neighborhood_records.jsonl",
        output_dir / "rlm_neighborhood_result.json",
        output_dir / "result_receipt.json",
        FINAL_RECEIPT,
    )
    existing = [str(path) for path in outcomes if path.exists()]
    if existing:
        raise RuntimeError(f"registered RLM outcome already exists: {existing}")
    return {
        "status": "valid",
        "config_sha256": config_hash,
        "task_count": 4,
        "architecture_count": 4,
        "cell_count": 16,
        "official_rlm_commit": head,
        "python_environment": environment,
        "restricted_repl": True,
        "outcomes_present": False,
    }


def _aggregate(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["architecture_id"])].append(row)
    aggregates = []
    for architecture_id, rows in sorted(grouped.items()):
        completed = [row for row in rows if row["status"] == "complete"]
        aggregates.append(
            {
                "architecture_id": architecture_id,
                "cells": len(rows),
                "completed_cells": len(completed),
                "accuracy": sum(bool(row["answer_match"]) for row in rows) / len(rows),
                "tasks_solved": sum(bool(row["answer_match"]) for row in rows),
                "total_calls": sum(int(row["usage"]["calls"]) for row in completed),
                "total_tokens": sum(int(row["usage"]["total_tokens"]) for row in completed),
                "execution_time": sum(float(row["execution_time"]) for row in completed),
                "maximum_observed_depth": max(
                    (int(row["trajectory_metrics"]["maximum_observed_depth"]) for row in completed),
                    default=0,
                ),
                "errors": len(rows) - len(completed),
            }
        )
    return aggregates


def run(config: Mapping[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    records_path = output_dir / "rlm_neighborhood_records.jsonl"
    result_path = output_dir / "rlm_neighborhood_result.json"
    event_path = output_dir / "rlm_neighborhood_events.jsonl"
    trajectory_dir = output_dir / "trajectories"
    if any(path.exists() for path in (records_path, result_path, event_path, trajectory_dir)):
        raise RuntimeError("refusing to overwrite RLM neighborhood outcomes")
    trajectory_dir.mkdir(parents=True)
    tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))["tasks"]
    architecture_by_id = {
        row["architecture_id"]: row for row in architecture_manifest()
    }
    records: list[dict[str, Any]] = []
    trajectory_manifest = []
    for architecture_id in config["architecture_order"]:
        architecture = architecture_by_id[architecture_id]
        for task in tasks:
            cell_id = f"{architecture_id}__{task['task_id']}"
            _append_event(
                event_path,
                {
                    "event": "cell_start",
                    "cell_id": cell_id,
                    "architecture_id": architecture_id,
                    "task_id": task["task_id"],
                },
            )
            try:
                response, trajectory = run_architecture(
                    architecture_id, task["prompt"], config["runtime"]
                )
                usage = usage_totals(trajectory["usage_summary"])
                status = "complete"
                error = None
            except Exception as exc:
                response = ""
                trajectory = {
                    "architecture": architecture_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "usage_summary": {"model_usage_summaries": {}},
                    "execution_time": 0.0,
                    "trajectory_metrics": {
                        "iterations": 0,
                        "code_blocks": 0,
                        "subcalls": 0,
                        "recursive_subcalls": 0,
                        "maximum_observed_depth": 0,
                        "repl_errors": 0,
                    },
                }
                usage = usage_totals(trajectory["usage_summary"])
                status = "error"
                error = {"type": type(exc).__name__, "message": str(exc)}
            trajectory_path = trajectory_dir / f"{cell_id}.json"
            _write_json(trajectory_path, trajectory)
            trajectory_sha = canonical_file_sha256(trajectory_path)
            trajectory_manifest.append(
                {
                    "cell_id": cell_id,
                    "path": trajectory_path.relative_to(ROOT).as_posix(),
                    "sha256": trajectory_sha,
                }
            )
            record = {
                "record_id": cell_id,
                "status": status,
                "architecture_id": architecture_id,
                "architecture_hash": architecture["architecture_hash"],
                "architecture_features": architecture["features"],
                "distance_from_rlm_depth_1": architecture["distance_from_rlm_depth_1"],
                "task_id": task["task_id"],
                "task_family": task["family"],
                "prompt_sha256": task["prompt_sha256"],
                "expected_answer": task["expected_answer"],
                "response": response,
                "answer_match": answer_matches(response, task["expected_answer"]),
                "usage": usage,
                "execution_time": float(trajectory["execution_time"]),
                "trajectory_metrics": trajectory["trajectory_metrics"],
                "trajectory_path": trajectory_path.relative_to(ROOT).as_posix(),
                "trajectory_sha256": trajectory_sha,
                "error": error,
            }
            records.append(record)
            _append_event(event_path, {"event": "cell_complete", **record})
    _write_records(records_path, records)
    manifest_path = output_dir / "trajectory_manifest.json"
    _write_json(manifest_path, trajectory_manifest)
    aggregates = _aggregate(records)
    complete_run = all(row["status"] == "complete" for row in records)
    result = {
        "protocol_id": config["protocol_id"],
        "status": "complete",
        "phase": "run",
        "config_sha256": config_hash,
        "git_head": _git_head(),
        "official_rlm_commit": config["official_rlm_source"]["commit"],
        "records_path": records_path.relative_to(ROOT).as_posix(),
        "records_sha256": canonical_file_sha256(records_path),
        "record_count": len(records),
        "events_path": event_path.relative_to(ROOT).as_posix(),
        "events_sha256": canonical_file_sha256(event_path),
        "trajectory_manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "trajectory_manifest_sha256": canonical_file_sha256(manifest_path),
        "summary": {
            "scientific_status": "complete" if complete_run else "completed_with_cell_errors",
            "architectures": aggregates,
            "pareto_frontier": pareto_frontier(aggregates),
            "cell_errors": [row for row in records if row["status"] != "complete"],
            "claim_scope": "architecture-neighborhood pilot",
        },
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(result_path, result)
    return result


def finalize(config: Mapping[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    result_path = output_dir / "rlm_neighborhood_result.json"
    records_path = output_dir / "rlm_neighborhood_records.jsonl"
    resource_path = output_dir / "run.resource_receipt.json"
    manifest_path = output_dir / "trajectory_manifest.json"
    if not all(path.exists() for path in (result_path, records_path, resource_path, manifest_path)):
        raise RuntimeError("RLM neighborhood run is incomplete")
    if (output_dir / "result_receipt.json").exists() or FINAL_RECEIPT.exists():
        raise RuntimeError("RLM neighborhood result already sealed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    if result["config_sha256"] != config_hash or result["status"] != "complete":
        raise RuntimeError("RLM neighborhood result does not match registered config")
    if not verify_file_sha256(records_path, result["records_sha256"]):
        raise RuntimeError("RLM neighborhood records failed re-verification")
    if not verify_file_sha256(manifest_path, result["trajectory_manifest_sha256"]):
        raise RuntimeError("RLM trajectory manifest failed re-verification")
    for entry in json.loads(manifest_path.read_text(encoding="utf-8")):
        if not verify_file_sha256(_root_path(entry["path"]), entry["sha256"]):
            raise RuntimeError(f"RLM trajectory failed re-verification: {entry['cell_id']}")
    if resource["status"] != "completed" or not resource["cleanup_passed"]:
        raise RuntimeError("RLM neighborhood resource receipt did not pass")
    receipt = {
        "protocol_id": config["protocol_id"],
        "status": "sealed",
        "config_path": DEFAULT_CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": config_hash,
        "result_path": result_path.relative_to(ROOT).as_posix(),
        "result_sha256": canonical_file_sha256(result_path),
        "records_path": records_path.relative_to(ROOT).as_posix(),
        "records_sha256": canonical_file_sha256(records_path),
        "record_count": result["record_count"],
        "trajectory_manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "trajectory_manifest_sha256": canonical_file_sha256(manifest_path),
        "resource_path": resource_path.relative_to(ROOT).as_posix(),
        "resource_sha256": canonical_file_sha256(resource_path),
        "cleanup_passed": True,
        "scientific_status": result["summary"]["scientific_status"],
        "pareto_frontier": result["summary"]["pareto_frontier"],
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(output_dir / "result_receipt.json", receipt)
    _write_json(FINAL_RECEIPT, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("materialize", "validate", "run", "finalize"), required=True
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks", type=Path, default=TASKS_PATH)
    parser.add_argument("--vram-fraction", type=float)
    args = parser.parse_args()
    if args.phase == "materialize":
        print(json.dumps(materialize(args.tasks.resolve()), sort_keys=True))
        return
    config, config_hash = load_registered_config(args.config.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        if args.phase == "validate":
            payload = validate(config, config_hash, output)
        elif args.phase == "run":
            payload = run(config, config_hash, output)
        else:
            payload = finalize(config, config_hash, output)
        print(json.dumps(payload, sort_keys=True))
    finally:
        gc.collect()


if __name__ == "__main__":
    main()
