"""Run the transport-qualified successor of the RLM neighborhood pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from research_gym.benchmarks.rlm_architecture_neighborhood import (
    architecture_manifest,
    run_architecture,
    usage_totals,
)
from research_gym.integrity import canonical_file_sha256, verify_file_sha256
from research_gym.scripts.bench_rlm_architecture_neighborhood_v0 import (
    _aggregate,
    _append_event,
    _external_path,
    _git_head,
    _root_path,
    _write_json,
    _write_records,
)
from research_gym.benchmarks.rlm_architecture_neighborhood import (
    answer_matches,
    pareto_frontier,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "rlm_architecture_neighborhood_v0_1.json"
REGISTRATION = ROOT / "configs" / "rlm_architecture_neighborhood_v0_1_registration.json"
TASKS_PATH = ROOT / "data" / "benchmarks" / "rlm_architecture_neighborhood_v0_tasks.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "rlm_architecture_neighborhood_v0_1"
FINAL_RECEIPT = ROOT / "data" / "benchmarks" / "rlm_architecture_neighborhood_v0_1_receipt.json"


def _provider_error_summary(exc: Exception) -> dict[str, Any]:
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    return {
        "error_type": type(exc).__name__,
        "status_code": getattr(exc, "status_code", None),
        "provider_code": error.get("code") if isinstance(error, dict) else None,
        "provider_type": error.get("type") if isinstance(error, dict) else None,
    }


def _is_global_provider_error(exc: Exception) -> bool:
    summary = _provider_error_summary(exc)
    return summary["status_code"] in {401, 403} or summary["provider_code"] in {
        "invalid_api_key",
        "model_not_found",
    }


def load_registered_config(path: Path) -> tuple[dict[str, Any], str]:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    config_hash = canonical_file_sha256(path)
    if config_hash != registration["config_sha256"]:
        raise RuntimeError("RLM neighborhood successor config does not match registration")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["protocol_id"] != registration["protocol_id"]:
        raise RuntimeError("RLM neighborhood successor protocol id changed")
    if not verify_file_sha256(TASKS_PATH, config["task_suite"]["sha256"]):
        raise RuntimeError("RLM task suite hash mismatch")
    if config["task_suite"]["sha256"] != config["preserved_from_v0"]["task_suite_sha256"]:
        raise RuntimeError("successor task suite does not preserve v0")
    manifest = {row["architecture_id"]: row["architecture_hash"] for row in architecture_manifest()}
    if manifest != config["preserved_from_v0"]["architecture_hashes"]:
        raise RuntimeError("successor architecture hashes do not preserve v0")
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
    expected = {
        "python": config["runtime"]["python_version"],
        "openai": config["runtime"]["openai_package_version"],
        "rlms": config["official_rlm_source"]["package"].split("==", 1)[1],
    }
    if environment != expected:
        raise RuntimeError(f"registered RLM Python environment changed: {environment} != {expected}")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is absent")
    outcomes = (
        output_dir / "provider_capability_receipt.json",
        output_dir / "rlm_neighborhood_records.jsonl",
        output_dir / "rlm_neighborhood_result.json",
        output_dir / "result_receipt.json",
        FINAL_RECEIPT,
    )
    existing = [str(path) for path in outcomes if path.exists()]
    if existing:
        raise RuntimeError(f"successor RLM outcome already exists: {existing}")
    return {
        "status": "valid",
        "config_sha256": config_hash,
        "task_count": 4,
        "architecture_count": 4,
        "cell_count": 16,
        "official_rlm_commit": head,
        "python_environment": environment,
        "model": config["runtime"]["model"],
        "restricted_repl": True,
        "outcomes_present": False,
    }


def _provider_capability_gate(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    from rlm.clients.openai import OpenAIClient

    gate = config["provider_capability_gate"]
    receipt_path = output_dir / "provider_capability_receipt.json"
    if receipt_path.exists():
        raise RuntimeError("refusing to overwrite provider capability receipt")
    started = datetime.now(timezone.utc)
    client = OpenAIClient(
        model_name=config["runtime"]["model"],
        max_retries=0,
        sampling_args={"max_tokens": int(gate["max_output_tokens"])},
    )
    try:
        response = client.completion(gate["prompt"])
        usage = usage_totals(client.get_usage_summary().to_dict())
        passed = response.strip() == gate["expected_response"]
        receipt = {
            "protocol_id": config["protocol_id"],
            "status": "passed" if passed else "failed",
            "model": config["runtime"]["model"],
            "transport": gate["transport"],
            "expected_response": gate["expected_response"],
            "response": response.strip(),
            "usage": usage,
            "excluded_from_architecture_metrics": True,
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        receipt = {
            "protocol_id": config["protocol_id"],
            "status": "failed",
            "model": config["runtime"]["model"],
            "transport": gate["transport"],
            "error": _provider_error_summary(exc),
            "excluded_from_architecture_metrics": True,
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(receipt_path, receipt)
        raise RuntimeError("provider capability gate failed") from exc
    _write_json(receipt_path, receipt)
    if receipt["status"] != "passed":
        raise RuntimeError("provider capability gate returned the wrong response")
    return receipt


def run(config: Mapping[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    records_path = output_dir / "rlm_neighborhood_records.jsonl"
    result_path = output_dir / "rlm_neighborhood_result.json"
    event_path = output_dir / "rlm_neighborhood_events.jsonl"
    trajectory_dir = output_dir / "trajectories"
    abort_path = output_dir / "provider_abort_receipt.json"
    if any(path.exists() for path in (records_path, result_path, event_path, trajectory_dir, abort_path)):
        raise RuntimeError("refusing to overwrite successor RLM outcomes")
    capability = _provider_capability_gate(config, output_dir)
    capability_path = output_dir / "provider_capability_receipt.json"
    trajectory_dir.mkdir(parents=True)
    tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))["tasks"]
    architecture_by_id = {row["architecture_id"]: row for row in architecture_manifest()}
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
                if _is_global_provider_error(exc):
                    abort = {
                        "protocol_id": config["protocol_id"],
                        "status": "provider_access_abort",
                        "cell_id": cell_id,
                        "completed_cells": len(records),
                        "error": _provider_error_summary(exc),
                        "finished_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    _write_json(abort_path, abort)
                    _append_event(event_path, {"event": "provider_access_abort", **abort})
                    raise RuntimeError("global provider access failed; stopped after first error") from exc
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
        "provider_capability_path": capability_path.relative_to(ROOT).as_posix(),
        "provider_capability_sha256": canonical_file_sha256(capability_path),
        "provider_capability_usage": capability["usage"],
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
    capability_path = output_dir / "provider_capability_receipt.json"
    required = (result_path, records_path, resource_path, manifest_path, capability_path)
    if not all(path.exists() for path in required):
        raise RuntimeError("successor RLM neighborhood run is incomplete")
    if (output_dir / "result_receipt.json").exists() or FINAL_RECEIPT.exists():
        raise RuntimeError("successor RLM neighborhood result already sealed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    capability = json.loads(capability_path.read_text(encoding="utf-8"))
    if result["config_sha256"] != config_hash or result["status"] != "complete":
        raise RuntimeError("successor RLM result does not match registered config")
    if capability["status"] != "passed":
        raise RuntimeError("provider capability gate did not pass")
    checks = (
        (records_path, result["records_sha256"]),
        (manifest_path, result["trajectory_manifest_sha256"]),
        (capability_path, result["provider_capability_sha256"]),
    )
    if not all(verify_file_sha256(path, digest) for path, digest in checks):
        raise RuntimeError("successor RLM artifact failed re-verification")
    for entry in json.loads(manifest_path.read_text(encoding="utf-8")):
        if not verify_file_sha256(_root_path(entry["path"]), entry["sha256"]):
            raise RuntimeError(f"RLM trajectory failed re-verification: {entry['cell_id']}")
    if resource["status"] != "completed" or not resource["cleanup_passed"]:
        raise RuntimeError("successor RLM resource receipt did not pass")
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
        "provider_capability_path": capability_path.relative_to(ROOT).as_posix(),
        "provider_capability_sha256": canonical_file_sha256(capability_path),
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
    parser.add_argument("--phase", choices=("validate", "run", "finalize"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vram-fraction", type=float)
    args = parser.parse_args()
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
