"""Run the calibration-informed, untouched-evaluation successor to RLM hybrid v1."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from research_gym.benchmarks.rlm_hybrid_neighborhood import architecture_hashes, architecture_ids
from research_gym.benchmarks.rlm_hybrid_runtime import API_ARCHITECTURES, LOCAL_ARCHITECTURES, run_api_architecture, run_local_architecture
from research_gym.integrity import canonical_file_sha256, verify_file_sha256
from research_gym.scripts import bench_rlm_trm_ldt_hybrid_neighborhood_v1 as v1


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "rlm_trm_ldt_hybrid_neighborhood_v1_1.json"
REGISTRATION = ROOT / "configs" / "rlm_trm_ldt_hybrid_neighborhood_v1_1_registration.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "rlm_trm_ldt_hybrid_neighborhood_v1_1" / "campaign"
FINAL_RECEIPT = ROOT / "data" / "benchmarks" / "rlm_trm_ldt_hybrid_neighborhood_v1_1_receipt.json"


def load_registered_config(path: Path) -> tuple[dict[str, Any], str]:
    if not REGISTRATION.exists():
        raise RuntimeError("RLM hybrid v1.1 registration is absent")
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    config_hash = canonical_file_sha256(path)
    if config_hash != registration["config_sha256"]:
        raise RuntimeError("RLM hybrid v1.1 config hash does not match registration")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["protocol_id"] != registration["protocol_id"]:
        raise RuntimeError("RLM hybrid v1.1 protocol id changed")
    if tuple(config["architecture_order"]) != architecture_ids():
        raise RuntimeError("RLM hybrid v1.1 architecture order changed")
    if registration["architecture_hashes"] != architecture_hashes():
        raise RuntimeError("RLM hybrid v1.1 architecture hashes changed")
    paths = [
        (config["calibration_source"]["path"], config["calibration_source"]["sha256"]),
        (config["task_suite"]["path"], config["task_suite"]["sha256"]),
        (config["trained_proposals"]["path"], config["trained_proposals"]["sha256"]),
        (config["trained_proposals"]["checkpoint_manifest_path"], config["trained_proposals"]["checkpoint_manifest_sha256"]),
    ]
    paths.extend((row["path"], row["sha256"]) for row in config["trained_proposals"]["final_checkpoints"])
    for value, digest in paths:
        if not verify_file_sha256(v1._root_path(value), digest):
            raise RuntimeError(f"registered v1.1 artifact hash mismatch: {value}")
    return config, config_hash


def validate(config: Mapping[str, Any], config_hash: str, output: Path) -> dict[str, Any]:
    checkout = Path(config["official_rlm_source"]["checkout_path"]).resolve()
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if head != config["official_rlm_source"]["commit"]:
        raise RuntimeError("official RLM checkout commit changed")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is absent")
    calibration = json.loads(v1._root_path(config["calibration_source"]["path"]).read_text(encoding="utf-8"))
    if calibration["held_out_evaluation_observed"] or calibration["promotion_passed"]:
        raise RuntimeError("v1.1 calibration source does not encode the registered v1 stop")
    tasks = v1._load_tasks(config)
    proposals = v1._load_proposals(config)
    if len(proposals) != len(tasks) * len(config["replicate_seeds"]):
        raise RuntimeError("trained proposal table is incomplete")
    if any(task.split != "eval" for stage in config["execution"]["stages"] for task in v1._stage_tasks(stage, tasks)[1]):
        raise RuntimeError("v1.1 stage selection escaped the untouched evaluation split")
    if list((output / "shards").glob("*/result.json")) or (output / "result.json").exists() or FINAL_RECEIPT.exists():
        raise RuntimeError("RLM hybrid v1.1 scientific outcomes already exist")
    return {
        "status": "valid",
        "config_sha256": config_hash,
        "official_rlm_commit": head,
        "task_count": len(tasks),
        "eval_count": sum(task.split == "eval" for task in tasks),
        "proposal_count": len(proposals),
        "stage_count": len(config["execution"]["stages"]),
        "architecture_count": len(config["architecture_order"]),
        "api_architecture_count": len(config["api_architecture_order"]),
        "calibration_stop_sha256": config["calibration_source"]["sha256"],
        "outcomes_present": False,
    }


def _prior_typed_unsafe(config: Mapping[str, Any], output: Path) -> int:
    total = 0
    for stage in config["execution"]["stages"]:
        path = output / "shards" / stage / "result.json"
        if path.exists():
            total += int(json.loads(path.read_text(encoding="utf-8"))["typed_unsafe_count"])
    return total


def run_next_stage(config: Mapping[str, Any], config_hash: str, output: Path) -> dict[str, Any]:
    stage = v1._pending_stage(config, output)
    if stage is None:
        raise RuntimeError("all registered RLM hybrid v1.1 stages already exist")
    if _prior_typed_unsafe(config, output):
        raise RuntimeError("prior v1.1 shard contains typed unsafe execution; continuation prohibited")
    shard = output / "shards" / stage
    if shard.exists():
        raise RuntimeError(f"refusing to overwrite partial v1.1 shard: {stage}")
    shard.mkdir(parents=True)
    capability = v1._capability_gate(config, shard / "provider_capability_receipt.json")
    tasks = v1._load_tasks(config)
    proposals = v1._load_proposals(config)
    replicate_seed, selected = v1._stage_tasks(stage, tasks)
    records: list[dict[str, Any]] = []
    trajectories = []
    events_path = shard / "events.jsonl"
    existing_tokens = v1._existing_provider_tokens(output)
    for task_index, task in enumerate(selected):
        trained_row = proposals[(replicate_seed, task.task_id)]
        for architecture_id in LOCAL_ARCHITECTURES:
            record, trajectory = run_local_architecture(architecture_id, task, trained_row, replicate_seed)
            record["record_id"] = f"{stage}__{architecture_id}__{task.task_id}"
            record["stage"] = stage
            record["architecture_hash"] = architecture_hashes()[architecture_id]
            records.append(record)
            trajectory_path = shard / "trajectories" / f"{architecture_id}__{task.task_id}.json"
            v1._write_json(trajectory_path, trajectory)
            trajectories.append({"record_id": record["record_id"], "path": trajectory_path.relative_to(ROOT).as_posix(), "sha256": canonical_file_sha256(trajectory_path)})
        for architecture_id in v1._rotated_api_order(config, task_index, replicate_seed):
            if existing_tokens + sum(int(row["usage"]["total_tokens"]) for row in records) >= int(config["execution"]["cumulative_provider_token_soft_cap"]):
                raise RuntimeError("cumulative provider token soft cap reached before next v1.1 cell")
            v1._append_event(events_path, {"event": "cell_start", "stage": stage, "architecture_id": architecture_id, "task_id": task.task_id})
            try:
                record, trajectory = run_api_architecture(architecture_id, task, trained_row, replicate_seed, config["runtime"])
            except Exception as exc:
                if v1._is_global_provider_error(exc):
                    v1._write_json(shard / "provider_abort_receipt.json", {"status": "provider_access_abort", "stage": stage, "architecture_id": architecture_id, "task_id": task.task_id, "error": v1._error_summary(exc)})
                    raise RuntimeError("global provider access failed; stopped after first error") from exc
                record = {
                    "architecture_id": architecture_id, "task_id": task.task_id, "task_family": task.family,
                    "replicate_seed": replicate_seed,
                    "checkpoint_seed": replicate_seed if "trained_trm" in architecture_id or "conductor" in architecture_id else None,
                    "proposal_action": None, "executed_action": None, "optimal_action": task.optimal_action,
                    "safe": False, "unsafe": False, "correct": False, "utility": 0.0,
                    "regret": float(task.utilities[task.optimal_action]) + 1.0, "fallback": False,
                    "decision_reason": "cell_error",
                    "usage": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reported_cost_usd": None},
                    "execution_time": 0.0, "manipulation": {"passed": False, "reason": "cell_error"},
                    "tool_calls": [], "recursive_child_count": 0, "error": v1._error_summary(exc),
                }
                trajectory = {"architecture": architecture_id, "error": v1._error_summary(exc)}
            record["record_id"] = f"{stage}__{architecture_id}__{task.task_id}"
            record["stage"] = stage
            record["architecture_hash"] = architecture_hashes()[architecture_id]
            records.append(record)
            trajectory_path = shard / "trajectories" / f"{architecture_id}__{task.task_id}.json"
            v1._write_json(trajectory_path, trajectory)
            trajectories.append({"record_id": record["record_id"], "path": trajectory_path.relative_to(ROOT).as_posix(), "sha256": canonical_file_sha256(trajectory_path)})
            v1._append_event(events_path, {"event": "cell_complete", **record})
    records = sorted(records, key=lambda row: row["record_id"])
    v1._write_jsonl(shard / "records.jsonl", records)
    v1._write_json(shard / "trajectory_manifest.json", sorted(trajectories, key=lambda row: row["record_id"]))
    summary = v1._stage_summary(records)
    typed_unsafe = sum(summary[name]["unsafe_count"] for name in v1.TYPED_ARCHITECTURES)
    conductor_manipulation = sum(summary[name]["manipulation_failures"] for name in ("rlm_tool_conductor", "rlm_recursive_conductor"))
    result = {
        "protocol_id": config["protocol_id"], "status": "complete", "stage": stage,
        "config_sha256": config_hash, "git_head": v1._git_head(), "replicate_seed": replicate_seed,
        "families": sorted({task.family for task in selected}), "task_count": len(selected), "record_count": len(records),
        "records_path": (shard / "records.jsonl").relative_to(ROOT).as_posix(), "records_sha256": canonical_file_sha256(shard / "records.jsonl"),
        "events_path": events_path.relative_to(ROOT).as_posix(), "events_sha256": canonical_file_sha256(events_path),
        "trajectory_manifest_path": (shard / "trajectory_manifest.json").relative_to(ROOT).as_posix(), "trajectory_manifest_sha256": canonical_file_sha256(shard / "trajectory_manifest.json"),
        "provider_capability_path": (shard / "provider_capability_receipt.json").relative_to(ROOT).as_posix(), "provider_capability_sha256": canonical_file_sha256(shard / "provider_capability_receipt.json"),
        "provider_capability_usage": capability["usage"], "summary": summary, "typed_unsafe_count": typed_unsafe,
        "conductor_manipulation_failures": conductor_manipulation,
        "cell_errors": sum(row.get("decision_reason") == "cell_error" for row in records),
        "calibration_informed_continuation": True,
    }
    v1._write_json(shard / "result.json", result)
    return result


def finalize(config: Mapping[str, Any], config_hash: str, output: Path) -> dict[str, Any]:
    v1.DEFAULT_CONFIG = DEFAULT_CONFIG
    v1.FINAL_RECEIPT = FINAL_RECEIPT
    return v1.finalize(config, config_hash, output)


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
            value = validate(config, config_hash, output)
        elif args.phase == "run":
            value = run_next_stage(config, config_hash, output)
        else:
            value = finalize(config, config_hash, output)
        print(json.dumps(value, sort_keys=True))
    finally:
        gc.collect()


if __name__ == "__main__":
    main()
