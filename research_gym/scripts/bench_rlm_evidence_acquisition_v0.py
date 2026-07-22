"""Calibrate, register, run, and seal the RLM evidence-acquisition mesh."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from statistics import mean
from typing import Any, Mapping, Sequence
from uuid import uuid4

from research_gym.benchmarks.rlm_evidence_acquisition import (
    ACQUISITION_CONTRACT_HASH,
    API_ARCHITECTURES,
    ARCHITECTURES,
    architecture_hashes,
    architecture_query_sequence,
    acquisition_context,
    build_initial_snapshot,
    context_privacy_violations,
    execute_acquisition,
    sequence_oracle,
    verify_evidence_receipt,
)
from research_gym.benchmarks.rlm_evidence_runtime import run_rlm_acquisition
from research_gym.benchmarks.rlm_hybrid_neighborhood import (
    FAMILIES,
    LongContextControlTask,
    canonical_sha256,
)
from research_gym.benchmarks.rlm_hybrid_runtime import _empty_usage
from research_gym.integrity import canonical_file_sha256, verify_file_sha256
from research_gym.scripts import bench_rlm_trm_ldt_hybrid_neighborhood_v1 as parent


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/rlm_evidence_acquisition_mesh_v0.json"
REGISTRATION = ROOT / "configs/rlm_evidence_acquisition_mesh_v0_registration.json"
DEFAULT_OUTPUT = ROOT / "experiments/rlm_evidence_acquisition_mesh_v0"
CALIBRATION_RECEIPT = DEFAULT_OUTPUT / "calibration/calibration_receipt.json"
FINAL_RECEIPT = ROOT / "data/benchmarks/rlm_evidence_acquisition_mesh_v0_receipt.json"
DEFAULT_REPORT = ROOT / "reports/rlm_evidence_acquisition_mesh_v0.md"


def _root_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"registered path escapes repository: {value}")
    return resolved


def _git_head() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def _official_rlm_gate(config: Mapping[str, Any]) -> str:
    source = config["official_rlm_source"]
    checkout = Path(source["checkout_path"]).resolve()
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if head != source["commit"]:
        raise RuntimeError("official RLM checkout commit changed")
    if subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("official RLM checkout is dirty")
    if Path(sys.executable).resolve() != Path(source["python_executable"]).resolve():
        raise RuntimeError("experiment is not running under the registered official-RLM interpreter")
    return head


def _raw_config(path: Path) -> tuple[dict[str, Any], str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if tuple(config["architecture_order"]) != ARCHITECTURES:
        raise RuntimeError("evidence-acquisition architecture order changed")
    _verify_declared_inputs(config)
    return config, canonical_file_sha256(path)


def _verify_declared_inputs(config: Mapping[str, Any]) -> None:
    declared = (
        config["task_suite"],
        config["trained_proposals"],
        config["evidence_truth"],
        config["materialization_receipt"],
    )
    for value in declared:
        if not verify_file_sha256(_root_path(value["path"]), str(value["sha256"])):
            raise RuntimeError(f"declared artifact hash mismatch: {value['path']}")
    checkpoint = config["trained_proposals"]
    if not verify_file_sha256(
        _root_path(checkpoint["checkpoint_path"]), str(checkpoint["checkpoint_sha256"])
    ):
        raise RuntimeError("declared ControlTRM checkpoint hash mismatch")


def _load_inputs(
    config: Mapping[str, Any],
) -> tuple[list[LongContextControlTask], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = json.loads(_root_path(config["task_suite"]["path"]).read_text(encoding="utf-8"))
    tasks = [LongContextControlTask.from_jsonable(row) for row in payload["tasks"]]
    proposals = {
        str(row["task_id"]): row
        for row in map(
            json.loads,
            _root_path(config["trained_proposals"]["path"]).read_text(encoding="utf-8").splitlines(),
        )
    }
    truth = {
        str(row["task_id"]): row
        for row in map(
            json.loads,
            _root_path(config["evidence_truth"]["path"]).read_text(encoding="utf-8").splitlines(),
        )
    }
    if {task.task_id for task in tasks} != set(proposals) or set(proposals) != set(truth):
        raise RuntimeError("task, proposal, and evidence-truth IDs differ")
    return tasks, proposals, truth


def _selected_tasks(config: Mapping[str, Any], split: str) -> list[LongContextControlTask]:
    tasks, _, _ = _load_inputs(config)
    candidates = [task for task in tasks if task.split == split]
    if split == "eval":
        return sorted(candidates, key=lambda task: (FAMILIES.index(task.family), task.task_id))
    count = int(config["calibration"]["tasks_per_family"])
    salt = str(config["calibration"]["selection_salt"])
    selected = []
    for family in FAMILIES:
        family_tasks = sorted(
            (task for task in candidates if task.family == family),
            key=lambda task: canonical_sha256((salt, task.task_id)),
        )
        selected.extend(family_tasks[:count])
    return selected


def _construction_gate(
    tasks: Sequence[LongContextControlTask],
    proposals: Mapping[str, Mapping[str, Any]],
    truth: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gains = []
    sequences = Counter()
    for task in tasks:
        baseline = execute_acquisition(task, proposals[task.task_id], truth[task.task_id], ())
        oracle = sequence_oracle(task, proposals[task.task_id], truth[task.task_id])
        gains.append(oracle.net_utility - baseline.net_utility)
        sequences[oracle.query_sequence] += 1
    two_query = {key: value for key, value in sequences.items() if len(key) == 2}
    thresholds = config["calibration"]["construction_gate"]
    mean_gain = mean(gains)
    positive = sum(value > 1e-12 for value in gains)
    distinct = len(two_query)
    max_share = max(two_query.values()) / sum(two_query.values()) if two_query else 1.0
    passed = (
        mean_gain >= float(thresholds["minimum_mean_oracle_net_gain"])
        and positive >= int(thresholds["minimum_positive_tasks"])
        and distinct >= int(thresholds["minimum_distinct_two_query_sequences"])
        and max_share < float(thresholds["maximum_two_query_sequence_share"])
    )
    return {
        "passed": passed,
        "task_count": len(tasks),
        "mean_sequence_oracle_net_gain": mean_gain,
        "positive_gain_tasks": positive,
        "oracle_sequence_counts": {
            "->".join(sequence) if sequence else "no_query": count
            for sequence, count in sorted(sequences.items())
        },
        "distinct_two_query_sequences": distinct,
        "maximum_two_query_sequence_share": max_share,
        "thresholds": thresholds,
    }


def _record(
    architecture: str,
    task: LongContextControlTask,
    proposal: Mapping[str, Any],
    truth: Mapping[str, Any],
    outcome,
    *,
    contract_passed: bool | None,
    valid_rounds: int,
    stopped: bool,
    invalid_decision: bool,
    provider_error: Mapping[str, Any] | None,
    usage: Mapping[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    oracle = sequence_oracle(task, proposal, truth)
    baseline = execute_acquisition(task, proposal, truth, ())
    return {
        "record_id": f"{architecture}__{task.task_id}",
        "architecture_id": architecture,
        "architecture_hash": architecture_hashes()[architecture],
        "task_id": task.task_id,
        "task_family": task.family,
        "executed_action": outcome.executed_action,
        "optimal_action": task.optimal_action,
        "safe": outcome.executed_action in task.exact_allowed,
        "unsafe": outcome.executed_action not in task.exact_allowed,
        "correct": outcome.executed_action == task.optimal_action,
        "raw_utility": outcome.raw_utility,
        "query_cost": outcome.query_cost,
        "net_utility": outcome.net_utility,
        "baseline_action": baseline.executed_action,
        "baseline_raw_utility": baseline.raw_utility,
        "oracle_action": oracle.executed_action,
        "oracle_query_sequence": list(oracle.query_sequence),
        "oracle_net_utility": oracle.net_utility,
        "net_regret_to_oracle": oracle.net_utility - outcome.net_utility,
        "action_changed": outcome.action_changed,
        "beneficial_change": outcome.raw_utility > baseline.raw_utility + 1e-12,
        "harmful_change": outcome.raw_utility < baseline.raw_utility - 1e-12,
        "query_sequence": list(outcome.query_sequence),
        "query_count": len(outcome.query_sequence),
        "evidence_receipts": [dict(value) for value in outcome.receipts],
        "fallback": outcome.fallback,
        "decision_reason": outcome.decision_reason,
        "wrapper_contract_passed": contract_passed,
        "valid_decision_rounds": valid_rounds,
        "stopped": stopped,
        "invalid_decision": invalid_decision,
        "provider_error": dict(provider_error) if provider_error else None,
        "usage": dict(usage),
        "execution_time": elapsed,
    }

def _run_architecture(
    architecture: str,
    task: LongContextControlTask,
    proposal: Mapping[str, Any],
    truth: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if architecture == "rlm_adaptive_two_query":
        state, trajectory = run_rlm_acquisition(task, proposal, truth, runtime)
        return (
            _record(
                architecture,
                task,
                proposal,
                truth,
                state["outcome"],
                contract_passed=bool(state["contract_passed"]),
                valid_rounds=int(state["valid_decision_rounds"]),
                stopped=bool(state["stopped"]),
                invalid_decision=bool(state["invalid_decision"]),
                provider_error=state["provider_error"],
                usage=state["usage"],
                elapsed=float(state["execution_time"]),
            ),
            trajectory,
        )
    sequence = architecture_query_sequence(architecture, task, proposal)
    outcome = execute_acquisition(task, proposal, truth, sequence)
    forced = architecture == "rlm_forced_failure"
    return (
        _record(
            architecture,
            task,
            proposal,
            truth,
            outcome,
            contract_passed=False if forced else None,
            valid_rounds=0,
            stopped=False,
            invalid_decision=forced,
            provider_error=None,
            usage=_empty_usage(),
            elapsed=0.0,
        ),
        {
            "architecture": architecture,
            "deterministic_local": True,
            "forced_rlm_failure": forced,
            "query_sequence": list(sequence),
            "outcome": outcome.to_jsonable(),
        },
    )


def _provider_capability(config: Mapping[str, Any], path: Path) -> dict[str, Any]:
    if not path.exists():
        return parent._capability_gate(config, path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != "passed":
        raise RuntimeError("existing provider capability receipt did not pass")
    if receipt.get("model") != config["runtime"]["model"]:
        raise RuntimeError("existing provider capability receipt model changed")
    if receipt.get("response") != config["provider_capability_gate"]["expected_response"]:
        raise RuntimeError("existing provider capability receipt response changed")
    return receipt


def _initial_context_gate(
    tasks: Sequence[LongContextControlTask],
    proposals: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    failures = []
    for task in tasks:
        context = acquisition_context(build_initial_snapshot(task, proposals[task.task_id]), [], [])
        violations = context_privacy_violations(task, context)
        if violations:
            failures.append({"task_id": task.task_id, "violations": list(violations)})
    return {
        "passed": not failures,
        "contexts_checked": len(tasks),
        "failures": failures,
    }


def _receipt_failures(
    records: Sequence[Mapping[str, Any]],
    tasks: Mapping[str, LongContextControlTask],
    proposals: Mapping[str, Mapping[str, Any]],
    truth: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    failures = []
    for record in records:
        task_id = str(record["task_id"])
        for index, receipt in enumerate(record["evidence_receipts"]):
            if not verify_evidence_receipt(tasks[task_id], proposals[task_id], truth[task_id], receipt):
                failures.append({"record_id": record["record_id"], "receipt_index": index})
    return failures


def _write_task_shard(
    output: Path,
    task: LongContextControlTask,
    records: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Mapping[str, Any]],
    config_hash: str,
) -> None:
    shards = output / "shards"
    final = shards / task.task_id
    if final.exists():
        raise RuntimeError(f"refusing to overwrite evaluation shard: {task.task_id}")
    temporary = shards / f".{task.task_id}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    records_path = temporary / "records.jsonl"
    trajectories_path = temporary / "trajectories.json"
    parent._write_jsonl(records_path, records)
    parent._write_json(trajectories_path, trajectories)
    receipt = {
        "status": "complete",
        "protocol_id": "rlm_adaptive_evidence_acquisition_mesh_v0",
        "config_sha256": config_hash,
        "task_id": task.task_id,
        "architecture_hashes": architecture_hashes(),
        "record_count": len(records),
        "records_sha256": canonical_file_sha256(records_path),
        "trajectories_sha256": canonical_file_sha256(trajectories_path),
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
    }
    parent._write_json(temporary / "receipt.json", receipt)
    temporary.rename(final)


def _load_task_shard(
    output: Path,
    task: LongContextControlTask,
    config_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    shard = output / "shards" / task.task_id
    if not shard.exists():
        return None
    receipt_path = shard / "receipt.json"
    if not receipt_path.exists():
        raise RuntimeError(f"evaluation shard is not sealed: {task.task_id}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete" or receipt.get("config_sha256") != config_hash:
        raise RuntimeError(f"evaluation shard registration mismatch: {task.task_id}")
    if receipt.get("task_id") != task.task_id or receipt.get("architecture_hashes") != architecture_hashes():
        raise RuntimeError(f"evaluation shard identity mismatch: {task.task_id}")
    records_path = shard / "records.jsonl"
    trajectories_path = shard / "trajectories.json"
    if not verify_file_sha256(records_path, receipt["records_sha256"]):
        raise RuntimeError(f"evaluation shard record hash mismatch: {task.task_id}")
    if not verify_file_sha256(trajectories_path, receipt["trajectories_sha256"]):
        raise RuntimeError(f"evaluation shard trajectory hash mismatch: {task.task_id}")
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    trajectories = json.loads(trajectories_path.read_text(encoding="utf-8"))
    if len(records) != len(ARCHITECTURES) or int(receipt["record_count"]) != len(records):
        raise RuntimeError(f"evaluation shard record count mismatch: {task.task_id}")
    if [row["architecture_id"] for row in records] != list(ARCHITECTURES):
        raise RuntimeError(f"evaluation shard architecture order mismatch: {task.task_id}")
    if any(row["task_id"] != task.task_id for row in records):
        raise RuntimeError(f"evaluation shard task mismatch: {task.task_id}")
    return records, trajectories


def calibrate(config: Mapping[str, Any], config_hash: str, output: Path) -> dict[str, Any]:
    calibration_dir = output / "calibration"
    if (calibration_dir / "result.json").exists():
        raise RuntimeError("refusing to overwrite evidence-acquisition calibration")
    all_tasks, proposals, truth = _load_inputs(config)
    calibration_all = [task for task in all_tasks if task.split == "calibration"]
    construction = _construction_gate(calibration_all, proposals, truth, config)
    if not construction["passed"]:
        raise RuntimeError("evidence-acquisition construction gate failed")
    official_rlm_commit = _official_rlm_gate(config)
    capability = _provider_capability(config, calibration_dir / "provider_capability_receipt.json")
    selected = _selected_tasks(config, "calibration")
    opacity = _initial_context_gate(selected, proposals)
    if not opacity["passed"]:
        raise RuntimeError("calibration provider context violated opacity contract")
    records = []
    trajectories = []
    for task in selected:
        record, trajectory = _run_architecture(
            "rlm_adaptive_two_query", task, proposals[task.task_id], truth[task.task_id], config["runtime"]
        )
        records.append(record)
        trajectories.append({"record_id": record["record_id"], **trajectory})
    contract_rate = mean(float(row["wrapper_contract_passed"]) for row in records)
    minimum = float(config["calibration"]["minimum_interface_contract_rate"])
    interface = {
        "passed": contract_rate >= minimum,
        "contract_rate": contract_rate,
        "minimum_contract_rate": minimum,
        "provider_error_count": sum(row["provider_error"] is not None for row in records),
        "invalid_decision_count": sum(bool(row["invalid_decision"]) for row in records),
        "query_sequence_counts": dict(sorted(Counter("->".join(row["query_sequence"]) or "no_query" for row in records).items())),
        "initial_context_opacity": opacity,
    }
    receipt_failures = _receipt_failures(
        records,
        {task.task_id: task for task in selected},
        proposals,
        truth,
    )
    if receipt_failures:
        raise RuntimeError("calibration evidence receipts failed replay")
    calibration_dir.mkdir(parents=True, exist_ok=True)
    records_path = calibration_dir / "records.jsonl"
    trajectories_path = calibration_dir / "trajectories.json"
    parent._write_jsonl(records_path, records)
    parent._write_json(trajectories_path, trajectories)
    result = {
        "status": "passed" if interface["passed"] else "failed",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "task_ids": [task.task_id for task in selected],
        "evaluation_task_ids_accessed": False,
        "official_rlm_commit": official_rlm_commit,
        "construction_gate": construction,
        "interface_gate": interface,
        "evidence_receipt_failures": len(receipt_failures),
        "provider_capability_usage": capability["usage"],
        "records_path": records_path.relative_to(ROOT).as_posix(),
        "records_sha256": canonical_file_sha256(records_path),
        "trajectories_path": trajectories_path.relative_to(ROOT).as_posix(),
        "trajectories_sha256": canonical_file_sha256(trajectories_path),
    }
    result_path = calibration_dir / "result.json"
    parent._write_json(result_path, result)
    receipt = {
        "status": result["status"],
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "result_path": result_path.relative_to(ROOT).as_posix(),
        "result_sha256": canonical_file_sha256(result_path),
        "records_path": result["records_path"],
        "records_sha256": result["records_sha256"],
        "trajectories_path": result["trajectories_path"],
        "trajectories_sha256": result["trajectories_sha256"],
        "evaluation_task_ids_accessed": False,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
    }
    parent._write_json(CALIBRATION_RECEIPT, receipt)
    return result


def register(config: Mapping[str, Any], config_hash: str) -> dict[str, Any]:
    if REGISTRATION.exists():
        raise RuntimeError("evidence-acquisition registration already exists")
    if not CALIBRATION_RECEIPT.exists():
        raise RuntimeError("calibration receipt is absent")
    calibration_receipt = json.loads(CALIBRATION_RECEIPT.read_text(encoding="utf-8"))
    if calibration_receipt["status"] != "passed":
        raise RuntimeError("calibration interface gate did not pass")
    if calibration_receipt["config_sha256"] != config_hash:
        raise RuntimeError("calibration config hash differs from registration config")
    if not verify_file_sha256(
        _root_path(calibration_receipt["result_path"]), calibration_receipt["result_sha256"]
    ):
        raise RuntimeError("calibration result hash mismatch")
    eval_tasks = _selected_tasks(config, "eval")
    _, proposals, _ = _load_inputs(config)
    opacity = _initial_context_gate(eval_tasks, proposals)
    if not opacity["passed"]:
        raise RuntimeError("evaluation initial context violated opacity contract")
    bound = []
    for value in [*config["registration_artifacts"], CALIBRATION_RECEIPT.relative_to(ROOT).as_posix()]:
        bound.append({"path": Path(value).as_posix(), "sha256": canonical_file_sha256(_root_path(value))})
    registration = {
        "status": "registered_before_evaluation_provider_outcomes",
        "protocol_id": config["protocol_id"],
        "config_path": DEFAULT_CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": config_hash,
        "architecture_hashes": architecture_hashes(),
        "acquisition_contract_hash": ACQUISITION_CONTRACT_HASH,
        "evaluation_task_ids": [task.task_id for task in eval_tasks],
        "evaluation_task_selection_sha256": canonical_sha256([task.task_id for task in eval_tasks]),
        "initial_context_opacity": opacity,
        "calibration_receipt_sha256": canonical_file_sha256(CALIBRATION_RECEIPT),
        "bound_artifacts": bound,
        "evaluation_provider_outcomes_present": False,
        "git_head_before_registration": _git_head(),
        "registered_utc": datetime.now(timezone.utc).isoformat(),
    }
    parent._write_json(REGISTRATION, registration)
    return registration


def _load_registered_config(path: Path) -> tuple[dict[str, Any], str]:
    if not REGISTRATION.exists():
        raise RuntimeError("evidence-acquisition registration is absent")
    config, config_hash = _raw_config(path)
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if registration["config_sha256"] != config_hash:
        raise RuntimeError("registered config hash mismatch")
    if registration["architecture_hashes"] != architecture_hashes():
        raise RuntimeError("registered architecture hashes changed")
    if registration["acquisition_contract_hash"] != ACQUISITION_CONTRACT_HASH:
        raise RuntimeError("registered acquisition contract changed")
    for artifact in registration["bound_artifacts"]:
        if not verify_file_sha256(_root_path(artifact["path"]), artifact["sha256"]):
            raise RuntimeError(f"registered artifact hash mismatch: {artifact['path']}")
    eval_ids = [task.task_id for task in _selected_tasks(config, "eval")]
    if eval_ids != registration["evaluation_task_ids"]:
        raise RuntimeError("registered evaluation task IDs changed")
    if canonical_sha256(eval_ids) != registration["evaluation_task_selection_sha256"]:
        raise RuntimeError("registered evaluation task selection hash changed")
    return config, config_hash


def validate(config: Mapping[str, Any], config_hash: str, output: Path) -> dict[str, Any]:
    head = _official_rlm_gate(config)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is absent")
    if any((output / name).exists() for name in ("result.json", "records.jsonl", "result_receipt.json")):
        raise RuntimeError("evaluation outcomes already exist")
    if FINAL_RECEIPT.exists():
        raise RuntimeError("final receipt already exists")
    tasks = _selected_tasks(config, "eval")
    return {
        "status": "valid",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "official_rlm_commit": head,
        "evaluation_task_ids": [task.task_id for task in tasks],
        "record_count_expected": len(tasks) * len(ARCHITECTURES),
        "api_episodes_expected": len(tasks),
        "api_rounds_maximum": len(tasks) * 2,
        "outcomes_present": False,
    }


def _macro(records: Sequence[Mapping[str, Any]], architecture: str, field: str) -> float:
    return mean(
        mean(
            float(row[field])
            for row in records
            if row["architecture_id"] == architecture and row["task_family"] == family
        )
        for family in FAMILIES
    )


def _summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for architecture in ARCHITECTURES:
        rows = [row for row in records if row["architecture_id"] == architecture]
        contracts = [row["wrapper_contract_passed"] for row in rows if row["wrapper_contract_passed"] is not None]
        output.append(
            {
                "architecture_id": architecture,
                "cells": len(rows),
                "macro_net_utility": _macro(records, architecture, "net_utility"),
                "macro_raw_utility": _macro(records, architecture, "raw_utility"),
                "mean_oracle_regret": mean(float(row["net_regret_to_oracle"]) for row in rows),
                "accuracy": mean(float(row["correct"]) for row in rows),
                "unsafe_count": sum(bool(row["unsafe"]) for row in rows),
                "contract_rate": mean(float(value) for value in contracts) if contracts else None,
                "mean_query_count": mean(float(row["query_count"]) for row in rows),
                "mean_query_cost": mean(float(row["query_cost"]) for row in rows),
                "second_query_rate": mean(float(row["query_count"] == 2) for row in rows),
                "action_change_rate": mean(float(row["action_changed"]) for row in rows),
                "beneficial_changes": sum(bool(row["beneficial_change"]) for row in rows),
                "harmful_changes": sum(bool(row["harmful_change"]) for row in rows),
                "provider_errors": sum(row["provider_error"] is not None for row in rows),
                "total_tokens": sum(int(row["usage"]["total_tokens"]) for row in rows),
                "execution_time": sum(float(row["execution_time"]) for row in rows),
                "query_sequences": dict(sorted(Counter("->".join(row["query_sequence"]) or "no_query" for row in rows).items())),
            }
        )
    return output


def _comparison(records: list[dict[str, Any]], treatment: str, control: str) -> dict[str, Any]:
    lookup = {(row["architecture_id"], row["task_id"]): row for row in records}
    family_net: dict[str, list[float]] = defaultdict(list)
    family_raw: dict[str, list[float]] = defaultdict(list)
    for row in records:
        if row["architecture_id"] != treatment:
            continue
        baseline = lookup[(control, row["task_id"])]
        family_net[row["task_family"]].append(float(row["net_utility"]) - float(baseline["net_utility"]))
        family_raw[row["task_family"]].append(float(row["raw_utility"]) - float(baseline["raw_utility"]))
    net = {family: mean(family_net[family]) for family in FAMILIES}
    raw = {family: mean(family_raw[family]) for family in FAMILIES}
    return {
        "treatment": treatment,
        "control": control,
        "macro_net_utility_delta": mean(net.values()),
        "macro_raw_utility_delta": mean(raw.values()),
        "family_net_deltas": net,
        "family_raw_deltas": raw,
    }


def run(config: Mapping[str, Any], config_hash: str, output: Path) -> dict[str, Any]:
    if (output / "result.json").exists() or (output / "records.jsonl").exists():
        raise RuntimeError("refusing to overwrite evaluation outcomes")
    output.mkdir(parents=True, exist_ok=True)
    capability = _provider_capability(config, output / "evaluation_provider_capability_receipt.json")
    tasks, proposals, truth = _load_inputs(config)
    selected = [task for task in tasks if task.split == "eval"]
    selected = sorted(selected, key=lambda task: (FAMILIES.index(task.family), task.task_id))
    records = []
    trajectories = []
    for task in selected:
        sealed = _load_task_shard(output, task, config_hash)
        if sealed is not None:
            shard_records, shard_trajectories = sealed
            records.extend(shard_records)
            trajectories.extend(shard_trajectories)
            continue
        shard_records = []
        shard_trajectories = []
        for architecture in ARCHITECTURES:
            record, trajectory = _run_architecture(
                architecture, task, proposals[task.task_id], truth[task.task_id], config["runtime"]
            )
            shard_records.append(record)
            shard_trajectories.append({"record_id": record["record_id"], **trajectory})
        failures = _receipt_failures(
            shard_records,
            {task.task_id: task},
            proposals,
            truth,
        )
        if failures:
            raise RuntimeError(f"evidence receipt replay failed before shard seal: {failures}")
        _write_task_shard(output, task, shard_records, shard_trajectories, config_hash)
        records.extend(shard_records)
        trajectories.extend(shard_trajectories)
    fixed = {row["task_id"]: row for row in records if row["architecture_id"] == "mesh_fixed_no_query"}
    forced = {row["task_id"]: row for row in records if row["architecture_id"] == "rlm_forced_failure"}
    no_op = all(
        fixed[task_id]["executed_action"] == forced[task_id]["executed_action"]
        and fixed[task_id]["net_utility"] == forced[task_id]["net_utility"]
        for task_id in fixed
    )
    receipt_failures = _receipt_failures(
        records,
        {task.task_id: task for task in selected},
        proposals,
        truth,
    )
    if receipt_failures:
        raise RuntimeError(f"evidence receipt replay failed at result seal: {receipt_failures}")
    typed_unsafe_count = sum(bool(row["unsafe"]) for row in records)
    if typed_unsafe_count:
        raise RuntimeError(f"typed execution safety gate failed: {typed_unsafe_count} unsafe actions")
    if not no_op:
        raise RuntimeError("forced-RLM-failure identity gate failed")
    records_path = output / "records.jsonl"
    trajectories_path = output / "trajectory_manifest.json"
    parent._write_jsonl(records_path, records)
    parent._write_json(trajectories_path, trajectories)
    comparisons = [
        _comparison(records, "rlm_adaptive_two_query", "mesh_fixed_no_query"),
        _comparison(records, "rlm_adaptive_two_query", "mesh_random_two_query"),
        _comparison(records, "rlm_adaptive_two_query", "mesh_deterministic_voi"),
        _comparison(records, "rlm_adaptive_two_query", "mesh_exact_then_rollout"),
    ]
    result = {
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "git_head": _git_head(),
        "acquisition_contract_hash": ACQUISITION_CONTRACT_HASH,
        "task_ids": [task.task_id for task in selected],
        "record_count": len(records),
        "records_path": records_path.relative_to(ROOT).as_posix(),
        "records_sha256": canonical_file_sha256(records_path),
        "trajectory_manifest_path": trajectories_path.relative_to(ROOT).as_posix(),
        "trajectory_manifest_sha256": canonical_file_sha256(trajectories_path),
        "provider_capability_usage": capability["usage"],
        "summary": _summaries(records),
        "comparisons": comparisons,
        "typed_unsafe_count": typed_unsafe_count,
        "forced_failure_no_op_passed": no_op,
        "evidence_receipt_failures": len(receipt_failures),
        "claim_boundary": config["claim_boundary"],
    }
    parent._write_json(output / "result.json", result)
    return result


def _render_report(result: Mapping[str, Any], receipt: Mapping[str, Any], path: Path) -> None:
    rows = [
        "| Architecture | Net utility | Raw utility | Oracle regret | Unsafe | Contract | Queries | Query cost | 2nd query | Changes | Provider errors | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(result["summary"], key=lambda value: float(value["macro_net_utility"]), reverse=True):
        contract = "-" if row["contract_rate"] is None else f'{float(row["contract_rate"]):.4f}'
        rows.append(
            f'| `{row["architecture_id"]}` | {float(row["macro_net_utility"]):.4f} | '
            f'{float(row["macro_raw_utility"]):.4f} | {float(row["mean_oracle_regret"]):.4f} | '
            f'{int(row["unsafe_count"])} | {contract} | {float(row["mean_query_count"]):.3f} | '
            f'{float(row["mean_query_cost"]):.4f} | {float(row["second_query_rate"]):.4f} | '
            f'{float(row["action_change_rate"]):.4f} | {int(row["provider_errors"])} | {int(row["total_tokens"]):,} |'
        )
    comparisons = "\n".join(
        f'- `{row["treatment"]}` vs `{row["control"]}`: net {float(row["macro_net_utility_delta"]):+.4f}, raw {float(row["macro_raw_utility_delta"]):+.4f}.'
        for row in result["comparisons"]
    )
    body = f"""# RLM Adaptive Evidence-Acquisition Mesh v0

## Construction

The official RLM sees an action-opaque canonical JSON snapshot and may select at most two universal evidence
queries. It cannot select actions, candidates, thresholds, or schedules. Typed receipts feed a deterministic
arbiter; the exact executor commits a safe action or executes fixed consensus. Provider and parser failures stop
acquisition without erasing baseline utility.

## Results

{chr(10).join(rows)}

Typed unsafe executions: `{result['typed_unsafe_count']}`. Forced-RLM-failure identity:
`{str(result['forced_failure_no_op_passed']).lower()}`.

## Matched Contrasts

{comparisons}

## Boundary

{result['claim_boundary']}

## Integrity

- Result SHA-256: `{receipt['result_sha256']}`
- Records SHA-256: `{receipt['records_sha256']}`
- Trajectory SHA-256: `{receipt['trajectory_manifest_sha256']}`
- Config SHA-256: `{receipt['config_sha256']}`
- Acquisition contract SHA-256: `{receipt['acquisition_contract_hash']}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")


def finalize(config: Mapping[str, Any], config_hash: str, output: Path) -> dict[str, Any]:
    result_path = output / "result.json"
    if not result_path.exists():
        raise RuntimeError("evaluation result is absent")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("config_sha256") != config_hash:
        raise RuntimeError("evaluation result config hash mismatch")
    if result.get("acquisition_contract_hash") != ACQUISITION_CONTRACT_HASH:
        raise RuntimeError("evaluation result acquisition contract changed")
    if result.get("typed_unsafe_count") != 0 or not result.get("forced_failure_no_op_passed"):
        raise RuntimeError("evaluation hard gates did not pass")
    if result.get("evidence_receipt_failures") != 0:
        raise RuntimeError("evaluation result records evidence receipt failures")
    records_path = _root_path(result["records_path"])
    trajectories_path = _root_path(result["trajectory_manifest_path"])
    if not verify_file_sha256(records_path, result["records_sha256"]):
        raise RuntimeError("evaluation records hash mismatch")
    if not verify_file_sha256(trajectories_path, result["trajectory_manifest_sha256"]):
        raise RuntimeError("evaluation trajectories hash mismatch")
    receipt = {
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "config_path": DEFAULT_CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": config_hash,
        "result_path": result_path.relative_to(ROOT).as_posix(),
        "result_sha256": canonical_file_sha256(result_path),
        "records_path": result["records_path"],
        "records_sha256": result["records_sha256"],
        "trajectory_manifest_path": result["trajectory_manifest_path"],
        "trajectory_manifest_sha256": result["trajectory_manifest_sha256"],
        "record_count": result["record_count"],
        "acquisition_contract_hash": ACQUISITION_CONTRACT_HASH,
        "typed_unsafe_count": result["typed_unsafe_count"],
        "forced_failure_no_op_passed": result["forced_failure_no_op_passed"],
        "claim_boundary": config["claim_boundary"],
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
    }
    parent._write_json(output / "result_receipt.json", receipt)
    parent._write_json(FINAL_RECEIPT, receipt)
    _render_report(result, receipt, DEFAULT_REPORT)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("calibrate", "register", "validate", "run", "finalize"),
        required=True,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vram-fraction", type=float, default=0.35)
    args = parser.parse_args()
    if args.phase in {"calibrate", "register"}:
        config, config_hash = _raw_config(args.config.resolve())
        value = (
            calibrate(config, config_hash, args.output.resolve())
            if args.phase == "calibrate"
            else register(config, config_hash)
        )
    else:
        config, config_hash = _load_registered_config(args.config.resolve())
        if args.phase == "validate":
            value = validate(config, config_hash, args.output.resolve())
        elif args.phase == "run":
            value = run(config, config_hash, args.output.resolve())
        else:
            value = finalize(config, config_hash, args.output.resolve())
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
