"""Validate, run, and seal the registered RLM-wrapped controller-mesh pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from statistics import mean
from typing import Any, Iterable, Mapping

from research_gym.benchmarks.rlm_hybrid_neighborhood import (
    FAMILIES,
    LongContextControlTask,
    canonical_sha256,
)
from research_gym.benchmarks.rlm_hybrid_runtime import run_local_architecture
from research_gym.benchmarks.rlm_wrapped_mesh import (
    API_ARCHITECTURES,
    ARCHITECTURES,
    MESH_TOPOLOGY_HASH,
    architecture_hashes,
    run_local_mesh_architecture,
    run_rlm_wrapped_mesh,
)
from research_gym.integrity import canonical_file_sha256, verify_file_sha256
from research_gym.scripts import bench_rlm_trm_ldt_hybrid_neighborhood_v1 as parent


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "rlm_wrapped_controller_mesh_v0.json"
REGISTRATION = ROOT / "configs" / "rlm_wrapped_controller_mesh_v0_registration.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "rlm_wrapped_controller_mesh_v0"
FINAL_RECEIPT = ROOT / "data" / "benchmarks" / "rlm_wrapped_controller_mesh_v0_receipt.json"
DEFAULT_REPORT = ROOT / "reports" / "rlm_wrapped_controller_mesh_v0.md"


def _root_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _git_head() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def _load_registered_config(path: Path) -> tuple[dict[str, Any], str]:
    if not REGISTRATION.exists():
        raise RuntimeError("RLM-wrapped mesh registration is absent")
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    config_hash = canonical_file_sha256(path)
    if config_hash != registration["config_sha256"]:
        raise RuntimeError("RLM-wrapped mesh config hash does not match registration")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["protocol_id"] != registration["protocol_id"]:
        raise RuntimeError("RLM-wrapped mesh protocol id changed")
    if tuple(config["architecture_order"]) != ARCHITECTURES:
        raise RuntimeError("RLM-wrapped mesh architecture order changed")
    if registration["architecture_hashes"] != architecture_hashes():
        raise RuntimeError("RLM-wrapped mesh architecture hashes changed")
    if registration["mesh_topology_hash"] != MESH_TOPOLOGY_HASH:
        raise RuntimeError("RLM-wrapped mesh topology changed")
    for artifact in registration["bound_artifacts"]:
        artifact_path = _root_path(artifact["path"])
        if not verify_file_sha256(artifact_path, artifact["sha256"]):
            raise RuntimeError(f"registered artifact hash mismatch: {artifact['path']}")
    selected_ids = [task.task_id for task in _load_tasks(config)]
    if selected_ids != registration["task_ids"]:
        raise RuntimeError("RLM-wrapped mesh selected task panel changed")
    if canonical_sha256(selected_ids) != registration["task_selection_sha256"]:
        raise RuntimeError("RLM-wrapped mesh task-selection hash changed")
    return config, config_hash


def _load_tasks(config: Mapping[str, Any]) -> list[LongContextControlTask]:
    payload = json.loads(_root_path(config["task_suite"]["path"]).read_text(encoding="utf-8"))
    candidates = [LongContextControlTask.from_jsonable(row) for row in payload["tasks"] if row["split"] == "eval"]
    selected = []
    for family in FAMILIES:
        values = sorted(
            (task for task in candidates if task.family == family),
            key=lambda task: canonical_sha256((config["selection"]["salt"], task.task_id)),
        )
        selected.extend(values[: int(config["selection"]["tasks_per_family"])])
    return selected


def _load_proposals(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    seed = int(config["selection"]["checkpoint_seed"])
    rows = [
        json.loads(line)
        for line in _root_path(config["trained_proposals"]["path"]).read_text(encoding="utf-8").splitlines()
    ]
    return {str(row["task_id"]): row for row in rows if int(row["seed"]) == seed}


def validate(config: Mapping[str, Any], config_hash: str, output: Path) -> dict[str, Any]:
    checkout = Path(config["official_rlm_source"]["checkout_path"])
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if head != config["official_rlm_source"]["commit"]:
        raise RuntimeError("official RLM checkout commit changed")
    checkout_status = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain"], text=True
    ).strip()
    if checkout_status:
        raise RuntimeError("official RLM checkout is dirty")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is absent")
    tasks = _load_tasks(config)
    proposals = _load_proposals(config)
    missing = sorted(task.task_id for task in tasks if task.task_id not in proposals)
    if missing:
        raise RuntimeError(f"trained proposal rows are missing: {missing}")
    scientific_outputs = ("result.json", "records.jsonl", "result_receipt.json")
    if any((output / name).exists() for name in scientific_outputs):
        raise RuntimeError("RLM-wrapped mesh scientific outcomes already exist")
    if FINAL_RECEIPT.exists():
        raise RuntimeError("RLM-wrapped mesh final receipt already exists")
    return {
        "status": "valid",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "official_rlm_commit": head,
        "task_ids": [task.task_id for task in tasks],
        "task_selection_sha256": canonical_sha256([task.task_id for task in tasks]),
        "architecture_hashes": architecture_hashes(),
        "record_count_expected": len(tasks) * len(ARCHITECTURES),
        "outcomes_present": False,
    }


def _error_record(
    architecture: str,
    task: LongContextControlTask,
    seed: int,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "architecture_id": architecture,
        "task_id": task.task_id,
        "task_family": task.family,
        "replicate_seed": seed,
        "checkpoint_seed": seed,
        "proposal_action": None,
        "executed_action": None,
        "optimal_action": task.optimal_action,
        "safe": False,
        "unsafe": False,
        "correct": False,
        "utility": 0.0,
        "regret": float(task.utilities[task.optimal_action]) + 1.0,
        "fallback": True,
        "decision_reason": "cell_error",
        "usage": {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reported_cost_usd": None,
        },
        "execution_time": 0.0,
        "policy_hint": "consensus",
        "mesh_certificate": None,
        "mesh_snapshot": None,
        "wrapper_contract_passed": False,
        "wrapper_fallback": True,
        "mesh_call_count": 0,
        "invalid_mesh_call_count": 0,
        "error": parent._error_summary(exc),
    }


def _macro_utility(records: list[Mapping[str, Any]], architecture: str) -> float:
    family_means = []
    for family in FAMILIES:
        values = [
            float(row["utility"])
            for row in records
            if row["architecture_id"] == architecture and row["task_family"] == family
        ]
        family_means.append(mean(values))
    return mean(family_means)


def _summarize(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for architecture in ARCHITECTURES:
        rows = [row for row in records if row["architecture_id"] == architecture]
        contract_values = [
            float(row["wrapper_contract_passed"])
            for row in rows
            if row.get("wrapper_contract_passed") is not None
        ]
        output.append(
            {
                "architecture_id": architecture,
                "cells": len(rows),
                "macro_utility": _macro_utility(records, architecture),
                "accuracy": mean(float(row["correct"]) for row in rows),
                "unsafe_count": sum(bool(row["unsafe"]) for row in rows),
                "wrapper_contract_rate": mean(contract_values) if contract_values else None,
                "fallback_rate": mean(float(row.get("wrapper_fallback", row["fallback"])) for row in rows),
                "cell_errors": sum(row.get("decision_reason") == "cell_error" for row in rows),
                "total_tokens": sum(int(row["usage"]["total_tokens"]) for row in rows),
                "execution_time": sum(float(row["execution_time"]) for row in rows),
                "policy_hints": dict(sorted(Counter(str(row.get("policy_hint")) for row in rows).items())),
            }
        )
    return output


def _comparison(records: list[Mapping[str, Any]], treatment: str, control: str) -> dict[str, Any]:
    lookup = {
        (str(row["architecture_id"]), str(row["task_id"])): float(row["utility"])
        for row in records
    }
    deltas: dict[str, list[float]] = defaultdict(list)
    for row in records:
        if row["architecture_id"] != treatment:
            continue
        deltas[str(row["task_family"])].append(
            float(row["utility"]) - lookup[(control, str(row["task_id"]))]
        )
    family_deltas = {family: mean(deltas[family]) for family in FAMILIES}
    return {
        "treatment": treatment,
        "control": control,
        "macro_utility_delta": mean(family_deltas.values()),
        "family_deltas": family_deltas,
    }


def _run_record(
    architecture: str,
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
    seed: int,
    runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if architecture in {"ldt_only", "trained_trm_ldt_fixed"}:
        return run_local_architecture(architecture, task, trained_row, seed)
    if architecture in {"mesh_fixed_consensus", "mesh_forced_no_tool_fallback"}:
        return run_local_mesh_architecture(architecture, task, trained_row, seed)
    return run_rlm_wrapped_mesh(architecture, task, trained_row, seed, runtime)


def run(config: Mapping[str, Any], config_hash: str, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists() or (output / "records.jsonl").exists():
        raise RuntimeError("refusing to overwrite RLM-wrapped mesh outcomes")
    capability = parent._capability_gate(config, output / "provider_capability_receipt.json")
    tasks = _load_tasks(config)
    proposals = _load_proposals(config)
    seed = int(config["selection"]["checkpoint_seed"])
    records: list[dict[str, Any]] = []
    trajectories = []
    local = [value for value in ARCHITECTURES if value not in API_ARCHITECTURES]
    for task_index, task in enumerate(tasks):
        api = list(API_ARCHITECTURES)
        if task_index % 2:
            api.reverse()
        for architecture in [*local, *api]:
            try:
                record, trajectory = _run_record(
                    architecture, task, proposals[task.task_id], seed, config["runtime"]
                )
            except Exception as exc:
                if parent._is_global_provider_error(exc):
                    parent._write_json(
                        output / "provider_abort_receipt.json",
                        {
                            "status": "provider_access_abort",
                            "architecture_id": architecture,
                            "task_id": task.task_id,
                            "error": parent._error_summary(exc),
                        },
                    )
                    raise RuntimeError("global provider access failed") from exc
                record = _error_record(architecture, task, seed, exc)
                trajectory = {"architecture": architecture, "error": parent._error_summary(exc)}
            record["record_id"] = f"{architecture}__{task.task_id}"
            record["architecture_hash"] = architecture_hashes()[architecture]
            records.append(record)
            trajectories.append({"record_id": record["record_id"], **trajectory})
    fixed = {row["task_id"]: row for row in records if row["architecture_id"] == "mesh_fixed_consensus"}
    forced = {row["task_id"]: row for row in records if row["architecture_id"] == "mesh_forced_no_tool_fallback"}
    no_op_control_passed = all(
        fixed[task_id]["executed_action"] == forced[task_id]["executed_action"]
        and fixed[task_id]["utility"] == forced[task_id]["utility"]
        for task_id in fixed
    )
    records_path = output / "records.jsonl"
    trajectories_path = output / "trajectory_manifest.json"
    parent._write_jsonl(records_path, records)
    parent._write_json(trajectories_path, trajectories)
    comparisons = [
        _comparison(records, "rlm_mesh_atomic_tool", "mesh_fixed_consensus"),
        _comparison(records, "rlm_mesh_text_router", "mesh_fixed_consensus"),
        _comparison(records, "rlm_mesh_atomic_tool", "rlm_mesh_text_router"),
        _comparison(records, "mesh_fixed_consensus", "trained_trm_ldt_fixed"),
    ]
    result = {
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "git_head": _git_head(),
        "mesh_topology_hash": MESH_TOPOLOGY_HASH,
        "task_ids": [task.task_id for task in tasks],
        "record_count": len(records),
        "records_path": records_path.relative_to(ROOT).as_posix(),
        "records_sha256": canonical_file_sha256(records_path),
        "trajectory_manifest_path": trajectories_path.relative_to(ROOT).as_posix(),
        "trajectory_manifest_sha256": canonical_file_sha256(trajectories_path),
        "provider_capability_usage": capability["usage"],
        "summary": _summarize(records),
        "comparisons": comparisons,
        "typed_unsafe_count": sum(bool(row["unsafe"]) for row in records),
        "cell_error_count": sum(row.get("decision_reason") == "cell_error" for row in records),
        "no_op_control_passed": no_op_control_passed,
        "claim_boundary": config["claim_boundary"],
    }
    parent._write_json(output / "result.json", result)
    return result


def _render_report(result: Mapping[str, Any], receipt: Mapping[str, Any], path: Path) -> None:
    rows = [
        "| Architecture | Utility | Accuracy | Unsafe | Contract | Fallback | Errors | Tokens | Wall s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(result["summary"], key=lambda value: float(value["macro_utility"]), reverse=True):
        contract = (
            "-"
            if row["wrapper_contract_rate"] is None
            else f'{float(row["wrapper_contract_rate"]):.4f}'
        )
        rows.append(
            f'| `{row["architecture_id"]}` | {float(row["macro_utility"]):.4f} | '
            f'{float(row["accuracy"]):.4f} | {int(row["unsafe_count"])} | '
            f'{contract} | {float(row["fallback_rate"]):.4f} | '
            f'{int(row["cell_errors"])} | {int(row["total_tokens"]):,} | '
            f'{float(row["execution_time"]):.1f} |'
        )
    comparisons = "\n".join(
        f'- `{row["treatment"]}` vs `{row["control"]}`: '
        f'{float(row["macro_utility_delta"]):+.4f}; family deltas '
        + ", ".join(f"{family}={float(value):+.4f}" for family, value in row["family_deltas"].items())
        + "."
        for row in result["comparisons"]
    )
    body = f"""# RLM-Wrapped Typed Controller Mesh v0

## Construction

An official RLM wraps one atomic `mesh_resolve(policy_hint)` capability. The inner mesh contains proxy-TRM,
trained-ControlTRM, exact-LDT, deterministic arbitration, and a certificate-bound executor. The RLM can choose
only `consensus`, `trained_first`, or `ldt_conservative`; it cannot submit an action or manufacture a commit.
Missing or invalid wrapper output executes the fixed consensus mesh.

This calibration-informed diagnostic uses {len(result['task_ids'])} hash-selected tasks from the previously opened
v1 evaluation suite. It is an architecture-extension pilot, not a fresh held-out efficacy test.

## Results

{chr(10).join(rows)}

Typed unsafe executions: `{result['typed_unsafe_count']}`. Cell errors remain zero-utility outcomes:
`{result['cell_error_count']}`. The forced no-tool control reproduced fixed-mesh actions and utility:
`{str(result['no_op_control_passed']).lower()}`.

## Matched Contrasts

{comparisons}

## Boundary

{result['claim_boundary']}

`Manipulation` is not an endpoint here. The atomic-tool contract measures capability invocation and certificate
completion only. The practical action mesh is topology-compatible with the controller-mesh program, but this
pilot does not compute a sheaf spectrum or test a spectral prediction.

## Integrity

- Result SHA-256: `{receipt['result_sha256']}`
- Records SHA-256: `{receipt['records_sha256']}`
- Trajectory manifest SHA-256: `{receipt['trajectory_manifest_sha256']}`
- Config SHA-256: `{receipt['config_sha256']}`
- Mesh topology SHA-256: `{receipt['mesh_topology_hash']}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")


def finalize(config: Mapping[str, Any], config_hash: str, output: Path) -> dict[str, Any]:
    result_path = output / "result.json"
    if not result_path.exists():
        raise RuntimeError("RLM-wrapped mesh result is absent")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    records_path = _root_path(result["records_path"])
    trajectories_path = _root_path(result["trajectory_manifest_path"])
    if not verify_file_sha256(records_path, result["records_sha256"]):
        raise RuntimeError("RLM-wrapped mesh records hash mismatch")
    if not verify_file_sha256(trajectories_path, result["trajectory_manifest_sha256"]):
        raise RuntimeError("RLM-wrapped mesh trajectory hash mismatch")
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
        "mesh_topology_hash": MESH_TOPOLOGY_HASH,
        "typed_unsafe_count": result["typed_unsafe_count"],
        "no_op_control_passed": result["no_op_control_passed"],
        "claim_boundary": config["claim_boundary"],
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
    }
    parent._write_json(output / "result_receipt.json", receipt)
    parent._write_json(FINAL_RECEIPT, receipt)
    _render_report(result, receipt, DEFAULT_REPORT)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("validate", "run", "finalize"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vram-fraction", type=float, default=0.35)
    args = parser.parse_args()
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
