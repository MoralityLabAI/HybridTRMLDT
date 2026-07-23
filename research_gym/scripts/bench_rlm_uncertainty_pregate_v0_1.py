"""Calibrate, register, run, and seal the RLM uncertainty pre-gate v0.1."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from statistics import mean
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

from research_gym.benchmarks.rlm_evidence_acquisition import (
    ACQUISITION_CONTRACT_HASH,
    execute_acquisition,
    sequence_oracle,
)
from research_gym.benchmarks.rlm_evidence_runtime import run_rlm_acquisition
from research_gym.benchmarks.rlm_hybrid_neighborhood import (
    FAMILIES,
    LongContextControlTask,
    canonical_sha256,
)
from research_gym.benchmarks.rlm_hybrid_runtime import _empty_usage
from research_gym.benchmarks.rlm_uncertainty_pregate import (
    API_ARCHITECTURES,
    ARCHITECTURES,
    GATE_ID,
    ProviderCostSpec,
    all_in_utility,
    architecture_hashes,
    gate_features,
    local_sequence,
    provider_utility_cost,
    should_invoke_rlm,
)
from research_gym.integrity import canonical_file_sha256, verify_file_sha256
from research_gym.scripts import bench_rlm_evidence_acquisition_v0 as base
from research_gym.scripts import bench_rlm_trm_ldt_hybrid_neighborhood_v1 as parent


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/rlm_uncertainty_pregate_v0_1.json"
REGISTRATION = ROOT / "configs/rlm_uncertainty_pregate_v0_1_registration.json"
DEFAULT_OUTPUT = ROOT / "experiments/rlm_uncertainty_pregate_v0_1"
CALIBRATION_RECEIPT = DEFAULT_OUTPUT / "calibration/calibration_receipt.json"
FINAL_RECEIPT = ROOT / "data/benchmarks/rlm_uncertainty_pregate_v0_1_receipt.json"
DEFAULT_REPORT = ROOT / "reports/rlm_uncertainty_pregate_v0_1.md"


def _root_path(value: str | Path) -> Path:
    return base._root_path(value)


def _git_head() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def _verify_declared_inputs(config: Mapping[str, Any]) -> None:
    for value in (
        config["task_suite"],
        config["trained_proposals"],
        config["evidence_truth"],
        config["materialization_receipt"],
    ):
        if not verify_file_sha256(_root_path(value["path"]), value["sha256"]):
            raise RuntimeError(f"declared artifact hash mismatch: {value['path']}")
    checkpoint = config["trained_proposals"]
    if not verify_file_sha256(
        _root_path(checkpoint["checkpoint_path"]), checkpoint["checkpoint_sha256"]
    ):
        raise RuntimeError("declared ControlTRM checkpoint hash mismatch")


def _raw_config(path: Path) -> tuple[dict[str, Any], str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if tuple(config["architecture_order"]) != ARCHITECTURES:
        raise RuntimeError("uncertainty-pregate architecture order changed")
    if config["gate"]["gate_id"] != GATE_ID:
        raise RuntimeError("uncertainty-pregate rule changed")
    if "primary" not in config["provider_cost_regimes"]:
        raise RuntimeError("primary provider cost regime is absent")
    _verify_declared_inputs(config)
    return config, canonical_file_sha256(path)


def _load_inputs(config: Mapping[str, Any]):
    return base._load_inputs(config)


def _selected_tasks(config: Mapping[str, Any], split: str) -> list[LongContextControlTask]:
    tasks, proposals, _ = _load_inputs(config)
    candidates = [task for task in tasks if task.split == split]
    if split == "eval":
        return sorted(candidates, key=lambda task: (FAMILIES.index(task.family), task.task_id))
    count = int(config["calibration"]["tasks_per_family"])
    salt = str(config["calibration"]["selection_salt"])
    selected = []
    for family in FAMILIES:
        invoked = [
            task
            for task in candidates
            if task.family == family and should_invoke_rlm(task, proposals[task.task_id])
        ]
        invoked.sort(key=lambda task: canonical_sha256((salt, task.task_id)))
        if len(invoked) < count:
            raise RuntimeError(f"insufficient gate-positive calibration tasks for {family}")
        selected.extend(invoked[:count])
    return selected


def _construction_gate(
    tasks: Sequence[LongContextControlTask],
    proposals: Mapping[str, Mapping[str, Any]],
    truth: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    invoked = [task for task in tasks if should_invoke_rlm(task, proposals[task.task_id])]
    gains = []
    for task in tasks:
        baseline = execute_acquisition(task, proposals[task.task_id], truth[task.task_id], ())
        oracle = sequence_oracle(task, proposals[task.task_id], truth[task.task_id])
        gains.append(oracle.net_utility - baseline.net_utility)
    thresholds = config["calibration"]["construction_gate"]
    represented = len({task.family for task in invoked})
    passed = (
        len(invoked) >= int(thresholds["minimum_invoked_tasks"])
        and len(invoked) <= int(thresholds["maximum_invoked_tasks"])
        and represented >= int(thresholds["minimum_families_represented"])
        and mean(gains) >= float(thresholds["minimum_mean_oracle_net_gain"])
        and sum(value > 1e-12 for value in gains)
        >= int(thresholds["minimum_positive_oracle_gain_tasks"])
    )
    return {
        "passed": passed,
        "task_count": len(tasks),
        "invoked_tasks": len(invoked),
        "invocation_rate": len(invoked) / len(tasks),
        "invoked_families": dict(sorted(Counter(task.family for task in invoked).items())),
        "families_represented": represented,
        "mean_oracle_net_gain": mean(gains),
        "positive_oracle_gain_tasks": sum(value > 1e-12 for value in gains),
        "thresholds": thresholds,
    }


def _record(
    architecture: str,
    task: LongContextControlTask,
    proposal: Mapping[str, Any],
    truth: Mapping[str, Any],
    outcome,
    config: Mapping[str, Any],
    *,
    gate_open: bool,
    provider_called: bool,
    contract_passed: bool | None,
    provider_error: Mapping[str, Any] | None,
    usage: Mapping[str, Any],
    wall_seconds: float,
) -> dict[str, Any]:
    baseline = execute_acquisition(task, proposal, truth, ())
    oracle = sequence_oracle(task, proposal, truth)
    costs = {
        name: provider_utility_cost(
            usage,
            wall_seconds,
            ProviderCostSpec.from_mapping(spec),
        )
        for name, spec in config["provider_cost_regimes"].items()
    }
    all_in = {
        name: all_in_utility(
            outcome.net_utility,
            usage,
            wall_seconds,
            ProviderCostSpec.from_mapping(spec),
        )
        for name, spec in config["provider_cost_regimes"].items()
    }
    return {
        "record_id": f"{architecture}__{task.task_id}",
        "architecture_id": architecture,
        "architecture_hash": architecture_hashes(config["provider_cost_regimes"])[architecture],
        "task_id": task.task_id,
        "task_family": task.family,
        "gate_open": gate_open,
        "gate_features": gate_features(task, proposal),
        "provider_called": provider_called,
        "executed_action": outcome.executed_action,
        "optimal_action": task.optimal_action,
        "unsafe": outcome.executed_action not in task.exact_allowed,
        "correct": outcome.executed_action == task.optimal_action,
        "raw_utility": outcome.raw_utility,
        "evidence_query_cost": outcome.query_cost,
        "evidence_net_utility": outcome.net_utility,
        "provider_utility_cost": costs,
        "all_in_utility": all_in,
        "baseline_action": baseline.executed_action,
        "baseline_raw_utility": baseline.raw_utility,
        "oracle_net_utility": oracle.net_utility,
        "oracle_query_sequence": list(oracle.query_sequence),
        "query_sequence": list(outcome.query_sequence),
        "query_count": len(outcome.query_sequence),
        "evidence_receipts": [dict(receipt) for receipt in outcome.receipts],
        "action_changed": outcome.executed_action != baseline.executed_action,
        "beneficial_change": outcome.raw_utility > baseline.raw_utility + 1e-12,
        "harmful_change": outcome.raw_utility < baseline.raw_utility - 1e-12,
        "fallback": outcome.fallback,
        "decision_reason": outcome.decision_reason,
        "wrapper_contract_passed": contract_passed,
        "provider_error": dict(provider_error) if provider_error else None,
        "usage": dict(usage),
        "provider_wall_seconds": wall_seconds,
    }


def _run_architecture(
    architecture: str,
    task: LongContextControlTask,
    proposal: Mapping[str, Any],
    truth: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    gate_open = should_invoke_rlm(task, proposal)
    should_call = architecture == "ungated_rlm" or (
        architecture == "gated_rlm" and gate_open
    )
    if should_call:
        started = time.perf_counter()
        state, trajectory = run_rlm_acquisition(task, proposal, truth, config["runtime"])
        wall = time.perf_counter() - started
        return (
            _record(
                architecture,
                task,
                proposal,
                truth,
                state["outcome"],
                config,
                gate_open=gate_open,
                provider_called=True,
                contract_passed=bool(state["contract_passed"]),
                provider_error=state["provider_error"],
                usage=state["usage"],
                wall_seconds=wall,
            ),
            {"provider_called": True, "provider_wall_seconds": wall, **trajectory},
        )
    sequence = local_sequence(architecture, task, proposal)
    outcome = execute_acquisition(task, proposal, truth, sequence)
    forced = architecture == "gated_forced_failure" and gate_open
    return (
        _record(
            architecture,
            task,
            proposal,
            truth,
            outcome,
            config,
            gate_open=gate_open,
            provider_called=False,
            contract_passed=False if forced else None,
            provider_error=None,
            usage=_empty_usage(),
            wall_seconds=0.0,
        ),
        {
            "provider_called": False,
            "gate_open": gate_open,
            "forced_failure": forced,
            "query_sequence": list(sequence),
            "outcome": outcome.to_jsonable(),
        },
    )


def _receipt_failures(records, tasks, proposals, truth):
    return base._receipt_failures(records, tasks, proposals, truth)


def calibrate(config: Mapping[str, Any], config_hash: str, output: Path) -> dict[str, Any]:
    calibration_dir = output / "calibration"
    if (calibration_dir / "result.json").exists():
        raise RuntimeError("refusing to overwrite pregate calibration")
    tasks, proposals, truth = _load_inputs(config)
    calibration_all = [task for task in tasks if task.split == "calibration"]
    construction = _construction_gate(calibration_all, proposals, truth, config)
    if not construction["passed"]:
        raise RuntimeError("pregate construction gate failed")
    official_commit = base._official_rlm_gate(config)
    capability = base._provider_capability(
        config, calibration_dir / "provider_capability_receipt.json"
    )
    selected = _selected_tasks(config, "calibration")
    opacity = base._initial_context_gate(selected, proposals)
    if not opacity["passed"]:
        raise RuntimeError("calibration context opacity gate failed")
    records, trajectories = [], []
    for task in selected:
        record, trajectory = _run_architecture(
            "gated_rlm", task, proposals[task.task_id], truth[task.task_id], config
        )
        if not record["provider_called"]:
            raise RuntimeError("gate-positive calibration task did not invoke provider")
        records.append(record)
        trajectories.append({"record_id": record["record_id"], **trajectory})
    contract_rate = mean(float(row["wrapper_contract_passed"]) for row in records)
    interface = {
        "passed": contract_rate >= float(config["calibration"]["minimum_interface_contract_rate"]),
        "contract_rate": contract_rate,
        "provider_error_count": sum(row["provider_error"] is not None for row in records),
        "query_sequences": dict(
            sorted(Counter("->".join(row["query_sequence"]) or "no_query" for row in records).items())
        ),
        "initial_context_opacity": opacity,
    }
    failures = _receipt_failures(
        records, {task.task_id: task for task in selected}, proposals, truth
    )
    if failures:
        raise RuntimeError("calibration evidence receipt replay failed")
    calibration_dir.mkdir(parents=True, exist_ok=True)
    records_path = calibration_dir / "records.jsonl"
    trajectories_path = calibration_dir / "trajectories.json"
    parent._write_jsonl(records_path, records)
    parent._write_json(trajectories_path, trajectories)
    result = {
        "status": "passed" if interface["passed"] else "failed",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "official_rlm_commit": official_commit,
        "evaluation_task_ids_accessed": False,
        "task_ids": [task.task_id for task in selected],
        "construction_gate": construction,
        "interface_gate": interface,
        "provider_capability_usage": capability["usage"],
        "evidence_receipt_failures": len(failures),
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
        raise RuntimeError("pregate registration already exists")
    if not CALIBRATION_RECEIPT.exists():
        raise RuntimeError("pregate calibration receipt is absent")
    calibration = json.loads(CALIBRATION_RECEIPT.read_text(encoding="utf-8"))
    if calibration["status"] != "passed" or calibration["config_sha256"] != config_hash:
        raise RuntimeError("pregate calibration is not valid for this config")
    if not verify_file_sha256(_root_path(calibration["result_path"]), calibration["result_sha256"]):
        raise RuntimeError("pregate calibration result hash mismatch")
    eval_tasks = _selected_tasks(config, "eval")
    _, proposals, _ = _load_inputs(config)
    opacity = base._initial_context_gate(eval_tasks, proposals)
    if not opacity["passed"]:
        raise RuntimeError("evaluation initial context opacity gate failed")
    gated_ids = [task.task_id for task in eval_tasks if should_invoke_rlm(task, proposals[task.task_id])]
    bound = []
    for value in [*config["registration_artifacts"], CALIBRATION_RECEIPT.relative_to(ROOT).as_posix()]:
        bound.append({"path": Path(value).as_posix(), "sha256": canonical_file_sha256(_root_path(value))})
    registration = {
        "status": "registered_before_evaluation_provider_outcomes",
        "protocol_id": config["protocol_id"],
        "config_path": DEFAULT_CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": config_hash,
        "architecture_hashes": architecture_hashes(config["provider_cost_regimes"]),
        "acquisition_contract_hash": ACQUISITION_CONTRACT_HASH,
        "gate_id": GATE_ID,
        "evaluation_task_ids": [task.task_id for task in eval_tasks],
        "evaluation_task_selection_sha256": canonical_sha256([task.task_id for task in eval_tasks]),
        "gated_provider_task_ids": gated_ids,
        "gated_provider_task_ids_sha256": canonical_sha256(gated_ids),
        "ungated_provider_calls_registered": len(eval_tasks),
        "gated_provider_calls_registered": len(gated_ids),
        "initial_context_opacity": opacity,
        "calibration_receipt_sha256": canonical_file_sha256(CALIBRATION_RECEIPT),
        "bound_artifacts": bound,
        "evaluation_provider_outcomes_present": False,
        "git_head_before_registration": _git_head(),
        "registered_utc": datetime.now(timezone.utc).isoformat(),
    }
    parent._write_json(REGISTRATION, registration)
    return registration


def _load_registered_config(path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if not REGISTRATION.exists():
        raise RuntimeError("pregate registration is absent")
    config, config_hash = _raw_config(path)
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if registration["config_sha256"] != config_hash:
        raise RuntimeError("pregate registered config hash mismatch")
    if registration["architecture_hashes"] != architecture_hashes(config["provider_cost_regimes"]):
        raise RuntimeError("pregate architecture hashes changed")
    if registration["gate_id"] != GATE_ID or registration["acquisition_contract_hash"] != ACQUISITION_CONTRACT_HASH:
        raise RuntimeError("pregate authority contract changed")
    for artifact in registration["bound_artifacts"]:
        if not verify_file_sha256(_root_path(artifact["path"]), artifact["sha256"]):
            raise RuntimeError(f"pregate bound artifact changed: {artifact['path']}")
    tasks = _selected_tasks(config, "eval")
    _, proposals, _ = _load_inputs(config)
    ids = [task.task_id for task in tasks]
    gated = [task.task_id for task in tasks if should_invoke_rlm(task, proposals[task.task_id])]
    if ids != registration["evaluation_task_ids"] or canonical_sha256(ids) != registration["evaluation_task_selection_sha256"]:
        raise RuntimeError("pregate evaluation task selection changed")
    if gated != registration["gated_provider_task_ids"] or canonical_sha256(gated) != registration["gated_provider_task_ids_sha256"]:
        raise RuntimeError("pregate provider invocation set changed")
    return config, config_hash, registration


def validate(config: Mapping[str, Any], config_hash: str, registration: Mapping[str, Any], output: Path) -> dict[str, Any]:
    head = base._official_rlm_gate(config)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is absent")
    if any((output / name).exists() for name in ("result.json", "records.jsonl", "result_receipt.json")):
        raise RuntimeError("pregate evaluation outcomes already exist")
    if FINAL_RECEIPT.exists():
        raise RuntimeError("pregate final receipt already exists")
    return {
        "status": "valid",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "official_rlm_commit": head,
        "evaluation_tasks": len(registration["evaluation_task_ids"]),
        "ungated_provider_calls": registration["ungated_provider_calls_registered"],
        "gated_provider_calls": registration["gated_provider_calls_registered"],
        "provider_calls_maximum": registration["ungated_provider_calls_registered"] + registration["gated_provider_calls_registered"],
        "record_count_expected": len(registration["evaluation_task_ids"]) * len(ARCHITECTURES),
        "outcomes_present": False,
    }


def _write_task_shard(output, task, records, trajectories, config_hash, hashes):
    shards = output / "shards"
    final = shards / task.task_id
    if final.exists():
        raise RuntimeError(f"refusing to overwrite pregate shard: {task.task_id}")
    temporary = shards / f".{task.task_id}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    records_path = temporary / "records.jsonl"
    trajectories_path = temporary / "trajectories.json"
    parent._write_jsonl(records_path, records)
    parent._write_json(trajectories_path, trajectories)
    parent._write_json(
        temporary / "receipt.json",
        {
            "status": "complete",
            "protocol_id": "rlm_uncertainty_pregate_v0_1",
            "config_sha256": config_hash,
            "task_id": task.task_id,
            "architecture_hashes": hashes,
            "record_count": len(records),
            "records_sha256": canonical_file_sha256(records_path),
            "trajectories_sha256": canonical_file_sha256(trajectories_path),
            "sealed_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    temporary.rename(final)


def _load_task_shard(output, task, config_hash, hashes):
    shard = output / "shards" / task.task_id
    if not shard.exists():
        return None
    receipt = json.loads((shard / "receipt.json").read_text(encoding="utf-8"))
    if receipt["config_sha256"] != config_hash or receipt["architecture_hashes"] != hashes:
        raise RuntimeError(f"pregate shard identity mismatch: {task.task_id}")
    records_path, trajectories_path = shard / "records.jsonl", shard / "trajectories.json"
    if not verify_file_sha256(records_path, receipt["records_sha256"]):
        raise RuntimeError(f"pregate shard record hash mismatch: {task.task_id}")
    if not verify_file_sha256(trajectories_path, receipt["trajectories_sha256"]):
        raise RuntimeError(f"pregate shard trajectory hash mismatch: {task.task_id}")
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    trajectories = json.loads(trajectories_path.read_text(encoding="utf-8"))
    if [row["architecture_id"] for row in records] != list(ARCHITECTURES):
        raise RuntimeError(f"pregate shard architecture order mismatch: {task.task_id}")
    return records, trajectories


def _macro(records, architecture, getter):
    return mean(
        mean(getter(row) for row in records if row["architecture_id"] == architecture and row["task_family"] == family)
        for family in FAMILIES
    )


def _summaries(records, config):
    output = []
    for architecture in ARCHITECTURES:
        rows = [row for row in records if row["architecture_id"] == architecture]
        contracts = [row["wrapper_contract_passed"] for row in rows if row["wrapper_contract_passed"] is not None]
        output.append({
            "architecture_id": architecture,
            "cells": len(rows),
            "macro_raw_utility": _macro(records, architecture, lambda row: float(row["raw_utility"])),
            "macro_evidence_net_utility": _macro(records, architecture, lambda row: float(row["evidence_net_utility"])),
            "macro_all_in_utility": {
                name: _macro(records, architecture, lambda row, key=name: float(row["all_in_utility"][key]))
                for name in config["provider_cost_regimes"]
            },
            "provider_calls": sum(bool(row["provider_called"]) for row in rows),
            "gate_open_tasks": sum(bool(row["gate_open"]) for row in rows),
            "total_tokens": sum(int(row["usage"]["total_tokens"]) for row in rows),
            "provider_wall_seconds": sum(float(row["provider_wall_seconds"]) for row in rows),
            "mean_query_count": mean(float(row["query_count"]) for row in rows),
            "action_change_rate": mean(float(row["action_changed"]) for row in rows),
            "unsafe_count": sum(bool(row["unsafe"]) for row in rows),
            "provider_errors": sum(row["provider_error"] is not None for row in rows),
            "contract_rate_on_calls": mean(float(value) for value in contracts) if contracts else None,
            "query_sequences": dict(sorted(Counter("->".join(row["query_sequence"]) or "no_query" for row in rows).items())),
        })
    return output


def _comparison(records, treatment, control, config):
    lookup = {(row["architecture_id"], row["task_id"]): row for row in records}
    fields = {"raw_utility": {}, "evidence_net_utility": {}}
    all_in = {name: {} for name in config["provider_cost_regimes"]}
    signs = Counter()
    primary = "primary"
    for family in FAMILIES:
        treated = [row for row in records if row["architecture_id"] == treatment and row["task_family"] == family]
        raw, evidence, primary_values = [], [], []
        regime_values = {name: [] for name in config["provider_cost_regimes"]}
        for row in treated:
            other = lookup[(control, row["task_id"])]
            raw.append(float(row["raw_utility"]) - float(other["raw_utility"]))
            evidence.append(float(row["evidence_net_utility"]) - float(other["evidence_net_utility"]))
            for name in regime_values:
                value = float(row["all_in_utility"][name]) - float(other["all_in_utility"][name])
                regime_values[name].append(value)
                if name == primary:
                    primary_values.append(value)
        fields["raw_utility"][family] = mean(raw)
        fields["evidence_net_utility"][family] = mean(evidence)
        for name in regime_values:
            all_in[name][family] = mean(regime_values[name])
        signs.update("positive" if value > 1e-12 else "negative" if value < -1e-12 else "zero" for value in primary_values)
    treatment_rows = [row for row in records if row["architecture_id"] == treatment]
    control_rows = [row for row in records if row["architecture_id"] == control]
    return {
        "treatment": treatment,
        "control": control,
        "macro_raw_utility_delta": mean(fields["raw_utility"].values()),
        "macro_evidence_net_utility_delta": mean(fields["evidence_net_utility"].values()),
        "macro_all_in_utility_delta": {name: mean(values.values()) for name, values in all_in.items()},
        "family_raw_deltas": fields["raw_utility"],
        "family_primary_all_in_deltas": all_in[primary],
        "primary_task_signs": dict(signs),
        "provider_call_delta": sum(bool(row["provider_called"]) for row in treatment_rows) - sum(bool(row["provider_called"]) for row in control_rows),
        "token_delta": sum(int(row["usage"]["total_tokens"]) for row in treatment_rows) - sum(int(row["usage"]["total_tokens"]) for row in control_rows),
        "provider_wall_seconds_delta": sum(float(row["provider_wall_seconds"]) for row in treatment_rows) - sum(float(row["provider_wall_seconds"]) for row in control_rows),
    }


def run(config, config_hash, registration, output):
    if (output / "result.json").exists() or (output / "records.jsonl").exists():
        raise RuntimeError("refusing to overwrite pregate evaluation")
    output.mkdir(parents=True, exist_ok=True)
    capability = base._provider_capability(config, output / "evaluation_provider_capability_receipt.json")
    tasks, proposals, truth = _load_inputs(config)
    selected = sorted([task for task in tasks if task.split == "eval"], key=lambda task: (FAMILIES.index(task.family), task.task_id))
    hashes = architecture_hashes(config["provider_cost_regimes"])
    records, trajectories = [], []
    for task_index, task in enumerate(selected):
        sealed = _load_task_shard(output, task, config_hash, hashes)
        if sealed is not None:
            records.extend(sealed[0]); trajectories.extend(sealed[1]); continue
        api_order = list(config["api_order"]["even_task_index" if task_index % 2 == 0 else "odd_task_index"])
        cache = {}
        for architecture in api_order:
            cache[architecture] = _run_architecture(architecture, task, proposals[task.task_id], truth[task.task_id], config)
        shard_records, shard_trajectories = [], []
        for architecture in ARCHITECTURES:
            record, trajectory = cache.get(architecture) or _run_architecture(
                architecture, task, proposals[task.task_id], truth[task.task_id], config
            )
            shard_records.append(record)
            shard_trajectories.append({"record_id": record["record_id"], **trajectory})
        failures = _receipt_failures(shard_records, {task.task_id: task}, proposals, truth)
        if failures:
            raise RuntimeError(f"pregate shard evidence receipt failure: {failures}")
        _write_task_shard(output, task, shard_records, shard_trajectories, config_hash, hashes)
        records.extend(shard_records); trajectories.extend(shard_trajectories)
    failures = _receipt_failures(records, {task.task_id: task for task in selected}, proposals, truth)
    fixed = {row["task_id"]: row for row in records if row["architecture_id"] == "fixed_no_query"}
    forced = {row["task_id"]: row for row in records if row["architecture_id"] == "gated_forced_failure"}
    forced_identity = all(
        fixed[task_id]["executed_action"] == forced[task_id]["executed_action"]
        and fixed[task_id]["raw_utility"] == forced[task_id]["raw_utility"]
        and fixed[task_id]["all_in_utility"] == forced[task_id]["all_in_utility"]
        for task_id in fixed
    )
    closed_call_violations = sum(
        row["architecture_id"] == "gated_rlm" and not row["gate_open"] and row["provider_called"]
        for row in records
    )
    observed_gated_calls = sum(
        row["architecture_id"] == "gated_rlm" and row["provider_called"] for row in records
    )
    observed_ungated_calls = sum(
        row["architecture_id"] == "ungated_rlm" and row["provider_called"] for row in records
    )
    invocation_identity = (
        observed_gated_calls == registration["gated_provider_calls_registered"]
        and observed_ungated_calls == registration["ungated_provider_calls_registered"]
    )
    unsafe = sum(bool(row["unsafe"]) for row in records)
    if failures or unsafe or not forced_identity or closed_call_violations or not invocation_identity:
        raise RuntimeError("pregate evaluation hard gate failed")
    records_path, trajectories_path = output / "records.jsonl", output / "trajectory_manifest.json"
    parent._write_jsonl(records_path, records)
    parent._write_json(trajectories_path, trajectories)
    comparisons = [
        _comparison(records, "gated_rlm", "ungated_rlm", config),
        _comparison(records, "gated_rlm", "fixed_no_query", config),
        _comparison(records, "gated_rlm", "gated_deterministic_voi", config),
    ]
    result = {
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "git_head": _git_head(),
        "registration_sha256": canonical_file_sha256(REGISTRATION),
        "record_count": len(records),
        "task_ids": [task.task_id for task in selected],
        "records_path": records_path.relative_to(ROOT).as_posix(),
        "records_sha256": canonical_file_sha256(records_path),
        "trajectory_manifest_path": trajectories_path.relative_to(ROOT).as_posix(),
        "trajectory_manifest_sha256": canonical_file_sha256(trajectories_path),
        "summary": _summaries(records, config),
        "comparisons": comparisons,
        "provider_capability_usage": capability["usage"],
        "evidence_receipt_failures": len(failures),
        "typed_unsafe_count": unsafe,
        "forced_failure_identity_passed": forced_identity,
        "closed_gate_provider_call_violations": closed_call_violations,
        "provider_invocation_identity_passed": invocation_identity,
        "registered_gated_provider_calls": registration["gated_provider_calls_registered"],
        "observed_gated_provider_calls": observed_gated_calls,
        "observed_ungated_provider_calls": observed_ungated_calls,
        "claim_boundary": config["claim_boundary"],
    }
    parent._write_json(output / "result.json", result)
    return result


def _render_report(result, receipt, path):
    rows = [
        "| Architecture | Raw | Evidence net | All-in primary | Calls | Tokens | Wall s | Unsafe | Contract |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(result["summary"], key=lambda value: value["macro_all_in_utility"]["primary"], reverse=True):
        contract = "-" if row["contract_rate_on_calls"] is None else f'{row["contract_rate_on_calls"]:.4f}'
        rows.append(
            f'| `{row["architecture_id"]}` | {row["macro_raw_utility"]:.4f} | '
            f'{row["macro_evidence_net_utility"]:.4f} | {row["macro_all_in_utility"]["primary"]:.4f} | '
            f'{row["provider_calls"]} | {row["total_tokens"]:,} | {row["provider_wall_seconds"]:.2f} | '
            f'{row["unsafe_count"]} | {contract} |'
        )
    contrasts = "\n".join(
        f'- `{row["treatment"]}` vs `{row["control"]}`: raw {row["macro_raw_utility_delta"]:+.4f}, '
        f'evidence net {row["macro_evidence_net_utility_delta"]:+.4f}, primary all-in '
        f'{row["macro_all_in_utility_delta"]["primary"]:+.4f}, calls {row["provider_call_delta"]:+d}, '
        f'tokens {row["token_delta"]:+d}, wall {row["provider_wall_seconds_delta"]:+.2f}s.'
        for row in result["comparisons"]
    )
    body = f"""# RLM Uncertainty Pre-Gate v0.1

## Result

{chr(10).join(rows)}

## Registered Contrasts

{contrasts}

The deterministic gate registered `{result['registered_gated_provider_calls']}` gated calls and observed
`{result['observed_gated_provider_calls']}`. Closed-gate call violations: `{result['closed_gate_provider_call_violations']}`.
Unsafe executions: `{result['typed_unsafe_count']}`. Forced-failure identity:
`{str(result['forced_failure_identity_passed']).lower()}`.

## Cost Definition

Primary all-in utility subtracts evidence cost, `0.001` utility per 1,000 provider tokens, and `0.0005` utility
per provider wall second. These are registered benchmark-local exchange rates, not actual dollar prices.

## Boundary

{result['claim_boundary']}

## Integrity

- Result SHA-256: `{receipt['result_sha256']}`
- Records SHA-256: `{receipt['records_sha256']}`
- Trajectory SHA-256: `{receipt['trajectory_manifest_sha256']}`
- Config SHA-256: `{receipt['config_sha256']}`
- Registration SHA-256: `{receipt['registration_sha256']}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")


def finalize(config, config_hash, registration, output):
    result_path = output / "result.json"
    if not result_path.exists():
        raise RuntimeError("pregate evaluation result is absent")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["config_sha256"] != config_hash or result["registration_sha256"] != canonical_file_sha256(REGISTRATION):
        raise RuntimeError("pregate result registration mismatch")
    if result["typed_unsafe_count"] or result["evidence_receipt_failures"] or result["closed_gate_provider_call_violations"] or not result["forced_failure_identity_passed"] or not result["provider_invocation_identity_passed"]:
        raise RuntimeError("pregate result hard gate failed")
    records_path = _root_path(result["records_path"])
    trajectories_path = _root_path(result["trajectory_manifest_path"])
    if not verify_file_sha256(records_path, result["records_sha256"]):
        raise RuntimeError("pregate records hash mismatch")
    if not verify_file_sha256(trajectories_path, result["trajectory_manifest_sha256"]):
        raise RuntimeError("pregate trajectory hash mismatch")
    receipt = {
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "config_path": DEFAULT_CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": config_hash,
        "registration_path": REGISTRATION.relative_to(ROOT).as_posix(),
        "registration_sha256": canonical_file_sha256(REGISTRATION),
        "result_path": result_path.relative_to(ROOT).as_posix(),
        "result_sha256": canonical_file_sha256(result_path),
        "records_path": result["records_path"],
        "records_sha256": result["records_sha256"],
        "trajectory_manifest_path": result["trajectory_manifest_path"],
        "trajectory_manifest_sha256": result["trajectory_manifest_sha256"],
        "record_count": result["record_count"],
        "typed_unsafe_count": result["typed_unsafe_count"],
        "forced_failure_identity_passed": result["forced_failure_identity_passed"],
        "closed_gate_provider_call_violations": result["closed_gate_provider_call_violations"],
        "provider_invocation_identity_passed": result["provider_invocation_identity_passed"],
        "claim_boundary": config["claim_boundary"],
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
    }
    parent._write_json(output / "result_receipt.json", receipt)
    parent._write_json(FINAL_RECEIPT, receipt)
    _render_report(result, receipt, DEFAULT_REPORT)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calibrate", "register", "validate", "run", "finalize"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vram-fraction", type=float, default=0.35)
    args = parser.parse_args()
    if args.phase in {"calibrate", "register"}:
        config, config_hash = _raw_config(args.config.resolve())
        value = calibrate(config, config_hash, args.output.resolve()) if args.phase == "calibrate" else register(config, config_hash)
    else:
        config, config_hash, registration = _load_registered_config(args.config.resolve())
        if args.phase == "validate":
            value = validate(config, config_hash, registration, args.output.resolve())
        elif args.phase == "run":
            value = run(config, config_hash, registration, args.output.resolve())
        else:
            value = finalize(config, config_hash, registration, args.output.resolve())
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
