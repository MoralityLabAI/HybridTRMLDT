from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import pytest

from research_gym.benchmarks.rlm_uncertainty_pregate import (
    ARCHITECTURES,
    ProviderCostSpec,
    all_in_utility,
    architecture_hashes,
    provider_utility_cost,
    should_invoke_rlm,
)
from research_gym.integrity import canonical_file_sha256
from research_gym.scripts import bench_rlm_evidence_acquisition_v0 as v0
from research_gym.scripts import bench_rlm_uncertainty_pregate_v0_1 as runner
from research_gym.scripts import audit_rlm_uncertainty_pregate_v0_1 as audit
from research_gym.scripts import materialize_rlm_uncertainty_pregate_v0_1 as materializer


ROOT = Path(__file__).resolve().parents[1]


def test_three_way_gate_is_frozen_from_v0_prior_and_retains_raw_gain() -> None:
    config, _ = v0._raw_config(v0.DEFAULT_CONFIG)
    tasks, proposals, _ = v0._load_inputs(config)
    evaluation = {task.task_id: task for task in tasks if task.split == "eval"}
    records = [
        json.loads(line)
        for line in (
            ROOT / "experiments/rlm_evidence_acquisition_mesh_v0/records.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    lookup = {(row["architecture_id"], row["task_id"]): row for row in records}
    gated_raw = []
    fixed_raw = []
    ungated_raw = []
    invoked = 0
    for task_id, task in evaluation.items():
        fixed = lookup[("mesh_fixed_no_query", task_id)]
        adaptive = lookup[("rlm_adaptive_two_query", task_id)]
        invoke = should_invoke_rlm(task, proposals[task_id])
        invoked += int(invoke)
        gated_raw.append(adaptive["raw_utility"] if invoke else fixed["raw_utility"])
        fixed_raw.append(fixed["raw_utility"])
        ungated_raw.append(adaptive["raw_utility"])
    assert invoked == 9
    assert mean(gated_raw) - mean(fixed_raw) == pytest.approx(
        mean(ungated_raw) - mean(fixed_raw)
    )


def test_provider_cost_is_explicit_and_all_in() -> None:
    cost = ProviderCostSpec(token_utility_per_1k=0.001, latency_utility_per_second=0.0005)
    usage = {"total_tokens": 7000}
    assert provider_utility_cost(usage, 4.0, cost) == pytest.approx(0.009)
    assert all_in_utility(0.70, usage, 4.0, cost) == pytest.approx(0.691)


def test_fresh_materializer_identity() -> None:
    assert materializer.GENERATOR_SEED == 494117
    assert materializer.TASK_SUFFIX == "__pregate_v0_1"
    assert materializer.CORPUS_ID == "rlm_uncertainty_pregate_v0_1"


def test_fresh_corpus_replays_without_provider_outcomes() -> None:
    receipt = json.loads(materializer.DEFAULT_RECEIPT.read_text(encoding="utf-8"))
    assert receipt["task_count"] == receipt["proposal_count"] == receipt["truth_count"] == 152
    assert receipt["provider_outcomes_present"] is False
    assert receipt["evaluation_outcomes_accessed"] is False
    for name in ("tasks", "proposals", "truth", "checkpoint", "source"):
        assert canonical_file_sha256(ROOT / receipt[f"{name}_path"]) == receipt[f"{name}_sha256"]


def test_protocol_construction_gate_and_calibration_selection() -> None:
    config, _ = runner._raw_config(runner.DEFAULT_CONFIG)
    tasks, proposals, truth = runner._load_inputs(config)
    calibration = [task for task in tasks if task.split == "calibration"]
    gate = runner._construction_gate(calibration, proposals, truth, config)
    selected = runner._selected_tasks(config, "calibration")
    assert gate["passed"]
    assert gate["invoked_tasks"] == 11
    assert gate["families_represented"] == 4
    assert len(selected) == 8
    assert all(should_invoke_rlm(task, proposals[task.task_id]) for task in selected)
    assert {family: sum(task.family == family for task in selected) for family in runner.FAMILIES} == {
        family: 2 for family in runner.FAMILIES
    }
    assert tuple(architecture_hashes(config["provider_cost_regimes"])) == ARCHITECTURES


def test_closed_gate_performs_no_provider_call(monkeypatch) -> None:
    config, _ = runner._raw_config(runner.DEFAULT_CONFIG)
    tasks, proposals, truth = runner._load_inputs(config)
    task = next(
        task
        for task in tasks
        if task.split == "calibration" and not should_invoke_rlm(task, proposals[task.task_id])
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("closed gate invoked provider")

    monkeypatch.setattr(runner, "run_rlm_acquisition", forbidden)
    record, _ = runner._run_architecture(
        "gated_rlm", task, proposals[task.task_id], truth[task.task_id], config
    )
    assert not record["gate_open"]
    assert not record["provider_called"]
    assert record["usage"]["total_tokens"] == 0
    assert record["provider_wall_seconds"] == 0


def test_pregate_shard_roundtrip(tmp_path) -> None:
    config, config_hash = runner._raw_config(runner.DEFAULT_CONFIG)
    task = runner._selected_tasks(config, "calibration")[0]
    hashes = architecture_hashes(config["provider_cost_regimes"])
    records = [
        {"architecture_id": architecture, "task_id": task.task_id}
        for architecture in ARCHITECTURES
    ]
    trajectories = [{"record_id": architecture} for architecture in ARCHITECTURES]
    runner._write_task_shard(tmp_path, task, records, trajectories, config_hash, hashes)
    assert runner._load_task_shard(tmp_path, task, config_hash, hashes) == (
        records,
        trajectories,
    )


def test_sealed_pregate_audit_recomputes_and_retains_negative_endpoint() -> None:
    result = audit.build_audit(audit.DEFAULT_OUTPUT)
    operational = result["registered_operational_result"]
    assert result["canonical_replay"]["records_replayed"] == 144
    assert result["canonical_replay"]["task_shards_replayed"] == 24
    assert result["canonical_replay"]["evidence_receipt_failures"] == 0
    assert operational["macro_all_in_utility_delta"]["primary"] == pytest.approx(
        -0.004927398583334475
    )
    assert result["repeatability"]["query_sequence_matches"] == 2
    assert result["repeatability"]["executed_action_matches"] == 7
    assert result["coupled_replay"]["macro_all_in_utility_delta_vs_ungated"][
        "primary"
    ] == pytest.approx(0.00610969902708283)
    assert result["winner"]["architecture"] == "gated_deterministic_voi"
