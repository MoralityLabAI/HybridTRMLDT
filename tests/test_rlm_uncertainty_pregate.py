from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import pytest

from research_gym.benchmarks.rlm_uncertainty_pregate import (
    ProviderCostSpec,
    all_in_utility,
    provider_utility_cost,
    should_invoke_rlm,
)
from research_gym.scripts import bench_rlm_evidence_acquisition_v0 as v0
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
