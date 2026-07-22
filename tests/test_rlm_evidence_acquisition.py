from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from statistics import mean

import pytest

from research_gym.benchmarks.rlm_evidence_acquisition import (
    ACQUISITION_CONTRACT_HASH,
    ARCHITECTURES,
    COUNTERFACTUAL_ROLLOUT,
    EXACT_MECHANICS,
    INDEPENDENT_PROBE,
    QUERY_COSTS,
    QUERY_IDS,
    RECEIPT_ATTESTATION,
    acquisition_context,
    architecture_hashes,
    architecture_query_sequence,
    build_initial_snapshot,
    execute_acquisition,
    make_evidence_receipt,
    parse_acquisition_decision,
    sequence_oracle,
    valid_query_sequences,
    verify_evidence_receipt,
)
from research_gym.benchmarks import rlm_evidence_runtime
from research_gym.benchmarks.rlm_hybrid_neighborhood import LongContextControlTask
from research_gym.integrity import canonical_file_sha256
from research_gym.scripts import bench_rlm_evidence_acquisition_v0 as runner
from research_gym.scripts import materialize_rlm_evidence_acquisition_v0 as materializer


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = ROOT / "data/benchmarks/rlm_evidence_acquisition_mesh_v0_materialization_receipt.json"


def _inputs():
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    payload = json.loads((ROOT / receipt["tasks_path"]).read_text(encoding="utf-8"))
    proposals = {
        row["task_id"]: row
        for row in map(json.loads, (ROOT / receipt["proposals_path"]).read_text(encoding="utf-8").splitlines())
    }
    truth = {
        row["task_id"]: row
        for row in map(json.loads, (ROOT / receipt["truth_path"]).read_text(encoding="utf-8").splitlines())
    }
    tasks = [LongContextControlTask.from_jsonable(row) for row in payload["tasks"]]
    return receipt, payload, tasks, proposals, truth


def test_fresh_corpus_replays_hashes_counts_and_isolated_ids() -> None:
    receipt, _, tasks, proposals, truth = _inputs()
    assert receipt["generator_seed"] == materializer.GENERATOR_SEED == 394117
    assert receipt["provider_outcomes_present"] is False
    assert len(tasks) == len(proposals) == len(truth) == 152
    assert all(task.task_id.endswith(materializer.TASK_SUFFIX) for task in tasks)
    assert len({task.task_id for task in tasks}) == len(tasks)
    for key in ("tasks", "proposals", "truth", "checkpoint", "source"):
        assert canonical_file_sha256(ROOT / receipt[f"{key}_path"]) == receipt[f"{key}_sha256"]


def test_currentness_strata_are_balanced_in_eval_and_train() -> None:
    _, _, tasks, _, truth = _inputs()
    by_id = {task.task_id: task for task in tasks}
    for split, expected in (("train", 8), ("eval", 2)):
        for family in {task.family for task in tasks}:
            counts = Counter(
                row["stratum"]
                for task_id, row in truth.items()
                if by_id[task_id].split == split and by_id[task_id].family == family
            )
            assert counts == {
                "proxy_stale": expected,
                "trained_stale": expected,
                "both_current": expected,
            }


def test_snapshot_and_context_hide_task_action_and_oracle_fields() -> None:
    _, _, tasks, proposals, truth = _inputs()
    for task in tasks[::17]:
        snapshot = build_initial_snapshot(task, proposals[task.task_id])
        receipt = make_evidence_receipt(task, proposals[task.task_id], truth[task.task_id], INDEPENDENT_PROBE)
        context = acquisition_context(snapshot, [receipt], [INDEPENDENT_PROBE])
        serialized = json.dumps(context, sort_keys=True)
        assert task.transcript not in serialized
        assert task.family not in serialized
        assert task.optimal_action not in _string_leaves(context)
        for action in task.candidates:
            assert not re.search(rf"(?<![A-Za-z0-9_]){re.escape(action)}(?![A-Za-z0-9_])", serialized)
        assert context["snapshot"]["contract_hash"] == ACQUISITION_CONTRACT_HASH


def _string_leaves(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        output: set[str] = set()
        for key, item in value.items():
            output.update(_string_leaves(key))
            output.update(_string_leaves(item))
        return output
    if isinstance(value, list):
        output = set()
        for item in value:
            output.update(_string_leaves(item))
        return output
    return set()


@pytest.mark.parametrize("query_id", QUERY_IDS)
def test_evidence_receipts_replay_and_reject_tampering(query_id: str) -> None:
    _, _, tasks, proposals, truth = _inputs()
    task = tasks[0]
    receipt = make_evidence_receipt(task, proposals[task.task_id], truth[task.task_id], query_id)
    assert verify_evidence_receipt(task, proposals[task.task_id], truth[task.task_id], receipt)
    tampered = json.loads(json.dumps(receipt))
    tampered["cost"] += 0.001
    assert not verify_evidence_receipt(task, proposals[task.task_id], truth[task.task_id], tampered)


def test_sequence_space_costs_safety_and_duplicate_rejection() -> None:
    _, _, tasks, proposals, truth = _inputs()
    assert len(valid_query_sequences()) == 17
    task = tasks[0]
    outcome = execute_acquisition(
        task,
        proposals[task.task_id],
        truth[task.task_id],
        (EXACT_MECHANICS, COUNTERFACTUAL_ROLLOUT),
    )
    assert outcome.executed_action in task.exact_allowed
    assert outcome.query_cost == pytest.approx(QUERY_COSTS[EXACT_MECHANICS] + QUERY_COSTS[COUNTERFACTUAL_ROLLOUT])
    assert outcome.net_utility == pytest.approx(outcome.raw_utility - outcome.query_cost)
    with pytest.raises(ValueError, match="repeats"):
        execute_acquisition(
            task,
            proposals[task.task_id],
            truth[task.task_id],
            (EXACT_MECHANICS, EXACT_MECHANICS),
        )


def test_decision_parser_accepts_only_registered_bounded_forms() -> None:
    available = list(QUERY_IDS)
    assert parse_acquisition_decision("STOP", available) == "STOP"
    assert parse_acquisition_decision(EXACT_MECHANICS, available) == EXACT_MECHANICS
    assert parse_acquisition_decision('{"decision":"query","query_id":"receipt_attestation"}', available) == RECEIPT_ATTESTATION
    assert parse_acquisition_decision('{"decision":"stop"}', available) == "STOP"
    assert parse_acquisition_decision("Use exact_mechanics", available) is None
    assert parse_acquisition_decision('```json\n{"decision":"stop"}\n```', available) is None
    assert parse_acquisition_decision('{"decision":"query","query_id":"exact_mechanics","action":"x"}', available) is None
    assert parse_acquisition_decision(EXACT_MECHANICS, [INDEPENDENT_PROBE]) is None


def test_calibration_construction_gate_passes_without_eval_access() -> None:
    _, _, tasks, proposals, truth = _inputs()
    calibration = [task for task in tasks if task.split == "calibration"]
    gains = []
    sequences = Counter()
    for task in calibration:
        baseline = execute_acquisition(task, proposals[task.task_id], truth[task.task_id], ())
        oracle = sequence_oracle(task, proposals[task.task_id], truth[task.task_id])
        gains.append(oracle.net_utility - baseline.net_utility)
        sequences[oracle.query_sequence] += 1
    two_query = {key: count for key, count in sequences.items() if len(key) == 2}
    assert mean(gains) >= 0.03
    assert len(two_query) >= 3
    assert max(two_query.values()) / sum(two_query.values()) < 0.70
    assert sum(value > 1e-12 for value in gains) >= 10


def test_architecture_hashes_and_static_sequences_are_bounded() -> None:
    _, _, tasks, proposals, _ = _inputs()
    hashes = architecture_hashes()
    assert tuple(hashes) == ARCHITECTURES
    assert len(set(hashes.values())) == len(ARCHITECTURES)
    task = tasks[0]
    for architecture in ARCHITECTURES:
        if architecture == "rlm_adaptive_two_query":
            continue
        sequence = architecture_query_sequence(architecture, task, proposals[task.task_id])
        assert len(sequence) <= 2
        assert len(sequence) == len(set(sequence))


def test_rlm_runtime_adapts_second_query_without_action_authority(monkeypatch) -> None:
    _, _, tasks, proposals, truth = _inputs()
    task = tasks[0]
    responses = iter((INDEPENDENT_PROBE, EXACT_MECHANICS))

    def fake_completion(context, runtime):
        return next(responses), {"calls": 1, "input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "reported_cost_usd": None}, {}, 0.1

    monkeypatch.setattr(rlm_evidence_runtime, "_completion", fake_completion)
    state, trajectory = rlm_evidence_runtime.run_rlm_acquisition(
        task,
        proposals[task.task_id],
        truth[task.task_id],
        {},
    )
    assert state["contract_passed"]
    assert state["outcome"].query_sequence == (INDEPENDENT_PROBE, EXACT_MECHANICS)
    assert trajectory["rounds"][1]["context"]["acquired_evidence"][0]["query_id"] == INDEPENDENT_PROBE
    assert state["outcome"].executed_action in task.exact_allowed


def test_rlm_provider_failure_executes_fixed_no_query_mesh(monkeypatch) -> None:
    _, _, tasks, proposals, truth = _inputs()
    task = tasks[0]

    def failed_completion(context, runtime):
        raise TimeoutError("synthetic timeout")

    monkeypatch.setattr(rlm_evidence_runtime, "_completion", failed_completion)
    state, _ = rlm_evidence_runtime.run_rlm_acquisition(
        task,
        proposals[task.task_id],
        truth[task.task_id],
        {},
    )
    baseline = execute_acquisition(task, proposals[task.task_id], truth[task.task_id], ())
    assert not state["contract_passed"]
    assert state["provider_error"]["error_type"] == "TimeoutError"
    assert state["outcome"].executed_action == baseline.executed_action
    assert state["outcome"].net_utility == baseline.net_utility


def test_rlm_second_round_failure_discards_acquired_authority_but_charges_query(monkeypatch) -> None:
    _, _, tasks, proposals, truth = _inputs()
    task = tasks[0]
    responses = iter((EXACT_MECHANICS, "not a valid decision"))

    def fake_completion(context, runtime):
        return next(responses), {"calls": 1, "input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "reported_cost_usd": None}, {}, 0.1

    monkeypatch.setattr(rlm_evidence_runtime, "_completion", fake_completion)
    state, _ = rlm_evidence_runtime.run_rlm_acquisition(
        task,
        proposals[task.task_id],
        truth[task.task_id],
        {},
    )
    baseline = execute_acquisition(task, proposals[task.task_id], truth[task.task_id], ())
    assert not state["contract_passed"]
    assert state["outcome"].executed_action == baseline.executed_action
    assert state["outcome"].raw_utility == baseline.raw_utility
    assert state["outcome"].query_cost == QUERY_COSTS[EXACT_MECHANICS]
    assert state["outcome"].net_utility == pytest.approx(
        baseline.raw_utility - QUERY_COSTS[EXACT_MECHANICS]
    )


def test_runner_config_hashes_selection_and_initial_opacity() -> None:
    config, _ = runner._raw_config(runner.DEFAULT_CONFIG)
    calibration = runner._selected_tasks(config, "calibration")
    evaluation = runner._selected_tasks(config, "eval")
    _, proposals, _ = runner._load_inputs(config)
    assert len(calibration) == 8
    assert Counter(task.family for task in calibration) == {family: 2 for family in runner.FAMILIES}
    assert len(evaluation) == 24
    assert Counter(task.family for task in evaluation) == {family: 6 for family in runner.FAMILIES}
    assert runner._initial_context_gate([*calibration, *evaluation], proposals) == {
        "passed": True,
        "contexts_checked": 32,
        "failures": [],
    }


def test_evaluation_task_shard_is_atomic_and_replays(tmp_path) -> None:
    config, config_hash = runner._raw_config(runner.DEFAULT_CONFIG)
    task = runner._selected_tasks(config, "calibration")[0]
    records = [
        {"architecture_id": architecture, "task_id": task.task_id}
        for architecture in ARCHITECTURES
    ]
    trajectories = [{"record_id": architecture} for architecture in ARCHITECTURES]
    runner._write_task_shard(tmp_path, task, records, trajectories, config_hash)
    loaded = runner._load_task_shard(tmp_path, task, config_hash)
    assert loaded == (records, trajectories)
    assert not list((tmp_path / "shards").glob(".*.tmp"))
