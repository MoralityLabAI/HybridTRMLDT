from __future__ import annotations

import json
from pathlib import Path

from research_gym.integrity import canonical_file_sha256
from research_gym.benchmarks.rlm_hybrid_neighborhood import LongContextControlTask, materialize_task_suite
from research_gym.benchmarks.rlm_hybrid_runtime import (
    API_ARCHITECTURES,
    LOCAL_ARCHITECTURES,
    _conductor_tools,
    run_local_architecture,
)
from research_gym.scripts import bench_rlm_trm_ldt_hybrid_neighborhood_v1 as runner


def _task() -> LongContextControlTask:
    payload = materialize_task_suite()
    return next(
        LongContextControlTask.from_jsonable(row)
        for row in payload["tasks"]
        if len(row["exact_allowed"]) < len(row["candidates"])
    )


def _trained_row(task: LongContextControlTask) -> dict[str, object]:
    return {
        "ranked_actions": list(task.candidates),
        "confidence": 0.7,
        "claimed_provenance": "model_sound",
    }


def test_runtime_partition_covers_all_architectures() -> None:
    assert len(API_ARCHITECTURES) == 6
    assert len(LOCAL_ARCHITECTURES) == 5
    assert not set(API_ARCHITECTURES) & set(LOCAL_ARCHITECTURES)


def test_fixed_typed_local_arms_never_execute_unsafe() -> None:
    task = _task()
    unsafe = next(action for action in task.candidates if action not in task.exact_allowed)
    row = _trained_row(task)
    row["ranked_actions"] = [unsafe, *[action for action in task.candidates if action != unsafe]]
    for architecture in ("proxy_trm_ldt_fixed", "trained_trm_ldt_fixed"):
        record, _ = run_local_architecture(architecture, task, row, 211)
        assert record["safe"]
        assert record["executed_action"] in task.exact_allowed


def test_conductor_tools_require_matching_proposal_and_certificate() -> None:
    task = _task()
    tools, state = _conductor_tools(task, _trained_row(task), {}, recursive=False)
    propose = tools["trm_propose"]["tool"]
    verify = tools["ldt_verify"]["tool"]
    commit = tools["hybrid_commit"]["tool"]
    proposal = propose(2)
    action = proposal["beam"][0]
    certificate = verify(action)
    result = commit(action, proposal["proposal_ref"], certificate["certificate_ref"])
    assert result["accepted"] == (action in task.exact_allowed)
    forged = commit(action, "missing", certificate["certificate_ref"])
    assert not forged["accepted"]
    assert state["calls"]


def test_screen_and_eval_stage_selection() -> None:
    config = json.loads(Path("configs/rlm_trm_ldt_hybrid_neighborhood_v1.json").read_text())
    tasks = [LongContextControlTask.from_jsonable(row) for row in materialize_task_suite()["tasks"]]
    seed, screen = runner._stage_tasks("screen", tasks)
    assert seed == 211
    assert len(screen) == 8
    seed, evaluation = runner._stage_tasks("eval_223_storyworld_control", tasks)
    assert seed == 223
    assert len(evaluation) == 6
    assert {task.family for task in evaluation} == {"storyworld_control"}
    orders = {tuple(runner._rotated_api_order(config, index, 211)) for index in range(6)}
    assert len(orders) == 6


def test_global_provider_access_errors_fail_fast() -> None:
    class Error(Exception):
        status_code = 403
        body = {"error": {"code": "model_not_found", "type": "invalid_request_error"}}

    assert runner._is_global_provider_error(Error())


def test_registration_hash_and_architectures_when_present() -> None:
    registration_path = Path("configs/rlm_trm_ldt_hybrid_neighborhood_v1_registration.json")
    if not registration_path.exists():
        return
    registration = json.loads(registration_path.read_text())
    assert canonical_file_sha256(registration["config_path"]) == registration["config_sha256"]
    from research_gym.benchmarks.rlm_hybrid_neighborhood import architecture_hashes

    assert registration["architecture_hashes"] == architecture_hashes()
    addendum_path = Path("configs/rlm_trm_ldt_hybrid_neighborhood_v1_construction_addendum_1.json")
    if addendum_path.exists():
        addendum = json.loads(addendum_path.read_text())
        assert canonical_file_sha256(addendum["parent_registration_path"]) == addendum["parent_registration_sha256"]
        assert canonical_file_sha256(addendum["triggering_receipt_path"]) == addendum["triggering_receipt_sha256"]
        evaluator_path = "research_gym/scripts/bench_rlm_trm_ldt_hybrid_neighborhood_v1.py"
        assert canonical_file_sha256(evaluator_path) == addendum["new_evaluator_sha256"]


def test_resource_receipt_indexing_retains_failures(tmp_path: Path) -> None:
    values = {
        1: {"status": "construction_failure", "cleanup_passed": True, "abort_reason": "foreign_gpu"},
        2: {"status": "completed", "cleanup_passed": True, "abort_reason": None},
        3: {"status": "completed", "cleanup_passed": True, "abort_reason": None},
    }
    for attempt, payload in values.items():
        (tmp_path / f"run.attempt-{attempt}.resource_receipt.json").write_text(json.dumps(payload))

    completed, failures = runner._run_resource_receipts(tmp_path)

    assert [attempt for attempt, _, _ in completed] == [2, 3]
    assert [row["attempt"] for row in failures] == [1]
    assert failures[0]["abort_reason"] == "foreign_gpu"
