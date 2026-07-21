from __future__ import annotations

import json
from pathlib import Path

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
