from __future__ import annotations

import json

import pytest

from research_gym.benchmarks.rlm_hybrid_neighborhood import (
    ACTION_VOCAB,
    FAMILIES,
    SPLIT_COUNTS,
    LongContextControlTask,
    architecture_hashes,
    architecture_ids,
    materialize_task_suite,
    parse_action,
    typed_execute,
)


def _tasks():
    payload = materialize_task_suite()
    return payload, [LongContextControlTask.from_jsonable(row) for row in payload["tasks"]]


def test_task_suite_is_deterministic_and_split_disjoint() -> None:
    left = materialize_task_suite()
    right = materialize_task_suite()
    assert left == right
    assert left["task_count"] == len(FAMILIES) * sum(SPLIT_COUNTS.values())
    groups = {
        split: {row["group_id"] for row in left["tasks"] if row["split"] == split}
        for split in SPLIT_COUNTS
    }
    assert not groups["train"] & groups["calibration"]
    assert not groups["train"] & groups["eval"]
    assert not groups["calibration"] & groups["eval"]


def test_every_task_has_long_context_and_model_view_excludes_oracle() -> None:
    _, tasks = _tasks()
    for task in tasks:
        assert len(task.transcript.splitlines()) >= 640
        assert set(task.exact_allowed) <= set(task.candidates)
        model_view = task.model_view()
        assert "exact_allowed" not in model_view
        assert "utilities" not in model_view
        assert "optimal_action" not in model_view
        assert task.optimal_action in task.exact_allowed


def test_task_roundtrip_and_prompt_hashes() -> None:
    payload, tasks = _tasks()
    restored = [LongContextControlTask.from_jsonable(row) for row in payload["tasks"]]
    assert tasks == restored
    assert len({task.prompt_sha256 for task in tasks}) == len(tasks)
    assert len(tasks[0].public_features) == payload["feature_dim"]
    assert len(ACTION_VOCAB) == len(set(ACTION_VOCAB))


def test_typed_execution_rejects_unsafe_and_uses_ldt_fallback() -> None:
    _, tasks = _tasks()
    task = next(task for task in tasks if len(task.exact_allowed) < len(task.candidates))
    unsafe = next(action for action in task.candidates if action not in task.exact_allowed)
    action, overridden, reason = typed_execute(task, unsafe)
    assert action in task.exact_allowed
    assert overridden
    assert reason == "exact_reject_ldt_fallback"


def test_action_parser_is_token_bounded() -> None:
    candidates = ("accept", "reject")
    assert parse_action("accept", candidates) == "accept"
    assert parse_action("I choose reject.", candidates) == "reject"
    assert parse_action("unacceptable", candidates) is None


def test_architecture_registry_contains_controls_and_three_hybrid_families() -> None:
    ids = architecture_ids()
    hashes = architecture_hashes()
    assert len(ids) == 11
    assert set(ids) == set(hashes)
    assert len(set(hashes.values())) == len(ids)
    assert "rlm_recursive_conductor" in ids


def test_control_trm_shapes_when_torch_available() -> None:
    torch = pytest.importorskip("torch")
    from research_gym.neural.control_trm import ControlTRMProposer

    payload = materialize_task_suite()
    model = ControlTRMProposer(payload["feature_dim"], len(ACTION_VOCAB), latent_dim=16, recurrence_steps=3)
    values = model(torch.zeros(2, payload["feature_dim"]))
    assert values.action_logits.shape == (2, len(ACTION_VOCAB))
    assert values.latents.shape == (2, 3, 16)
