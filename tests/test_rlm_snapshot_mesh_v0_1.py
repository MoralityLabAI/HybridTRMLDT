from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re

import pytest

from research_gym.benchmarks.rlm_snapshot_mesh import (
    ARCHITECTURES,
    SNAPSHOT_CONTRACT_HASH,
    architecture_hashes,
    build_opaque_snapshot,
    snapshot_prompt,
    verify_opaque_snapshot,
)
from research_gym.benchmarks.rlm_wrapped_mesh import POLICY_HINTS, resolve_mesh
from research_gym.integrity import canonical_file_sha256
from research_gym.scripts import bench_rlm_snapshot_mesh_v0_1 as runner
from research_gym.scripts import materialize_rlm_snapshot_mesh_v0_1 as materializer


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = ROOT / "data/benchmarks/rlm_snapshot_mesh_materialization_v0_1_receipt.json"


def _canonical_sha256(value: object) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return sha256(payload).hexdigest()


def test_materialized_corpus_replays_bound_hashes_and_counts() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    tasks_path = ROOT / receipt["tasks_path"]
    proposals_path = ROOT / receipt["proposals_path"]
    checkpoint_path = ROOT / receipt["checkpoint_path"]

    assert receipt["generator_seed"] == materializer.GENERATOR_SEED
    assert receipt["task_suffix"] == materializer.TASK_SUFFIX
    assert receipt["provider_outcomes_present"] is False
    assert canonical_file_sha256(tasks_path) == receipt["tasks_sha256"]
    assert canonical_file_sha256(proposals_path) == receipt["proposals_sha256"]
    assert canonical_file_sha256(checkpoint_path) == receipt["checkpoint_sha256"]
    assert canonical_file_sha256(ROOT / receipt["source_path"]) == receipt["source_sha256"]

    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    proposals = [json.loads(line) for line in proposals_path.read_text(encoding="utf-8").splitlines()]
    assert len(tasks["tasks"]) == receipt["task_count"] == 152
    assert len(proposals) == receipt["proposal_count"] == 152


def test_materialized_corpus_has_isolated_ids_and_consistent_splits() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    payload = json.loads((ROOT / receipt["tasks_path"]).read_text(encoding="utf-8"))
    proposals = [
        json.loads(line)
        for line in (ROOT / receipt["proposals_path"]).read_text(encoding="utf-8").splitlines()
    ]
    task_ids = [row["task_id"] for row in payload["tasks"]]
    group_ids = [row["group_id"] for row in payload["tasks"]]

    assert len(task_ids) == len(set(task_ids))
    assert all(task_id.endswith(materializer.TASK_SUFFIX) for task_id in task_ids)
    assert all(group_id.endswith(materializer.TASK_SUFFIX) for group_id in group_ids)
    assert {row["task_id"] for row in proposals} == set(task_ids)
    assert len({row["record_id"] for row in proposals}) == len(proposals)
    for split in ("train", "calibration", "eval"):
        expected = sorted(row["group_id"] for row in payload["tasks"] if row["split"] == split)
        assert payload["split_group_sha256"][split] == _canonical_sha256(expected)


def test_materializer_refuses_to_overwrite_sealed_outputs() -> None:
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        materializer.materialize(
            materializer.DEFAULT_TASKS,
            materializer.DEFAULT_PROPOSALS,
            materializer.DEFAULT_RECEIPT,
        )


def _panel():
    config = json.loads((ROOT / "configs/rlm_snapshot_mesh_v0_1.json").read_text(encoding="utf-8"))
    return config, runner._load_tasks(config), runner._load_proposals(config)


def test_snapshot_architecture_hashes_are_unique_and_stable() -> None:
    assert tuple(architecture_hashes()) == ARCHITECTURES
    assert len(set(architecture_hashes().values())) == len(ARCHITECTURES)
    assert len(SNAPSHOT_CONTRACT_HASH) == 64


def test_fresh_panel_has_two_policy_divergent_tasks_per_family() -> None:
    _, tasks, proposals = _panel()
    assert len(tasks) == 8
    divergent_count = 0
    for family in runner.FAMILIES:
        family_tasks = [task for task in tasks if task.family == family]
        assert len(family_tasks) == 2
        for task in family_tasks:
            actions = {
                resolve_mesh(task, proposals[task.task_id], hint).action
                for hint in POLICY_HINTS
            }
            divergent_count += int(len(actions) > 1)
            if family == "multi_hop_reachability":
                assert len(actions) == 1
            else:
                assert len(actions) > 1
    assert divergent_count == 6


def test_opaque_snapshot_is_replayable_and_rejects_tampering() -> None:
    _, tasks, proposals = _panel()
    for task in tasks:
        snapshot = build_opaque_snapshot(task, proposals[task.task_id])
        assert verify_opaque_snapshot(task, proposals[task.task_id], snapshot)
        tampered = json.loads(json.dumps(snapshot))
        tampered["agreement"]["distinct_top_count"] += 1
        assert not verify_opaque_snapshot(task, proposals[task.task_id], tampered)


@pytest.mark.parametrize("atomic", [False, True])
def test_visible_snapshot_prompt_contains_no_task_or_action_tokens(atomic: bool) -> None:
    _, tasks, proposals = _panel()
    for task in tasks:
        snapshot = build_opaque_snapshot(task, proposals[task.task_id])
        prompt = snapshot_prompt(snapshot, atomic=atomic)
        assert task.model_prompt() not in prompt
        assert task.transcript not in prompt
        for action in task.candidates:
            assert not re.search(rf"(?<![A-Za-z0-9_]){re.escape(action)}(?![A-Za-z0-9_])", prompt)
        serialized_values = json.dumps(snapshot, sort_keys=True)
        assert task.optimal_action not in _string_leaves(json.loads(serialized_values))


def test_registration_replays_all_bound_artifacts() -> None:
    path = ROOT / "configs/rlm_snapshot_mesh_v0_1_registration.json"
    registration = json.loads(path.read_text(encoding="utf-8"))
    assert registration["provider_outcomes_present"] is False
    assert registration["snapshot_contract_hash"] == SNAPSHOT_CONTRACT_HASH
    assert canonical_file_sha256(ROOT / registration["config_path"]) == registration["config_sha256"]
    assert registration["architecture_hashes"] == architecture_hashes()
    _, tasks, _ = _panel()
    task_ids = [task.task_id for task in tasks]
    assert task_ids == registration["task_ids"]
    for artifact in registration["bound_artifacts"]:
        assert canonical_file_sha256(ROOT / artifact["path"]) == artifact["sha256"]


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
