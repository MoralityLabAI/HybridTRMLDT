from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from research_gym.benchmarks.rlm_hybrid_neighborhood import LongContextControlTask
from research_gym.benchmarks.rlm_wrapped_mesh import (
    ARCHITECTURES,
    MESH_TOPOLOGY_HASH,
    POLICY_HINTS,
    _parse_policy_hint,
    architecture_hashes,
    resolve_mesh,
    run_local_mesh_architecture,
    verify_mesh_resolution,
)
from research_gym.integrity import canonical_file_sha256
from research_gym.scripts import bench_rlm_wrapped_mesh as runner


ROOT = Path(__file__).resolve().parents[1]


def _inputs() -> tuple[list[LongContextControlTask], dict[str, dict[str, object]]]:
    config = json.loads((ROOT / "configs/rlm_wrapped_controller_mesh_v0.json").read_text())
    tasks = runner._load_tasks(config)
    proposals = runner._load_proposals(config)
    return tasks, proposals


def test_architecture_and_topology_hashes_are_unique_and_stable() -> None:
    assert tuple(architecture_hashes()) == ARCHITECTURES
    assert len(set(architecture_hashes().values())) == len(ARCHITECTURES)
    assert len(MESH_TOPOLOGY_HASH) == 64


def test_hash_selected_diagnostic_panel_has_two_tasks_per_family() -> None:
    tasks, _ = _inputs()
    assert len(tasks) == 8
    assert {family: sum(task.family == family for task in tasks) for family in runner.FAMILIES} == {
        family: 2 for family in runner.FAMILIES
    }


@pytest.mark.parametrize("policy_hint", POLICY_HINTS)
def test_mesh_resolution_is_exact_safe_and_certificate_replayable(policy_hint: str) -> None:
    tasks, proposals = _inputs()
    for task in tasks:
        resolution = resolve_mesh(task, proposals[task.task_id], policy_hint)
        assert resolution.action in task.exact_allowed
        assert verify_mesh_resolution(task, proposals[task.task_id], resolution)


def test_mesh_certificate_rejects_tampering() -> None:
    tasks, proposals = _inputs()
    task = tasks[0]
    resolution = resolve_mesh(task, proposals[task.task_id], "consensus")
    tampered = replace(resolution, certificate="0" * 64)
    assert not verify_mesh_resolution(task, proposals[task.task_id], tampered)


def test_forced_no_tool_fallback_matches_fixed_mesh() -> None:
    tasks, proposals = _inputs()
    for task in tasks:
        fixed, _ = run_local_mesh_architecture(
            "mesh_fixed_consensus", task, proposals[task.task_id], 211
        )
        forced, _ = run_local_mesh_architecture(
            "mesh_forced_no_tool_fallback", task, proposals[task.task_id], 211
        )
        assert fixed["executed_action"] == forced["executed_action"]
        assert fixed["utility"] == forced["utility"]
        assert forced["wrapper_fallback"]


def test_policy_hint_parser_cannot_parse_actions_as_authority() -> None:
    assert _parse_policy_hint("trained_first") == "trained_first"
    assert _parse_policy_hint("approve") is None
    assert _parse_policy_hint("consensus then trained_first") == "consensus"


def test_registration_hashes_when_registration_exists() -> None:
    path = ROOT / "configs/rlm_wrapped_controller_mesh_v0_registration.json"
    if not path.exists():
        return
    registration = json.loads(path.read_text())
    assert canonical_file_sha256(ROOT / registration["config_path"]) == registration["config_sha256"]
    assert registration["architecture_hashes"] == architecture_hashes()
    assert registration["mesh_topology_hash"] == MESH_TOPOLOGY_HASH
    for artifact in registration["bound_artifacts"]:
        assert canonical_file_sha256(ROOT / artifact["path"]) == artifact["sha256"]
