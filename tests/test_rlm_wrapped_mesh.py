from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from statistics import mean

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


def test_sealed_pilot_preserves_containment_and_routing_null() -> None:
    receipt_path = ROOT / "data/benchmarks/rlm_wrapped_controller_mesh_v0_receipt.json"
    if not receipt_path.exists():
        return
    receipt = json.loads(receipt_path.read_text())
    assert canonical_file_sha256(ROOT / receipt["result_path"]) == receipt["result_sha256"]
    assert canonical_file_sha256(ROOT / receipt["records_path"]) == receipt["records_sha256"]
    records = [
        json.loads(line)
        for line in (ROOT / receipt["records_path"]).read_text().splitlines()
    ]
    assert sum(bool(row["unsafe"]) for row in records) == 0
    assert sum(row.get("decision_reason") == "cell_error" for row in records) == 3
    fixed = {
        row["task_id"]: row
        for row in records
        if row["architecture_id"] == "mesh_fixed_consensus"
    }
    for architecture in ("rlm_mesh_atomic_tool", "rlm_mesh_text_router"):
        completed = [
            row
            for row in records
            if row["architecture_id"] == architecture
            and row["decision_reason"] != "cell_error"
        ]
        assert all(row["executed_action"] == fixed[row["task_id"]]["executed_action"] for row in completed)
    atomic = [row for row in records if row["architecture_id"] == "rlm_mesh_atomic_tool"]
    text = [row for row in records if row["architecture_id"] == "rlm_mesh_text_router"]
    assert sum(bool(row["wrapper_contract_passed"]) for row in atomic) == 2
    assert sum(bool(row["wrapper_contract_passed"]) for row in text) == 0


def test_sealed_panel_contains_posthoc_routing_opportunity() -> None:
    receipt_path = ROOT / "data/benchmarks/rlm_wrapped_controller_mesh_v0_receipt.json"
    if not receipt_path.exists():
        return
    tasks, proposals = _inputs()
    by_family: dict[str, list[tuple[float, float]]] = {family: [] for family in runner.FAMILIES}
    opportunity_count = 0
    for task in tasks:
        utilities = {
            hint: task.utilities[resolve_mesh(task, proposals[task.task_id], hint).action]
            for hint in POLICY_HINTS
        }
        opportunity_count += int(len(set(utilities.values())) > 1)
        by_family[task.family].append((utilities["consensus"], max(utilities.values())))
    consensus = mean(mean(value[0] for value in by_family[family]) for family in runner.FAMILIES)
    oracle = mean(mean(value[1] for value in by_family[family]) for family in runner.FAMILIES)
    assert opportunity_count == 6
    assert oracle - consensus == pytest.approx(0.03625)
