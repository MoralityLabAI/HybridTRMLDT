from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_gym.architecture_discovery.campaign import (
    build_stage_cells,
    read_stage_manifest,
    write_stage_manifest,
)
from research_gym.architecture_discovery.planner import read_proposals
from research_gym.scripts.run_lspg_architecture_stage import (
    _launched_attempt_count,
    _next_attempt,
)


ROOT = Path(__file__).resolve().parents[1]


def _proposals():
    return {
        value.proposal_id: value
        for value in read_proposals(
            ROOT / "experiments/loop_schedule_architecture_discovery_v1/proposals"
        )
    }


def _policy():
    return json.loads(
        (ROOT / "configs/lsa/architecture_promotion_policy_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_a1_contains_twelve_candidates_and_six_deduplicated_controls() -> None:
    proposals = _proposals()
    candidates = sorted(
        value.proposal_id for value in proposals.values() if value.batch == "batch_1"
    )
    cells = build_stage_cells(
        stage_id="A1",
        candidate_ids=candidates,
        proposals=proposals,
        policy=_policy(),
        selected_profile="minimum",
    )

    assert len(cells) == 18
    assert len([value for value in cells if value.role == "candidate"]) == 12
    assert len({value.cell_id for value in cells}) == 18


def test_d_adds_matched_controls_nulls_and_two_context_models() -> None:
    proposals = _proposals()
    candidates = ["LSAD-B1-K2L6-01", "LSAD-B1-K3L8-01"]
    cells = build_stage_cells(
        stage_id="D",
        candidate_ids=candidates,
        proposals=proposals,
        policy=_policy(),
        selected_profile="minimum",
    )

    assert len(cells) == 24
    assert all(value.allow_locked_evaluation for value in cells)
    assert {value.role for value in cells} == {
        "candidate",
        "matched_periodic_control",
        "balanced_schedule_null",
        "fully_tied_context",
        "fully_untied_context",
    }


def test_stage_manifest_round_trip_rejects_tampering(tmp_path: Path) -> None:
    proposals = _proposals()
    cells = build_stage_cells(
        stage_id="A2",
        candidate_ids=["LSAD-B1-K2L6-01"],
        proposals=proposals,
        policy=_policy(),
        selected_profile="minimum",
    )
    path = tmp_path / "stage.json"
    manifest = write_stage_manifest(
        cells,
        path,
        stage_id="A2",
        selected_profile="minimum",
        profile_selection_hash="selection",
        proposal_manifest_sha256="proposal",
        code_commit="commit",
    )

    assert read_stage_manifest(path)["manifest_hash"] == manifest["manifest_hash"]
    value = json.loads(path.read_text(encoding="utf-8"))
    value["cells"][0]["token_visit_budget"] += 1
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        read_stage_manifest(path)


def test_stage_retry_preserves_existing_attempt_receipts(tmp_path: Path) -> None:
    cell_id = "cell-A1"
    (tmp_path / f"{cell_id}.attempt-1.resource_receipt.json").write_text("{}")
    (tmp_path / f"{cell_id}.attempt-2.resource_receipt.json").write_text("{}")
    (tmp_path / f"{cell_id}.attempt-invalid.resource_receipt.json").write_text("{}")

    assert _next_attempt(tmp_path, cell_id) == 3
    assert _next_attempt(tmp_path, "unseen-cell") == 1


def test_foreign_gpu_preflight_does_not_consume_training_attempt(tmp_path: Path) -> None:
    cell_id = "cell-A1"
    (tmp_path / f"{cell_id}.attempt-1.resource_receipt.json").write_text(
        json.dumps({"status": "construction_failure", "owned_pid": None})
    )
    (tmp_path / f"{cell_id}.attempt-2.resource_receipt.json").write_text(
        json.dumps({"status": "completed", "owned_pid": 42})
    )

    assert _launched_attempt_count(tmp_path, cell_id) == 1
