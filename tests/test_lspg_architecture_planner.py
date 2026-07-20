from __future__ import annotations

import json
from pathlib import Path

import pytest

from lsa.canonical import digest
from research_gym.architecture_discovery.planner import (
    generate_architecture_proposals,
    read_proposals,
    write_proposals,
)
from research_gym.architecture_discovery.promotion import (
    PromotionDecision,
    extension_required,
    select_distinct_winners,
)
from research_gym.integrity import verify_file_sha256


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _proposals():
    return generate_architecture_proposals(
        code_commit="test",
        task_manifest=_load("experiments/loop_schedule_architecture_discovery_v1/datasets/manifest.json"),
        scale_ladder=_load("configs/lsa/scale_ladder_v0.json"),
        resource_profile={
            "sequence_length": 64,
            "effective_batch_size": 32,
            "microbatch_by_scale": {"S0": 8, "S1": 4, "S2": 2},
            "vram_bytes": 2_621_440_000,
        },
    )


def test_planner_materializes_discovery_reserve_controls_and_nulls() -> None:
    proposals = _proposals()

    assert len([value for value in proposals if value.batch == "batch_1"]) == 12
    assert len([value for value in proposals if value.batch == "batch_2_reserve"]) == 12
    assert len([value for value in proposals if value.batch == "control"]) == 6
    assert len([value for value in proposals if value.batch == "null"]) == 6
    assert len([value for value in proposals if value.batch == "context"]) == 2


def test_candidate_and_matched_control_have_equal_parameters_and_compute() -> None:
    proposals = {value.proposal_id: value for value in _proposals()}
    for candidate in (value for value in proposals.values() if value.role == "candidate"):
        control = proposals[candidate.matched_control_id]
        for scale in ("S0", "S1", "S2"):
            assert candidate.models[scale]["unique_parameters"] == control.models[scale]["unique_parameters"]
            candidate_resource = next(value for value in candidate.resources if value.scale_rung == scale)
            control_resource = next(value for value in control.resources if value.scale_rung == scale)
            assert candidate_resource.estimated_flops_per_example == control_resource.estimated_flops_per_example


def test_proposals_are_deterministic_and_above_30m_is_not_executable() -> None:
    first = _proposals()
    second = _proposals()

    assert [value.proposal_hash for value in first] == [value.proposal_hash for value in second]
    assert all(
        not resource.executable
        for proposal in first
        for resource in proposal.resources
        if resource.scale_rung in {"S3", "S4", "S5"}
    )


def test_frozen_search_space_drives_the_same_proposal_corpus() -> None:
    search_space = _load("configs/lsa/architecture_search_space_v1.json")
    task_manifest = _load(
        "experiments/loop_schedule_architecture_discovery_v1/datasets/manifest.json"
    )
    scale_ladder = _load("configs/lsa/scale_ladder_v0.json")
    resource_profile = {
        "sequence_length": 64,
        "effective_batch_size": 32,
        "microbatch_by_scale": {"S0": 8, "S1": 4, "S2": 2},
        "vram_bytes": 2_621_440_000,
    }

    frozen = generate_architecture_proposals(
        code_commit="test",
        task_manifest=task_manifest,
        scale_ladder=scale_ladder,
        resource_profile=resource_profile,
        search_space=search_space,
    )

    assert [value.proposal_hash for value in frozen] == [
        value.proposal_hash for value in _proposals()
    ]


def test_proposal_bundle_round_trips_and_rejects_tampering(tmp_path: Path) -> None:
    proposals = _proposals()
    input_path = ROOT / "configs/lsa/scale_ladder_v0.json"
    manifest = write_proposals(
        proposals,
        tmp_path,
        input_files={"scale_ladder": input_path},
        input_root=ROOT,
    )

    assert manifest["proposal_count"] == 38
    assert manifest["inputs"]["scale_ladder"]["path"] == "configs/lsa/scale_ladder_v0.json"
    assert [value.proposal_hash for value in read_proposals(tmp_path)] == [
        value.proposal_hash for value in proposals
    ]

    prediction_path = tmp_path / "theory_predictions.json"
    prediction_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="derived proposal artifact hash mismatch"):
        read_proposals(tmp_path)


def test_architecture_registration_hashes_all_frozen_inputs() -> None:
    registration = _load("configs/lsa/architecture_discovery_v1_registration.json")
    claimed_hash = registration.pop("frozen_config_sha256")

    assert registration["status"] == "frozen_before_proposals_and_outcomes"
    assert digest(registration) == claimed_hash
    for receipt in registration["inputs"].values():
        path = ROOT / receipt["path"]
        assert verify_file_sha256(path, receipt["sha256"])


def test_extension_runs_at_most_once() -> None:
    assert extension_required(passing_candidates=1, reserve_already_run=False)
    assert not extension_required(passing_candidates=1, reserve_already_run=True)
    assert not extension_required(passing_candidates=2, reserve_already_run=False)


def test_winner_selection_can_return_fewer_than_two() -> None:
    proposals = {value.proposal_id: value for value in _proposals()}
    candidate_ids = [value.proposal_id for value in proposals.values() if value.role == "candidate"]
    decisions = [
        PromotionDecision(candidate_ids[0], True, 0.1, 3, 0.0, 1.0, ()),
        PromotionDecision(candidate_ids[1], False, 0.2, 1, -0.1, 1.0, ("failed",)),
    ]

    assert select_distinct_winners(decisions, proposals) == (candidate_ids[0],)
