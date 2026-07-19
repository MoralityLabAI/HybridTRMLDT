from __future__ import annotations

import json
from pathlib import Path

from research_gym.prognostics.planner import LoopSchedulePlanner
from research_gym.prognostics.schemas import ResourceProfile, TRAINABILITY_ONLY


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def _planner(*, vram_bytes: int = 1_572_864_000) -> LoopSchedulePlanner:
    return LoopSchedulePlanner(
        base_receipt=ROOT / "data/benchmarks/lsa_v0_receipt.json",
        scale_ladder=_load("configs/lsa/scale_ladder_v0.json"),
        mutation_space=_load("configs/lsa/mutation_space_v0.json"),
        resource_profile=ResourceProfile(
            ram_bytes=2_147_483_648,
            vram_bytes=vram_bytes,
            cpu_pct=50,
            io_bytes_per_second=52_428_800,
            timeout_seconds=1800,
            max_stage_a_proposals=12,
        ),
        promotion_policy=_load("configs/lsa/promotion_policy_v0.json"),
        code_commit="test-commit",
    )


def test_planner_emits_twelve_deterministic_diverse_s0_proposals() -> None:
    first = _planner().propose(12)
    second = _planner().propose(12)

    assert len(first) == len(second) == 12
    assert [row.proposal_hash for row in first] == [row.proposal_hash for row in second]
    assert len({row.algebra_hash for row in first}) == 12
    assert all(row.scale_rung == "S0" for row in first)


def test_edge_censoring_prioritizes_grid_extension_not_scale_promotion() -> None:
    proposals = _planner().propose(12)

    assert proposals[0].decision.decision_kind == "grid_extension"
    assert proposals[1].decision.decision_kind == "grid_extension"
    assert all(row.scale_rung != "S5" for row in proposals)


def test_all_rungs_are_materialized_but_only_s0_is_locally_executable() -> None:
    proposal = _planner().propose(1)[0]
    by_rung = {row.scale_rung: row for row in proposal.resources}

    assert set(by_rung) == {"S0", "S1", "S2", "S3", "S4", "S5"}
    assert by_rung["S0"].locally_executable
    assert not by_rung["S5"].locally_executable
    assert "stage_a_executes_s0_only" in by_rung["S5"].exclusion_reasons


def test_resource_limit_eliminates_otherwise_high_scoring_proposals() -> None:
    assert _planner(vram_bytes=1).propose(12) == ()


def test_trainability_scope_emits_no_task_performance_claim() -> None:
    proposal = _planner().propose(1)[0]
    serialized = json.dumps(proposal.to_dict(), sort_keys=True).lower()

    assert proposal.claim_scope == TRAINABILITY_ONLY
    for forbidden in ("task accuracy", "reasoning quality", "sample efficiency"):
        assert forbidden not in serialized
