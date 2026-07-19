from __future__ import annotations

from research_gym.prognostics.promotion import promotion_decision


def test_edge_censoring_blocks_scale_promotion() -> None:
    decision = promotion_decision(
        integrity_passed=True,
        stable_sign_or_ordering=True,
        posterior_contracted=True,
        edge_censored=True,
        redundant=False,
    )

    assert decision.action == "stop_or_extend_grid"
    assert "unresolved_grid_edge_censoring" in decision.reasons


def test_all_registered_gates_are_required_for_promotion() -> None:
    decision = promotion_decision(
        integrity_passed=True,
        stable_sign_or_ordering=True,
        posterior_contracted=True,
        edge_censored=False,
        redundant=False,
    )

    assert decision.action == "promote"
    assert not decision.reasons
