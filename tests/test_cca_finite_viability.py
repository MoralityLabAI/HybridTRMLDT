"""Regression against registered CCA-05 action totals and storyworld geometry."""
from experiments.cca11_recoverability_threshold.finite_viability import (
    ACTIONS, ALL, State, authorized, manhattan, one_step_resilience,
    radius_from_winning_set, target, win,
)


def test_geometry_and_goal():
    assert len(ALL) == 1050
    assert manhattan(State(0, 0, 0, 0), State(1, 1, 1, 1)) == 4
    assert target(State(1, 3, 2, 5))
    assert not target(State(0, 3, 2, 5))


def test_horizon_six_matches_cca05_registered_action_totals():
    for robust, viable, certified, r0, r1, r2 in (
        (False, 226, 618, 134, 69, 23),
        (True, 40, 44, 34, 6, 0),
    ):
        winners = {s for s in ALL if win(s, 6, robust)}
        losers = tuple(s for s in ALL if s not in winners)
        assert len(winners) == viable
        assert sum(authorized(s, a, 6, robust)
                   for s in ALL for a in ACTIONS) == certified
        radii = [radius_from_winning_set(s, losers) for s in winners]
        assert radii.count(0) == r0
        assert radii.count(1) == r1
        assert radii.count(2) == r2


def test_depth_ranking_improves_next_step_resilience():
    assert one_step_resilience(6, False) == {
        "decision_states": 208,
        "priority_action_one_step_resilient": 85,
        "depth_selected_one_step_resilient": 99,
        "states_with_depth_improvement": 20,
    }
    assert one_step_resilience(6, True) == {
        "decision_states": 22,
        "priority_action_one_step_resilient": 13,
        "depth_selected_one_step_resilient": 15,
        "states_with_depth_improvement": 2,
    }
