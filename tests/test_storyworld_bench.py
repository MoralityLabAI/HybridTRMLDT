from research_gym.benchmarks.storyworld_bench import (
    hybrid_action,
    ldt_certified_action,
    play_storyworld,
    run_storyworld_benchmark,
    sample_start_states,
    trm_heuristic_action,
)
from research_gym.envs.coupled_storyworld import CoupledStoryworldEnv, StoryState


def test_trm_heuristic_prioritizes_defusing_high_heat():
    env = CoupledStoryworldEnv()

    assert trm_heuristic_action(env, StoryState(trust=2, evidence=3, heat=4, scene=2)) == "defuse"


def test_ldt_certified_action_preserves_reachability():
    env = CoupledStoryworldEnv()
    state = StoryState(trust=2, evidence=3, heat=0, scene=3)

    action = ldt_certified_action(env, state, horizon=3)

    assert action in env.self_actions
    next_state = env.step(state, action, env.other_policy(state))
    assert env.target(next_state) or env.reachable(next_state, 2, "model")[0]


def test_hybrid_action_returns_action_and_override_flag():
    env = CoupledStoryworldEnv()
    state = StoryState(trust=0, evidence=2, heat=2, scene=2)

    action, overridden = hybrid_action(env, state, horizon=5)

    assert action in env.self_actions
    assert isinstance(overridden, bool)


def test_play_storyworld_result_shape():
    result = play_storyworld("hybrid", StoryState(trust=2, evidence=3, heat=0, scene=3), horizon=4)

    assert result.player == "hybrid"
    assert result.steps <= 4
    assert set(result.final_state) == {"trust", "evidence", "heat", "scene"}


def test_sample_start_states_are_modeled_reachable():
    env = CoupledStoryworldEnv()
    starts = sample_start_states(env, n=8, horizon=6, seed=1)

    assert len(starts) == 8
    assert all(env.reachable(state, 6, "model")[0] for state in starts)


def test_storyworld_benchmark_shape():
    payload = run_storyworld_benchmark(n=8, horizon=6, seed=2)

    assert payload["n"] == 8
    assert len(payload["runs"]) == 24
    assert set(payload["summary"]) == {"ldt", "trm", "hybrid"}
