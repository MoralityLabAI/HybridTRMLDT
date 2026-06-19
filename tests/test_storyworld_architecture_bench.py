from research_gym.benchmarks.storyworld_architecture_bench import (
    confidence_action,
    moral_score,
    play_architecture_policy,
    run_storyworld_architecture_benchmark,
)
from research_gym.envs.coupled_storyworld import CoupledStoryworldEnv, StoryState


def test_moral_score_rewards_trust_and_penalizes_heat():
    calm = StoryState(trust=2, evidence=2, heat=0, scene=3)
    hot = StoryState(trust=2, evidence=2, heat=3, scene=3)

    assert moral_score(calm) > moral_score(hot)


def test_confidence_action_reports_margin():
    env = CoupledStoryworldEnv()
    action, overridden, margin = confidence_action(env, StoryState(3, 1, 4, 1), 6, gamma=2.0)

    assert action in env.self_actions
    assert isinstance(overridden, bool)
    assert margin >= 0.0


def test_secret_ending_high_confidence_trap_split():
    start = StoryState(trust=3, evidence=1, heat=4, scene=1)
    confidence = play_architecture_policy("secret_ending", "confidence_arbitration", start, gamma=2.0)
    typed = play_architecture_policy("secret_ending", "typed_membrane", start, gamma=2.0)

    assert not confidence.success
    assert typed.success


def test_storyworld_architecture_benchmark_shows_tradeoff():
    payload = run_storyworld_architecture_benchmark(n=64, horizon=6, seed=7, gamma=2.0)
    summary = payload["summary"]
    secret = summary["secret_ending"]
    moral = summary["moral_optimization"]

    assert secret["typed_membrane"]["success_rate"] > secret["confidence_arbitration"]["success_rate"]
    assert moral["confidence_arbitration"]["avg_score"] > moral["typed_membrane"]["avg_score"]
