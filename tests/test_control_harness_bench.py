from research_gym.benchmarks.control_harness_bench import (
    evaluate_decision,
    run_control_harness_benchmark,
)
from research_gym.envs.control_tasks import generate_control_tasks


def test_typed_membrane_blocks_environment_unsound_top_proposal():
    tasks = generate_control_tasks(n_train=1, n_eval=3, seed=23)
    task = next(
        task
        for task in tasks
        if task.application == "oracle_control"
        and task.split == "eval"
        and max(task.trm_scores, key=lambda action: task.trm_scores[action]) not in task.environment_allowed
    )

    trm = evaluate_decision(task, "trm", gamma=0.2, beam_width=2)
    typed = evaluate_decision(task, "typed_membrane", gamma=0.2, beam_width=2)

    assert not trm.safe
    assert typed.safe


def test_benchmark_uses_held_out_skill_routes_and_shows_hard_gate_risk():
    payload = run_control_harness_benchmark(n_train=24, n_eval=24, seed=23)
    summary = payload["summary"]

    assert len(payload["skill_routes"]) == 9
    assert summary["typed_membrane"]["unsafe_rate"] == 0.0
    assert summary["typed_confidence"]["unsafe_rate"] == 0.0
    assert summary["hard_gate"]["unsafe_rate"] > 0.0
    assert summary["skill_router"]["selection_objective"] >= summary["trm"]["selection_objective"]
    assert payload["tuned"]["beam_width"] in {1, 2, 3, 4}
