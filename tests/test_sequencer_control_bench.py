from dataclasses import replace

from research_gym.benchmarks.sequencer_control_bench import (
    CONTROL_MATH,
    GLOBAL_SIGNED,
    LINEAGE_ONLY,
    PROPOSAL_ONLY,
    TYPED_PROPOSE_CERTIFY,
    BenchmarkEpisode,
    CandidateOutcome,
    SequencerBenchmarkConfig,
    _sign_flip_pvalue,
    evaluate_sequencers,
    fit_context_plans,
    run_sequencer_control_benchmark,
    v1_replay_rows,
)
from research_gym.envs.routing import RoutingExample


def _outcome(utility: float) -> CandidateOutcome:
    return CandidateOutcome(True, utility, 1.0, False, str(utility))


def _episode(context: str, index: int, scores: tuple[float, float, float], split: str = "calibration"):
    return BenchmarkEpisode(
        episode_id=f"{context}-{split}-{index}",
        family=context.split(":")[0],
        context=context,
        split=split,
        prompt="task",
        expected_answer="answer",
        outcomes={
            PROPOSAL_ONLY: _outcome(scores[0]),
            "deduction_only": _outcome(scores[1]),
            TYPED_PROPOSE_CERTIFY: _outcome(scores[2]),
        },
    )


def _routing_examples() -> list[RoutingExample]:
    rows = []
    for env_id, token in (("alpha", "sort letters"), ("math", "solve number")):
        for index in range(30):
            rows.append(RoutingExample(prompt=f"{token} item {index} shared", env_id=env_id))
    return rows


def test_orientation_reversal_is_invisible_to_lineage_only_gate():
    calibration = []
    for context in ("domain:clean-a", "domain:clean-b", "domain:clean-c"):
        calibration.extend(_episode(context, index, (0.0, 0.0, 1.0)) for index in range(12))
    calibration.extend(_episode("domain:reversed-x", index, (1.0, 0.9, 0.0)) for index in range(12))
    plans, fit = fit_context_plans(calibration, SequencerBenchmarkConfig())
    reversed_plan = plans["domain:reversed-x"]

    assert fit["global_sequence"] == TYPED_PROPOSE_CERTIFY
    assert reversed_plan.orientation_reversal
    assert reversed_plan.lineage_only_authorized
    assert not reversed_plan.full_control_authorized
    assert reversed_plan.selection(LINEAGE_ONLY) == TYPED_PROPOSE_CERTIFY
    assert reversed_plan.selection(CONTROL_MATH) == PROPOSAL_ONLY


def test_plans_are_fit_only_from_calibration_rows():
    calibration = [_episode("domain:a", index, (0.0, 0.2, 1.0)) for index in range(12)]
    plans, _ = fit_context_plans(calibration, SequencerBenchmarkConfig())
    evaluation = [replace(_episode("domain:a", 0, (1.0, 0.0, 0.0), "eval"), episode_id="held-out")]

    rows = evaluate_sequencers(evaluation, plans)

    assert rows[0]["selections"][GLOBAL_SIGNED] == TYPED_PROPOSE_CERTIFY
    assert rows[0]["selected_outcomes"][GLOBAL_SIGNED]["utility"] == 0.0


def test_global_fit_macro_weights_families_before_contexts():
    calibration = []
    for context in ("heavy:a", "heavy:b", "heavy:c"):
        calibration.extend(_episode(context, index, (1.0, 0.0, 0.0)) for index in range(12))
    calibration.extend(_episode("light:a", index, (0.0, 0.0, 1.0)) for index in range(12))

    _, fit = fit_context_plans(calibration, SequencerBenchmarkConfig())

    assert fit["global_scores"][PROPOSAL_ONLY] == 0.5
    assert fit["global_scores"][TYPED_PROPOSE_CERTIFY] == 0.5
    assert fit["global_sequence"] == TYPED_PROPOSE_CERTIFY


def test_sign_flip_uses_context_clusters_not_episode_count():
    def row(family: str, context: str, delta: float) -> dict[str, object]:
        return {
            "family": family,
            "context": context,
            "selected_outcomes": {
                CONTROL_MATH: {"utility": delta},
                GLOBAL_SIGNED: {"utility": 0.0},
            },
        }

    rows = [
        row("a", "a:one", 1.0),
        row("a", "a:two", -0.25),
        row("b", "b:one", 0.5),
        row("b", "b:two", 0.2),
    ]
    duplicated = [item for item in rows for _ in range(20)]

    original = _sign_flip_pvalue(
        rows, CONTROL_MATH, GLOBAL_SIGNED, "utility", iterations=1000, seed=17
    )
    repeated = _sign_flip_pvalue(
        duplicated, CONTROL_MATH, GLOBAL_SIGNED, "utility", iterations=1000, seed=17
    )

    assert repeated == original


def test_small_matched_benchmark_emits_statistics_and_v1_rows():
    config = SequencerBenchmarkConfig(
        sudoku_calibration_per_base=2,
        sudoku_eval_per_base=2,
        arc1_calibration_per_rule=2,
        arc1_eval_per_rule=2,
        arc2_calibration_per_rule=2,
        arc2_eval_per_rule=2,
        story_calibration_per_scenario=4,
        story_eval_per_scenario=4,
        bootstrap_iterations=100,
        permutation_iterations=100,
    )
    payload, rows = run_sequencer_control_benchmark(_routing_examples(), config)
    replay = v1_replay_rows(rows)

    assert payload["fit_receipt"]["global_sequence"] == TYPED_PROPOSE_CERTIFY
    assert payload["eval_episode_count"] == len(rows) == len(replay)
    assert set(payload["families"]) == {"arc1", "arc2", "routing", "story_moral", "story_secret", "sudoku"}
    assert {item["control"] for item in payload["comparisons"]} == {
        "global_signed",
        "lineage_only",
        "fixed_typed",
    }
    assert [row["signed_error_budget"] for row in payload["error_budget_sensitivity"]] == [
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
    ]
    assert all("selections" in row and "candidate_outcomes" in row for row in replay)


def test_evaluation_ids_and_hash_are_deterministic():
    config = SequencerBenchmarkConfig(
        sudoku_calibration_per_base=1,
        sudoku_eval_per_base=1,
        arc1_calibration_per_rule=1,
        arc1_eval_per_rule=1,
        arc2_calibration_per_rule=1,
        arc2_eval_per_rule=1,
        story_calibration_per_scenario=2,
        story_eval_per_scenario=2,
        bootstrap_iterations=20,
        permutation_iterations=20,
    )
    first, first_rows = run_sequencer_control_benchmark(_routing_examples(), config)
    second, second_rows = run_sequencer_control_benchmark(_routing_examples(), config)

    assert first["eval_sha256"] == second["eval_sha256"]
    assert first_rows == second_rows
