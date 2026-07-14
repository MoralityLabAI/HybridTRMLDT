"""Matched benchmark for topology-aware control of infused skill sequences."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import random
from statistics import mean
from typing import Iterable, Mapping, Sequence

from research_gym.benchmarks.arc_bench import (
    solve_arc1_hybrid,
    solve_arc1_ldt,
    solve_arc1_trm,
    solve_arc2_hybrid,
    solve_arc2_ldt,
    solve_arc2_trm,
)
from research_gym.benchmarks.routing_bench import (
    HybridRoutingModel,
    LexicalTRMRouter,
    TokenLatticeRouter,
)
from research_gym.benchmarks.storyworld_architecture_bench import play_architecture_policy
from research_gym.benchmarks.sudoku_bench import solve_hybrid, solve_ldt, solve_trm
from research_gym.core.training_review import (
    ControlRiskMeasurement,
    EdgeRiskReceipt,
    LoopRiskReceipt,
    control_risk_bound,
)
from research_gym.envs.arc_tasks import ArcExample, ArcRule, ArcTask, Grid as ArcGrid, primitive_rule_pairs, primitive_rules
from research_gym.envs.coupled_storyworld import CoupledStoryworldEnv
from research_gym.envs.routing import RoutingExample
from research_gym.envs.sudoku import Grid as SudokuGrid, SudokuPuzzle, default_sudoku_puzzles, grid_from_rows


PROPOSAL_ONLY = "proposal_only"
DEDUCTION_ONLY = "deduction_only"
TYPED_PROPOSE_CERTIFY = "typed_propose_certify"
SKILL_SEQUENCES = (PROPOSAL_ONLY, DEDUCTION_ONLY, TYPED_PROPOSE_CERTIFY)

GLOBAL_SIGNED = "global_signed"
LINEAGE_ONLY = "lineage_only"
FIXED_TYPED = "fixed_typed"
CONTROL_MATH = "control_math"
LOCAL_CALIBRATED = "local_calibrated"
SEQUENCERS = (GLOBAL_SIGNED, LINEAGE_ONLY, FIXED_TYPED, CONTROL_MATH, LOCAL_CALIBRATED)


@dataclass(frozen=True)
class SequencerBenchmarkConfig:
    seed: int = 20260713
    signed_error_budget: float = 0.5
    delta: float = 0.05
    sudoku_calibration_per_base: int = 16
    sudoku_eval_per_base: int = 32
    arc1_calibration_per_rule: int = 12
    arc1_eval_per_rule: int = 24
    arc2_calibration_per_rule: int = 10
    arc2_eval_per_rule: int = 20
    story_calibration_per_scenario: int = 64
    story_eval_per_scenario: int = 64
    routing_max_per_env: int = 80
    bootstrap_iterations: int = 5000
    permutation_iterations: int = 10000
    inference_unit: str = "family_context_cluster"


@dataclass(frozen=True)
class CandidateOutcome:
    correct: bool
    utility: float
    cost: float
    constraint_violation: bool
    answer: str

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkEpisode:
    episode_id: str
    family: str
    context: str
    split: str
    prompt: str
    expected_answer: str
    outcomes: Mapping[str, CandidateOutcome]

    def to_jsonable(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "family": self.family,
            "context": self.context,
            "split": self.split,
            "prompt": self.prompt,
            "expected_answer": self.expected_answer,
            "outcomes": {name: outcome.to_jsonable() for name, outcome in sorted(self.outcomes.items())},
        }


@dataclass(frozen=True)
class ContextPlan:
    context: str
    global_sequence: str
    local_sequence: str
    full_control_authorized: bool
    lineage_only_authorized: bool
    route: str
    point_risk_score: float
    signed_control_loss_upper_bound: float
    lineage_contraction_bound: float
    holonomy_displacement_bound: float
    orientation_reversal: bool
    matched_noise_null_passed: bool
    simultaneous_coverage: float
    maximum_canonical_angle_degrees: float
    calibration_count: int
    local_scores: Mapping[str, float]

    def selection(self, sequencer: str) -> str:
        if sequencer == GLOBAL_SIGNED:
            return self.global_sequence
        if sequencer == LINEAGE_ONLY:
            return self.global_sequence if self.lineage_only_authorized else self.local_sequence
        if sequencer == FIXED_TYPED:
            return TYPED_PROPOSE_CERTIFY
        if sequencer == CONTROL_MATH:
            return self.global_sequence if self.full_control_authorized else self.local_sequence
        if sequencer == LOCAL_CALIBRATED:
            return self.local_sequence
        raise ValueError(f"unknown sequencer: {sequencer}")

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


def _stable_int(value: str) -> int:
    return int(sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _grid_text(grid: Sequence[Sequence[int]]) -> str:
    return "/".join("".join(str(value) for value in row) for row in grid)


def _candidate_utility(correct: bool, cost: float, cost_scale: float, penalty: float = 0.08) -> float:
    return float(correct) - penalty * min(2.0, cost / max(cost_scale, 1e-9))


def _sudoku_order(rng: random.Random) -> list[int]:
    bands = [0, 1]
    rng.shuffle(bands)
    order: list[int] = []
    for band in bands:
        rows = [2 * band, 2 * band + 1]
        rng.shuffle(rows)
        order.extend(rows)
    return order


def _transform_sudoku(base: SudokuPuzzle, rng: random.Random, episode_id: str) -> SudokuPuzzle:
    digits = [1, 2, 3, 4]
    rng.shuffle(digits)
    mapping = {index + 1: value for index, value in enumerate(digits)}
    rows = _sudoku_order(rng)
    cols = _sudoku_order(rng)
    transpose = rng.random() < 0.5

    def apply(grid: SudokuGrid) -> SudokuGrid:
        values = [[grid[row][col] for col in cols] for row in rows]
        if transpose:
            values = [list(row) for row in zip(*values)]
        return grid_from_rows([[mapping.get(value, 0) if value else 0 for value in row] for row in values])

    return SudokuPuzzle(episode_id, apply(base.givens), apply(base.solution))


def _sudoku_episode(puzzle: SudokuPuzzle, base_id: str, split: str) -> BenchmarkEpisode:
    runs = {
        PROPOSAL_ONLY: solve_trm(puzzle),
        DEDUCTION_ONLY: solve_ldt(puzzle),
        TYPED_PROPOSE_CERTIFY: solve_hybrid(puzzle),
    }
    outcomes = {}
    for sequence, run in runs.items():
        cost = float(run.steps + run.guesses + 0.5 * run.conflicts)
        outcomes[sequence] = CandidateOutcome(
            correct=run.solved,
            utility=_candidate_utility(run.solved, cost, 24.0),
            cost=cost,
            constraint_violation=not run.valid,
            answer=_grid_text(run.final_grid),
        )
    return BenchmarkEpisode(
        episode_id=puzzle.puzzle_id,
        family="sudoku",
        context=f"sudoku:{base_id}",
        split=split,
        prompt=f"Solve 4x4 Sudoku {_grid_text(puzzle.givens)}",
        expected_answer=_grid_text(puzzle.solution),
        outcomes=outcomes,
    )


def generate_sudoku_episodes(config: SequencerBenchmarkConfig) -> list[BenchmarkEpisode]:
    episodes = []
    for base_index, base in enumerate(default_sudoku_puzzles()):
        rng = random.Random(config.seed + 101 * (base_index + 1))
        counts = (
            ("calibration", config.sudoku_calibration_per_base),
            ("eval", config.sudoku_eval_per_base),
        )
        for split, count in counts:
            for index in range(count):
                episode_id = f"sudoku-{base.puzzle_id}-{split}-{index:04d}"
                episodes.append(_sudoku_episode(_transform_sudoku(base, rng, episode_id), base.puzzle_id, split))
    return episodes


def _random_arc_grid(rng: random.Random) -> ArcGrid:
    rows = rng.choice((2, 3))
    cols = rng.choice((2, 3))
    values = [rng.randrange(5) for _ in range(rows * cols)]
    for index, required in enumerate((0, 1, 2)):
        if index < len(values):
            values[index] = required
    rng.shuffle(values)
    return tuple(tuple(values[row * cols + col] for col in range(cols)) for row in range(rows))


def _arc_task(rule: ArcRule, rng: random.Random, task_id: str) -> ArcTask:
    inputs = [_random_arc_grid(rng) for _ in range(3)]
    return ArcTask(
        task_id=task_id,
        train=tuple(ArcExample(value, rule.apply(value)) for value in inputs[:2]),
        test_input=inputs[2],
        test_output=rule.apply(inputs[2]),
        rule_name=rule.name,
        rule_names=tuple(rule.name.split("+")),
    )


def _arc_episode(task: ArcTask, split: str, order: int) -> BenchmarkEpisode:
    if order == 1:
        runs = {
            PROPOSAL_ONLY: solve_arc1_trm(task),
            DEDUCTION_ONLY: solve_arc1_ldt(task),
            TYPED_PROPOSE_CERTIFY: solve_arc1_hybrid(task),
        }
        cost_scale = 7.0
    else:
        runs = {
            PROPOSAL_ONLY: solve_arc2_trm(task),
            DEDUCTION_ONLY: solve_arc2_ldt(task),
            TYPED_PROPOSE_CERTIFY: solve_arc2_hybrid(task),
        }
        cost_scale = 49.0
    outcomes = {}
    for sequence, run in runs.items():
        cost = float(run.steps + 0.25 * run.rejected)
        outcomes[sequence] = CandidateOutcome(
            correct=run.solved,
            utility=_candidate_utility(run.solved, cost, cost_scale),
            cost=cost,
            constraint_violation=False,
            answer=_grid_text(run.predicted),
        )
    family = f"arc{order}"
    return BenchmarkEpisode(
        episode_id=task.task_id,
        family=family,
        context=f"{family}:{task.rule_name}",
        split=split,
        prompt=f"Apply the inferred ARC-{order} transformation to {_grid_text(task.test_input)}",
        expected_answer=_grid_text(task.test_output),
        outcomes=outcomes,
    )


def _balanced_arc_episodes(
    rules: Sequence[ArcRule],
    *,
    order: int,
    calibration_per_rule: int,
    eval_per_rule: int,
    seed: int,
) -> list[BenchmarkEpisode]:
    episodes = []
    for rule_index, rule in enumerate(rules):
        rng = random.Random(seed + 1009 * (rule_index + 1) + order)
        for split, target_count in (("calibration", calibration_per_rule), ("eval", eval_per_rule)):
            accepted = 0
            attempts = 0
            while accepted < target_count:
                attempts += 1
                if attempts > target_count * 500:
                    raise RuntimeError(f"could not generate unambiguous {rule.name} tasks")
                task_id = f"arc{order}-{rule.name}-{split}-{accepted:04d}"
                task = _arc_task(rule, rng, task_id)
                episode = _arc_episode(task, split, order)
                if not all(outcome.correct for outcome in episode.outcomes.values()):
                    continue
                episodes.append(episode)
                accepted += 1
    return episodes


def generate_arc_episodes(config: SequencerBenchmarkConfig) -> list[BenchmarkEpisode]:
    arc1 = _balanced_arc_episodes(
        primitive_rules(),
        order=1,
        calibration_per_rule=config.arc1_calibration_per_rule,
        eval_per_rule=config.arc1_eval_per_rule,
        seed=config.seed + 2000,
    )
    pair_names = (
        "flip_h+map_1_to_2",
        "fill_zero_with_1+rotate_180",
        "map_2_to_3+flip_h",
        "flip_v+map_2_to_3",
        "fill_zero_with_2+flip_v",
        "map_1_to_2+rotate_180",
        "rotate_180+fill_zero_with_1",
        "flip_h+fill_zero_with_2",
    )
    pairs_by_name = {rule.name: rule for rule in primitive_rule_pairs()}
    arc2_rules = [pairs_by_name[name] for name in pair_names]
    arc2 = _balanced_arc_episodes(
        arc2_rules,
        order=2,
        calibration_per_rule=config.arc2_calibration_per_rule,
        eval_per_rule=config.arc2_eval_per_rule,
        seed=config.seed + 3000,
    )
    return arc1 + arc2


def _routing_splits(
    examples: Sequence[RoutingExample], seed: int
) -> tuple[list[RoutingExample], list[RoutingExample], list[RoutingExample]]:
    grouped: dict[str, list[RoutingExample]] = defaultdict(list)
    for example in examples:
        grouped[example.env_id].append(example)
    train: list[RoutingExample] = []
    calibration: list[RoutingExample] = []
    evaluation: list[RoutingExample] = []
    for env_id, rows in sorted(grouped.items()):
        values = list(rows)
        random.Random(seed + _stable_int(env_id) % 100000).shuffle(values)
        train_end = max(1, int(0.6 * len(values)))
        calibration_end = max(train_end + 1, int(0.8 * len(values)))
        train.extend(values[:train_end])
        calibration.extend(values[train_end:calibration_end])
        evaluation.extend(values[calibration_end:])
    return train, calibration, evaluation


def generate_routing_episodes(
    examples: Sequence[RoutingExample], config: SequencerBenchmarkConfig
) -> list[BenchmarkEpisode]:
    train, calibration, evaluation = _routing_splits(examples, config.seed + 4000)
    if not train or not calibration or not evaluation:
        raise ValueError("routing benchmark requires non-empty train, calibration, and eval splits")
    trm = LexicalTRMRouter().fit(train)
    ldt = TokenLatticeRouter().fit(train)
    hybrid = HybridRoutingModel(ldt, trm)
    episodes = []
    for split, rows in (("calibration", calibration), ("eval", evaluation)):
        for index, example in enumerate(rows):
            predictions = {
                PROPOSAL_ONLY: trm.predict(example.prompt),
                DEDUCTION_ONLY: ldt.predict(example.prompt),
                TYPED_PROPOSE_CERTIFY: hybrid.predict(example.prompt)[0],
            }
            costs = {PROPOSAL_ONLY: 1.0, DEDUCTION_ONLY: 1.2, TYPED_PROPOSE_CERTIFY: 1.6}
            outcomes = {}
            for sequence, prediction in predictions.items():
                answer = prediction or "<abstain>"
                correct = answer == example.env_id
                outcomes[sequence] = CandidateOutcome(
                    correct=correct,
                    utility=_candidate_utility(correct, costs[sequence], 1.6, penalty=0.03),
                    cost=costs[sequence],
                    constraint_violation=False,
                    answer=answer,
                )
            episode_id = f"routing-{example.env_id}-{split}-{index:04d}-{_stable_int(example.prompt) % 1000000:06d}"
            episodes.append(
                BenchmarkEpisode(
                    episode_id=episode_id,
                    family="routing",
                    context=f"routing:{example.env_id}",
                    split=split,
                    prompt=example.prompt,
                    expected_answer=example.env_id,
                    outcomes=outcomes,
                )
            )
    return episodes


def generate_storyworld_episodes(config: SequencerBenchmarkConfig) -> list[BenchmarkEpisode]:
    env = CoupledStoryworldEnv()
    total = config.story_calibration_per_scenario + config.story_eval_per_scenario
    from research_gym.benchmarks.storyworld_bench import sample_start_states

    starts = sample_start_states(env, n=total, horizon=6, seed=config.seed + 5000)
    if len(starts) < total:
        raise ValueError(f"only {len(starts)} viable storyworld starts for requested {total}")
    episodes = []
    for scenario in ("secret_ending", "moral_optimization"):
        for index, start in enumerate(starts):
            split = "calibration" if index < config.story_calibration_per_scenario else "eval"
            policies = {
                PROPOSAL_ONLY: "trm",
                DEDUCTION_ONLY: "hard_gate",
                TYPED_PROPOSE_CERTIFY: "typed_membrane",
            }
            outcomes = {}
            for sequence, policy in policies.items():
                run = play_architecture_policy(scenario, policy, start, env=env, horizon=6)
                unit_cost = {PROPOSAL_ONLY: 1.0, DEDUCTION_ONLY: 1.8, TYPED_PROPOSE_CERTIFY: 2.2}[sequence]
                cost = unit_cost * len(run.actions)
                if scenario == "secret_ending":
                    base_utility = float(run.success)
                    violation = not run.success
                    expected = "secret_ending"
                else:
                    base_utility = min(1.0, max(0.0, (run.score + 21.0) / 43.0))
                    violation = False
                    expected = "maximize_moral_score"
                utility = base_utility - 0.03 * min(1.0, cost / 13.2)
                outcomes[sequence] = CandidateOutcome(
                    correct=run.success,
                    utility=utility,
                    cost=cost,
                    constraint_violation=violation,
                    answer=",".join(run.actions),
                )
            episode_id = f"story-{scenario}-{split}-{index:04d}"
            episodes.append(
                BenchmarkEpisode(
                    episode_id=episode_id,
                    family=f"story_{'secret' if scenario == 'secret_ending' else 'moral'}",
                    context=f"story:{scenario}",
                    split=split,
                    prompt=f"Play {scenario} from {json.dumps(start.to_dict(), sort_keys=True)}",
                    expected_answer=expected,
                    outcomes=outcomes,
                )
            )
    return episodes


def build_benchmark_episodes(
    config: SequencerBenchmarkConfig,
    routing_examples: Sequence[RoutingExample],
) -> list[BenchmarkEpisode]:
    episodes = generate_sudoku_episodes(config)
    episodes.extend(generate_arc_episodes(config))
    episodes.extend(generate_routing_episodes(routing_examples, config))
    episodes.extend(generate_storyworld_episodes(config))
    ids = [episode.episode_id for episode in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("benchmark episode IDs must be unique")
    return episodes


def _score_sequences(episodes: Sequence[BenchmarkEpisode]) -> dict[str, float]:
    return {
        sequence: mean(episode.outcomes[sequence].utility for episode in episodes)
        for sequence in SKILL_SEQUENCES
    }


def _best_sequence(scores: Mapping[str, float]) -> str:
    preference = {PROPOSAL_ONLY: 0, DEDUCTION_ONLY: 1, TYPED_PROPOSE_CERTIFY: 2}
    return max(SKILL_SEQUENCES, key=lambda name: (scores[name], preference[name]))


def _preference_vector(scores: Mapping[str, float]) -> tuple[float, float]:
    return (
        scores[TYPED_PROPOSE_CERTIFY] - scores[PROPOSAL_ONLY],
        scores[TYPED_PROPOSE_CERTIFY] - scores[DEDUCTION_ONLY],
    )


def _angle(vector: tuple[float, float]) -> float:
    if math.hypot(*vector) < 1e-12:
        return 0.0
    return math.degrees(math.atan2(vector[1], vector[0]))


def _wrapped_angle(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _alignment(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, bool]:
    norm = math.hypot(*a) * math.hypot(*b)
    if norm < 1e-12:
        return 0.0, False
    cosine = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / norm))
    return cosine * cosine, cosine < 0.0


def _macro_scores(by_context: Mapping[str, Sequence[BenchmarkEpisode]]) -> dict[str, float]:
    family_context_scores: dict[str, list[dict[str, float]]] = defaultdict(list)
    for rows in by_context.values():
        if not rows:
            continue
        families = {episode.family for episode in rows}
        if len(families) != 1:
            raise ValueError("one sequencer context cannot span benchmark families")
        family_context_scores[next(iter(families))].append(_score_sequences(rows))
    if not family_context_scores:
        raise ValueError("cannot fit sequencer without calibration contexts")
    return {
        sequence: mean(
            mean(scores[sequence] for scores in context_scores)
            for context_scores in family_context_scores.values()
        )
        for sequence in SKILL_SEQUENCES
    }


def fit_context_plans(
    calibration: Sequence[BenchmarkEpisode],
    config: SequencerBenchmarkConfig,
) -> tuple[dict[str, ContextPlan], dict[str, object]]:
    if not calibration or any(episode.split != "calibration" for episode in calibration):
        raise ValueError("fit_context_plans accepts calibration episodes only")
    by_context: dict[str, list[BenchmarkEpisode]] = defaultdict(list)
    for episode in calibration:
        by_context[episode.context].append(episode)
    fold_contexts: list[dict[str, list[BenchmarkEpisode]]] = [defaultdict(list), defaultdict(list)]
    for context, rows in by_context.items():
        ordered = sorted(rows, key=lambda item: (_stable_int(item.episode_id), item.episode_id))
        for index, episode in enumerate(ordered):
            fold_contexts[index % 2][context].append(episode)

    global_scores = _macro_scores(by_context)
    global_sequence = _best_sequence(global_scores)
    root_fold_scores = [_macro_scores(fold or by_context) for fold in fold_contexts]
    plans = {}
    for context, rows in sorted(by_context.items()):
        local_scores = _score_sequences(rows)
        local_sequence = _best_sequence(local_scores)
        local_fold_scores = [
            _score_sequences(fold_contexts[index][context] or rows) for index in range(2)
        ]
        root_vectors = [_preference_vector(value) for value in root_fold_scores]
        local_vectors = [_preference_vector(value) for value in local_fold_scores]
        alignments = [_alignment(root_vectors[index], local_vectors[index]) for index in range(2)]
        retentions = [item[0] for item in alignments]
        orientation_reversal = any(item[1] for item in alignments)
        deltas = [
            _wrapped_angle(_angle(local_vectors[index]) - _angle(root_vectors[index]))
            for index in range(2)
        ]
        loop_angle = abs(_wrapped_angle(deltas[0] - deltas[1]))
        retention_disagreement = abs(retentions[0] - retentions[1])
        retention_uncertainty = min(min(retentions), 0.01 + 0.25 * retention_disagreement)
        angle_uncertainty = min(30.0, 2.0 + 90.0 / math.sqrt(len(rows)))
        null_limit = 15.0 + 180.0 / math.sqrt(len(rows))
        matched_noise_null_passed = loop_angle <= null_limit
        coverage = 0.99 if len(rows) >= 10 else 0.95
        measurement = ControlRiskMeasurement(
            measurement_id=f"sequencer-loop:{context}",
            edges=tuple(
                EdgeRiskReceipt(
                    edge_id=f"{context}:path-{index}",
                    minimum_edge_worst_direction_retention=retention,
                    retention_uncertainty=retention_uncertainty,
                )
                for index, retention in enumerate(retentions)
            ),
            loop=LoopRiskReceipt(
                maximum_canonical_angle_degrees=loop_angle,
                angle_uncertainty_degrees=angle_uncertainty,
                det_h_flag=orientation_reversal,
                measured=True,
                matched_noise_null_passed=matched_noise_null_passed,
            ),
            simultaneous_coverage=coverage,
        )
        bound = control_risk_bound(measurement)
        lineage_authorized = bound.lineage_contraction_bound <= config.signed_error_budget
        full_authorized = (
            not bound.audit_gap
            and not bound.orientation_reversal
            and bound.noise_null_passed
            and bound.simultaneous_coverage >= 1.0 - config.delta
            and bound.signed_control_loss_upper_bound <= config.signed_error_budget
        )
        plans[context] = ContextPlan(
            context=context,
            global_sequence=global_sequence,
            local_sequence=local_sequence,
            full_control_authorized=full_authorized,
            lineage_only_authorized=lineage_authorized,
            route="global_signed" if full_authorized else "local_section",
            point_risk_score=bound.point_risk_score,
            signed_control_loss_upper_bound=bound.signed_control_loss_upper_bound,
            lineage_contraction_bound=bound.lineage_contraction_bound,
            holonomy_displacement_bound=bound.holonomy_displacement_bound,
            orientation_reversal=bound.orientation_reversal,
            matched_noise_null_passed=bound.noise_null_passed,
            simultaneous_coverage=bound.simultaneous_coverage,
            maximum_canonical_angle_degrees=loop_angle,
            calibration_count=len(rows),
            local_scores=local_scores,
        )
    fit_receipt = {
        "global_scores": global_scores,
        "global_sequence": global_sequence,
        "calibration_episode_count": len(calibration),
        "calibration_sha256": _episodes_hash(calibration),
    }
    return plans, fit_receipt


def _episodes_hash(episodes: Sequence[BenchmarkEpisode]) -> str:
    payload = [episode.to_jsonable() for episode in sorted(episodes, key=lambda item: item.episode_id)]
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def evaluate_sequencers(
    evaluation: Sequence[BenchmarkEpisode],
    plans: Mapping[str, ContextPlan],
) -> list[dict[str, object]]:
    if not evaluation or any(episode.split != "eval" for episode in evaluation):
        raise ValueError("evaluate_sequencers accepts eval episodes only")
    rows = []
    for episode in evaluation:
        plan = plans[episode.context]
        selections = {sequencer: plan.selection(sequencer) for sequencer in SEQUENCERS}
        selected = {sequencer: episode.outcomes[sequence] for sequencer, sequence in selections.items()}
        rows.append(
            {
                "episode_id": episode.episode_id,
                "family": episode.family,
                "context": episode.context,
                "prompt": episode.prompt,
                "expected_answer": episode.expected_answer,
                "candidate_outcomes": {
                    name: outcome.to_jsonable() for name, outcome in sorted(episode.outcomes.items())
                },
                "selections": selections,
                "selected_outcomes": {
                    name: outcome.to_jsonable() for name, outcome in sorted(selected.items())
                },
                "topology": {
                    "route": plan.route,
                    "signed_control_loss_upper_bound": plan.signed_control_loss_upper_bound,
                    "orientation_reversal": plan.orientation_reversal,
                    "maximum_canonical_angle_degrees": plan.maximum_canonical_angle_degrees,
                },
            }
        )
    return rows


def _metric(row: Mapping[str, object], sequencer: str, name: str) -> float:
    selected = row["selected_outcomes"]
    assert isinstance(selected, Mapping)
    outcome = selected[sequencer]
    assert isinstance(outcome, Mapping)
    value = outcome[name]
    return float(value)


def _summarize_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_family: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family"])].append(row)
    summary = {}
    for sequencer in SEQUENCERS:
        families = {}
        for family, family_rows in sorted(by_family.items()):
            families[family] = {
                "episodes": len(family_rows),
                "accuracy": mean(_metric(row, sequencer, "correct") for row in family_rows),
                "mean_utility": mean(_metric(row, sequencer, "utility") for row in family_rows),
                "mean_cost": mean(_metric(row, sequencer, "cost") for row in family_rows),
                "constraint_violation_rate": mean(
                    _metric(row, sequencer, "constraint_violation") for row in family_rows
                ),
            }
        summary[sequencer] = {
            "episodes": len(rows),
            "micro_accuracy": mean(_metric(row, sequencer, "correct") for row in rows),
            "macro_accuracy": mean(value["accuracy"] for value in families.values()),
            "macro_utility": mean(value["mean_utility"] for value in families.values()),
            "macro_cost": mean(value["mean_cost"] for value in families.values()),
            "macro_constraint_violation_rate": mean(
                value["constraint_violation_rate"] for value in families.values()
            ),
            "families": families,
        }
    return summary


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _macro_delta(rows: Sequence[Mapping[str, object]], treatment: str, control: str, metric: str) -> float:
    by_family: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family"])].append(
            _metric(row, treatment, metric) - _metric(row, control, metric)
        )
    return mean(mean(values) for values in by_family.values())


def _clustered_stratified_bootstrap_ci(
    rows: Sequence[Mapping[str, object]],
    treatment: str,
    control: str,
    metric: str,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    by_family_context: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_family_context[str(row["family"])][str(row["context"])].append(
            _metric(row, treatment, metric) - _metric(row, control, metric)
        )
    rng = random.Random(seed)
    draws = []
    families = sorted(by_family_context)
    for _ in range(iterations):
        family_means = []
        for family in families:
            clusters = by_family_context[family]
            contexts = sorted(clusters)
            sampled_values = []
            for _ in contexts:
                values = clusters[contexts[rng.randrange(len(contexts))]]
                sampled_values.extend(values[rng.randrange(len(values))] for _ in values)
            family_means.append(mean(sampled_values))
        draws.append(mean(family_means))
    return _percentile(draws, 0.025), _percentile(draws, 0.975)


def _sign_flip_pvalue(
    rows: Sequence[Mapping[str, object]],
    treatment: str,
    control: str,
    metric: str,
    *,
    iterations: int,
    seed: int,
) -> float:
    by_family_context: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_family_context[str(row["family"])][str(row["context"])].append(
            _metric(row, treatment, metric) - _metric(row, control, metric)
        )
    observed = abs(_macro_delta(rows, treatment, control, metric))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        permuted = []
        for family in sorted(by_family_context):
            family_values = []
            for context in sorted(by_family_context[family]):
                sign = 1.0 if rng.random() < 0.5 else -1.0
                family_values.extend(sign * value for value in by_family_context[family][context])
            permuted.append(mean(family_values))
        extreme += int(abs(mean(permuted)) >= observed - 1e-15)
    return (extreme + 1.0) / (iterations + 1.0)


def _mcnemar_exact(rows: Sequence[Mapping[str, object]], treatment: str, control: str) -> dict[str, float | int]:
    treatment_only = 0
    control_only = 0
    for row in rows:
        treatment_correct = bool(_metric(row, treatment, "correct"))
        control_correct = bool(_metric(row, control, "correct"))
        treatment_only += int(treatment_correct and not control_correct)
        control_only += int(control_correct and not treatment_correct)
    discordant = treatment_only + control_only
    if discordant == 0:
        pvalue = 1.0
    else:
        tail = sum(math.comb(discordant, value) for value in range(min(treatment_only, control_only) + 1))
        pvalue = min(1.0, 2.0 * tail / (2.0**discordant))
    return {
        "treatment_only_correct": treatment_only,
        "control_only_correct": control_only,
        "p_value": pvalue,
    }


def _cohen_dz(rows: Sequence[Mapping[str, object]], treatment: str, control: str) -> float:
    by_family_context: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_family_context[str(row["family"])][str(row["context"])].append(
            _metric(row, treatment, "utility") - _metric(row, control, "utility")
        )
    values_and_weights = []
    family_weight = 1.0 / len(by_family_context)
    for contexts in by_family_context.values():
        context_weight = family_weight / len(contexts)
        values_and_weights.extend((mean(values), context_weight) for values in contexts.values())
    weighted_mean = sum(value * weight for value, weight in values_and_weights)
    variance = sum(weight * (value - weighted_mean) ** 2 for value, weight in values_and_weights)
    deviation = math.sqrt(variance)
    if deviation < 1e-15:
        return 0.0 if abs(weighted_mean) < 1e-15 else math.copysign(math.inf, weighted_mean)
    return weighted_mean / deviation


def _comparisons(
    rows: Sequence[Mapping[str, object]], config: SequencerBenchmarkConfig
) -> list[dict[str, object]]:
    controls = (GLOBAL_SIGNED, LINEAGE_ONLY, FIXED_TYPED)
    comparisons = []
    for index, control in enumerate(controls):
        ci_low, ci_high = _clustered_stratified_bootstrap_ci(
            rows,
            CONTROL_MATH,
            control,
            "utility",
            iterations=config.bootstrap_iterations,
            seed=config.seed + 7000 + index,
        )
        pvalue = _sign_flip_pvalue(
            rows,
            CONTROL_MATH,
            control,
            "utility",
            iterations=config.permutation_iterations,
            seed=config.seed + 8000 + index,
        )
        comparisons.append(
            {
                "treatment": CONTROL_MATH,
                "control": control,
                "macro_utility_delta": _macro_delta(rows, CONTROL_MATH, control, "utility"),
                "utility_delta_ci_95": [ci_low, ci_high],
                "macro_accuracy_delta": _macro_delta(rows, CONTROL_MATH, control, "correct"),
                "macro_cost_delta": _macro_delta(rows, CONTROL_MATH, control, "cost"),
                "macro_constraint_violation_delta": _macro_delta(
                    rows, CONTROL_MATH, control, "constraint_violation"
                ),
                "paired_sign_flip_p_value": pvalue,
                "cohen_dz_utility": _cohen_dz(rows, CONTROL_MATH, control),
                "mcnemar_accuracy": _mcnemar_exact(rows, CONTROL_MATH, control),
            }
        )
    ordered = sorted(enumerate(comparisons), key=lambda item: item[1]["paired_sign_flip_p_value"])
    running = 0.0
    for rank, (original_index, comparison) in enumerate(ordered):
        adjusted = min(1.0, float(comparison["paired_sign_flip_p_value"]) * (len(ordered) - rank))
        running = max(running, adjusted)
        comparisons[original_index]["holm_adjusted_p_value"] = running
    return comparisons


def _sensitivity_frontier(
    rows: Sequence[Mapping[str, object]],
    plans: Mapping[str, ContextPlan],
    config: SequencerBenchmarkConfig,
) -> list[dict[str, float]]:
    frontier = []
    for epsilon in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        authorized = {
            context: (
                not plan.orientation_reversal
                and plan.matched_noise_null_passed
                and plan.simultaneous_coverage >= 1.0 - config.delta
                and plan.signed_control_loss_upper_bound <= epsilon
            )
            for context, plan in plans.items()
        }
        by_family: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in rows:
            by_family[str(row["family"])].append(row)
        family_metrics = []
        for family_rows in by_family.values():
            utilities = []
            accuracies = []
            costs = []
            violations = []
            for row in family_rows:
                plan = plans[str(row["context"])]
                sequence = plan.global_sequence if authorized[plan.context] else plan.local_sequence
                candidates = row["candidate_outcomes"]
                assert isinstance(candidates, Mapping)
                outcome = candidates[sequence]
                assert isinstance(outcome, Mapping)
                utilities.append(float(outcome["utility"]))
                accuracies.append(float(outcome["correct"]))
                costs.append(float(outcome["cost"]))
                violations.append(float(outcome["constraint_violation"]))
            family_metrics.append(
                (mean(utilities), mean(accuracies), mean(costs), mean(violations))
            )
        frontier.append(
            {
                "signed_error_budget": epsilon,
                "section_rate": mean(not value for value in authorized.values()),
                "macro_utility": mean(item[0] for item in family_metrics),
                "macro_accuracy": mean(item[1] for item in family_metrics),
                "macro_cost": mean(item[2] for item in family_metrics),
                "macro_constraint_violation_rate": mean(item[3] for item in family_metrics),
            }
        )
    return frontier


def run_sequencer_control_benchmark(
    routing_examples: Sequence[RoutingExample],
    config: SequencerBenchmarkConfig | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    config = config or SequencerBenchmarkConfig()
    if config.inference_unit != "family_context_cluster":
        raise ValueError(f"unsupported inference unit: {config.inference_unit}")
    protocol_payload = asdict(config)
    protocol_hash = sha256(
        json.dumps(protocol_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    episodes = build_benchmark_episodes(config, routing_examples)
    calibration = [episode for episode in episodes if episode.split == "calibration"]
    evaluation = [episode for episode in episodes if episode.split == "eval"]
    plans, fit_receipt = fit_context_plans(calibration, config)
    rows = evaluate_sequencers(evaluation, plans)
    summary = _summarize_rows(rows)
    context_counts: dict[str, int] = defaultdict(int)
    for episode in evaluation:
        context_counts[episode.context] += 1
    result = {
        "schema_version": "1.1.0",
        "study_id": "rsi_control_math_infused_skill_sequencers",
        "protocol": protocol_payload,
        "protocol_sha256": protocol_hash,
        "inference": {
            "estimand": "equal-family macro mean of paired episode utility deltas",
            "unit": config.inference_unit,
            "bootstrap": "family-stratified hierarchical context-cluster bootstrap",
            "randomization": "context-cluster sign flip",
            "multiple_comparisons": "Holm over three registered controls",
        },
        "fit_receipt": fit_receipt,
        "eval_episode_count": len(evaluation),
        "eval_sha256": _episodes_hash(evaluation),
        "families": sorted({episode.family for episode in evaluation}),
        "contexts": {
            context: {**plan.to_jsonable(), "eval_count": context_counts[context]}
            for context, plan in sorted(plans.items())
        },
        "summary": summary,
        "comparisons": _comparisons(rows, config),
        "error_budget_sensitivity": _sensitivity_frontier(rows, plans, config),
        "section_rate": mean(not plan.full_control_authorized for plan in plans.values()),
        "orientation_reversal_rate": mean(plan.orientation_reversal for plan in plans.values()),
        "claim_boundary": (
            "Matched deterministic solver and policy replay over procedural Sudoku/ARC-style tasks, local "
            "Tesseract routing trajectories, and the finite-state storyworld. It measures sequencer control, "
            "not neural weight infusion or official ARC/INTELLECT-3 leaderboard performance."
        ),
    }
    return result, rows


def summary_markdown(payload: Mapping[str, object]) -> str:
    summary = payload["summary"]
    assert isinstance(summary, Mapping)
    lines = [
        "# Control Math on Infused Hybrid Skill Sequencers",
        "",
        f"Evaluation episodes: `{payload['eval_episode_count']}`",
        f"Protocol SHA-256: `{payload['protocol_sha256']}`",
        f"Context section rate: `{float(payload['section_rate']):.3f}`",
        f"Orientation reversal rate: `{float(payload['orientation_reversal_rate']):.3f}`",
        "",
        "| Sequencer | Macro utility | Macro accuracy | Macro cost | Constraint violations |",
        "|---|---:|---:|---:|---:|",
    ]
    for sequencer in SEQUENCERS:
        metrics = summary[sequencer]
        assert isinstance(metrics, Mapping)
        lines.append(
            f"| `{sequencer}` | {float(metrics['macro_utility']):.4f} | "
            f"{float(metrics['macro_accuracy']):.4f} | {float(metrics['macro_cost']):.3f} | "
            f"{float(metrics['macro_constraint_violation_rate']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "| Control | Utility delta | 95% bootstrap CI | Sign-flip p | Holm p | Accuracy delta | Cost delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    comparisons = payload["comparisons"]
    assert isinstance(comparisons, list)
    for comparison in comparisons:
        ci = comparison["utility_delta_ci_95"]
        lines.append(
            f"| `{comparison['control']}` | {float(comparison['macro_utility_delta']):+.4f} | "
            f"[{float(ci[0]):+.4f}, {float(ci[1]):+.4f}] | "
            f"{float(comparison['paired_sign_flip_p_value']):.4g} | "
            f"{float(comparison['holm_adjusted_p_value']):.4g} | "
            f"{float(comparison['macro_accuracy_delta']):+.4f} | "
            f"{float(comparison['macro_cost_delta']):+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Post-registration error-budget sensitivity",
            "",
            "| Error budget | Section rate | Macro utility | Macro accuracy | Macro cost |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    sensitivity = payload["error_budget_sensitivity"]
    assert isinstance(sensitivity, list)
    for row in sensitivity:
        lines.append(
            f"| {float(row['signed_error_budget']):.2f} | {float(row['section_rate']):.3f} | "
            f"{float(row['macro_utility']):.4f} | {float(row['macro_accuracy']):.4f} | "
            f"{float(row['macro_cost']):.3f} |"
        )
    lines.extend(["", "## Family breakdown", ""])
    control_metrics = summary[CONTROL_MATH]
    assert isinstance(control_metrics, Mapping)
    families = control_metrics["families"]
    assert isinstance(families, Mapping)
    lines.extend(
        [
            "| Family | Episodes | Utility | Accuracy | Cost | Violations |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for family, metrics in families.items():
        lines.append(
            f"| `{family}` | {int(metrics['episodes'])} | {float(metrics['mean_utility']):.4f} | "
            f"{float(metrics['accuracy']):.4f} | {float(metrics['mean_cost']):.3f} | "
            f"{float(metrics['constraint_violation_rate']):.4f} |"
        )
    lines.extend(["", f"Claim boundary: {payload['claim_boundary']}", ""])
    return "\n".join(lines)


def v1_replay_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    for row in rows:
        replay_row = {
            "example_id": row["episode_id"],
            "system_prompt": (
                "Execute the configured hybrid skill sequencer from its sealed topology receipt. "
                "Do not replace typed authority with confidence."
            ),
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "answer": row["expected_answer"],
            "family": row["family"],
            "context": row["context"],
            "candidate_outcomes": row["candidate_outcomes"],
            "selections": row["selections"],
            "topology": row["topology"],
            "max_turns": 1,
        }
        if "airis_das" in row:
            replay_row["airis_das"] = row["airis_das"]
        output.append(replay_row)
    return output
