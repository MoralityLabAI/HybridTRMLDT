from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from statistics import mean
from typing import Sequence

from research_gym.benchmarks.gaming_vs_improvement_bench import (
    EXACT_MECHANICS,
    EXPOSED_FROZEN_PROBE,
    STATE_CONDITIONED_FALLBACK,
    GamingBenchmarkConfig,
    _adapt_model,
    _context_factory,
    _fit_probe,
    _oracle_power_metrics,
    _policy_for,
    _undertrained_base,
    build_region_heldout_examples,
    validate_frozen_config,
)
from research_gym.envs.coupled_storyworld import CoupledStoryworldEnv
from research_gym.neural.rollout import StoryExample, action_candidate_state, rollout_examples
from research_gym.neural.trm import TRMProposer
from research_gym.scripts.bench_gaming_vs_improvement import _write_json


CONFIG_PATH = Path("configs/gaming_vs_improvement_v2.json")
RESULT_PATH = Path("data/benchmarks/gaming_vs_improvement_v2_calibration.json")
REPORT_PATH = Path("reports/gaming_vs_improvement_v2_calibration.md")


def _development_regions(
    config: GamingBenchmarkConfig,
    *,
    seed: int,
) -> dict[str, list[StoryExample]]:
    regions = build_region_heldout_examples(config, seed=seed)
    # The calibration path never computes an oracle label on final buckets 8-9.
    regions["heldout_high_trust"] = regions.pop("power_development")
    return regions


def _actions(model: TRMProposer, examples: Sequence[StoryExample]) -> list[str]:
    candidates = action_candidate_state(model.action_vocab)
    return [
        str(
            model.propose(
                row.scenario,
                candidates,
                row.state,
                force_mode=None,
            ).metadata["selected_action"]
        )
        for row in examples
    ]


def run_calibration(registration: dict[str, object]) -> dict[str, object]:
    config_sha256 = validate_frozen_config(registration)
    base_config = GamingBenchmarkConfig.from_registration(registration)
    calibration = registration["calibration"]
    development_seeds = tuple(int(seed) for seed in calibration["development_seeds"])
    adaptation_seeds = set(int(seed) for seed in calibration["adaptation_capacity_seeds"])
    env = CoupledStoryworldEnv()
    power_rows = []
    movement_rows = []

    for seed in development_seeds:
        config = replace(base_config, seeds=(seed,))
        regions = _development_regions(config, seed=seed)
        base_model, metrics = _undertrained_base(config, regions, seed=seed)
        power_rows.append({"seed": seed, **metrics})
        if seed not in adaptation_seeds:
            continue

        eval_examples = list(regions["heldout_high_trust"])
        initial_actions = _actions(base_model, eval_examples)
        exposed = _fit_probe(
            base_model,
            list(regions["probe_calibration"])[::2],
            config,
            seed=seed + 2000,
        )
        train_by_id = {row.episode_id: row for row in regions["proposer_train"]}
        for evidence_source in (EXACT_MECHANICS, EXPOSED_FROZEN_PROBE):
            model = deepcopy(base_model)
            policy = _policy_for(
                evidence_source,
                exposed_probe=exposed.probe,
                rotating_probe=exposed.probe,
            )
            retained_counts = []
            for round_index in range(config.expert_iteration_rounds):
                records = rollout_examples(
                    model,
                    regions["proposer_train"],
                    policy=policy,
                    rejection_action=STATE_CONDITIONED_FALLBACK,
                    seed=seed + 7000 + round_index,
                    verifier_context=_context_factory(env),
                )
                rows = []
                for record in records:
                    row = record.to_jsonable()
                    row["accepted"] = record.accepted
                    rows.append(row)
                retained_counts.append(
                    _adapt_model(
                        model,
                        train_by_id,
                        rows,
                        config,
                        seed=seed + 9000 + round_index,
                    )
                )
            final_actions = _actions(model, eval_examples)
            action_change = mean(
                initial != final
                for initial, final in zip(initial_actions, final_actions)
            )
            final_metrics = _oracle_power_metrics(model, eval_examples, env)
            movement_rows.append(
                {
                    "seed": seed,
                    "evidence_source": evidence_source,
                    "action_change_rate": action_change,
                    "oracle_accuracy_delta": float(final_metrics["oracle_accuracy"])
                    - float(metrics["oracle_accuracy"]),
                    "retained_training_counts": retained_counts,
                }
            )

    minimum_power = min(float(row["accuracy_above_majority"]) for row in power_rows)
    minimum_action_change = min(float(row["action_change_rate"]) for row in movement_rows)
    if minimum_power + 1e-12 < base_config.round0_min_accuracy_above_majority:
        raise AssertionError("development power confirmation failed")
    if minimum_action_change + 1e-12 < 0.1:
        raise AssertionError("adaptation action-movement confirmation failed")
    return {
        "schema": "gaming_vs_improvement_v2_calibration_v1",
        "config_sha256": config_sha256,
        "development_bucket": list(base_config.development_hash_buckets),
        "final_eval_buckets": list(base_config.eval_hash_buckets),
        "final_evaluation_oracle_labels_accessed": False,
        "selected_round0_steps": base_config.round0_steps,
        "selected_adaptation_steps": base_config.adaptation_steps,
        "frozen_note_correction": (
            "The frozen adaptation_calibration_note lower bound 0.102 came from the wider "
            "multi-budget sweep. The canonical selected-six-step confirmation minimum is "
            f"{minimum_action_change:.6f}; the selection rule still passes."
        ),
        "minimum_development_accuracy_above_majority": minimum_power,
        "minimum_development_action_change": minimum_action_change,
        "power_confirmation": power_rows,
        "adaptation_capacity_confirmation": movement_rows,
        "selection_history": {
            "stage_1": calibration["step_sweep_stage_1"],
            "stage_2": calibration["step_sweep_stage_2"],
            "selection_rule": calibration["selection_rule"],
        },
    }


def calibration_markdown(result: dict[str, object]) -> str:
    lines = [
        "# Above-Majority Calibration",
        "",
        f"Config SHA-256: `{result['config_sha256']}`",
        "Final evaluation oracle labels accessed: "
        f"`{result['final_evaluation_oracle_labels_accessed']}`",
        "",
        "## Power Confirmation",
        "",
        "| Seed | Accuracy | Majority | Delta | Ceiling margin |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in result["power_confirmation"]:
        lines.append(
            f"| {row['seed']} | {float(row['oracle_accuracy']):.3f} | "
            f"{float(row['majority_class_accuracy']):.3f} | "
            f"{float(row['accuracy_above_majority']):+.3f} | "
            f"{float(row['margin_to_ceiling']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Adaptation Capacity",
            "",
            "| Seed | Evidence | Action change | Oracle accuracy delta |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in result["adaptation_capacity_confirmation"]:
        lines.append(
            f"| {row['seed']} | {row['evidence_source']} | "
            f"{float(row['action_change_rate']):.3f} | "
            f"{float(row['oracle_accuracy_delta']):+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Registration Note Correction",
            "",
            str(result["frozen_note_correction"]),
            "",
            "Selection was based on development power and action movement, not on a preferred improvement "
            "or evasion outcome. Buckets 8-9 remained outcome-blind until the frozen benchmark run.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    registration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    result = run_calibration(registration)
    _write_json(result, RESULT_PATH)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(calibration_markdown(result), encoding="utf-8")
    print(calibration_markdown(result))


if __name__ == "__main__":
    main()
