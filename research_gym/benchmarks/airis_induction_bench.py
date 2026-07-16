"""Calibration-transition induction and held-out evaluation for AIRIS-style rules."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from statistics import mean
from typing import Mapping, Sequence

from research_gym.adapters.airis_das import (
    AIRIS_BRIDGE_SCHEMA,
    AIRIS_RULE_SCHEMA,
    AIRIS_RULESET_SCHEMA,
    airis_condition_features,
    airis_observation,
    embedded_forecast,
    resolve_airis_decision,
    trusted_rule_sha256,
)
from research_gym.benchmarks.sequencer_control_bench import (
    CONTROL_MATH,
    SKILL_SEQUENCES,
    BenchmarkEpisode,
)


RAW_AIRIS = "raw_airis"
GUARDED_AIRIS = "guarded_airis"
EPISODE_ORACLE = "episode_oracle"


def episodes_sha256(episodes: Sequence[BenchmarkEpisode]) -> str:
    payload = [episode.to_jsonable() for episode in sorted(episodes, key=lambda row: row.episode_id)]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _preference(tie_break_order: Sequence[str]) -> dict[str, int]:
    if set(tie_break_order) != set(SKILL_SEQUENCES):
        raise ValueError("tie_break_order must contain every skill sequence exactly once")
    return {name: len(tie_break_order) - index for index, name in enumerate(tie_break_order)}


def _winner(outcomes: Mapping[str, object], tie_break_order: Sequence[str]) -> str:
    preference = _preference(tie_break_order)
    return max(
        SKILL_SEQUENCES,
        key=lambda name: (float(outcomes[name].utility), preference[name])
        if hasattr(outcomes[name], "utility")
        else (float(outcomes[name]["utility"]), preference[name]),
    )


def _mean_utilities(episodes: Sequence[BenchmarkEpisode]) -> dict[str, float]:
    return {
        sequence: mean(episode.outcomes[sequence].utility for episode in episodes)
        for sequence in SKILL_SEQUENCES
    }


def induce_airis_ruleset(
    payload: Mapping[str, object],
    calibration: Sequence[BenchmarkEpisode],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    if not calibration or any(episode.split != "calibration" for episode in calibration):
        raise ValueError("AIRIS induction accepts calibration episodes only")
    protocol_sha256 = str(payload.get("protocol_sha256") or "")
    contexts = payload.get("contexts")
    fit = payload.get("fit_receipt")
    if not protocol_sha256 or not isinstance(contexts, Mapping) or not isinstance(fit, Mapping):
        raise ValueError("frozen benchmark protocol, contexts, and fit receipt are required")
    calibration_sha256 = episodes_sha256(calibration)
    if calibration_sha256 != str(fit.get("calibration_sha256") or ""):
        raise ValueError("regenerated calibration episodes do not match the frozen fit receipt")

    tie_break_order = tuple(str(value) for value in protocol.get("tie_break_order", []))
    preference = _preference(tie_break_order)
    grouped: dict[str, list[BenchmarkEpisode]] = defaultdict(list)
    for episode in calibration:
        grouped[episode.context].append(episode)
    if set(grouped) != set(contexts):
        raise ValueError("calibration contexts do not match frozen topology contexts")

    rules = []
    for context, episodes in sorted(grouped.items()):
        raw_plan = contexts[context]
        if not isinstance(raw_plan, Mapping):
            raise ValueError(f"invalid topology plan for {context}")
        families = {episode.family for episode in episodes}
        if len(families) != 1:
            raise ValueError(f"induction context spans families: {context}")
        family = next(iter(families))
        winners = [_winner(episode.outcomes, tie_break_order) for episode in episodes]
        label_counts = Counter(winners)
        selected = max(
            SKILL_SEQUENCES,
            key=lambda name: (label_counts[name], preference[name]),
        )
        support = label_counts[selected]
        counterexample_count = len(episodes) - support
        confidence = support / len(episodes)
        mean_utilities = _mean_utilities(episodes)
        alternatives = sorted(
            (value for name, value in mean_utilities.items() if name != selected),
            reverse=True,
        )
        mean_utility_margin = mean_utilities[selected] - alternatives[0]
        route = str(raw_plan.get("route") or "")
        features = airis_condition_features(
            protocol_sha256=protocol_sha256,
            family=family,
            context=context,
            global_sequence=str(raw_plan.get("global_sequence") or ""),
            topology_route=route,
            orientation_reversal=bool(raw_plan.get("orientation_reversal")),
            signed_control_loss_upper_bound=raw_plan.get(
                "signed_control_loss_upper_bound", 2.0
            ),
        )
        counterexamples = []
        for episode, winner in zip(episodes, winners):
            if winner == selected:
                continue
            counterexamples.append(
                {
                    "row_id": episode.episode_id,
                    "context": context,
                    "action": selected,
                    "observed": {
                        "outcome": (
                            f"selected_sequence={winner};"
                            f"predicted_utility={episode.outcomes[selected].utility:.8f};"
                            f"winner_utility={episode.outcomes[winner].utility:.8f}"
                        ),
                        "value_signs": {
                            "rule_match": "false",
                            "utility_regret": "positive",
                        },
                    },
                }
            )
        basis = "|".join(
            [
                "calibration_episode_induction_v1",
                protocol_sha256,
                calibration_sha256,
                context,
                selected,
                *features,
            ]
        )
        rule_id = f"induced_airis_rule_{sha256(basis.encode('utf-8')).hexdigest()[:12]}"
        rules.append(
            {
                "schema": AIRIS_RULE_SCHEMA,
                "rule_id": rule_id,
                "preconditions": features,
                "predicts": {
                    "outcome": (
                        f"selected_sequence={selected};control_route={route};"
                        "authority=topology_membrane"
                    ),
                    "value_signs": {
                        "calibration_majority": "positive" if confidence > 0.5 else "weak",
                        "mean_utility_margin": (
                            "positive" if mean_utility_margin > 1e-12 else "non_positive"
                        ),
                    },
                },
                "confidence": confidence,
                "support": support,
                "counterexample_count": counterexample_count,
                "counterexamples": counterexamples,
                "metadata": {
                    "source": "calibration_episode_outcome_induction",
                    "protocol_sha256": protocol_sha256,
                    "calibration_sha256": calibration_sha256,
                    "context": context,
                    "family": family,
                    "calibration_episode_count": len(episodes),
                    "label_counts": {
                        name: label_counts[name] for name in SKILL_SEQUENCES
                    },
                    "mean_utilities": mean_utilities,
                    "mean_utility_margin": mean_utility_margin,
                    "label_target": protocol.get("rule_target"),
                },
            }
        )

    registry = trusted_rule_sha256(rules)
    return {
        "schema": AIRIS_RULESET_SCHEMA,
        "summary": {
            "schema": AIRIS_BRIDGE_SCHEMA,
            "study_id": protocol.get("study_id"),
            "rule_count": len(rules),
            "calibration_episode_count": len(calibration),
            "source": "calibration_episode_outcome_induction",
            "protocol_sha256": protocol_sha256,
            "calibration_sha256": calibration_sha256,
            "primary_confidence_threshold": protocol.get(
                "primary_confidence_threshold"
            ),
            "claim_boundary": protocol.get("claim_boundary"),
        },
        "trusted_rule_sha256": registry,
        "rules": rules,
    }


def _macro_metrics(
    records: Sequence[Mapping[str, object]], selection_key: str
) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in records:
        grouped[str(row["family"])].append(row)
    families = {}
    for family, rows in sorted(grouped.items()):
        outcomes = [row["candidate_outcomes"][row[selection_key]] for row in rows]
        families[family] = {
            "episodes": len(rows),
            "accuracy": mean(float(outcome["correct"]) for outcome in outcomes),
            "mean_utility": mean(float(outcome["utility"]) for outcome in outcomes),
            "mean_cost": mean(float(outcome["cost"]) for outcome in outcomes),
            "constraint_violation_rate": mean(
                float(outcome["constraint_violation"]) for outcome in outcomes
            ),
        }
    return {
        "macro_accuracy": mean(value["accuracy"] for value in families.values()),
        "macro_utility": mean(value["mean_utility"] for value in families.values()),
        "macro_cost": mean(value["mean_cost"] for value in families.values()),
        "macro_constraint_violation_rate": mean(
            value["constraint_violation_rate"] for value in families.values()
        ),
        "families": families,
    }


def _threshold_summary(
    records: Sequence[Mapping[str, object]], threshold_key: str
) -> dict[str, object]:
    selection_key = f"selection_{threshold_key}"
    metrics = _macro_metrics(records, selection_key)
    accepted = [row for row in records if row[f"accepted_{threshold_key}"]]
    changed = [row for row in accepted if row[selection_key] != row["control_selection"]]
    harmful = [
        row
        for row in changed
        if float(row["candidate_outcomes"][row[selection_key]]["utility"])
        < float(row["candidate_outcomes"][row["control_selection"]]["utility"])
        - 1e-15
    ]
    beneficial = [
        row
        for row in changed
        if float(row["candidate_outcomes"][row[selection_key]]["utility"])
        > float(row["candidate_outcomes"][row["control_selection"]]["utility"])
        + 1e-15
    ]
    raw_wrong = [row for row in records if row["raw_selection"] != row["oracle_selection"]]
    raw_below_control = [
        row
        for row in records
        if float(row["candidate_outcomes"][row["raw_selection"]]["utility"])
        < float(row["candidate_outcomes"][row["control_selection"]]["utility"])
        - 1e-15
    ]
    return {
        **metrics,
        "acceptance_rate": len(accepted) / len(records),
        "fallback_rate": (len(records) - len(accepted)) / len(records),
        "control_parity_rate": mean(
            row[selection_key] == row["control_selection"] for row in records
        ),
        "accepted_oracle_label_accuracy": (
            mean(row[selection_key] == row["oracle_selection"] for row in accepted)
            if accepted
            else 0.0
        ),
        "accepted_change_count": len(changed),
        "beneficial_change_count": len(beneficial),
        "harmful_change_count": len(harmful),
        "neutral_change_count": len(changed) - len(beneficial) - len(harmful),
        "harmful_change_episode_rate": len(harmful) / len(records),
        "intact_wrong_rule_count": len(raw_wrong),
        "intact_wrong_rule_acceptance_rate": (
            mean(bool(row[f"accepted_{threshold_key}"]) for row in raw_wrong)
            if raw_wrong
            else 0.0
        ),
        "raw_below_control_count": len(raw_below_control),
        "fallback_save_rate": (
            mean(not bool(row[f"accepted_{threshold_key}"]) for row in raw_below_control)
            if raw_below_control
            else 0.0
        ),
    }


def evaluate_induced_airis(
    payload: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    ruleset: Mapping[str, object],
    protocol: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rules = ruleset.get("rules")
    if not isinstance(rules, list) or not rows:
        raise ValueError("induced AIRIS rules and held-out rows are required")
    protocol_sha256 = str(payload.get("protocol_sha256") or "")
    registry = trusted_rule_sha256(rules)
    declared_registry = ruleset.get("trusted_rule_sha256")
    if not isinstance(declared_registry, Mapping) or dict(declared_registry) != registry:
        raise ValueError("induced rule registry is missing or inconsistent")
    tie_break_order = tuple(str(value) for value in protocol.get("tie_break_order", []))
    thresholds = tuple(float(value) for value in protocol.get("sensitivity_thresholds", []))
    primary_threshold = float(protocol.get("primary_confidence_threshold", 0.5))
    minimum_support = int(protocol.get("minimum_support", 2))
    if primary_threshold not in thresholds:
        raise ValueError("primary confidence threshold must appear in sensitivity thresholds")

    records = []
    for raw_row in rows:
        row = dict(raw_row)
        candidates = row["candidate_outcomes"]
        observation = airis_observation(row, protocol_sha256)
        forecast = embedded_forecast(observation, rules, limit=3)
        matches = forecast.get("matches")
        if not isinstance(matches, list) or not matches:
            raise ValueError(f"induced AIRIS rule did not cover {row.get('episode_id')}")
        top = matches[0]
        raw_selection = str(top["outcome_fields"]["selected_sequence"])
        oracle_selection = _winner(candidates, tie_break_order)
        control_selection = str(row["selections"][CONTROL_MATH])
        record: dict[str, object] = {
            "episode_id": row.get("episode_id"),
            "family": row.get("family"),
            "context": row.get("context"),
            "candidate_outcomes": candidates,
            "oracle_selection": oracle_selection,
            "control_selection": control_selection,
            "raw_selection": raw_selection,
            "raw_oracle_label_correct": raw_selection == oracle_selection,
            "raw_control_parity": raw_selection == control_selection,
            "rule_id": top.get("rule_id"),
            "rule_sha256": registry.get(str(top.get("rule_id") or "")),
            "confidence": float(top.get("confidence") or 0.0),
            "support": int(top.get("support") or 0),
            "counterexample_count": int(top.get("counterexample_count") or 0),
        }
        for threshold in thresholds:
            key = f"t{threshold:.2f}".replace(".", "_")
            decision = resolve_airis_decision(
                row,
                forecast,
                expected_protocol_sha256=protocol_sha256,
                trusted_rules=registry,
                min_confidence=threshold,
                min_support=minimum_support,
            )
            record[f"selection_{key}"] = decision.selected_sequence
            record[f"accepted_{key}"] = decision.accepted
            record[f"reasons_{key}"] = list(decision.reasons)
        records.append(record)

    raw_metrics = _macro_metrics(records, "raw_selection")
    control_metrics = _macro_metrics(records, "control_selection")
    oracle_metrics = _macro_metrics(records, "oracle_selection")
    sensitivity = []
    for threshold in thresholds:
        key = f"t{threshold:.2f}".replace(".", "_")
        sensitivity.append(
            {
                "confidence_threshold": threshold,
                **_threshold_summary(records, key),
            }
        )
    primary = next(
        item for item in sensitivity if item["confidence_threshold"] == primary_threshold
    )

    by_context: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in records:
        by_context[str(row["context"])].append(row)
    context_generalization = []
    for context, context_rows in sorted(by_context.items()):
        confidence = float(context_rows[0]["confidence"])
        heldout_precision = mean(
            bool(row["raw_oracle_label_correct"]) for row in context_rows
        )
        context_generalization.append(
            {
                "context": context,
                "family": context_rows[0]["family"],
                "calibration_confidence": confidence,
                "heldout_rule_precision": heldout_precision,
                "calibration_minus_heldout_gap": confidence - heldout_precision,
                "eval_count": len(context_rows),
                "support": context_rows[0]["support"],
                "counterexample_count": context_rows[0]["counterexample_count"],
            }
        )

    label_rows: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in records:
        label_rows[str(row["family"])].append(row)
    label_precision_by_family = {}
    for family, family_rows in sorted(label_rows.items()):
        label_precision_by_family[family] = {
            "eval_count": len(family_rows),
            "heldout_rule_precision": mean(
                bool(row["raw_oracle_label_correct"]) for row in family_rows
            ),
            "wrong_rule_count": sum(
                not bool(row["raw_oracle_label_correct"]) for row in family_rows
            ),
        }

    result = {
        "schema": "hybrid_airis_induction_benchmark_v1",
        "study_id": protocol.get("study_id"),
        "protocol_sha256": protocol_sha256,
        "calibration_sha256": ruleset["summary"]["calibration_sha256"],
        "eval_sha256": payload.get("eval_sha256"),
        "rule_count": len(rules),
        "eval_episode_count": len(records),
        "primary_confidence_threshold": primary_threshold,
        "raw_airis": {
            **raw_metrics,
            "oracle_label_accuracy": mean(
                bool(row["raw_oracle_label_correct"]) for row in records
            ),
            "control_parity_rate": mean(bool(row["raw_control_parity"]) for row in records),
            "label_precision_by_family": label_precision_by_family,
        },
        "control_math": control_metrics,
        "episode_oracle": oracle_metrics,
        "guarded_airis": primary,
        "primary_macro_utility_delta_vs_control": (
            float(primary["macro_utility"]) - float(control_metrics["macro_utility"])
        ),
        "threshold_sensitivity": sensitivity,
        "context_generalization": context_generalization,
        "calibration_confidence_brier": mean(
            (float(row["confidence"]) - float(row["raw_oracle_label_correct"])) ** 2
            for row in records
        ),
        "mean_absolute_context_calibration_gap": mean(
            abs(float(item["calibration_minus_heldout_gap"]))
            for item in context_generalization
        ),
        "claim_boundary": protocol.get("claim_boundary"),
    }
    return result, records


def induction_markdown(result: Mapping[str, object]) -> str:
    raw = result["raw_airis"]
    guarded = result["guarded_airis"]
    control = result["control_math"]
    oracle = result["episode_oracle"]
    lines = [
        "# Independently Induced AIRIS Rules",
        "",
        f"Calibration-derived rules: `{result['rule_count']}`",
        f"Held-out episodes: `{result['eval_episode_count']}`",
        f"Primary confidence threshold: `{result['primary_confidence_threshold']:.2f}`",
        "",
        "| Controller | Macro utility | Macro accuracy | Macro cost |",
        "|---|---:|---:|---:|",
        f"| `raw_airis` | {float(raw['macro_utility']):.4f} | {float(raw['macro_accuracy']):.4f} | {float(raw['macro_cost']):.3f} |",
        f"| `guarded_airis` | {float(guarded['macro_utility']):.4f} | {float(guarded['macro_accuracy']):.4f} | {float(guarded['macro_cost']):.3f} |",
        f"| `control_math` | {float(control['macro_utility']):.4f} | {float(control['macro_accuracy']):.4f} | {float(control['macro_cost']):.3f} |",
        f"| `episode_oracle` | {float(oracle['macro_utility']):.4f} | {float(oracle['macro_accuracy']):.4f} | {float(oracle['macro_cost']):.3f} |",
        "",
        f"Raw held-out oracle-label accuracy: `{float(raw['oracle_label_accuracy']):.3f}`",
        f"Guarded acceptance rate: `{float(guarded['acceptance_rate']):.3f}`",
        f"Accepted-rule oracle-label accuracy: `{float(guarded['accepted_oracle_label_accuracy']):.3f}`",
        f"Intact-but-wrong held-out rules: `{int(guarded['intact_wrong_rule_count'])}`",
        f"Guarded harmful changes: `{int(guarded['harmful_change_count'])}`",
        f"Raw proposals below control: `{int(guarded['raw_below_control_count'])}`",
        f"Fallback save rate: `{float(guarded['fallback_save_rate']):.3f}`",
        f"Macro utility delta vs control: `{float(result['primary_macro_utility_delta_vs_control']):+.6f}`",
        "",
        "## Held-Out Rule Precision",
        "",
        "| Family | Episodes | Rule precision | Wrong rules |",
        "|---|---:|---:|---:|",
    ]
    for family, item in raw["label_precision_by_family"].items():
        lines.append(
            f"| `{family}` | {int(item['eval_count'])} | "
            f"{float(item['heldout_rule_precision']):.3f} | {int(item['wrong_rule_count'])} |"
        )
    lines.extend([
        "",
        "## Confidence Sensitivity",
        "",
        "| Threshold | Acceptance | Macro utility | Harmful changes | Wrong-rule acceptance |",
        "|---:|---:|---:|---:|---:|",
    ])
    for item in result["threshold_sensitivity"]:
        lines.append(
            f"| {float(item['confidence_threshold']):.2f} | {float(item['acceptance_rate']):.3f} | "
            f"{float(item['macro_utility']):.4f} | {int(item['harmful_change_count'])} | "
            f"{float(item['intact_wrong_rule_acceptance_rate']):.3f} |"
        )
    lines.extend(["", f"Claim boundary: {result['claim_boundary']}", ""])
    return "\n".join(lines)
