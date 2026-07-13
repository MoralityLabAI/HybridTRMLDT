from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from research_gym.envs.control_tasks import ControlTask, generate_control_tasks


POLICIES = (
    "trm",
    "ldt",
    "hard_gate",
    "confidence_arbitration",
    "typed_membrane",
    "typed_confidence",
    "counterfactual_beam",
)

ARCHITECTURE_NOTES = {
    "trm": "Latent proposal only; cheapest path and no explicit certification.",
    "ldt": "Explicit scorer only; robust on exact mechanics but proxy-bound on preferences.",
    "hard_gate": "Treat all LDT/model candidates as mandatory filters, regardless of provenance.",
    "confidence_arbitration": "Use TRM above a calibrated margin and otherwise defer to LDT.",
    "typed_membrane": "Use TRM unless an environment-sound constraint rejects it, then use LDT.",
    "typed_confidence": "Apply environment constraints first, then confidence arbitration inside the safe set.",
    "counterfactual_beam": "Certify a bounded TRM beam and rank surviving branches by combined TRM/LDT score.",
    "skill_router": "Select a hybrid structure per skill using only train-split control utility.",
}


@dataclass(frozen=True)
class ControlDecision:
    task_id: str
    policy: str
    action: str
    optimal_action: str
    safe: bool
    correct: bool
    utility: float
    regret: float
    deliberation_cost: float
    consulted_ldt: bool
    reason: str

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


def _rank(scores: dict[str, float], actions: tuple[str, ...] | list[str]) -> list[str]:
    order = {action: index for index, action in enumerate(actions)}
    return sorted(actions, key=lambda action: (-scores[action], order[action]))


def _margin(scores: dict[str, float], actions: tuple[str, ...] | list[str]) -> float:
    ranked = _rank(scores, actions)
    if len(ranked) < 2:
        return float("inf")
    return scores[ranked[0]] - scores[ranked[1]]


def choose_action(
    task: ControlTask,
    policy: str,
    *,
    gamma: float,
    beam_width: int,
) -> tuple[str, float, bool, str]:
    trm_ranked = _rank(task.trm_scores, task.actions)
    ldt_ranked = _rank(task.ldt_scores, task.actions)
    allowed = list(task.environment_allowed)

    if policy == "trm":
        return trm_ranked[0], 1.0, False, "latent_top"
    if policy == "ldt":
        return ldt_ranked[0], 2.0, True, "explicit_top"
    if policy == "hard_gate":
        candidates = list(task.soft_candidates) or list(task.actions)
        action = _rank(task.trm_scores, candidates)[0]
        return action, 2.0, True, f"hard_filter_{task.soft_soundness.value}"
    if policy == "confidence_arbitration":
        margin = _margin(task.trm_scores, task.actions)
        if margin >= gamma:
            return trm_ranked[0], 1.0, False, f"trm_margin={margin:.3f}"
        return ldt_ranked[0], 2.0, True, f"ldt_margin={margin:.3f}"
    if policy == "typed_membrane":
        if trm_ranked[0] in task.environment_allowed:
            return trm_ranked[0], 1.2, False, "environment_check_accept"
        action = _rank(task.ldt_scores, allowed)[0]
        return action, 2.2, True, "environment_check_override"
    if policy == "typed_confidence":
        safe_trm = _rank(task.trm_scores, allowed)
        margin = _margin(task.trm_scores, allowed)
        if margin >= gamma:
            return safe_trm[0], 1.4, False, f"typed_trm_margin={margin:.3f}"
        action = _rank(task.ldt_scores, allowed)[0]
        return action, 2.4, True, f"typed_ldt_margin={margin:.3f}"
    if policy == "counterfactual_beam":
        beam = trm_ranked[:beam_width]
        surviving = [action for action in beam if action in task.environment_allowed]
        if not surviving:
            action = _rank(task.ldt_scores, allowed)[0]
            return action, 2.0 + beam_width, True, "beam_empty_ldt_fallback"
        combined = {
            action: 0.65 * task.trm_scores[action] + 0.35 * task.ldt_scores[action]
            for action in surviving
        }
        action = _rank(combined, surviving)[0]
        return action, 1.5 + beam_width, True, f"certified_beam={len(surviving)}/{len(beam)}"
    raise ValueError(f"Unknown control policy: {policy}")


def evaluate_decision(
    task: ControlTask,
    policy: str,
    *,
    gamma: float,
    beam_width: int,
) -> ControlDecision:
    action, cost, consulted_ldt, reason = choose_action(task, policy, gamma=gamma, beam_width=beam_width)
    optimal_utility = task.utilities[task.optimal_action]
    utility = task.utilities[action]
    safe = action in task.environment_allowed
    return ControlDecision(
        task_id=task.task_id,
        policy=policy,
        action=action,
        optimal_action=task.optimal_action,
        safe=safe,
        correct=safe and action == task.optimal_action,
        utility=utility,
        regret=optimal_utility - utility if safe else optimal_utility + 1.0,
        deliberation_cost=cost,
        consulted_ldt=consulted_ldt,
        reason=reason,
    )


def _metrics(decisions: list[ControlDecision]) -> dict[str, float | int]:
    total = len(decisions)
    if not total:
        return {
            "episodes": 0,
            "accuracy": 0.0,
            "mean_utility": 0.0,
            "unsafe_rate": 0.0,
            "mean_regret": 0.0,
            "mean_cost": 0.0,
            "ldt_consult_rate": 0.0,
            "selection_objective": 0.0,
        }
    accuracy = sum(decision.correct for decision in decisions) / total
    mean_utility = sum(decision.utility for decision in decisions) / total
    unsafe_rate = sum(not decision.safe for decision in decisions) / total
    mean_regret = sum(decision.regret for decision in decisions) / total
    mean_cost = sum(decision.deliberation_cost for decision in decisions) / total
    ldt_consult_rate = sum(decision.consulted_ldt for decision in decisions) / total
    objective = mean_utility + 0.25 * accuracy - 2.0 * unsafe_rate - 0.025 * mean_cost
    return {
        "episodes": total,
        "accuracy": accuracy,
        "mean_utility": mean_utility,
        "unsafe_rate": unsafe_rate,
        "mean_regret": mean_regret,
        "mean_cost": mean_cost,
        "ldt_consult_rate": ldt_consult_rate,
        "selection_objective": objective,
    }


def _evaluate_tasks(
    tasks: list[ControlTask],
    policy: str,
    *,
    gamma: float,
    beam_width: int,
) -> tuple[list[ControlDecision], dict[str, float | int]]:
    decisions = [
        evaluate_decision(task, policy, gamma=gamma, beam_width=beam_width)
        for task in tasks
    ]
    return decisions, _metrics(decisions)


def _tune_parameters(train_tasks: list[ControlTask]) -> tuple[float, int, list[dict[str, float | int]]]:
    search: list[dict[str, float | int]] = []
    gamma_candidates = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
    for gamma in gamma_candidates:
        _, confidence = _evaluate_tasks(
            train_tasks, "typed_confidence", gamma=gamma, beam_width=2
        )
        search.append({"family": "typed_confidence", "gamma": gamma, **confidence})
    best_confidence = max(
        (row for row in search if row["family"] == "typed_confidence"),
        key=lambda row: (float(row["selection_objective"]), -float(row["gamma"])),
    )

    for beam_width in (1, 2, 3, 4):
        _, beam = _evaluate_tasks(
            train_tasks,
            "counterfactual_beam",
            gamma=float(best_confidence["gamma"]),
            beam_width=beam_width,
        )
        search.append({"family": "counterfactual_beam", "beam_width": beam_width, **beam})
    best_beam = max(
        (row for row in search if row["family"] == "counterfactual_beam"),
        key=lambda row: (float(row["selection_objective"]), -int(row["beam_width"])),
    )
    return float(best_confidence["gamma"]), int(best_beam["beam_width"]), search


def _learn_skill_routes(
    train_tasks: list[ControlTask],
    *,
    gamma: float,
    beam_width: int,
) -> tuple[dict[str, str], dict[str, dict[str, dict[str, float | int]]]]:
    grouped: dict[str, list[ControlTask]] = defaultdict(list)
    for task in train_tasks:
        grouped[task.skill].append(task)
    routes: dict[str, str] = {}
    diagnostics: dict[str, dict[str, dict[str, float | int]]] = {}
    for skill, skill_tasks in sorted(grouped.items()):
        diagnostics[skill] = {}
        for policy in POLICIES:
            _, metrics = _evaluate_tasks(skill_tasks, policy, gamma=gamma, beam_width=beam_width)
            diagnostics[skill][policy] = metrics
        routes[skill] = max(
            POLICIES,
            key=lambda policy: float(diagnostics[skill][policy]["selection_objective"]),
        )
    return routes, diagnostics


def run_control_harness_benchmark(
    *,
    n_train: int = 48,
    n_eval: int = 48,
    seed: int = 23,
) -> dict[str, object]:
    tasks = generate_control_tasks(n_train=n_train, n_eval=n_eval, seed=seed)
    train_tasks = [task for task in tasks if task.split == "train"]
    eval_tasks = [task for task in tasks if task.split == "eval"]
    gamma, beam_width, search = _tune_parameters(train_tasks)
    routes, route_diagnostics = _learn_skill_routes(
        train_tasks, gamma=gamma, beam_width=beam_width
    )

    all_decisions: list[tuple[ControlTask, ControlDecision]] = []
    summary: dict[str, dict[str, float | int]] = {}
    by_skill: dict[str, dict[str, dict[str, float | int]]] = {}
    for policy in POLICIES:
        decisions, metrics = _evaluate_tasks(
            eval_tasks, policy, gamma=gamma, beam_width=beam_width
        )
        summary[policy] = metrics
        all_decisions.extend(zip(eval_tasks, decisions))

    routed_decisions: list[ControlDecision] = []
    for task in eval_tasks:
        delegated = routes[task.skill]
        decision = evaluate_decision(task, delegated, gamma=gamma, beam_width=beam_width)
        routed_decisions.append(
            ControlDecision(
                **{
                    **decision.to_jsonable(),
                    "policy": "skill_router",
                    "deliberation_cost": decision.deliberation_cost + 0.1,
                    "reason": f"skill={task.skill};delegate={delegated};{decision.reason}",
                }
            )
        )
    summary["skill_router"] = _metrics(routed_decisions)
    all_decisions.extend(zip(eval_tasks, routed_decisions))

    for skill in sorted(routes):
        by_skill[skill] = {}
        skill_pairs = [(task, decision) for task, decision in all_decisions if task.skill == skill]
        for policy in (*POLICIES, "skill_router"):
            decisions = [decision for _, decision in skill_pairs if decision.policy == policy]
            by_skill[skill][policy] = _metrics(decisions)

    runs = []
    task_by_id = {task.task_id: task for task in eval_tasks}
    for _, decision in all_decisions:
        task = task_by_id[decision.task_id]
        runs.append(
            {
                "application": task.application,
                "source_repo": task.source_repo,
                "skill": task.skill,
                "soft_soundness": task.soft_soundness.value,
                **decision.to_jsonable(),
            }
        )

    return {
        "schema": "hybrid_control_harness_benchmark_v1",
        "seed": seed,
        "n_train_per_application": n_train,
        "n_eval_per_application": n_eval,
        "applications": sorted({task.application for task in tasks}),
        "source_repositories": sorted({task.source_repo for task in tasks}),
        "architecture_notes": ARCHITECTURE_NOTES,
        "tuned": {"gamma": gamma, "beam_width": beam_width},
        "architecture_search": search,
        "skill_routes": routes,
        "skill_route_train_diagnostics": route_diagnostics,
        "summary": summary,
        "by_skill": by_skill,
        "runs": runs,
        "task_cards": [task.to_jsonable() for task in eval_tasks],
    }


def summary_markdown(payload: dict[str, object]) -> str:
    summary = payload["summary"]
    routes = payload["skill_routes"]
    by_skill = payload["by_skill"]
    assert isinstance(summary, dict)
    assert isinstance(routes, dict)
    assert isinstance(by_skill, dict)
    lines = [
        "# Cross-Project Hybrid Control Harness Benchmark",
        "",
        "These are deterministic source-inspired proxy tasks, not direct performance claims about the neighboring repositories.",
        "They preserve the control distinction found in those repos: exact mechanics/provenance versus model- or replay-derived guidance.",
        "",
        f"Train cases per application: `{payload['n_train_per_application']}`",
        f"Held-out cases per application: `{payload['n_eval_per_application']}`",
        f"Calibrated confidence margin: `{payload['tuned']['gamma']:.2f}`",
        f"Calibrated beam width: `{payload['tuned']['beam_width']}`",
        "",
        "## Held-out aggregate",
        "",
        "| Architecture | Accuracy | Utility | Unsafe Rate | Regret | Cost | LDT Consult |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, metrics in summary.items():
        assert isinstance(metrics, dict)
        lines.append(
            f"| `{policy}` | {float(metrics['accuracy']):.3f} | {float(metrics['mean_utility']):.3f} | "
            f"{float(metrics['unsafe_rate']):.3f} | {float(metrics['mean_regret']):.3f} | "
            f"{float(metrics['mean_cost']):.2f} | {float(metrics['ldt_consult_rate']):.3f} |"
        )

    lines.extend(["", "## Learned skill routes", "", "| Skill | Selected architecture |", "|---|---|"])
    for skill, policy in sorted(routes.items()):
        lines.append(f"| `{skill}` | `{policy}` |")

    lines.extend(
        [
            "",
            "## Held-out skill behavior",
            "",
            "| Skill | Route | Accuracy | Utility | Unsafe Rate | Cost |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for skill, policy in sorted(routes.items()):
        skill_rows = by_skill[skill]
        assert isinstance(skill_rows, dict)
        metrics = skill_rows["skill_router"]
        assert isinstance(metrics, dict)
        lines.append(
            f"| `{skill}` | `{policy}` | {float(metrics['accuracy']):.3f} | "
            f"{float(metrics['mean_utility']):.3f} | {float(metrics['unsafe_rate']):.3f} | "
            f"{float(metrics['mean_cost']):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Hard gating tests the failure mode where conditional evidence is promoted into elimination authority.",
            "- Typed confidence composes provenance checks with calibration rather than choosing one arbitration mechanism globally.",
            "- Counterfactual beam tests whether a small proposal budget can recover utility after certification.",
            "- Skill routing tests whether architecture choice itself is an optimizable game-playing skill.",
            "",
        ]
    )
    return "\n".join(lines)


def write_experiment_bundle(payload: dict[str, object], out_dir: Path, *, notes: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "training_notes.md").write_text(notes, encoding="utf-8")
