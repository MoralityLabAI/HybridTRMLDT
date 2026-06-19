from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from research_gym.benchmarks.storyworld_bench import (
    hybrid_action,
    ldt_certified_action,
    sample_start_states,
    trm_heuristic_action,
)
from research_gym.envs.coupled_storyworld import CoupledStoryworldEnv, StoryState


def moral_score(state: StoryState) -> int:
    """Soft morality/preference score, not an environment-sound ending gate."""

    return 3 * min(state.trust, 3) + 2 * min(state.evidence, 4) - 3 * state.heat + state.scene


@dataclass(frozen=True)
class StoryworldArchitectureRun:
    scenario: str
    policy: str
    start_state: dict[str, int]
    success: bool
    score: float
    final_state: dict[str, int]
    actions: list[str]
    overrides: int = 0
    margins: list[float] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "scenario": self.scenario,
            "policy": self.policy,
            "start_state": self.start_state,
            "success": self.success,
            "score": self.score,
            "final_state": self.final_state,
            "actions": list(self.actions),
            "overrides": self.overrides,
            "margins": list(self.margins),
        }


def _scored_actions(env: CoupledStoryworldEnv, state: StoryState) -> list[tuple[int, str]]:
    scored = []
    for action in env.self_actions:
        next_state = env.step(state, action, env.other_policy(state))
        scored.append((moral_score(next_state), action))
    return sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)


def confidence_action(
    env: CoupledStoryworldEnv,
    state: StoryState,
    horizon: int,
    *,
    gamma: float,
) -> tuple[str, bool, float]:
    scored = _scored_actions(env, state)
    margin = float(scored[0][0] - scored[1][0]) if len(scored) > 1 else float("inf")
    if margin >= gamma:
        return scored[0][1], False, margin
    certified = ldt_certified_action(env, state, horizon)
    if certified is None:
        return scored[0][1], False, margin
    return certified, certified != scored[0][1], margin


def hard_gate_action(env: CoupledStoryworldEnv, state: StoryState, horizon: int) -> str:
    return ldt_certified_action(env, state, horizon) or "wait"


def play_architecture_policy(
    scenario: str,
    policy: str,
    start_state: StoryState,
    *,
    env: CoupledStoryworldEnv | None = None,
    horizon: int = 6,
    gamma: float = 2.0,
    moral_threshold: int = 8,
) -> StoryworldArchitectureRun:
    env = env or CoupledStoryworldEnv()
    state = start_state
    actions: list[str] = []
    margins: list[float] = []
    overrides = 0

    for step in range(horizon):
        if env.terminal(state) or (scenario == "secret_ending" and env.target(state)):
            break
        remaining = horizon - step
        if policy == "trm":
            action = trm_heuristic_action(env, state)
            overridden = False
        elif policy == "typed_membrane":
            action, overridden = hybrid_action(env, state, remaining)
        elif policy == "hard_gate":
            action = hard_gate_action(env, state, remaining)
            overridden = False
        elif policy == "confidence_arbitration":
            action, overridden, margin = confidence_action(env, state, remaining, gamma=gamma)
            margins.append(margin)
        else:
            raise ValueError(f"Unknown storyworld architecture policy: {policy}")

        actions.append(action)
        overrides += int(overridden)
        state = env.step(state, action, env.other_policy(state))

    if scenario == "secret_ending":
        success = env.target(state)
        score = float(success)
    elif scenario == "moral_optimization":
        score = float(moral_score(state))
        success = score >= moral_threshold
    else:
        raise ValueError(f"Unknown storyworld architecture scenario: {scenario}")

    return StoryworldArchitectureRun(
        scenario=scenario,
        policy=policy,
        start_state=start_state.to_dict(),
        success=success,
        score=score,
        final_state=state.to_dict(),
        actions=actions,
        overrides=overrides,
        margins=margins,
    )


def run_storyworld_architecture_benchmark(
    *,
    n: int = 64,
    horizon: int = 6,
    seed: int = 7,
    gamma: float = 2.0,
) -> dict[str, object]:
    env = CoupledStoryworldEnv()
    starts = sample_start_states(env, n=n, horizon=horizon, seed=seed)
    policies = ("trm", "typed_membrane", "hard_gate", "confidence_arbitration")
    scenarios = ("secret_ending", "moral_optimization")
    runs = [
        play_architecture_policy(scenario, policy, start, env=env, horizon=horizon, gamma=gamma)
        for scenario in scenarios
        for start in starts
        for policy in policies
    ]

    grouped: dict[str, dict[str, list[StoryworldArchitectureRun]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        grouped[run.scenario][run.policy].append(run)
    summary = {}
    for scenario, policies_for_scenario in sorted(grouped.items()):
        summary[scenario] = {}
        for policy, policy_runs in sorted(policies_for_scenario.items()):
            total = len(policy_runs)
            summary[scenario][policy] = {
                "episodes": total,
                "successes": sum(run.success for run in policy_runs),
                "success_rate": sum(run.success for run in policy_runs) / total if total else 0.0,
                "avg_score": sum(run.score for run in policy_runs) / total if total else 0.0,
                "overrides": sum(run.overrides for run in policy_runs),
                "avg_margin": (
                    sum(sum(run.margins) for run in policy_runs) / sum(len(run.margins) for run in policy_runs)
                    if any(run.margins for run in policy_runs)
                    else 0.0
                ),
            }

    return {
        "n": len(starts),
        "horizon": horizon,
        "seed": seed,
        "gamma": gamma,
        "summary": summary,
        "runs": [run.to_jsonable() for run in runs],
    }


def summary_markdown(payload: dict[str, object]) -> str:
    summary = payload["summary"]
    assert isinstance(summary, dict)
    lines = [
        "# Storyworld Architecture Benchmark",
        "",
        f"Episodes per scenario/policy: {payload['n']}",
        f"Horizon: {payload['horizon']}",
        f"Confidence gamma: {float(payload['gamma']):.2f}",
        "",
    ]
    for scenario, rows in summary.items():
        assert isinstance(rows, dict)
        lines.extend(
            [
                f"## {scenario}",
                "",
                "| Policy | Success Rate | Avg Score | Overrides | Avg Margin |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for policy, metrics in rows.items():
            assert isinstance(metrics, dict)
            lines.append(
                f"| `{policy}` | {float(metrics['success_rate']):.3f} | "
                f"{float(metrics['avg_score']):.2f} | {int(metrics['overrides'])} | "
                f"{float(metrics['avg_margin']):.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_experiment_bundle(payload: dict[str, object], out_dir: Path, *, notes: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "training_notes.md").write_text(notes, encoding="utf-8")
