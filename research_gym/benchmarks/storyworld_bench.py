from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from research_gym.envs.coupled_storyworld import CoupledStoryworldEnv, StoryState


@dataclass(frozen=True)
class StoryworldRunResult:
    player: str
    start_state: dict[str, int]
    success: bool
    terminal: bool
    steps: int
    actions: list[str]
    final_state: dict[str, int]
    overrides: int = 0
    trace: list[str] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "player": self.player,
            "start_state": self.start_state,
            "success": self.success,
            "terminal": self.terminal,
            "steps": self.steps,
            "actions": list(self.actions),
            "final_state": self.final_state,
            "overrides": self.overrides,
            "trace": list(self.trace),
        }


def ldt_certified_action(env: CoupledStoryworldEnv, state: StoryState, horizon: int) -> str | None:
    """Pick the first action that preserves modeled reachability to target."""

    if env.target(state):
        return None
    candidates: list[tuple[int, str]] = []
    for action in env.self_actions:
        next_state = env.step(state, action, env.other_policy(state))
        reachable, _ = env.reachable(next_state, max(0, horizon - 1), other_mode="model")
        if reachable or env.target(next_state):
            score = _goal_score(env, next_state)
            candidates.append((score, action))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], -env.self_actions.index(item[1])))[1]


def trm_heuristic_action(env: CoupledStoryworldEnv, state: StoryState) -> str:
    """Small persistent-policy analogue: advance local deficits greedily."""

    if state.heat >= 3:
        return "defuse"
    if state.evidence < 3:
        return "investigate"
    if state.trust < 1:
        return "befriend"
    if state.scene < env.max_scene:
        return "rush"
    return "wait"


def hybrid_action(env: CoupledStoryworldEnv, state: StoryState, horizon: int) -> tuple[str, bool]:
    proposed = trm_heuristic_action(env, state)
    proposed_state = env.step(state, proposed, env.other_policy(state))
    reachable, _ = env.reachable(proposed_state, max(0, horizon - 1), other_mode="model")
    if reachable or env.target(proposed_state):
        return proposed, False
    certified = ldt_certified_action(env, state, horizon)
    if certified is None:
        return proposed, False
    return certified, certified != proposed


def _goal_score(env: CoupledStoryworldEnv, state: StoryState) -> int:
    return (
        10 * int(env.target(state))
        + 2 * min(state.evidence, 3)
        + 2 * min(state.trust, 1)
        - state.heat
        + state.scene
    )


def play_storyworld(
    player: str,
    start_state: StoryState,
    *,
    env: CoupledStoryworldEnv | None = None,
    horizon: int = 6,
) -> StoryworldRunResult:
    env = env or CoupledStoryworldEnv()
    state = start_state
    actions: list[str] = []
    trace: list[str] = []
    overrides = 0

    for step in range(horizon):
        if env.target(state) or env.terminal(state):
            break
        remaining = horizon - step
        if player == "ldt":
            action = ldt_certified_action(env, state, remaining) or "wait"
            overridden = False
        elif player == "trm":
            action = trm_heuristic_action(env, state)
            overridden = False
        elif player == "hybrid":
            action, overridden = hybrid_action(env, state, remaining)
        else:
            raise ValueError(f"Unknown storyworld player: {player}")

        if overridden:
            overrides += 1
        other = env.other_policy(state)
        next_state = env.step(state, action, other)
        actions.append(action)
        trace.append(f"{step}:{state.to_dict()} action={action} other={other} next={next_state.to_dict()}")
        state = next_state

    return StoryworldRunResult(
        player=player,
        start_state=start_state.to_dict(),
        success=env.target(state),
        terminal=env.terminal(state),
        steps=len(actions),
        actions=actions,
        final_state=state.to_dict(),
        overrides=overrides,
        trace=trace,
    )


def sample_start_states(env: CoupledStoryworldEnv, *, n: int = 64, horizon: int = 6, seed: int = 7) -> list[StoryState]:
    rng = random.Random(seed)
    viable = [state for state in env.all_states() if not env.terminal(state) and env.reachable(state, horizon, "model")[0]]
    rng.shuffle(viable)
    return viable[:n]


def run_storyworld_benchmark(*, n: int = 64, horizon: int = 6, seed: int = 7) -> dict[str, object]:
    env = CoupledStoryworldEnv()
    starts = sample_start_states(env, n=n, horizon=horizon, seed=seed)
    runs = [
        play_storyworld(player, state, env=env, horizon=horizon)
        for state in starts
        for player in ("ldt", "trm", "hybrid")
    ]
    grouped: dict[str, list[StoryworldRunResult]] = defaultdict(list)
    for run in runs:
        grouped[run.player].append(run)
    summary = {}
    for player, player_runs in sorted(grouped.items()):
        total = len(player_runs)
        summary[player] = {
            "episodes": total,
            "successes": sum(run.success for run in player_runs),
            "success_rate": sum(run.success for run in player_runs) / total if total else 0.0,
            "avg_steps": sum(run.steps for run in player_runs) / total if total else 0.0,
            "overrides": sum(run.overrides for run in player_runs),
        }
    return {
        "n": len(starts),
        "horizon": horizon,
        "seed": seed,
        "summary": summary,
        "runs": [run.to_jsonable() for run in runs],
    }


def summary_markdown(payload: dict[str, object]) -> str:
    summary = payload["summary"]
    assert isinstance(summary, dict)
    lines = [
        "# Storyworld Playing Benchmark",
        "",
        f"Episodes per player: {payload['n']}",
        f"Horizon: {payload['horizon']}",
        "",
        "| Player | Successes | Success Rate | Avg Steps | Overrides |",
        "|---|---:|---:|---:|---:|",
    ]
    for player, metrics in summary.items():
        assert isinstance(metrics, dict)
        lines.append(
            f"| `{player}` | {int(metrics['successes'])}/{int(metrics['episodes'])} | "
            f"{float(metrics['success_rate']):.3f} | {float(metrics['avg_steps']):.2f} | "
            f"{int(metrics['overrides'])} |"
        )
    return "\n".join(lines) + "\n"


def write_experiment_bundle(payload: dict[str, object], out_dir: Path, *, notes: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "training_notes.md").write_text(notes, encoding="utf-8")
