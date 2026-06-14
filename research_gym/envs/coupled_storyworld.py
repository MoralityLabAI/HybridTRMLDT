from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from research_gym.core.frames import ReachabilityFrame
from research_gym.core.typed_soundness import SoundnessType


@dataclass(frozen=True, order=True)
class StoryState:
    """Small bounded state for the first gym environment.

    trust: relationship / cooperation variable.
    evidence: progress toward discovering the target ending.
    heat: instability or rival pressure variable.
    scene: progress through the story.
    """

    trust: int
    evidence: int
    heat: int
    scene: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


class CoupledStoryworldEnv:
    """Exact finite transition system with one self agent and one modeled agent.

    This is deliberately tiny. Its job is to generate dense frames and typed
    labels, not to be a final benchmark.
    """

    self_actions: Tuple[str, ...] = ("befriend", "investigate", "defuse", "rush", "wait")
    other_actions: Tuple[str, ...] = ("rival_sabotage", "rival_flatter", "rival_wait")

    def __init__(self, max_scene: int = 5) -> None:
        self.max_scene = max_scene
        self.bounds = {
            "trust": (-3, 3),
            "evidence": (0, 4),
            "heat": (0, 4),
            "scene": (0, max_scene),
        }

    def clamp(self, name: str, value: int) -> int:
        lo, hi = self.bounds[name]
        return max(lo, min(hi, value))

    def target(self, s: StoryState) -> bool:
        """Secret ending predicate."""
        return s.scene >= self.max_scene and s.trust >= 1 and s.evidence >= 3 and s.heat <= 2

    def terminal(self, s: StoryState) -> bool:
        return s.scene >= self.max_scene

    def step(self, s: StoryState, self_action: str, other_action: str = "rival_wait") -> StoryState:
        trust, evidence, heat, scene = s.trust, s.evidence, s.heat, s.scene

        if self_action == "befriend":
            trust += 1
            heat = max(0, heat - 1)
        elif self_action == "investigate":
            evidence += 1
            heat += 1
            trust -= 1 if heat >= 2 else 0
        elif self_action == "defuse":
            heat -= 2
        elif self_action == "rush":
            scene += 1
            heat += 1
        elif self_action == "wait":
            heat = max(0, heat - 1)
        else:
            raise ValueError(f"Unknown self action: {self_action}")

        if other_action == "rival_sabotage":
            trust -= 1
            heat += 1
        elif other_action == "rival_flatter":
            trust += 1
            evidence = max(0, evidence - 1)
        elif other_action == "rival_wait":
            pass
        else:
            raise ValueError(f"Unknown other action: {other_action}")

        # Time passes unless the player waited to lower heat.
        if self_action != "wait":
            scene += 1

        return StoryState(
            trust=self.clamp("trust", trust),
            evidence=self.clamp("evidence", evidence),
            heat=self.clamp("heat", heat),
            scene=self.clamp("scene", scene),
        )

    def other_policy(self, s: StoryState) -> str:
        """Simple model of another agent's objective pressure.

        The rival wants low evidence and low player trust. It sabotages when
        evidence is high enough to threaten its objective, otherwise it flatters
        if trust is already low, otherwise it waits.
        """
        if s.evidence >= 2 and s.heat <= 3:
            return "rival_sabotage"
        if s.trust <= -1 and s.evidence >= 1:
            return "rival_flatter"
        return "rival_wait"

    def all_states(self) -> Iterable[StoryState]:
        for trust in range(self.bounds["trust"][0], self.bounds["trust"][1] + 1):
            for evidence in range(self.bounds["evidence"][0], self.bounds["evidence"][1] + 1):
                for heat in range(self.bounds["heat"][0], self.bounds["heat"][1] + 1):
                    for scene in range(self.bounds["scene"][0], self.bounds["scene"][1] + 1):
                        yield StoryState(trust, evidence, heat, scene)

    def reachable(
        self,
        start: StoryState,
        horizon: int,
        other_mode: str = "frozen",
    ) -> Tuple[bool, List[str]]:
        """Exact finite-horizon reachability to target.

        other_mode:
          frozen: other agent always waits.
          model: other agent follows `other_policy`.
          nondet: any other action is possible.
        """
        frontier: Dict[StoryState, List[str]] = {start: []}
        if self.target(start):
            return True, []

        for _ in range(horizon):
            next_frontier: Dict[StoryState, List[str]] = {}
            for state, path in frontier.items():
                if self.terminal(state):
                    continue
                for self_action in self.self_actions:
                    if other_mode == "frozen":
                        others = ("rival_wait",)
                    elif other_mode == "model":
                        others = (self.other_policy(state),)
                    elif other_mode == "nondet":
                        others = self.other_actions
                    else:
                        raise ValueError(f"Unknown other_mode: {other_mode}")
                    for other_action in others:
                        ns = self.step(state, self_action, other_action)
                        npath = path + [f"{self_action}/{other_action}"]
                        if self.target(ns):
                            return True, npath
                        if ns not in next_frontier:
                            next_frontier[ns] = npath
            frontier = next_frontier
        return False, []

    def label_frame(self, state: StoryState, horizon: int, frame_id: str) -> ReachabilityFrame:
        reachable_frozen, witness_frozen = self.reachable(state, horizon, other_mode="frozen")
        reachable_model, witness_model = self.reachable(state, horizon, other_mode="model")
        reachable_nondet, _ = self.reachable(state, horizon, other_mode="nondet")

        if not reachable_frozen:
            label = SoundnessType.ENV_SOUND_DEAD
        elif reachable_frozen and not reachable_model:
            label = SoundnessType.MODEL_SOUND_DEAD
        else:
            label = SoundnessType.LIVE

        return ReachabilityFrame(
            frame_id=frame_id,
            state=state.to_dict(),
            horizon=horizon,
            label=label,
            target_reachable_self_only=reachable_frozen,
            target_reachable_with_other_model=reachable_model,
            target_reachable_with_other_frozen=reachable_frozen,
            witness_actions=witness_model or witness_frozen,
            counterfactual={"reachable_with_other_nondet": reachable_nondet},
        )
