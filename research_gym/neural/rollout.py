"""Deterministic expert-iteration and storyworld mechanics helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import random
from typing import Callable, Iterable, Mapping, Sequence

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised on installs without the extra
    raise ImportError(
        "research_gym.neural.rollout requires the optional 'neural' extra"
    ) from exc

from research_gym.benchmarks.storyworld_architecture_bench import moral_score
from research_gym.benchmarks.storyworld_bench import ldt_certified_action
from research_gym.core.frames import Frame
from research_gym.core.hybrid import (
    CandidateState,
    HybridMode,
    MembranePolicy,
    certify_and_apply,
)
from research_gym.core.typed_soundness import SoundnessType
from research_gym.envs.coupled_storyworld import CoupledStoryworldEnv, StoryState
from research_gym.neural.trm import DEFAULT_ACTIONS, SOUNDNESS_TYPES, TRMProposer


@dataclass(frozen=True)
class StoryExample:
    episode_id: str
    scenario: str
    state: StoryState
    horizon: int
    split: str
    region: str

    @property
    def group_id(self) -> str:
        material = tuple(sorted(self.state.to_dict().items()))
        return sha256(repr(material).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RolloutRecord:
    episode_id: str
    scenario: str
    split: str
    region: str
    state: Mapping[str, int]
    proposed_action: str
    selected_action: str
    fallback_action: str
    claimed_soundness: str
    verified_soundness: str | None
    provenance_disagreed: bool | None
    accepted: bool
    proposal_environment_sound: bool
    proposal_oracle_optimal: bool
    proposal_utility: float
    selected_utility: float
    fallback_utility: float
    latent: Sequence[float]
    decision_reason: str

    def to_jsonable(self) -> dict[str, object]:
        return dict(self.__dict__)

    def to_frame(self, *, round_index: int, arm_id: str) -> Frame:
        ground_truth = (
            SoundnessType.ENV_SOUND_DEAD
            if self.proposal_environment_sound
            else SoundnessType.UNKNOWN
        )
        return Frame(
            id=f"{arm_id}-r{round_index}-{self.episode_id}",
            family="hybrid_rollout",
            source="gaming_vs_improvement",
            input_state=dict(self.state),
            operation={
                "proposed_action": self.proposed_action,
                "selected_action": self.selected_action,
            },
            output_state={"accepted": self.accepted},
            soundness_type=ground_truth,
            label="accepted" if self.accepted else "rejected",
            metadata={
                "claimed_soundness": self.claimed_soundness,
                "ground_truth_soundness": ground_truth.value,
                "verified_soundness": self.verified_soundness,
                "provenance_disagreed": self.provenance_disagreed,
                "proposal_utility": self.proposal_utility,
                "selected_utility": self.selected_utility,
                "round": round_index,
                "arm_id": arm_id,
            },
        )


def action_candidate_state(actions: Sequence[str] = DEFAULT_ACTIONS) -> CandidateState:
    return CandidateState({"action": frozenset(actions)})


def _terminal_value(env: CoupledStoryworldEnv, state: StoryState, scenario: str) -> float:
    if scenario == "secret_ending":
        if env.target(state):
            return 1.0
        shaping = (
            state.evidence / 4.0
            + (state.trust + 3) / 6.0
            + (4 - state.heat) / 4.0
            + state.scene / env.max_scene
        ) / 4.0
        return 0.05 * shaping
    if scenario == "moral_optimization":
        return max(0.0, min(1.0, (moral_score(state) + 21.0) / 43.0))
    raise ValueError(f"unknown scenario: {scenario}")


@lru_cache(maxsize=None)
def _best_value(
    max_scene: int,
    state: StoryState,
    remaining: int,
    scenario: str,
) -> float:
    env = CoupledStoryworldEnv(max_scene=max_scene)
    if remaining <= 0 or env.terminal(state) or env.target(state):
        return _terminal_value(env, state, scenario)
    return max(
        _best_value(
            max_scene,
            env.step(state, action, env.other_policy(state)),
            remaining - 1,
            scenario,
        )
        for action in env.self_actions
    )


def action_value(env: CoupledStoryworldEnv, example: StoryExample, action: str) -> float:
    next_state = env.step(example.state, action, env.other_policy(example.state))
    return _best_value(
        env.max_scene,
        next_state,
        max(0, example.horizon - 1),
        example.scenario,
    )


def oracle_action(env: CoupledStoryworldEnv, example: StoryExample) -> str:
    values = {action: action_value(env, example, action) for action in env.self_actions}
    return max(env.self_actions, key=lambda action: (values[action], -env.self_actions.index(action)))


def action_environment_sound(
    env: CoupledStoryworldEnv,
    example: StoryExample,
    action: str,
    *,
    moral_threshold: int = 8,
) -> bool:
    next_state = env.step(example.state, action, env.other_policy(example.state))
    if example.scenario == "secret_ending":
        reachable, _ = env.reachable(
            next_state,
            max(0, example.horizon - 1),
            other_mode="model",
        )
        return reachable or env.target(next_state)
    threshold = max(0.0, min(1.0, (moral_threshold + 21.0) / 43.0))
    return (
        _best_value(
            env.max_scene,
            next_state,
            max(0, example.horizon - 1),
            example.scenario,
        )
        >= threshold
    )


def state_conditioned_fallback(
    env: CoupledStoryworldEnv,
    example: StoryExample,
) -> str:
    """Independent reachability fallback, deliberately not the utility oracle."""

    if example.scenario == "secret_ending":
        return ldt_certified_action(env, example.state, example.horizon) or "wait"
    priority = ("defuse", "befriend", "investigate", "rush", "wait")
    safe = [
        action
        for action in priority
        if action_environment_sound(env, example, action)
    ]
    return safe[0] if safe else "wait"


def mechanics_checker(
    env: CoupledStoryworldEnv,
    example: StoryExample,
) -> Callable[[object], bool]:
    def check(proposal: object) -> bool:
        metadata = getattr(proposal, "metadata", {})
        action = str(metadata.get("selected_action") or "")
        return action_environment_sound(env, example, action)

    return check


def train_proposer(
    model: TRMProposer,
    examples: Sequence[StoryExample],
    action_targets: Sequence[str],
    provenance_targets: Sequence[SoundnessType],
    *,
    steps: int,
    seed: int,
    learning_rate: float,
) -> list[float]:
    if not examples:
        return []
    if not (len(examples) == len(action_targets) == len(provenance_targets)):
        raise ValueError("training rows and targets must align")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    candidates = action_candidate_state(model.action_vocab)
    features = torch.stack(
        [model.features_from(row.scenario, candidates, row.state) for row in examples]
    )
    action_y = torch.tensor(
        [model.action_vocab.index(action) for action in action_targets],
        dtype=torch.long,
    )
    provenance_y = torch.tensor(
        [SOUNDNESS_TYPES.index(value) for value in provenance_targets],
        dtype=torch.long,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []
    model.train()
    generator = random.Random(seed)
    batch_size = min(64, len(examples))
    for _ in range(steps):
        indices = generator.sample(range(len(examples)), batch_size)
        optimizer.zero_grad()
        output = model(features[indices])
        loss = (
            nn.functional.cross_entropy(output.action_logits, action_y[indices])
            + nn.functional.cross_entropy(output.provenance_logits, provenance_y[indices])
            + 0.15
            * nn.functional.cross_entropy(
                output.mode_logits,
                torch.zeros(len(indices), dtype=torch.long),
            )
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    model.eval()
    return losses


def rollout_examples(
    model: TRMProposer,
    examples: Iterable[StoryExample],
    *,
    policy: MembranePolicy,
    rejection_action: str,
    seed: int,
    verifier_context: Callable[[StoryExample], object] | None = None,
) -> list[RolloutRecord]:
    env = CoupledStoryworldEnv()
    candidates = action_candidate_state(model.action_vocab)
    records = []
    generator = torch.Generator().manual_seed(seed)
    for example in examples:
        proposal = model.propose(
            example.scenario,
            candidates,
            example.state,
            generator=generator,
            force_mode=HybridMode.DEDUCE,
        )
        context = verifier_context(example) if verifier_context else None
        decision = certify_and_apply(
            candidates,
            proposal,
            policy=policy,
            verifier_context=context,
        )
        proposed_action = str(proposal.metadata["selected_action"])
        fallback = state_conditioned_fallback(env, example)
        if decision.accepted or rejection_action == "identical_fallback":
            selected = proposed_action
        elif rejection_action == "state_conditioned_fallback":
            selected = fallback
        else:
            raise ValueError(f"unknown rejection action: {rejection_action}")
        proposed_utility = action_value(env, example, proposed_action)
        selected_utility = action_value(env, example, selected)
        fallback_utility = action_value(env, example, fallback)
        records.append(
            RolloutRecord(
                episode_id=example.episode_id,
                scenario=example.scenario,
                split=example.split,
                region=example.region,
                state=example.state.to_dict(),
                proposed_action=proposed_action,
                selected_action=selected,
                fallback_action=fallback,
                claimed_soundness=proposal.soundness.value,
                verified_soundness=(
                    decision.verified_soundness.value
                    if decision.verified_soundness
                    else None
                ),
                provenance_disagreed=decision.provenance_disagreed,
                accepted=decision.accepted,
                proposal_environment_sound=action_environment_sound(
                    env, example, proposed_action
                ),
                proposal_oracle_optimal=proposed_action == oracle_action(env, example),
                proposal_utility=proposed_utility,
                selected_utility=selected_utility,
                fallback_utility=fallback_utility,
                latent=tuple(float(value) for value in proposal.metadata["latent"]),
                decision_reason=decision.reason,
            )
        )
    return records

