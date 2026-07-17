from copy import deepcopy

import pytest


torch = pytest.importorskip("torch", exc_type=ImportError)

from research_gym.core.hybrid import HybridMode, MembranePolicy
from research_gym.envs.coupled_storyworld import CoupledStoryworldEnv, StoryState
from research_gym.neural.probes import (
    direction_cosine,
    fit_grouped_probe,
    matched_control_directions,
    probe_verifier,
)
from research_gym.neural.rollout import (
    StoryExample,
    action_candidate_state,
    mechanics_checker,
    oracle_action,
    rollout_examples,
    train_proposer,
)
from research_gym.neural.trm import TRMProposer
from research_gym.core.hybrid import exact_mechanics_verifier
from research_gym.core.typed_soundness import SoundnessType


def _examples() -> list[StoryExample]:
    return [
        StoryExample(
            episode_id=f"example-{index}",
            scenario="secret_ending" if index % 2 == 0 else "moral_optimization",
            state=StoryState(
                trust=-1 + index % 3,
                evidence=index % 4,
                heat=(index * 2) % 4,
                scene=index % 3,
            ),
            horizon=5,
            split="train",
            region="unit",
        )
        for index in range(12)
    ]


def test_trm_proposer_shapes_and_lattice_emission():
    torch.manual_seed(3)
    model = TRMProposer(latent_dim=24, recurrence_steps=4)
    candidates = action_candidate_state()
    proposal, latents = model.propose(
        "secret_ending",
        candidates,
        StoryState(0, 1, 2, 1),
        return_latents=True,
        force_mode=HybridMode.DEDUCE,
    )

    assert latents.shape == (4, 24)
    assert proposal.proposed_state == candidates
    assert proposal.metadata["selected_action"] in candidates.domains["action"]
    assert len(proposal.metadata["latent"]) == 24


def test_storyworld_rollout_is_deterministic_given_model_and_seed():
    torch.manual_seed(5)
    model = TRMProposer(latent_dim=16, recurrence_steps=3)
    env = CoupledStoryworldEnv()
    examples = _examples()[:4]
    policy = MembranePolicy(provenance_verifier=exact_mechanics_verifier)
    context = lambda example: {"mechanics_checker": mechanics_checker(env, example)}

    first = rollout_examples(
        model,
        examples,
        policy=policy,
        rejection_action="state_conditioned_fallback",
        seed=11,
        verifier_context=context,
    )
    second = rollout_examples(
        deepcopy(model),
        examples,
        policy=policy,
        rejection_action="state_conditioned_fallback",
        seed=11,
        verifier_context=context,
    )

    assert first == second


def test_short_training_changes_model_but_keeps_valid_emission():
    torch.manual_seed(7)
    model = TRMProposer(latent_dim=16, recurrence_steps=3)
    env = CoupledStoryworldEnv()
    examples = _examples()
    actions = [oracle_action(env, example) for example in examples]
    losses = train_proposer(
        model,
        examples,
        actions,
        [SoundnessType.ENV_SOUND_DEAD] * len(examples),
        steps=3,
        seed=7,
        learning_rate=0.01,
    )

    assert len(losses) == 3
    assert all(value >= 0 for value in losses)


def test_grouped_probe_and_matched_direction_controls():
    generator = torch.Generator().manual_seed(13)
    latents = torch.randn((48, 12), generator=generator)
    groups = [f"state-{index // 3}" for index in range(48)]
    labels = [int(value > 0) for value in latents[:, 0].tolist()]
    fit = fit_grouped_probe(latents, labels, groups, seed=13, steps=20)
    controls = matched_control_directions(
        fit.probe.weight,
        seed=17,
        random_count=3,
        orthogonal_count=3,
    )

    assert 0.0 <= fit.auroc <= 1.0
    assert 0.0 <= fit.shuffled_floor_auroc <= 1.0
    assert len(controls["random"]) == 3
    assert len(controls["orthogonal"]) == 3
    assert all(
        abs(direction_cosine(fit.probe.weight, value)) < 1e-5
        for value in controls["orthogonal"]
    )


def test_probe_verifier_emits_typed_pass_and_fail_verdicts():
    from research_gym.neural.probes import LinearProbe

    verifier = probe_verifier(
        LinearProbe(weight=torch.tensor([1.0, 0.0]), bias=0.0, threshold=0.6)
    )
    candidates = action_candidate_state()
    model = TRMProposer(latent_dim=2, recurrence_steps=1)
    passed = model.propose(
        "secret_ending", candidates, StoryState(0, 0, 0, 0), force_mode=HybridMode.DEDUCE
    )
    failed = model.propose(
        "secret_ending", candidates, StoryState(0, 0, 0, 0), force_mode=HybridMode.DEDUCE
    )
    passed.metadata["latent"] = [2.0, 0.0]
    failed.metadata["latent"] = [-2.0, 0.0]

    assert verifier(passed, None) == SoundnessType.ENV_SOUND_DEAD
    assert verifier(failed, None) == SoundnessType.UNKNOWN
