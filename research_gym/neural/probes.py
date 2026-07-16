"""Linear probes, grouped controls, and membrane adapters."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import random
from typing import Mapping, Sequence

try:
    import torch
    from torch import Tensor
except ImportError as exc:  # pragma: no cover - exercised on installs without the extra
    raise ImportError(
        "research_gym.neural.probes requires the optional 'neural' extra"
    ) from exc

from research_gym.core.hybrid import LatticeProposal, ProvenanceVerifier
from research_gym.core.typed_soundness import SoundnessType


def stable_group_hash(value: object) -> str:
    return sha256(repr(value).encode("utf-8")).hexdigest()


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


@dataclass(frozen=True)
class LinearProbe:
    weight: Tensor
    bias: float
    threshold: float = 0.5

    def scores(self, latents: Tensor) -> Tensor:
        if latents.ndim == 1:
            latents = latents.unsqueeze(0)
        return torch.sigmoid(latents @ self.weight + self.bias)

    def predict(self, latent: Tensor | Sequence[float]) -> bool:
        value = latent if isinstance(latent, Tensor) else torch.tensor(latent)
        return bool(self.scores(value.float())[0].item() >= self.threshold)


@dataclass(frozen=True)
class ProbeFit:
    probe: LinearProbe
    auroc: float
    shuffled_floor_auroc: float
    train_groups: tuple[str, ...]
    eval_groups: tuple[str, ...]


def _fit_weights(
    latents: Tensor,
    labels: Tensor,
    *,
    steps: int,
    lr: float,
    seed: int,
) -> LinearProbe:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(latents.shape[1], generator=generator) * 0.01
    weight.requires_grad_(True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        logits = latents @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
    return LinearProbe(weight.detach(), float(bias.detach().item()))


def _balanced_threshold(labels: Sequence[int], scores: Sequence[float]) -> float:
    if not labels or not any(labels) or all(labels):
        return 0.5
    candidates = sorted(set(float(score) for score in scores))
    candidates = [0.0, *candidates, 1.0]
    best = (float("-inf"), 0.5)
    for threshold in candidates:
        true_positive = sum(
            label == 1 and score >= threshold for label, score in zip(labels, scores)
        )
        true_negative = sum(
            label == 0 and score < threshold for label, score in zip(labels, scores)
        )
        positives = sum(labels)
        negatives = len(labels) - positives
        balanced = 0.5 * (true_positive / positives + true_negative / negatives)
        candidate = (balanced, -abs(threshold - 0.5))
        if candidate > (best[0], -abs(best[1] - 0.5)):
            best = (balanced, threshold)
    return float(best[1])


def _group_split(groups: Sequence[str], eval_fraction: float) -> tuple[set[str], set[str]]:
    unique = sorted(set(groups))
    eval_count = max(1, int(round(len(unique) * eval_fraction)))
    eval_groups = {
        group
        for group in unique
        if int(sha256(group.encode("utf-8")).hexdigest()[:8], 16) % len(unique)
        < eval_count
    }
    if not eval_groups or len(eval_groups) == len(unique):
        eval_groups = set(unique[-eval_count:])
    return set(unique) - eval_groups, eval_groups


def _within_group_shuffle(
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[group].append(index)
    shuffled = list(labels)
    for indices in by_group.values():
        values = [labels[index] for index in indices]
        rng.shuffle(values)
        for index, value in zip(indices, values):
            shuffled[index] = value
    return shuffled


def fit_grouped_probe(
    latents: Tensor,
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    seed: int,
    steps: int = 120,
    lr: float = 0.05,
    eval_fraction: float = 0.25,
) -> ProbeFit:
    if len(latents) != len(labels) or len(labels) != len(groups):
        raise ValueError("latents, labels, and groups must have equal lengths")
    train_groups, eval_groups = _group_split(groups, eval_fraction)
    train_indices = [index for index, group in enumerate(groups) if group in train_groups]
    eval_indices = [index for index, group in enumerate(groups) if group in eval_groups]
    y = torch.tensor(labels, dtype=torch.float32)
    probe = _fit_weights(
        latents[train_indices].float(),
        y[train_indices],
        steps=steps,
        lr=lr,
        seed=seed,
    )
    scores = probe.scores(latents[eval_indices].float()).tolist()
    eval_labels = [labels[index] for index in eval_indices]
    probe = LinearProbe(
        weight=probe.weight,
        bias=probe.bias,
        threshold=_balanced_threshold(eval_labels, scores),
    )

    shuffled = _within_group_shuffle(labels, groups, seed=seed + 104729)
    shuffled_y = torch.tensor(shuffled, dtype=torch.float32)
    floor_probe = _fit_weights(
        latents[train_indices].float(),
        shuffled_y[train_indices],
        steps=steps,
        lr=lr,
        seed=seed + 1,
    )
    floor_scores = floor_probe.scores(latents[eval_indices].float()).tolist()
    floor_labels = [shuffled[index] for index in eval_indices]
    return ProbeFit(
        probe=probe,
        auroc=binary_auroc(eval_labels, scores),
        shuffled_floor_auroc=binary_auroc(floor_labels, floor_scores),
        train_groups=tuple(sorted(train_groups)),
        eval_groups=tuple(sorted(eval_groups)),
    )


def probe_verifier(probe: LinearProbe) -> ProvenanceVerifier:
    def verify(proposal: LatticeProposal, context: object) -> SoundnessType:
        latent = proposal.metadata.get("latent")
        if latent is None and isinstance(context, Mapping):
            latent = context.get("latent")
        if latent is None:
            return SoundnessType.UNKNOWN
        return (
            SoundnessType.ENV_SOUND_DEAD
            if probe.predict(latent)
            else SoundnessType.UNKNOWN
        )

    return verify


def direction_cosine(left: Tensor, right: Tensor) -> float:
    denominator = float(left.norm().item() * right.norm().item())
    if denominator == 0:
        return 0.0
    return float(torch.dot(left.flatten(), right.flatten()).item() / denominator)


def matched_control_directions(
    direction: Tensor,
    *,
    seed: int,
    random_count: int = 8,
    orthogonal_count: int = 8,
) -> dict[str, list[Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    norm = direction.norm().clamp_min(1e-12)
    unit = direction / norm
    random_directions = []
    orthogonal_directions = []
    for _ in range(random_count):
        value = torch.randn(direction.shape, generator=generator)
        random_directions.append(value / value.norm().clamp_min(1e-12) * norm)
    for _ in range(orthogonal_count):
        value = torch.randn(direction.shape, generator=generator)
        value = value - torch.dot(value.flatten(), unit.flatten()) * unit
        orthogonal_directions.append(value / value.norm().clamp_min(1e-12) * norm)
    return {"random": random_directions, "orthogonal": orthogonal_directions}
