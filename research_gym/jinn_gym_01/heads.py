"""Compact CPU persona heads trained on verified examples.

Head = the v1 additive three-lane candidate head (`ldt_experiment.AdditiveCandidateHead`,
vendored verbatim). Every head sees every tag in ALL_TAGS (character_ldt_heads_v1 rule),
including boundary tags; the boundary itself is never trained and stays in host.py.

Training rows are "verified examples": (host-audited tag set, persona rubric lane) pairs
taken only from development tasks whose stored utilities passed `verify_task`, plus seeded
compositional tag sets, labeled by the compiled persona rubric parsed with the pinned
`parse_policy` / `score_policy`.

Training uses a NumPy implementation of exactly the vendored `train_head` update rule
(same initialization, shuffle order, minibatch gradient, L2 term). `train_reference` calls
the vendored stdlib trainer; tests check the two agree to 1e-9 on a small case.
"""
from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

import numpy as np

from ._vendor.ldt_experiment import LANES, AdditiveCandidateHead, train_head, vectorize
from .tags import ALL_TAGS, PERSONA_TAGS, BOUNDARY_TAGS
from .tasks import Task, verify_task

FEATURES = ALL_TAGS
EPOCHS, BATCH, LR, L2 = 40, 64, 0.08, 0.0005
N_COMPOSITIONAL = 1200


class VetoedHead:
    """heads_v3.VetoHead semantics: a critical tag in the input gives lanes [0, 0, 1]."""

    def __init__(self, base: AdditiveCandidateHead, veto_tags: frozenset):
        missing = set(veto_tags) - set(base.feature_names)
        if missing:
            raise ValueError(f"veto tags are not features: {sorted(missing)}")
        self.base, self.veto_tags = base, frozenset(veto_tags)
        self.feature_names = base.feature_names

    def probabilities(self, features: Sequence[float]) -> list[float]:
        if any(v and n in self.veto_tags for n, v in zip(self.feature_names, features)):
            return [0.0, 0.0, 1.0]
        return self.base.probabilities(features)

    def to_jsonable(self) -> dict:
        v = self.base.to_jsonable()
        v["schema_version"] = "additive_candidate_lattice_head_with_critical_veto_v1"
        v["critical_veto_tags"] = sorted(self.veto_tags)
        return v


def compositional_tagsets(seed: int, n: int = N_COMPOSITIONAL) -> list[list[str]]:
    rng = random.Random(seed)
    pool = list(PERSONA_TAGS) + list(BOUNDARY_TAGS)
    return [sorted(rng.sample(pool, rng.randint(1, 5))) for _ in range(n)]


def training_rows(rubric_lane, tasks: Sequence[Task], seed: int) -> list[dict]:
    """rubric_lane(tags) -> lane. Only verified tasks contribute task rows."""
    rows, seen = [], set()
    for t in tasks:
        if not verify_task(t):
            continue
        for a in t.actions:
            key = tuple(sorted(a.tags))
            if key in seen:
                continue
            seen.add(key)
            rows.append({"tags": list(key), "split": "train", "labels": {"persona": {"lane": rubric_lane(key)}},
                         "source": "task", "task_id": t.task_id})
    for tags in compositional_tagsets(seed):
        rows.append({"tags": tags, "split": "train", "labels": {"persona": {"lane": rubric_lane(tags)}},
                     "source": "compositional"})
    return rows


def train_numpy(rows: Sequence[Mapping[str, Any]], seed: int, epochs: int = EPOCHS, batch_size: int = BATCH,
                learning_rate: float = LR, l2: float = L2, feature_names: Sequence[str] = FEATURES,
                frame: str = "persona") -> AdditiveCandidateHead:
    """Same arithmetic as ldt_experiment.train_head, vectorized."""
    training = [r for r in rows if r["split"] == "train"]
    if not training:
        raise ValueError("training split is empty")
    model = AdditiveCandidateHead.initialized(feature_names, seed)
    W = np.array(model.weights, dtype=float)
    b = np.array(model.bias, dtype=float)
    X = np.array([vectorize(r["tags"], feature_names) for r in training], dtype=float)
    lane_index = {lane: i for i, lane in enumerate(LANES)}
    y = np.array([lane_index[r["labels"][frame]["lane"]] for r in training])
    Y = np.eye(len(LANES))[y]
    rng = random.Random(seed + 1)
    indices = list(range(len(training)))
    for _ in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            bi = indices[start:start + batch_size]
            xb, yb = X[bi], Y[bi]
            logits = xb @ W.T + b
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            err = p - yb
            scale = 1.0 / len(bi)
            b -= learning_rate * err.sum(axis=0) * scale
            W -= learning_rate * (err.T @ xb * scale + l2 * W)
    model.weights = W.tolist()
    model.bias = b.tolist()
    return model


def train_reference(rows, seed, epochs=EPOCHS, batch_size=BATCH, learning_rate=LR, l2=L2,
                    feature_names=FEATURES) -> AdditiveCandidateHead:
    return train_head(rows, frame="persona", feature_names=feature_names, seed=seed, epochs=epochs,
                      batch_size=batch_size, learning_rate=learning_rate, l2=l2, checkpoint_interval=10 ** 9)


def head_agreement(head, rubric_lane, rows: Sequence[Mapping]) -> float:
    hit = 0
    for r in rows:
        p = head.probabilities(vectorize(r["tags"], head.feature_names))
        hit += LANES[max(range(3), key=lambda i: p[i])] == rubric_lane(r["tags"])
    return hit / max(1, len(rows))


def load_head(value: Mapping) -> Any:
    base = AdditiveCandidateHead.from_jsonable(value)
    if "critical_veto_tags" in value:
        return VetoedHead(base, frozenset(value["critical_veto_tags"]))
    return base


def train_log_prob(head, tags) -> float:
    p = head.probabilities(vectorize(tags, head.feature_names))
    return math.log(max(p[0], 1e-6))
