from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


LANES = ("train", "hold", "reject")
FORM_RE = re.compile(r"^\(([^()\s]+)(?:\s+(.*?))?\)$")
KNOWN_FORMS = {
    "benchmark",
    "constitution",
    "critical-cap",
    "dimension",
    "tag-score",
    "critical-tag",
    "ablation",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


@dataclass(frozen=True)
class Policy:
    benchmark_id: str
    constitution_id: str
    critical_cap: float
    dimensions: Mapping[str, float]
    tag_scores: Mapping[str, tuple[str, float]]
    critical_tags: frozenset[str]
    source_path: str
    source_sha256: str

    @property
    def tags(self) -> frozenset[str]:
        return frozenset(self.tag_scores)

    def signs_by_dimension(self) -> dict[str, dict[str, tuple[str, ...]]]:
        grouped: dict[str, dict[str, list[str]]] = {
            dimension: {"positive": [], "negative": []}
            for dimension in self.dimensions
        }
        for tag, (dimension, value) in self.tag_scores.items():
            grouped[dimension]["positive" if value > 0 else "negative"].append(tag)
        return {
            dimension: {
                sign: tuple(sorted(tags))
                for sign, tags in signs.items()
            }
            for dimension, signs in grouped.items()
        }


def parse_policy(path: Path) -> Policy:
    benchmark_ids: list[str] = []
    constitution_ids: list[str] = []
    critical_caps: list[float] = []
    dimensions: dict[str, float] = {}
    tag_scores: dict[str, tuple[str, float]] = {}
    critical_tags: set[str] = set()

    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        match = FORM_RE.fullmatch(line)
        if not match:
            raise ValueError(f"{path}:{line_number}: unsupported MeTTa syntax")
        form = match.group(1)
        if form not in KNOWN_FORMS:
            raise ValueError(f"{path}:{line_number}: unknown form {form!r}")
        tokens = (match.group(2) or "").split()
        if form == "benchmark":
            if len(tokens) != 1:
                raise ValueError(f"{path}:{line_number}: benchmark expects one id")
            benchmark_ids.append(tokens[0])
        elif form == "constitution":
            if len(tokens) != 1:
                raise ValueError(f"{path}:{line_number}: constitution expects one id")
            constitution_ids.append(tokens[0])
        elif form == "critical-cap":
            if len(tokens) != 1:
                raise ValueError(f"{path}:{line_number}: critical-cap expects one number")
            critical_caps.append(float(tokens[0]))
        elif form == "dimension":
            if len(tokens) != 2 or tokens[0] in dimensions:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate dimension")
            dimensions[tokens[0]] = float(tokens[1])
        elif form == "tag-score":
            if len(tokens) != 3 or tokens[0] in tag_scores:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate tag-score")
            tag_scores[tokens[0]] = (tokens[1], float(tokens[2]))
        elif form == "critical-tag":
            if len(tokens) != 1:
                raise ValueError(f"{path}:{line_number}: critical-tag expects one tag")
            critical_tags.add(tokens[0])
        elif form == "ablation":
            if len(tokens) < 2:
                raise ValueError(f"{path}:{line_number}: ablation expects an id and tags")

    if len(benchmark_ids) != 1 or len(constitution_ids) != 1 or len(critical_caps) != 1:
        raise ValueError(f"{path}: policy requires one benchmark, constitution, and cap")
    if not dimensions or abs(sum(dimensions.values()) - 1.0) > 1e-9:
        raise ValueError(f"{path}: dimension weights must sum to one")
    for tag, (dimension, value) in tag_scores.items():
        if dimension not in dimensions:
            raise ValueError(f"{path}: tag {tag!r} uses unknown dimension")
        if value not in {-1.0, 1.0}:
            raise ValueError(f"{path}: tag {tag!r} score must be -1 or 1")
    if not critical_tags.issubset(tag_scores):
        raise ValueError(f"{path}: critical tags must have tag-score facts")

    return Policy(
        benchmark_id=benchmark_ids[0],
        constitution_id=constitution_ids[0],
        critical_cap=critical_caps[0],
        dimensions=dimensions,
        tag_scores=tag_scores,
        critical_tags=frozenset(critical_tags),
        source_path=path.as_posix(),
        source_sha256=sha256_file(path),
    )


def score_policy(
    tags: Sequence[str],
    policy: Policy,
    *,
    allowed_universe: frozenset[str],
    train_minimum: float,
    reject_maximum: float,
) -> dict[str, Any]:
    if len(tags) != len(set(tags)):
        raise ValueError("candidate tags must be unique")
    unknown = sorted(set(tags).difference(allowed_universe))
    if unknown:
        raise ValueError(f"candidate uses unknown tags: {unknown}")
    relevant = sorted(set(tags).intersection(policy.tags))
    if not relevant:
        return {
            "score": 0.5,
            "lane": "hold",
            "relevant_tags": [],
            "critical_hits": [],
            "annotation_conflicts": [],
            "route_soundness": "unknown",
            "route_reason": "no_frame_relevant_audited_tags",
        }

    by_dimension: dict[str, list[float]] = {key: [] for key in policy.dimensions}
    for tag in relevant:
        dimension, value = policy.tag_scores[tag]
        by_dimension[dimension].append(value)
    conflicts = sorted(
        dimension
        for dimension, values in by_dimension.items()
        if any(value > 0 for value in values) and any(value < 0 for value in values)
    )
    if conflicts:
        return {
            "score": 0.5,
            "lane": "hold",
            "relevant_tags": relevant,
            "critical_hits": sorted(set(relevant).intersection(policy.critical_tags)),
            "annotation_conflicts": conflicts,
            "route_soundness": "unknown",
            "route_reason": "contradictory_dimension_annotations",
        }

    dimension_scores = {
        dimension: max(-1.0, min(1.0, sum(values)))
        for dimension, values in by_dimension.items()
    }
    weighted_raw = sum(
        policy.dimensions[dimension] * value
        for dimension, value in dimension_scores.items()
    )
    score = (weighted_raw + 1.0) / 2.0
    critical_hits = sorted(set(relevant).intersection(policy.critical_tags))
    if critical_hits:
        score = min(score, policy.critical_cap)
        lane = "reject"
        reason = "critical_tag"
    elif score >= train_minimum:
        lane = "train"
        reason = "score_at_or_above_train_boundary"
    elif score <= reject_maximum:
        lane = "reject"
        reason = "score_at_or_below_reject_boundary"
    else:
        lane = "hold"
        reason = "score_inside_review_band"
    return {
        "score": round(score, 6),
        "lane": lane,
        "relevant_tags": relevant,
        "critical_hits": critical_hits,
        "annotation_conflicts": [],
        "dimension_scores": {
            key: round(value, 6) for key, value in dimension_scores.items()
        },
        "route_soundness": "environment_sound_within_frozen_policy",
        "route_reason": reason,
    }


def _sample_tags(policy: Policy, rng: random.Random) -> set[str]:
    tags: set[str] = set()
    for signs in policy.signs_by_dimension().values():
        state = rng.randrange(3)
        if state == 1:
            tags.add(rng.choice(signs["positive"]))
        elif state == 2:
            tags.add(rng.choice(signs["negative"]))
    return tags


def build_joint_balanced_dataset(
    policies: Mapping[str, Policy],
    *,
    seed: int,
    split_counts: Mapping[str, int],
    train_minimum: float,
    reject_maximum: float,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    if set(policies) != {"jinn", "beast"}:
        raise ValueError("the frozen design requires exactly jinn and beast policies")
    target = sum(split_counts.values())
    if target <= 0:
        raise ValueError("split counts must be positive")
    feature_names = tuple(sorted(set().union(*(policy.tags for policy in policies.values()))))
    universe = frozenset(feature_names)
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], dict[str, dict[str, Any]]] = {
        (jinn_lane, beast_lane): {}
        for jinn_lane in LANES
        for beast_lane in LANES
    }

    attempts = 0
    while any(len(rows) < target for rows in buckets.values()):
        attempts += 1
        if attempts > 2_000_000:
            missing = {str(key): target - len(rows) for key, rows in buckets.items() if len(rows) < target}
            raise RuntimeError(f"could not fill joint-label cells: {missing}")
        tags = sorted(_sample_tags(policies["jinn"], rng) | _sample_tags(policies["beast"], rng))
        if not tags:
            continue
        scored = {
            alias: score_policy(
                tags,
                policy,
                allowed_universe=universe,
                train_minimum=train_minimum,
                reject_maximum=reject_maximum,
            )
            for alias, policy in policies.items()
        }
        if any(value["annotation_conflicts"] for value in scored.values()):
            continue
        key = (scored["jinn"]["lane"], scored["beast"]["lane"])
        signature = sha256_bytes(canonical_json(tags).encode("utf-8"))
        if len(buckets[key]) >= target or signature in buckets[key]:
            continue
        buckets[key][signature] = {
            "row_id": f"mb-{signature[:16]}",
            "schema_version": "two_frame_tagged_candidate_v1",
            "tags": tags,
            "labels": scored,
            "joint_label": {"jinn": key[0], "beast": key[1]},
        }

    rows: list[dict[str, Any]] = []
    split_order = tuple(split_counts)
    for key in sorted(buckets):
        ordered = [buckets[key][signature] for signature in sorted(buckets[key])]
        offset = 0
        for split in split_order:
            count = split_counts[split]
            for row in ordered[offset : offset + count]:
                row["split"] = split
                rows.append(row)
            offset += count
    rows.sort(key=lambda row: row["row_id"])
    return rows, feature_names


def vectorize(tags: Sequence[str], feature_names: Sequence[str]) -> list[float]:
    active = set(tags)
    return [float(name in active) for name in feature_names]


@dataclass
class AdditiveCandidateHead:
    feature_names: tuple[str, ...]
    weights: list[list[float]]
    bias: list[float]

    @classmethod
    def initialized(cls, feature_names: Sequence[str], seed: int) -> "AdditiveCandidateHead":
        rng = random.Random(seed)
        weights = [
            [rng.uniform(-0.01, 0.01) for _ in feature_names]
            for _ in LANES
        ]
        return cls(tuple(feature_names), weights, [0.0 for _ in LANES])

    def probabilities(self, features: Sequence[float]) -> list[float]:
        logits = [
            self.bias[class_index]
            + sum(weight * value for weight, value in zip(self.weights[class_index], features))
            for class_index in range(len(LANES))
        ]
        maximum = max(logits)
        exponents = [math.exp(value - maximum) for value in logits]
        total = sum(exponents)
        return [value / total for value in exponents]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema_version": "additive_candidate_lattice_head_v1",
            "feature_names": list(self.feature_names),
            "classes": list(LANES),
            "weights": self.weights,
            "bias": self.bias,
            "proposal_soundness": "model_sound",
        }

    @classmethod
    def from_jsonable(cls, value: Mapping[str, Any]) -> "AdditiveCandidateHead":
        if tuple(value["classes"]) != LANES:
            raise ValueError("model classes do not match the frozen lane order")
        return cls(
            tuple(map(str, value["feature_names"])),
            [[float(item) for item in row] for row in value["weights"]],
            [float(item) for item in value["bias"]],
        )


def train_head(
    rows: Sequence[Mapping[str, Any]],
    *,
    frame: str,
    feature_names: Sequence[str],
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    l2: float,
    checkpoint_interval: int,
    checkpoint_callback: Callable[[int, AdditiveCandidateHead, float], None] | None = None,
    epoch_callback: Callable[[int, float], None] | None = None,
) -> AdditiveCandidateHead:
    training = [row for row in rows if row["split"] == "train"]
    if not training:
        raise ValueError("training split is empty")
    model = AdditiveCandidateHead.initialized(feature_names, seed)
    rng = random.Random(seed + 1)
    lane_index = {lane: index for index, lane in enumerate(LANES)}
    indices = list(range(len(training)))

    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        epoch_loss = 0.0
        observed = 0
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            grad_w = [[0.0 for _ in feature_names] for _ in LANES]
            grad_b = [0.0 for _ in LANES]
            for row_index in batch_indices:
                row = training[row_index]
                features = vectorize(row["tags"], feature_names)
                target = lane_index[row["labels"][frame]["lane"]]
                probabilities = model.probabilities(features)
                epoch_loss -= math.log(max(probabilities[target], 1e-12))
                observed += 1
                for class_index in range(len(LANES)):
                    error = probabilities[class_index] - float(class_index == target)
                    grad_b[class_index] += error
                    for feature_index, value in enumerate(features):
                        grad_w[class_index][feature_index] += error * value
            scale = 1.0 / len(batch_indices)
            for class_index in range(len(LANES)):
                model.bias[class_index] -= learning_rate * grad_b[class_index] * scale
                for feature_index in range(len(feature_names)):
                    gradient = (
                        grad_w[class_index][feature_index] * scale
                        + l2 * model.weights[class_index][feature_index]
                    )
                    model.weights[class_index][feature_index] -= learning_rate * gradient
        mean_loss = epoch_loss / observed
        if epoch_callback is not None:
            epoch_callback(epoch, mean_loss)
        if checkpoint_callback is not None and (
            epoch % checkpoint_interval == 0 or epoch == epochs
        ):
            checkpoint_callback(epoch, model, mean_loss)
    return model


def candidate_lanes(
    probabilities: Sequence[float],
    *,
    singleton_threshold: float,
    otherwise_top_k: int,
) -> list[str]:
    ordered = sorted(range(len(LANES)), key=lambda index: (-probabilities[index], LANES[index]))
    if probabilities[ordered[0]] >= singleton_threshold:
        return [LANES[ordered[0]]]
    return [LANES[index] for index in ordered[:otherwise_top_k]]


def evaluate_head(
    model: AdditiveCandidateHead,
    rows: Sequence[Mapping[str, Any]],
    *,
    frame: str,
    policies: Mapping[str, Policy],
    split: str,
    singleton_threshold: float,
    otherwise_top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"split {split!r} is empty")
    opposite = "beast" if frame == "jinn" else "jinn"
    own_tags = policies[frame].tags
    confusion = {truth: {prediction: 0 for prediction in LANES} for truth in LANES}
    correct = 0
    covered = 0
    singleton = 0
    critical_total = 0
    critical_rejected = 0
    nuisance_stable = 0
    nuisance_l1: list[float] = []
    predictions: list[dict[str, Any]] = []

    for row in selected:
        features = vectorize(row["tags"], model.feature_names)
        probabilities = model.probabilities(features)
        prediction = LANES[max(range(len(LANES)), key=lambda index: probabilities[index])]
        truth = row["labels"][frame]["lane"]
        candidates = candidate_lanes(
            probabilities,
            singleton_threshold=singleton_threshold,
            otherwise_top_k=otherwise_top_k,
        )
        confusion[truth][prediction] += 1
        correct += int(prediction == truth)
        covered += int(truth in candidates)
        singleton += int(len(candidates) == 1)
        if row["labels"][frame]["critical_hits"]:
            critical_total += 1
            critical_rejected += int(prediction == "reject")

        own_only_tags = [tag for tag in row["tags"] if tag in own_tags]
        own_probabilities = model.probabilities(vectorize(own_only_tags, model.feature_names))
        own_prediction = LANES[
            max(range(len(LANES)), key=lambda index: own_probabilities[index])
        ]
        nuisance_stable += int(own_prediction == prediction)
        nuisance_l1.append(
            sum(abs(left - right) for left, right in zip(probabilities, own_probabilities))
        )
        predictions.append(
            {
                "row_id": row["row_id"],
                "frame": frame,
                "split": split,
                "truth_lane": truth,
                "top_lane": prediction,
                "candidate_lanes": candidates,
                "probabilities": {
                    lane: round(probabilities[index], 8)
                    for index, lane in enumerate(LANES)
                },
                "proposal_soundness": "model_sound",
                "hard_applied": False,
                "exact_policy_score": row["labels"][frame]["score"],
                "exact_policy_route_soundness": row["labels"][frame]["route_soundness"],
            }
        )

    f1_values = []
    for lane in LANES:
        tp = confusion[lane][lane]
        fp = sum(confusion[truth][lane] for truth in LANES if truth != lane)
        fn = sum(confusion[lane][prediction] for prediction in LANES if prediction != lane)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    own_indices = {
        index for index, name in enumerate(model.feature_names) if name in own_tags
    }
    total_weight = sum(abs(value) for row in model.weights for value in row)
    irrelevant_weight = sum(
        abs(value)
        for row in model.weights
        for index, value in enumerate(row)
        if index not in own_indices
    )
    count = len(selected)
    metrics = {
        "frame": frame,
        "opposite_frame": opposite,
        "split": split,
        "rows": count,
        "accuracy": round(correct / count, 8),
        "macro_f1": round(sum(f1_values) / len(f1_values), 8),
        "candidate_coverage": round(covered / count, 8),
        "false_elimination_rate": round(1.0 - covered / count, 8),
        "singleton_rate": round(singleton / count, 8),
        "critical_reject_recall": round(
            critical_rejected / critical_total if critical_total else 1.0,
            8,
        ),
        "critical_rows": critical_total,
        "opposite_frame_top_lane_invariance": round(nuisance_stable / count, 8),
        "mean_opposite_frame_probability_l1": round(sum(nuisance_l1) / count, 8),
        "irrelevant_weight_fraction": round(
            irrelevant_weight / total_weight if total_weight else 0.0,
            8,
        ),
        "confusion": confusion,
    }
    return metrics, predictions


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows
