from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_TESSERACT_DATA_ROOT = Path("D:/Research_Engine/tesseract_persistent/data")
DEFAULT_ROUTING_ENVS = (
    "alphabet_sort",
    "arc_challenge",
    "arc_easy",
    "gsm8k",
    "intellect_3_logic",
    "intellect_3_math",
    "mbpp",
    "wiki_search",
)


@dataclass(frozen=True)
class RoutingExample:
    prompt: str
    env_id: str
    source: str = "synthetic"


@dataclass(frozen=True)
class RoutingRunResult:
    router: str
    accuracy: float
    correct: int
    total: int
    abstained: int
    avg_candidates: float
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "router": self.router,
            "accuracy": self.accuracy,
            "correct": self.correct,
            "total": self.total,
            "abstained": self.abstained,
            "avg_candidates": self.avg_candidates,
            "confusion": self.confusion,
        }


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+|\d+", text.lower())


def load_tesseract_routing_examples(
    data_root: Path = DEFAULT_TESSERACT_DATA_ROOT,
    envs: tuple[str, ...] = DEFAULT_ROUTING_ENVS,
    *,
    max_per_env: int = 80,
) -> list[RoutingExample]:
    data_dir = data_root / "normalized_trajectories"
    examples: list[RoutingExample] = []
    for env_id in envs:
        path = data_dir / f"{env_id}.jsonl"
        if not path.exists():
            continue
        count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if count >= max_per_env:
                    break
                row = json.loads(line)
                prompt = row.get("state_prompt")
                if isinstance(prompt, str) and prompt.strip():
                    examples.append(RoutingExample(prompt=prompt, env_id=env_id, source=str(path)))
                    count += 1
    return examples


def split_examples(
    examples: list[RoutingExample],
    *,
    train_ratio: float = 0.7,
    seed: int = 7,
) -> tuple[list[RoutingExample], list[RoutingExample]]:
    rng = random.Random(seed)
    grouped: dict[str, list[RoutingExample]] = {}
    for example in examples:
        grouped.setdefault(example.env_id, []).append(example)

    train: list[RoutingExample] = []
    test: list[RoutingExample] = []
    for env_examples in grouped.values():
        shuffled = list(env_examples)
        rng.shuffle(shuffled)
        split_at = max(1, int(len(shuffled) * train_ratio))
        train.extend(shuffled[:split_at])
        test.extend(shuffled[split_at:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test
