"""Deterministic five-family task bundle for loop-schedule discovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

from lsa.canonical import digest
from research_gym.envs.routing import (
    DEFAULT_ROUTING_ENVS,
    DEFAULT_TESSERACT_DATA_ROOT,
    load_tesseract_routing_examples,
)


PAD_TOKEN = 0
BOS_TOKEN = 1
ANSWER_TOKEN = 2
TASK_TOKENS = {
    "pointer_chase": 8,
    "modular_recurrence": 9,
    "rewrite_normalization": 10,
    "sudoku": 11,
    "routing": 12,
}
BYTE_BASE = 32
NODE_BASE = 300
NUMBER_BASE = 340
SYMBOL_BASE = 400
SUDOKU_INPUT_BASE = 440
POINTER_TARGET_BASE = 800
MODULAR_TARGET_BASE = 832
REWRITE_TARGET_BASE = 896
SUDOKU_TARGET_BASE = 928
ROUTING_TARGET_BASE = 960


DEFAULT_COUNTS: Mapping[str, Mapping[str, int]] = {
    "pointer_chase": {"train": 2048, "calibration": 256, "evaluation": 512},
    "modular_recurrence": {"train": 2048, "calibration": 256, "evaluation": 512},
    "rewrite_normalization": {"train": 2048, "calibration": 256, "evaluation": 512},
    "sudoku": {"train": 1024, "calibration": 128, "evaluation": 256},
}


@dataclass(frozen=True)
class TaskExample:
    example_id: str
    fingerprint: str
    family: str
    split: str
    tokens: tuple[int, ...]
    target_token: int
    difficulty: int
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.family not in TASK_TOKENS:
            raise ValueError(f"unknown task family: {self.family}")
        if self.split not in {"train", "calibration", "evaluation"}:
            raise ValueError(f"unknown split: {self.split}")
        if not self.tokens or self.tokens[-1] != ANSWER_TOKEN:
            raise ValueError("task sequences must end in the answer token")
        if self.target_token < 0 or self.target_token >= 2048:
            raise ValueError("target token must fit the registered vocabulary")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["tokens"] = list(self.tokens)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskExample":
        return cls(
            example_id=str(value["example_id"]),
            fingerprint=str(value["fingerprint"]),
            family=str(value["family"]),
            split=str(value["split"]),
            tokens=tuple(int(token) for token in value["tokens"]),
            target_token=int(value["target_token"]),
            difficulty=int(value["difficulty"]),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class TaskBundle:
    examples: tuple[TaskExample, ...]
    sequence_length: int
    vocabulary_size: int
    source_files: Mapping[str, str]

    def split(self, name: str, families: Iterable[str] | None = None) -> tuple[TaskExample, ...]:
        allowed = set(families) if families is not None else None
        return tuple(
            example
            for example in self.examples
            if example.split == name and (allowed is None or example.family in allowed)
        )

    def validate(self) -> None:
        ids = [example.example_id for example in self.examples]
        if len(ids) != len(set(ids)):
            raise ValueError("task example IDs overlap")
        seen_by_split: dict[str, set[str]] = {}
        for example in self.examples:
            if len(example.tokens) != self.sequence_length:
                raise ValueError("task sequence length mismatch")
            seen_by_split.setdefault(example.split, set()).add(example.fingerprint)
        names = sorted(seen_by_split)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap = seen_by_split[left] & seen_by_split[right]
                if overlap:
                    raise ValueError(f"task split overlap between {left} and {right}")

    def manifest(self) -> dict[str, Any]:
        self.validate()
        counts: dict[str, dict[str, int]] = {}
        split_hashes: dict[str, str] = {}
        for split in ("train", "calibration", "evaluation"):
            rows = sorted(
                (example.to_dict() for example in self.split(split)),
                key=lambda row: row["example_id"],
            )
            split_hashes[split] = digest(rows)
            for row in rows:
                family = str(row["family"])
                counts.setdefault(family, {})[split] = counts.setdefault(family, {}).get(split, 0) + 1
        return {
            "schema_version": 1,
            "sequence_length": self.sequence_length,
            "vocabulary_size": self.vocabulary_size,
            "counts": counts,
            "split_hashes": split_hashes,
            "source_files": dict(sorted(self.source_files.items())),
            "bundle_hash": digest(
                {
                    "sequence_length": self.sequence_length,
                    "vocabulary_size": self.vocabulary_size,
                    "split_hashes": split_hashes,
                    "source_files": dict(sorted(self.source_files.items())),
                }
            ),
        }


def _pack(family: str, payload: Sequence[int], sequence_length: int) -> tuple[int, ...]:
    if sequence_length < 8:
        raise ValueError("sequence_length is too short")
    prefix = [BOS_TOKEN, TASK_TOKENS[family]]
    available = sequence_length - len(prefix) - 1
    values = prefix + [int(token) for token in payload[:available]]
    values.extend([PAD_TOKEN] * (sequence_length - len(values) - 1))
    values.append(ANSWER_TOKEN)
    return tuple(values)


def _make_example(
    *,
    family: str,
    split: str,
    payload: Sequence[int],
    target_token: int,
    difficulty: int,
    semantic: Mapping[str, Any],
    sequence_length: int,
    metadata: Mapping[str, Any] | None = None,
) -> TaskExample:
    fingerprint = digest({"family": family, "semantic": semantic})
    return TaskExample(
        example_id=f"{family}-{split}-{fingerprint[:16]}",
        fingerprint=fingerprint,
        family=family,
        split=split,
        tokens=_pack(family, payload, sequence_length),
        target_token=target_token,
        difficulty=difficulty,
        metadata={**dict(metadata or {}), "semantic_hash": digest(semantic)},
    )


def _rng(seed: int, family: str, split: str, trial: int) -> random.Random:
    value = int(digest({"seed": seed, "family": family, "split": split, "trial": trial})[:16], 16)
    return random.Random(value)


def _depth_range(split: str) -> tuple[int, int]:
    return {
        "train": (1, 4),
        "calibration": (5, 6),
        "evaluation": (7, 8),
    }[split]


def _pointer_example(
    *, split: str, seed: int, trial: int, sequence_length: int
) -> TaskExample:
    rng = _rng(seed, "pointer_chase", split, trial)
    nodes = 8
    mapping = list(range(nodes))
    rng.shuffle(mapping)
    start = rng.randrange(nodes)
    low, high = _depth_range(split)
    hops = rng.randint(low, high)
    target = start
    for _ in range(hops):
        target = mapping[target]
    payload = [16]
    for source, destination in enumerate(mapping):
        payload.extend((NODE_BASE + source, NODE_BASE + destination))
    payload.extend((17, NODE_BASE + start, 18, NUMBER_BASE + hops))
    semantic = {"mapping": mapping, "start": start, "hops": hops}
    return _make_example(
        family="pointer_chase",
        split=split,
        payload=payload,
        target_token=POINTER_TARGET_BASE + target,
        difficulty=hops,
        semantic=semantic,
        sequence_length=sequence_length,
        metadata={"hops": hops},
    )


def _modular_example(
    *, split: str, seed: int, trial: int, sequence_length: int
) -> TaskExample:
    rng = _rng(seed, "modular_recurrence", split, trial)
    modulus = 17
    low, high = _depth_range(split)
    steps = rng.randint(low, high)
    start = rng.randrange(modulus)
    operations = [(rng.randrange(1, modulus), rng.randrange(modulus)) for _ in range(steps)]
    target = start
    for multiplier, offset in operations:
        target = (multiplier * target + offset) % modulus
    payload = [19, NUMBER_BASE + start, 20, NUMBER_BASE + modulus]
    for multiplier, offset in operations:
        payload.extend((21, NUMBER_BASE + multiplier, NUMBER_BASE + offset))
    semantic = {"modulus": modulus, "start": start, "operations": operations}
    return _make_example(
        family="modular_recurrence",
        split=split,
        payload=payload,
        target_token=MODULAR_TARGET_BASE + target,
        difficulty=steps,
        semantic=semantic,
        sequence_length=sequence_length,
        metadata={"steps": steps},
    )


def _rewrite_example(
    *, split: str, seed: int, trial: int, sequence_length: int
) -> TaskExample:
    rng = _rng(seed, "rewrite_normalization", split, trial)
    alphabet = 6
    width = 8
    low, high = _depth_range(split)
    steps = rng.randint(low, high)
    state = [rng.randrange(alphabet) for _ in range(width)]
    rules = []
    for _ in range(3):
        left = (rng.randrange(alphabet), rng.randrange(alphabet))
        right = (rng.randrange(alphabet), rng.randrange(alphabet))
        rules.append((left, right))
    original = list(state)
    applied = 0
    for _ in range(steps):
        changed = False
        for position in range(width - 1):
            pair = (state[position], state[position + 1])
            for left, right in rules:
                if pair == left:
                    state[position : position + 2] = right
                    changed = True
                    applied += 1
                    break
            if changed:
                break
    query = rng.randrange(width)
    payload = [22]
    for left, right in rules:
        payload.extend(
            (SYMBOL_BASE + left[0], SYMBOL_BASE + left[1], 23, SYMBOL_BASE + right[0], SYMBOL_BASE + right[1])
        )
    payload.append(24)
    payload.extend(SYMBOL_BASE + value for value in original)
    payload.extend((25, NUMBER_BASE + steps, 26, NUMBER_BASE + query))
    semantic = {"rules": rules, "state": original, "steps": steps, "query": query}
    return _make_example(
        family="rewrite_normalization",
        split=split,
        payload=payload,
        target_token=REWRITE_TARGET_BASE + state[query],
        difficulty=steps,
        semantic=semantic,
        sequence_length=sequence_length,
        metadata={"steps": steps, "applied_rewrites": applied, "query": query},
    )


_BASE_SUDOKU = (
    (1, 2, 3, 4),
    (3, 4, 1, 2),
    (2, 1, 4, 3),
    (4, 3, 2, 1),
)


def _sudoku_solution(rng: random.Random) -> tuple[tuple[int, ...], ...]:
    digits = [1, 2, 3, 4]
    rng.shuffle(digits)
    bands = [[0, 1], [2, 3]]
    stacks = [[0, 1], [2, 3]]
    rng.shuffle(bands)
    rng.shuffle(stacks)
    for group in bands:
        rng.shuffle(group)
    for group in stacks:
        rng.shuffle(group)
    rows = [value for group in bands for value in group]
    columns = [value for group in stacks for value in group]
    return tuple(
        tuple(digits[_BASE_SUDOKU[row][column] - 1] for column in columns)
        for row in rows
    )


def _sudoku_candidates(grid: Sequence[Sequence[int]], row: int, column: int) -> tuple[int, ...]:
    used = set(grid[row])
    used.update(grid[index][column] for index in range(4))
    box_row = (row // 2) * 2
    box_column = (column // 2) * 2
    used.update(
        grid[r][c]
        for r in range(box_row, box_row + 2)
        for c in range(box_column, box_column + 2)
    )
    return tuple(value for value in range(1, 5) if value not in used)


def _solution_count(grid: Sequence[Sequence[int]], *, limit: int = 2) -> int:
    mutable = [list(row) for row in grid]
    total = 0

    def search() -> None:
        nonlocal total
        if total >= limit:
            return
        empty = [
            (len(_sudoku_candidates(mutable, row, column)), row, column)
            for row in range(4)
            for column in range(4)
            if mutable[row][column] == 0
        ]
        if not empty:
            total += 1
            return
        _, row, column = min(empty)
        for value in _sudoku_candidates(mutable, row, column):
            mutable[row][column] = value
            search()
            mutable[row][column] = 0

    search()
    return total


def _sudoku_example(
    *, split: str, seed: int, trial: int, sequence_length: int
) -> TaskExample | None:
    rng = _rng(seed, "sudoku", split, trial)
    solution = _sudoku_solution(rng)
    clue_range = {
        "train": (8, 12),
        "calibration": (7, 9),
        "evaluation": (6, 8),
    }[split]
    clue_count = rng.randint(*clue_range)
    positions = list(range(16))
    rng.shuffle(positions)
    clues = set(positions[:clue_count])
    puzzle = tuple(
        tuple(solution[row][column] if row * 4 + column in clues else 0 for column in range(4))
        for row in range(4)
    )
    if _solution_count(puzzle) != 1:
        return None
    blanks = [position for position in range(16) if position not in clues]
    query = rng.choice(blanks)
    payload = [27]
    payload.extend(SUDOKU_INPUT_BASE + value for row in puzzle for value in row)
    payload.extend((28, NUMBER_BASE + query))
    target = solution[query // 4][query % 4]
    puzzle_id = digest({"puzzle": puzzle, "solution": solution})[:16]
    semantic = {"puzzle": puzzle, "query": query}
    return _make_example(
        family="sudoku",
        split=split,
        payload=payload,
        target_token=SUDOKU_TARGET_BASE + target,
        difficulty=16 - clue_count,
        semantic=semantic,
        sequence_length=sequence_length,
        metadata={
            "puzzle_id": puzzle_id,
            "query": query,
            "clues": clue_count,
            "solution": [list(row) for row in solution],
        },
    )


def _generate_unique(
    family: str,
    split: str,
    count: int,
    *,
    seed: int,
    sequence_length: int,
    forbidden: set[str],
) -> list[TaskExample]:
    factory = {
        "pointer_chase": _pointer_example,
        "modular_recurrence": _modular_example,
        "rewrite_normalization": _rewrite_example,
        "sudoku": _sudoku_example,
    }[family]
    output: list[TaskExample] = []
    trial = 0
    max_trials = max(10_000, count * 200)
    while len(output) < count and trial < max_trials:
        example = factory(split=split, seed=seed, trial=trial, sequence_length=sequence_length)
        trial += 1
        if example is None or example.fingerprint in forbidden:
            continue
        forbidden.add(example.fingerprint)
        output.append(example)
    if len(output) != count:
        raise RuntimeError(f"could not generate {count} unique {family}/{split} examples")
    return output


def _routing_payload(text: str, available: int) -> list[int]:
    encoded = [BYTE_BASE + value for value in text.encode("utf-8", errors="replace")]
    if len(encoded) <= available:
        return encoded
    left = available // 2
    return encoded[:left] + encoded[-(available - left) :]


def _routing_examples(
    *, routing_root: Path, max_per_env: int, sequence_length: int
) -> tuple[list[TaskExample], dict[str, str]]:
    loaded = load_tesseract_routing_examples(
        routing_root, DEFAULT_ROUTING_ENVS, max_per_env=max_per_env
    )
    grouped: dict[str, list[Any]] = {env: [] for env in DEFAULT_ROUTING_ENVS}
    for example in loaded:
        grouped.setdefault(example.env_id, []).append(example)
    output: list[TaskExample] = []
    sources: dict[str, str] = {}
    for env_index, env in enumerate(DEFAULT_ROUTING_ENVS):
        deduplicated = {
            hashlib.sha256(item.prompt.encode("utf-8")).hexdigest(): item
            for item in grouped.get(env, [])
        }
        examples = [deduplicated[key] for key in sorted(deduplicated)]
        if len(examples) < 3:
            raise RuntimeError(f"routing environment {env} needs at least three examples")
        train_end = max(1, int(len(examples) * 0.6))
        calibration_end = max(train_end + 1, int(len(examples) * 0.8))
        calibration_end = min(calibration_end, len(examples) - 1)
        source_paths = sorted({str(item.source) for item in examples})
        for source in source_paths:
            path = Path(source)
            if path.exists():
                sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for index, example in enumerate(examples):
            split = "train" if index < train_end else "calibration" if index < calibration_end else "evaluation"
            source_hash = hashlib.sha256(example.prompt.encode("utf-8")).hexdigest()
            payload = _routing_payload(example.prompt, sequence_length - 3)
            output.append(
                _make_example(
                    family="routing",
                    split=split,
                    payload=payload,
                    target_token=ROUTING_TARGET_BASE + env_index,
                    difficulty=len(example.prompt.encode("utf-8")),
                    semantic={"prompt_sha256": source_hash, "env_id": env},
                    sequence_length=sequence_length,
                    metadata={
                        "env_id": env,
                        "source": str(example.source),
                        "source_sha256": source_hash,
                        "source_text": example.prompt,
                    },
                )
            )
    return output, sources


def build_task_bundle(
    *,
    seed: int = 503,
    sequence_length: int = 64,
    counts: Mapping[str, Mapping[str, int]] = DEFAULT_COUNTS,
    routing_root: Path = DEFAULT_TESSERACT_DATA_ROOT,
    routing_max_per_env: int = 80,
) -> TaskBundle:
    examples: list[TaskExample] = []
    forbidden: set[str] = set()
    for family in (
        "pointer_chase",
        "modular_recurrence",
        "rewrite_normalization",
        "sudoku",
    ):
        for split in ("train", "calibration", "evaluation"):
            examples.extend(
                _generate_unique(
                    family,
                    split,
                    int(counts[family][split]),
                    seed=seed,
                    sequence_length=sequence_length,
                    forbidden=forbidden,
                )
            )
    routing, source_files = _routing_examples(
        routing_root=routing_root,
        max_per_env=routing_max_per_env,
        sequence_length=sequence_length,
    )
    examples.extend(routing)
    bundle = TaskBundle(
        examples=tuple(sorted(examples, key=lambda value: value.example_id)),
        sequence_length=sequence_length,
        vocabulary_size=2048,
        source_files=source_files,
    )
    bundle.validate()
    return bundle


def write_task_bundle(bundle: TaskBundle, directory: Path) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    for split in ("train", "calibration", "evaluation"):
        path = directory / f"{split}.jsonl"
        rows = sorted(bundle.split(split), key=lambda value: value.example_id)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for example in rows:
                handle.write(json.dumps(example.to_dict(), sort_keys=True, separators=(",", ":")))
                handle.write("\n")
    manifest = bundle.manifest()
    manifest["files"] = {
        split: {
            "path": f"{split}.jsonl",
            "sha256": hashlib.sha256((directory / f"{split}.jsonl").read_bytes()).hexdigest(),
        }
        for split in ("train", "calibration", "evaluation")
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def read_task_bundle(directory: Path) -> TaskBundle:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    examples: list[TaskExample] = []
    for split in ("train", "calibration", "evaluation"):
        path = directory / manifest["files"][split]["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != manifest["files"][split]["sha256"]:
            raise ValueError(f"task shard hash mismatch: {split}")
        for line in path.read_text(encoding="utf-8").splitlines():
            examples.append(TaskExample.from_mapping(json.loads(line)))
    bundle = TaskBundle(
        examples=tuple(examples),
        sequence_length=int(manifest["sequence_length"]),
        vocabulary_size=int(manifest["vocabulary_size"]),
        source_files=dict(manifest.get("source_files", {})),
    )
    bundle.validate()
    if bundle.manifest()["bundle_hash"] != manifest["bundle_hash"]:
        raise ValueError("task bundle manifest hash mismatch")
    return bundle
