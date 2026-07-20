"""Canonical schedule topologies and bounded grammar discovery for LSPG v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import math
from typing import Iterable, Iterator, Mapping, Sequence

from .canonical import digest


def canonical_module_word(word: Iterable[int]) -> tuple[int, ...]:
    """Relabel modules by first appearance so cosmetic labels share an identity."""

    labels: dict[int, int] = {}
    canonical: list[int] = []
    for value in word:
        value = int(value)
        if value < 0:
            raise ValueError("module labels must be non-negative")
        if value not in labels:
            labels[value] = len(labels)
        canonical.append(labels[value])
    if not canonical:
        raise ValueError("schedule word must be non-empty")
    return tuple(canonical)


def minimal_period(word: Sequence[int]) -> int:
    if not word:
        raise ValueError("schedule word must be non-empty")
    for period in range(1, len(word) + 1):
        if all(word[index] == word[index % period] for index in range(len(word))):
            return period
    return len(word)


def repeated_prefix(word: Sequence[int], visits: int) -> tuple[int, ...]:
    if not word:
        raise ValueError("schedule word must be non-empty")
    if visits <= 0:
        raise ValueError("visits must be positive")
    return tuple(word[index % len(word)] for index in range(visits))


def _entropy(values: Iterable[int]) -> float:
    counts: dict[int, int] = {}
    total = 0
    for value in values:
        counts[value] = counts.get(value, 0) + 1
        total += 1
    if total <= 1 or len(counts) <= 1:
        return 0.0
    entropy = -sum(
        (count / total) * math.log(count / total) for count in counts.values()
    )
    return entropy / math.log(len(counts))


def _jacobi_eigenvalues(matrix: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Return eigenvalues of a tiny symmetric matrix without a NumPy dependency."""

    values = [list(map(float, row)) for row in matrix]
    size = len(values)
    if any(len(row) != size for row in values):
        raise ValueError("matrix must be square")
    for _ in range(64):
        pivot = max(
            ((abs(values[i][j]), i, j) for i in range(size) for j in range(i + 1, size)),
            default=(0.0, 0, 0),
        )
        magnitude, p, q = pivot
        if magnitude < 1e-12:
            break
        angle = 0.5 * math.atan2(
            2.0 * values[p][q], values[q][q] - values[p][p]
        )
        cosine = math.cos(angle)
        sine = math.sin(angle)
        app = values[p][p]
        aqq = values[q][q]
        apq = values[p][q]
        values[p][p] = cosine * cosine * app - 2 * sine * cosine * apq + sine * sine * aqq
        values[q][q] = sine * sine * app + 2 * sine * cosine * apq + cosine * cosine * aqq
        values[p][q] = values[q][p] = 0.0
        for index in range(size):
            if index in {p, q}:
                continue
            aip = values[index][p]
            aiq = values[index][q]
            values[index][p] = values[p][index] = cosine * aip - sine * aiq
            values[index][q] = values[q][index] = sine * aip + cosine * aiq
    return tuple(sorted(values[index][index] for index in range(size)))


def transition_spectral_gap(word: Sequence[int]) -> float:
    modules = max(word) + 1
    if modules <= 1:
        return 0.0
    adjacency = [[0.0 for _ in range(modules)] for _ in range(modules)]
    cyclic = tuple(word) + (word[0],)
    for left, right in zip(cyclic, cyclic[1:]):
        adjacency[left][right] += 1.0
        adjacency[right][left] += 1.0
    degrees = [sum(row) for row in adjacency]
    laplacian = [[0.0 for _ in range(modules)] for _ in range(modules)]
    for row in range(modules):
        laplacian[row][row] = 1.0
        for column in range(modules):
            if adjacency[row][column] and degrees[row] and degrees[column]:
                laplacian[row][column] -= adjacency[row][column] / math.sqrt(
                    degrees[row] * degrees[column]
                )
    eigenvalues = _jacobi_eigenvalues(laplacian)
    return max(0.0, min(2.0, eigenvalues[1]))


@dataclass(frozen=True)
class ScheduleDescriptors:
    transition_entropy: float
    transition_spectral_gap: float
    adjacent_repeat_fraction: float
    return_gap_cv: float
    reversal_symmetry: float
    boundary_repeat: float
    module_balance: float

    def vector(self) -> tuple[float, ...]:
        return tuple(float(value) for value in asdict(self).values())


@dataclass(frozen=True)
class ScheduleTopology:
    macro_word: tuple[int, ...]
    physical_modules: int
    train_visits: int
    inference_visits: tuple[int, ...]
    topology_kind: str = "candidate"
    schema_version: int = 1

    def __post_init__(self) -> None:
        canonical = canonical_module_word(self.macro_word)
        if canonical != self.macro_word:
            raise ValueError("macro_word must use canonical first-appearance labels")
        if self.physical_modules != max(canonical) + 1:
            raise ValueError("physical_modules must match macro_word labels")
        if self.train_visits <= 0:
            raise ValueError("train_visits must be positive")
        if any(value <= 0 for value in self.inference_visits):
            raise ValueError("inference visits must be positive")

    @property
    def expanded_word(self) -> tuple[int, ...]:
        return repeated_prefix(self.macro_word, self.train_visits)

    @property
    def topology_hash(self) -> str:
        return digest(self.to_dict())

    @property
    def descriptors(self) -> ScheduleDescriptors:
        return schedule_descriptors(self.expanded_word)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "macro_word": list(self.macro_word),
            "physical_modules": self.physical_modules,
            "train_visits": self.train_visits,
            "inference_visits": list(self.inference_visits),
            "topology_kind": self.topology_kind,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ScheduleTopology":
        return cls(
            macro_word=tuple(int(item) for item in value["macro_word"]),  # type: ignore[arg-type]
            physical_modules=int(value["physical_modules"]),
            train_visits=int(value.get("train_visits", len(value["macro_word"]))),  # type: ignore[arg-type]
            inference_visits=tuple(
                int(item) for item in value.get("inference_visits", (value.get("train_visits"),))  # type: ignore[arg-type]
            ),
            topology_kind=str(value.get("topology_kind", "candidate")),
            schema_version=int(value.get("schema_version", 1)),
        )


def schedule_descriptors(word: Sequence[int]) -> ScheduleDescriptors:
    word = canonical_module_word(word)
    modules = max(word) + 1
    transitions = [left * modules + right for left, right in zip(word, word[1:] + word[:1])]
    adjacent_repeats = sum(left == right for left, right in zip(word, word[1:]))
    positions: dict[int, list[int]] = {module: [] for module in range(modules)}
    for index, module in enumerate(word):
        positions[module].append(index)
    gaps: list[float] = []
    for indexes in positions.values():
        cyclic_indexes = indexes + [indexes[0] + len(word)]
        gaps.extend(float(right - left) for left, right in zip(cyclic_indexes, cyclic_indexes[1:]))
    gap_mean = sum(gaps) / len(gaps)
    gap_variance = sum((gap - gap_mean) ** 2 for gap in gaps) / len(gaps)
    reversed_word = canonical_module_word(reversed(word))
    reversal_matches = sum(left == right for left, right in zip(word, reversed_word))
    counts = [len(positions[module]) for module in range(modules)]
    balance = 1.0 - ((max(counts) - min(counts)) / max(counts))
    return ScheduleDescriptors(
        transition_entropy=_entropy(transitions),
        transition_spectral_gap=transition_spectral_gap(word),
        adjacent_repeat_fraction=adjacent_repeats / max(1, len(word) - 1),
        return_gap_cv=math.sqrt(gap_variance) / gap_mean if gap_mean else 0.0,
        reversal_symmetry=reversal_matches / len(word),
        boundary_repeat=float(word[-1] == word[0]),
        module_balance=balance,
    )


def _restricted_growth_words(length: int, modules: int) -> Iterator[tuple[int, ...]]:
    if length <= 0 or modules <= 0 or modules > length:
        return
    word = [0]

    def visit(position: int, maximum: int) -> Iterator[tuple[int, ...]]:
        if position == length:
            if maximum + 1 == modules:
                yield tuple(word)
            return
        for value in range(min(maximum + 1, modules - 1) + 1):
            word.append(value)
            yield from visit(position + 1, max(maximum, value))
            word.pop()

    yield from visit(1, 0)


def is_known_schedule_pattern(word: Sequence[int]) -> bool:
    canonical = canonical_module_word(word)
    modules = max(canonical) + 1
    if modules == 1 or modules == len(canonical):
        return True
    if minimal_period(canonical) < len(canonical):
        return True
    periodic = tuple(index % modules for index in range(len(canonical)))
    if canonical == canonical_module_word(periodic):
        return True
    return False


def enumerate_schedule_grammar(
    *, modules: Sequence[int] = (2, 3, 4), lengths: Sequence[int] = (6, 8)
) -> tuple[ScheduleTopology, ...]:
    output: list[ScheduleTopology] = []
    for module_count, length in product(modules, lengths):
        for word in _restricted_growth_words(length, module_count):
            counts = [word.count(module) for module in range(module_count)]
            if max(counts) - min(counts) > 1:
                continue
            if is_known_schedule_pattern(word):
                continue
            output.append(
                ScheduleTopology(
                    macro_word=word,
                    physical_modules=module_count,
                    train_visits=length,
                    inference_visits=tuple(sorted({max(1, length // 2), length, length + length // 2, length * 2})),
                )
            )
    output.sort(key=lambda value: value.topology_hash)
    return tuple(output)


def periodic_control(modules: int, length: int) -> ScheduleTopology:
    word = canonical_module_word(index % modules for index in range(length))
    return ScheduleTopology(
        macro_word=word,
        physical_modules=modules,
        train_visits=length,
        inference_visits=tuple(sorted({max(1, length // 2), length, length + length // 2, length * 2})),
        topology_kind="fixed_periodic_control",
    )


def _standardized_vectors(
    topologies: Sequence[ScheduleTopology],
) -> dict[str, tuple[float, ...]]:
    raw = [topology.descriptors.vector() for topology in topologies]
    dimensions = len(raw[0]) if raw else 0
    means = [sum(row[index] for row in raw) / len(raw) for index in range(dimensions)]
    scales = []
    for index in range(dimensions):
        variance = sum((row[index] - means[index]) ** 2 for row in raw) / len(raw)
        scales.append(math.sqrt(variance) or 1.0)
    return {
        topology.topology_hash: tuple(
            (value - means[index]) / scales[index] for index, value in enumerate(row)
        )
        for topology, row in zip(topologies, raw)
    }


def topology_distance(left: ScheduleTopology, right: ScheduleTopology) -> float:
    vectors = _standardized_vectors((left, right))
    left_vector = vectors[left.topology_hash]
    right_vector = vectors[right.topology_hash]
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left_vector, right_vector)))


def select_discovery_batches(
    pool: Sequence[ScheduleTopology], *, per_stratum: int = 4
) -> tuple[tuple[ScheduleTopology, ...], tuple[ScheduleTopology, ...]]:
    """Choose two candidates per stratum for each of two presealed batches."""

    if per_stratum != 4:
        raise ValueError("v1 freezes four selected schedules per (K,L) stratum")
    vectors = _standardized_vectors(pool)
    first: list[ScheduleTopology] = []
    reserve: list[ScheduleTopology] = []
    strata = sorted({(item.physical_modules, item.train_visits) for item in pool})
    for modules, length in strata:
        candidates = [
            item
            for item in pool
            if item.physical_modules == modules and item.train_visits == length
        ]
        if len(candidates) < per_stratum:
            raise ValueError(f"not enough candidates in stratum {(modules, length)}")
        control = periodic_control(modules, length)
        comparison = tuple(candidates) + (control,)
        local_vectors = _standardized_vectors(comparison)
        selected: list[ScheduleTopology] = []
        while len(selected) < per_stratum:
            def score(candidate: ScheduleTopology) -> tuple[float, str]:
                anchors = selected or [control]
                distance = min(
                    math.sqrt(
                        sum(
                            (a - b) ** 2
                            for a, b in zip(
                                local_vectors[candidate.topology_hash],
                                local_vectors[anchor.topology_hash],
                            )
                        )
                    )
                    for anchor in anchors
                )
                global_magnitude = math.sqrt(
                    sum(value * value for value in vectors[candidate.topology_hash])
                )
                return distance + 1e-6 * global_magnitude, candidate.topology_hash

            choice = max(
                (item for item in candidates if item not in selected), key=score
            )
            selected.append(choice)
        first.extend(selected[:2])
        reserve.extend(selected[2:])
    first.sort(key=lambda value: (value.physical_modules, value.train_visits, value.topology_hash))
    reserve.sort(key=lambda value: (value.physical_modules, value.train_visits, value.topology_hash))
    return tuple(first), tuple(reserve)


def random_schedule_null(
    pool: Sequence[ScheduleTopology], *, modules: int, length: int
) -> ScheduleTopology:
    candidates = [
        item for item in pool if item.physical_modules == modules and item.train_visits == length
    ]
    if not candidates:
        raise ValueError("schedule-null stratum is empty")
    selected = min(candidates, key=lambda item: digest({"null": item.topology_hash}))
    return ScheduleTopology(
        macro_word=selected.macro_word,
        physical_modules=modules,
        train_visits=length,
        inference_visits=selected.inference_visits,
        topology_kind="balanced_schedule_null",
    )
