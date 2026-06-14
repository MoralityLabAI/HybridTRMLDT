from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


Grid = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ArcExample:
    input_grid: Grid
    output_grid: Grid


@dataclass(frozen=True)
class ArcTask:
    task_id: str
    train: tuple[ArcExample, ...]
    test_input: Grid
    test_output: Grid
    rule_name: str
    rule_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArcRunResult:
    solver: str
    task_id: str
    solved: bool
    predicted: Grid
    expected: Grid
    steps: int
    proposals: int
    rejected: int
    trace: list[str] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "solver": self.solver,
            "task_id": self.task_id,
            "solved": self.solved,
            "predicted": [list(row) for row in self.predicted],
            "expected": [list(row) for row in self.expected],
            "steps": self.steps,
            "proposals": self.proposals,
            "rejected": self.rejected,
            "trace": list(self.trace),
        }


def grid(rows: list[list[int]]) -> Grid:
    return tuple(tuple(row) for row in rows)


def map_color(g: Grid, src: int, dst: int) -> Grid:
    return tuple(tuple(dst if value == src else value for value in row) for row in g)


def flip_h(g: Grid) -> Grid:
    return tuple(tuple(reversed(row)) for row in g)


def flip_v(g: Grid) -> Grid:
    return tuple(reversed(g))


def rotate_180(g: Grid) -> Grid:
    return flip_v(flip_h(g))


def fill_zero_with(g: Grid, value: int) -> Grid:
    return tuple(tuple(value if cell == 0 else cell for cell in row) for row in g)


@dataclass(frozen=True)
class ArcRule:
    name: str
    apply: Callable[[Grid], Grid]
    family: str


def primitive_rules() -> list[ArcRule]:
    return [
        ArcRule("map_1_to_2", lambda g: map_color(g, 1, 2), "color_map"),
        ArcRule("map_2_to_3", lambda g: map_color(g, 2, 3), "color_map"),
        ArcRule("fill_zero_with_1", lambda g: fill_zero_with(g, 1), "fill"),
        ArcRule("fill_zero_with_2", lambda g: fill_zero_with(g, 2), "fill"),
        ArcRule("flip_h", flip_h, "spatial"),
        ArcRule("flip_v", flip_v, "spatial"),
        ArcRule("rotate_180", rotate_180, "spatial"),
    ]


def default_arc1_tasks() -> list[ArcTask]:
    return [
        ArcTask(
            task_id="arc1_color_map",
            train=(
                ArcExample(grid([[1, 0], [0, 1]]), grid([[2, 0], [0, 2]])),
                ArcExample(grid([[1, 1], [0, 0]]), grid([[2, 2], [0, 0]])),
            ),
            test_input=grid([[0, 1], [1, 0]]),
            test_output=grid([[0, 2], [2, 0]]),
            rule_name="map_1_to_2",
        ),
        ArcTask(
            task_id="arc1_horizontal_flip",
            train=(
                ArcExample(grid([[1, 0, 2], [0, 3, 0]]), grid([[2, 0, 1], [0, 3, 0]])),
                ArcExample(grid([[4, 0, 0], [1, 2, 3]]), grid([[0, 0, 4], [3, 2, 1]])),
            ),
            test_input=grid([[7, 0, 8], [5, 6, 0]]),
            test_output=grid([[8, 0, 7], [0, 6, 5]]),
            rule_name="flip_h",
        ),
        ArcTask(
            task_id="arc1_fill_background",
            train=(
                ArcExample(grid([[0, 3], [3, 0]]), grid([[1, 3], [3, 1]])),
                ArcExample(grid([[0, 0], [2, 0]]), grid([[1, 1], [2, 1]])),
            ),
            test_input=grid([[4, 0], [0, 4]]),
            test_output=grid([[4, 1], [1, 4]]),
            rule_name="fill_zero_with_1",
        ),
    ]


def default_arc2_tasks() -> list[ArcTask]:
    return [
        ArcTask(
            task_id="arc2_flip_then_color",
            train=(
                ArcExample(grid([[1, 0, 3], [0, 1, 2]]), grid([[3, 0, 2], [2, 2, 0]])),
                ArcExample(grid([[0, 1, 4], [2, 0, 1]]), grid([[4, 2, 0], [2, 0, 2]])),
            ),
            test_input=grid([[1, 3, 0], [0, 2, 1]]),
            test_output=grid([[0, 3, 2], [2, 2, 0]]),
            rule_name="flip_h+map_1_to_2",
            rule_names=("flip_h", "map_1_to_2"),
        ),
        ArcTask(
            task_id="arc2_fill_then_rotate",
            train=(
                ArcExample(grid([[0, 2], [3, 0]]), grid([[1, 3], [2, 1]])),
                ArcExample(grid([[4, 0], [0, 2]]), grid([[2, 1], [1, 4]])),
            ),
            test_input=grid([[0, 5], [6, 0]]),
            test_output=grid([[1, 6], [5, 1]]),
            rule_name="fill_zero_with_1+rotate_180",
            rule_names=("fill_zero_with_1", "rotate_180"),
        ),
        ArcTask(
            task_id="arc2_color_then_flip",
            train=(
                ArcExample(grid([[2, 0, 2], [1, 0, 0]]), grid([[3, 0, 3], [0, 0, 1]])),
                ArcExample(grid([[0, 2, 1], [2, 0, 0]]), grid([[1, 3, 0], [0, 0, 3]])),
            ),
            test_input=grid([[2, 1, 0], [0, 2, 0]]),
            test_output=grid([[0, 1, 3], [0, 3, 0]]),
            rule_name="map_2_to_3+flip_h",
            rule_names=("map_2_to_3", "flip_h"),
        ),
    ]


def rule_fits(rule: ArcRule, task: ArcTask) -> bool:
    return all(rule.apply(example.input_grid) == example.output_grid for example in task.train)


def compose_rules(first: ArcRule, second: ArcRule) -> ArcRule:
    return ArcRule(
        name=f"{first.name}+{second.name}",
        apply=lambda g: second.apply(first.apply(g)),
        family=f"{first.family}+{second.family}",
    )


def primitive_rule_pairs() -> list[ArcRule]:
    rules = primitive_rules()
    return [compose_rules(first, second) for first in rules for second in rules]
