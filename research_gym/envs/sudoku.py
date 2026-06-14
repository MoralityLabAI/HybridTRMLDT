from __future__ import annotations

from dataclasses import dataclass, field
from math import isqrt
from typing import Iterable


Grid = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class SudokuPuzzle:
    """Small square Sudoku puzzle. Zero denotes an empty cell."""

    puzzle_id: str
    givens: Grid
    solution: Grid

    @property
    def size(self) -> int:
        return len(self.givens)

    @property
    def box_size(self) -> int:
        return isqrt(self.size)


@dataclass(frozen=True)
class SudokuRunResult:
    solver: str
    puzzle_id: str
    solved: bool
    valid: bool
    steps: int
    guesses: int
    conflicts: int
    final_grid: Grid
    trace: list[str] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "solver": self.solver,
            "puzzle_id": self.puzzle_id,
            "solved": self.solved,
            "valid": self.valid,
            "steps": self.steps,
            "guesses": self.guesses,
            "conflicts": self.conflicts,
            "final_grid": [list(row) for row in self.final_grid],
            "trace": list(self.trace),
        }


def grid_from_rows(rows: Iterable[Iterable[int]]) -> Grid:
    return tuple(tuple(int(value) for value in row) for row in rows)


def empty_cells(grid: Grid) -> list[tuple[int, int]]:
    return [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == 0]


def peers(size: int, row: int, col: int) -> set[tuple[int, int]]:
    box_size = isqrt(size)
    box_row = row // box_size
    box_col = col // box_size
    out = {(row, c) for c in range(size) if c != col}
    out.update((r, col) for r in range(size) if r != row)
    out.update(
        (r, c)
        for r in range(box_row * box_size, (box_row + 1) * box_size)
        for c in range(box_col * box_size, (box_col + 1) * box_size)
        if (r, c) != (row, col)
    )
    return out


def candidates(grid: Grid, row: int, col: int) -> set[int]:
    if grid[row][col] != 0:
        return {grid[row][col]}
    size = len(grid)
    used = {grid[r][c] for r, c in peers(size, row, col) if grid[r][c] != 0}
    return set(range(1, size + 1)) - used


def assign(grid: Grid, row: int, col: int, value: int) -> Grid:
    rows = [list(r) for r in grid]
    rows[row][col] = value
    return grid_from_rows(rows)


def is_consistent(grid: Grid) -> bool:
    size = len(grid)
    box_size = isqrt(size)
    expected_len = size

    def ok(values: list[int]) -> bool:
        filled = [value for value in values if value != 0]
        return len(filled) == len(set(filled))

    if any(len(row) != expected_len for row in grid):
        return False
    for row in grid:
        if not ok(list(row)):
            return False
    for col in range(size):
        if not ok([grid[row][col] for row in range(size)]):
            return False
    for box_row in range(0, size, box_size):
        for box_col in range(0, size, box_size):
            values = [
                grid[r][c]
                for r in range(box_row, box_row + box_size)
                for c in range(box_col, box_col + box_size)
            ]
            if not ok(values):
                return False
    return True


def is_solved(grid: Grid) -> bool:
    return is_consistent(grid) and all(value != 0 for row in grid for value in row)


def propagate_singles(grid: Grid) -> tuple[Grid, int, bool, list[str]]:
    """LDT-style monotone candidate elimination via naked singles."""

    steps = 0
    trace: list[str] = []
    changed = True
    while changed:
        changed = False
        if not is_consistent(grid):
            return grid, steps, True, trace + ["conflict:duplicate"]
        for row, col in empty_cells(grid):
            cell_candidates = candidates(grid, row, col)
            if not cell_candidates:
                return grid, steps, True, trace + [f"conflict:no_candidates:{row},{col}"]
            if len(cell_candidates) == 1:
                value = next(iter(cell_candidates))
                grid = assign(grid, row, col, value)
                steps += 1
                changed = True
                trace.append(f"single:{row},{col}={value}")
                break
    return grid, steps, False, trace


def select_mrv_cell(grid: Grid) -> tuple[int, int] | None:
    cells = empty_cells(grid)
    if not cells:
        return None
    return min(cells, key=lambda cell: (len(candidates(grid, cell[0], cell[1])), cell[0], cell[1]))


def default_sudoku_puzzles() -> list[SudokuPuzzle]:
    return [
        SudokuPuzzle(
            puzzle_id="s4_single_chain",
            givens=grid_from_rows(
                [
                    [1, 0, 0, 4],
                    [0, 4, 1, 0],
                    [2, 0, 4, 3],
                    [0, 3, 0, 1],
                ]
            ),
            solution=grid_from_rows(
                [
                    [1, 2, 3, 4],
                    [3, 4, 1, 2],
                    [2, 1, 4, 3],
                    [4, 3, 2, 1],
                ]
            ),
        ),
        SudokuPuzzle(
            puzzle_id="s4_requires_search",
            givens=grid_from_rows(
                [
                    [0, 0, 0, 4],
                    [0, 4, 0, 0],
                    [0, 0, 4, 0],
                    [4, 0, 0, 0],
                ]
            ),
            solution=grid_from_rows(
                [
                    [1, 2, 3, 4],
                    [3, 4, 1, 2],
                    [2, 1, 4, 3],
                    [4, 3, 2, 1],
                ]
            ),
        ),
        SudokuPuzzle(
            puzzle_id="s4_sparse",
            givens=grid_from_rows(
                [
                    [0, 0, 3, 0],
                    [0, 4, 0, 0],
                    [0, 0, 0, 3],
                    [4, 0, 0, 0],
                ]
            ),
            solution=grid_from_rows(
                [
                    [1, 2, 3, 4],
                    [3, 4, 1, 2],
                    [2, 1, 4, 3],
                    [4, 3, 2, 1],
                ]
            ),
        ),
    ]
