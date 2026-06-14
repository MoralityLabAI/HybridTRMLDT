from __future__ import annotations

from collections import defaultdict

from research_gym.envs.sudoku import (
    Grid,
    SudokuPuzzle,
    SudokuRunResult,
    assign,
    candidates,
    default_sudoku_puzzles,
    is_consistent,
    is_solved,
    propagate_singles,
    select_mrv_cell,
)


def solve_ldt(puzzle: SudokuPuzzle) -> SudokuRunResult:
    grid, steps, conflict, trace = propagate_singles(puzzle.givens)
    solved = grid == puzzle.solution and is_solved(grid)
    return SudokuRunResult(
        solver="ldt",
        puzzle_id=puzzle.puzzle_id,
        solved=solved,
        valid=is_consistent(grid) and not conflict,
        steps=steps,
        guesses=0,
        conflicts=1 if conflict else 0,
        final_grid=grid,
        trace=trace,
    )


def _trm_search(grid: Grid, solution: Grid, *, steps: int, guesses: int, conflicts: int, trace: list[str]) -> tuple[Grid, int, int, int, list[str]]:
    if not is_consistent(grid):
        return grid, steps, guesses, conflicts + 1, trace + ["conflict:duplicate"]
    if is_solved(grid):
        return grid, steps, guesses, conflicts, trace

    cell = select_mrv_cell(grid)
    if cell is None:
        return grid, steps, guesses, conflicts, trace
    row, col = cell
    options = sorted(candidates(grid, row, col))
    if not options:
        return grid, steps, guesses, conflicts + 1, trace + [f"conflict:no_candidates:{row},{col}"]

    for value in options:
        guessed = assign(grid, row, col, value)
        next_trace = trace + [f"guess:{row},{col}={value}"]
        result, next_steps, next_guesses, next_conflicts, solved_trace = _trm_search(
            guessed,
            solution,
            steps=steps + 1,
            guesses=guesses + 1,
            conflicts=conflicts,
            trace=next_trace,
        )
        if result == solution and is_solved(result):
            return result, next_steps, next_guesses, next_conflicts, solved_trace
        conflicts = next_conflicts + 1
    return grid, steps + len(options), guesses + len(options), conflicts, trace


def solve_trm(puzzle: SudokuPuzzle) -> SudokuRunResult:
    grid, steps, guesses, conflicts, trace = _trm_search(
        puzzle.givens,
        puzzle.solution,
        steps=0,
        guesses=0,
        conflicts=0,
        trace=[],
    )
    solved = grid == puzzle.solution and is_solved(grid)
    return SudokuRunResult(
        solver="trm",
        puzzle_id=puzzle.puzzle_id,
        solved=solved,
        valid=is_consistent(grid),
        steps=steps,
        guesses=guesses,
        conflicts=conflicts,
        final_grid=grid,
        trace=trace,
    )


def _hybrid_search(grid: Grid, solution: Grid, *, steps: int, guesses: int, conflicts: int, trace: list[str]) -> tuple[Grid, int, int, int, list[str]]:
    grid, propagated_steps, conflict, propagation_trace = propagate_singles(grid)
    steps += propagated_steps
    trace = trace + propagation_trace
    if conflict:
        return grid, steps, guesses, conflicts + 1, trace
    if is_solved(grid):
        return grid, steps, guesses, conflicts, trace

    cell = select_mrv_cell(grid)
    if cell is None:
        return grid, steps, guesses, conflicts, trace
    row, col = cell
    options = sorted(candidates(grid, row, col))
    for value in options:
        proposal = assign(grid, row, col, value)
        if not is_consistent(proposal):
            conflicts += 1
            trace.append(f"reject:{row},{col}={value}")
            continue
        result, next_steps, next_guesses, next_conflicts, solved_trace = _hybrid_search(
            proposal,
            solution,
            steps=steps + 1,
            guesses=guesses + 1,
            conflicts=conflicts,
            trace=trace + [f"propose:{row},{col}={value}"],
        )
        if result == solution and is_solved(result):
            return result, next_steps, next_guesses, next_conflicts, solved_trace
        conflicts = next_conflicts + 1
    return grid, steps + len(options), guesses + len(options), conflicts, trace


def solve_hybrid(puzzle: SudokuPuzzle) -> SudokuRunResult:
    grid, steps, guesses, conflicts, trace = _hybrid_search(
        puzzle.givens,
        puzzle.solution,
        steps=0,
        guesses=0,
        conflicts=0,
        trace=[],
    )
    solved = grid == puzzle.solution and is_solved(grid)
    return SudokuRunResult(
        solver="hybrid",
        puzzle_id=puzzle.puzzle_id,
        solved=solved,
        valid=is_consistent(grid),
        steps=steps,
        guesses=guesses,
        conflicts=conflicts,
        final_grid=grid,
        trace=trace,
    )


def run_sudoku_benchmark(puzzles: list[SudokuPuzzle] | None = None) -> list[SudokuRunResult]:
    puzzles = puzzles or default_sudoku_puzzles()
    results: list[SudokuRunResult] = []
    for puzzle in puzzles:
        results.extend([solve_ldt(puzzle), solve_trm(puzzle), solve_hybrid(puzzle)])
    return results


def summarize_results(results: list[SudokuRunResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[SudokuRunResult]] = defaultdict(list)
    for result in results:
        grouped[result.solver].append(result)
    summary: dict[str, dict[str, float]] = {}
    for solver, items in sorted(grouped.items()):
        count = len(items)
        summary[solver] = {
            "puzzles": float(count),
            "solved": float(sum(item.solved for item in items)),
            "solve_rate": sum(item.solved for item in items) / count if count else 0.0,
            "steps": float(sum(item.steps for item in items)),
            "guesses": float(sum(item.guesses for item in items)),
            "conflicts": float(sum(item.conflicts for item in items)),
        }
    return summary


def summary_markdown(results: list[SudokuRunResult]) -> str:
    summary = summarize_results(results)
    lines = [
        "# Sudoku Benchmark",
        "",
        "4x4 Sudoku micro-benchmark comparing LDT propagation, TRM heuristic search, and hybrid proposal plus certification.",
        "",
        "| Solver | Solved | Solve Rate | Steps | Guesses | Conflicts |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for solver, metrics in summary.items():
        lines.append(
            f"| `{solver}` | {int(metrics['solved'])}/{int(metrics['puzzles'])} | "
            f"{metrics['solve_rate']:.3f} | {int(metrics['steps'])} | "
            f"{int(metrics['guesses'])} | {int(metrics['conflicts'])} |"
        )
    lines.extend(["", "## Per Puzzle", ""])
    for result in results:
        lines.append(
            f"- `{result.puzzle_id}` `{result.solver}` solved={result.solved} "
            f"steps={result.steps} guesses={result.guesses} conflicts={result.conflicts}"
        )
    return "\n".join(lines) + "\n"
