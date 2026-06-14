from research_gym.benchmarks.sudoku_bench import run_sudoku_benchmark, solve_hybrid, solve_ldt, solve_trm
from research_gym.envs.sudoku import candidates, default_sudoku_puzzles, is_consistent, is_solved, propagate_singles


def test_sudoku_candidates_and_consistency():
    puzzle = default_sudoku_puzzles()[0]

    assert is_consistent(puzzle.givens)
    assert is_solved(puzzle.solution)
    assert candidates(puzzle.givens, 0, 1) == {2}


def test_ldt_solves_single_chain_with_propagation():
    puzzle = default_sudoku_puzzles()[0]
    result = solve_ldt(puzzle)

    assert result.solved
    assert result.guesses == 0
    assert result.final_grid == puzzle.solution


def test_ldt_does_not_solve_search_puzzle_without_guessing():
    puzzle = default_sudoku_puzzles()[1]
    result = solve_ldt(puzzle)

    assert not result.solved
    assert result.valid
    assert result.guesses == 0


def test_trm_and_hybrid_solve_all_default_puzzles():
    for puzzle in default_sudoku_puzzles():
        trm = solve_trm(puzzle)
        hybrid = solve_hybrid(puzzle)

        assert trm.solved
        assert hybrid.solved
        assert trm.final_grid == puzzle.solution
        assert hybrid.final_grid == puzzle.solution


def test_hybrid_uses_propagation_trace():
    puzzle = default_sudoku_puzzles()[0]
    result = solve_hybrid(puzzle)

    assert result.solved
    assert any(item.startswith("single:") for item in result.trace)


def test_propagate_singles_reports_conflict():
    grid = ((1, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))

    _, _, conflict, trace = propagate_singles(grid)

    assert conflict
    assert trace == ["conflict:duplicate"]


def test_run_sudoku_benchmark_shape():
    results = run_sudoku_benchmark()

    assert len(results) == 9
    assert {result.solver for result in results} == {"ldt", "trm", "hybrid"}
    assert {result.puzzle_id for result in results} == {puzzle.puzzle_id for puzzle in default_sudoku_puzzles()}
