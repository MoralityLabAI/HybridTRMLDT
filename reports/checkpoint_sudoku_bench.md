# Checkpoint Sudoku Bench

## Completed

- Added dependency-free 4x4 Sudoku environment.
- Added LDT, TRM, and hybrid Sudoku solvers.
- Added Sudoku benchmark CLI.
- Added benchmark report and JSONL result output.
- Added tests covering candidates, propagation, solver behavior, conflicts, and benchmark shape.
- Added benchmark plan that explicitly excludes VPD from this hybrid exercise.

## Commands run

```text
python -m pytest tests -q
python -m research_gym.scripts.bench_sudoku --out data/benchmarks/sudoku_results.jsonl --report reports/sudoku_bench.md
```

## Files changed

```text
research_gym/envs/sudoku.py
research_gym/benchmarks/__init__.py
research_gym/benchmarks/sudoku_bench.py
research_gym/scripts/bench_sudoku.py
tests/test_sudoku_bench.py
docs/benchmark_plan.md
reports/sudoku_bench.md
reports/checkpoint_sudoku_bench.md
data/benchmarks/sudoku_results.jsonl
```

## Tests

```text
45 passed
```

## Results

```text
hybrid: 3/3 solved, 31 steps, 6 guesses, 0 conflicts
ldt: 1/3 solved, 7 steps, 0 guesses, 0 conflicts
trm: 3/3 solved, 31 steps, 31 guesses, 0 conflicts
```

## Decisions made

- Use 4x4 Sudoku first to keep the bench deterministic and fast.
- Treat LDT as propagation-only.
- Treat TRM as heuristic search without propagation after each guess.
- Treat hybrid as heuristic proposal plus propagation/certification.
- Exclude VPD from this benchmark track.

## Open questions

- Whether the next Sudoku step should move to 9x9 puzzles or add more 4x4 puzzle classes with known search-depth labels.

## Recommended next step

Add a larger Sudoku suite with puzzle difficulty tags and per-frame traces, then decide whether to move to ARC-1.
