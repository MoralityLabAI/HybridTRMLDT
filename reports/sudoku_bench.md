# Sudoku Benchmark

4x4 Sudoku micro-benchmark comparing LDT propagation, TRM heuristic search, and hybrid proposal plus certification.

| Solver | Solved | Solve Rate | Steps | Guesses | Conflicts |
|---|---:|---:|---:|---:|---:|
| `hybrid` | 3/3 | 1.000 | 31 | 6 | 0 |
| `ldt` | 1/3 | 0.333 | 7 | 0 | 0 |
| `trm` | 3/3 | 1.000 | 31 | 31 | 0 |

## Per Puzzle

- `s4_single_chain` `ldt` solved=True steps=7 guesses=0 conflicts=0
- `s4_single_chain` `trm` solved=True steps=7 guesses=7 conflicts=0
- `s4_single_chain` `hybrid` solved=True steps=7 guesses=0 conflicts=0
- `s4_requires_search` `ldt` solved=False steps=0 guesses=0 conflicts=0
- `s4_requires_search` `trm` solved=True steps=12 guesses=12 conflicts=0
- `s4_requires_search` `hybrid` solved=True steps=12 guesses=4 conflicts=0
- `s4_sparse` `ldt` solved=False steps=0 guesses=0 conflicts=0
- `s4_sparse` `trm` solved=True steps=12 guesses=12 conflicts=0
- `s4_sparse` `hybrid` solved=True steps=12 guesses=2 conflicts=0
