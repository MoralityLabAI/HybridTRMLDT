# Benchmark Plan

This benchmark track compares:

- LDT: explicit lattice or constraint-state deduction.
- TRM: heuristic recurrent/search-style proposal.
- Hybrid: TRM-style proposal passed through checkable LDT-style state updates.

VPD is out of scope for this exercise. The paper path for this benchmark should focus on hybrid reasoning behavior, typed state updates, and empirical task performance.

## Task Ladder

Start small and add task families only after the previous one is stable:

```text
1. Sudoku: implemented
2. ARC-1: implemented
3. ARC-2: implemented
4. int-3 / INTELLECT-3 logic-env: integration inspected
5. routing: implemented
6. storyworld playing
```

## Current Sudoku Bench

The first implemented task is a dependency-free 4x4 Sudoku micro-benchmark.

Run:

```bash
python -m research_gym.scripts.bench_sudoku --out data/benchmarks/sudoku_results.jsonl --report reports/sudoku_bench.md
```

Current solver definitions:

- `ldt`: naked-single constraint propagation only.
- `trm`: heuristic MRV backtracking search without propagation after each guess.
- `hybrid`: MRV proposal plus LDT single propagation after each proposal.

The benchmark is intentionally small. Its job is to establish the comparison harness and expected qualitative separation before moving to larger task families.

## Current ARC-1 Bench

ARC-1 is a dependency-free single-rule grid transformation benchmark.

Run:

```bash
python -m research_gym.scripts.bench_arc1 --out data/benchmarks/arc1_results.jsonl --report reports/arc1_bench.md
```

Current solver definitions:

- `ldt`: derive a rule family from input/output invariants, then certify a unique fitting primitive.
- `trm`: search primitive rules in fixed heuristic order.
- `hybrid`: use LDT-derived rule families to restrict TRM proposals, then reject proposals that fail training examples.

The current primitive set includes color mapping, background fill, horizontal flip, vertical flip, and 180-degree rotation.

## Current ARC-2 Bench

ARC-2 uses ordered two-rule compositions over the ARC-1 primitive library.

Run:

```bash
python -m research_gym.scripts.bench_arc2 --out data/benchmarks/arc2_results.jsonl --report reports/arc2_bench.md
```

Current solver definitions:

- `ldt`: derive broad family constraints and certify a unique fitting composed rule.
- `trm`: search all ordered primitive pairs.
- `hybrid`: restrict ordered-pair proposals using LDT family constraints and certify each proposal against train examples.

## INTELLECT-3 Logic Env

The `int-3` item refers to the local Prime Intellect research environment fork, not a synthetic integer task.

Local path:

```text
C:/projects/prime_intellect_research_environments/environments/logic_env
```

Inspect from this repo:

```bash
python -m research_gym.scripts.inspect_intellect_logic --out reports/intellect_logic_env.md --json-out data/benchmarks/intellect_logic_env.json
```

Canonical smoke command from the environment repo:

```bash
uv run vf-eval --env logic-env -n1 -r1 -d -v
```

The environment uses the `logic` subset of `PrimeIntellect/INTELLECT-3-RL` and task-specific verifiers from `logic_env/task2verifier.py`.

## Current Routing Bench

The routing bench trains env-pointer routers on Tesseract normalized trajectories.

Run:

```bash
python -m research_gym.scripts.bench_routing --experiment-dir experiments/routing_env_pointer
```

Current model definitions:

- `ldt`: token lattice router. Prompt tokens monotonically refine candidate environment IDs.
- `trm`: dependency-light lexical TRM analogue over bag-of-token evidence, aligned with Tesseract's TF-IDF router objective.
- `hybrid`: LDT candidate filtering followed by TRM scoring among surviving candidates.

Outputs:

```text
data/benchmarks/routing_results.json
reports/routing_bench.md
experiments/routing_env_pointer/results.json
experiments/routing_env_pointer/training_notes.md
```
