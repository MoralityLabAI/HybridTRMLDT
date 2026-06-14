# Checkpoint ARC-2 Bench

## Completed

- Added ARC-2 two-rule composition tasks.
- Added ordered primitive-pair rule generation.
- Added LDT, TRM, and hybrid ARC-2 solvers.
- Added ARC-2 benchmark CLI.
- Added ARC-2 tests.
- Updated benchmark plan.

## Commands run

```text
python -m pytest tests/test_arc2_bench.py -q
python -m research_gym.scripts.bench_arc2 --out data/benchmarks/arc2_results.jsonl --report reports/arc2_bench.md
python -m pytest tests -q
```

## Files changed

```text
research_gym/envs/arc_tasks.py
research_gym/benchmarks/arc_bench.py
research_gym/benchmarks/__init__.py
research_gym/scripts/bench_arc2.py
tests/test_arc2_bench.py
docs/benchmark_plan.md
reports/arc2_bench.md
reports/checkpoint_arc2_bench.md
data/benchmarks/arc2_results.jsonl
```

## Tests

```text
55 passed
```

## Results

```text
hybrid: 3/3 solved, 32 steps, 32 proposals, 29 rejected
ldt: 3/3 solved, 120 steps, 0 proposals, 0 rejected
trm: 3/3 solved, 38 steps, 38 proposals, 35 rejected
```

## Decisions made

- ARC-2 uses ordered pairs of ARC-1 primitives.
- LDT certifies fitting composed rules and applies only when fitting rules imply a unique test prediction.
- TRM searches all ordered primitive pairs in fixed order.
- Hybrid filters and orders proposals by LDT-style family constraints, then certifies against training pairs.
- The benchmark remains synthetic and dependency-free.

## Open questions

- Whether to add object-centric ARC-2 transforms later or keep ARC-2 as pure two-rule composition for the first paper scaffold.

## Recommended next step

Proceed to `int-3`: small integer reasoning tasks with three-operation chains.
