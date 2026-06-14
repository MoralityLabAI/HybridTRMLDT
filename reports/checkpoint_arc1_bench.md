# Checkpoint ARC-1 Bench

## Completed

- Added ARC-style grid task primitives.
- Added ARC-1 single-rule task suite.
- Added LDT, TRM, and hybrid ARC-1 solvers.
- Added ARC-1 benchmark CLI.
- Added tests for primitives, fitting, solvers, and benchmark output shape.
- Updated benchmark plan.

## Commands run

```text
python -m pytest tests -q
python -m research_gym.scripts.bench_arc1 --out data/benchmarks/arc1_results.jsonl --report reports/arc1_bench.md
```

## Files changed

```text
research_gym/envs/arc_tasks.py
research_gym/benchmarks/arc_bench.py
research_gym/benchmarks/__init__.py
research_gym/scripts/bench_arc1.py
tests/test_arc1_bench.py
docs/benchmark_plan.md
reports/arc1_bench.md
reports/checkpoint_arc1_bench.md
data/benchmarks/arc1_results.jsonl
```

## Tests

```text
50 passed
```

## Results

```text
hybrid: 3/3 solved, 5 steps, 5 proposals, 2 rejected
ldt: 3/3 solved, 11 steps, 0 proposals, 0 rejected
trm: 3/3 solved, 9 steps, 9 proposals, 6 rejected
```

## Decisions made

- ARC-1 is single-rule only.
- LDT derives candidate rule families from train input/output invariants.
- TRM searches primitive rules in fixed order.
- Hybrid uses LDT family filtering to constrain TRM proposals and certify against train pairs.

## Open questions

- Whether ARC-2 should mean two-rule compositions over the same primitive library or a broader set of object-centric transforms.

## Recommended next step

Proceed to ARC-2 with two-step rule compositions.
