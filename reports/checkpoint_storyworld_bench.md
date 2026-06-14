# Checkpoint Storyworld Playing Bench

## Completed

- Added storyworld playing benchmark over `CoupledStoryworldEnv`.
- Added three player policies:
  - `ldt`: exact finite-horizon planner under modeled rival policy.
  - `trm`: heuristic policy over heat, evidence, trust, and scene deficits.
  - `hybrid`: TRM proposal plus LDT reachability check and override.
- Added CLI runner.
- Added tests for policy behavior, reachable start sampling, run shape, and benchmark shape.
- Saved experiment outputs and model notes under `experiments/storyworld_playing/`.
- Updated benchmark plan.

## Commands run

```text
python -m pytest tests/test_storyworld_bench.py -q
python -m research_gym.scripts.bench_storyworld --n 64 --horizon 6 --seed 7 --out data/benchmarks/storyworld_results.json --report reports/storyworld_bench.md --experiment-dir experiments/storyworld_playing
python -m pytest tests -q
```

## Files changed

```text
research_gym/benchmarks/storyworld_bench.py
research_gym/benchmarks/__init__.py
research_gym/scripts/bench_storyworld.py
tests/test_storyworld_bench.py
docs/benchmark_plan.md
reports/storyworld_bench.md
reports/checkpoint_storyworld_bench.md
data/benchmarks/storyworld_results.json
experiments/storyworld_playing/results.json
experiments/storyworld_playing/training_notes.md
```

## Tests

```text
69 passed
```

## Results

```text
hybrid: 64/64 solved, success_rate 1.000, avg_steps 3.28, overrides 124
ldt: 64/64 solved, success_rate 1.000, avg_steps 3.50
trm: 11/64 solved, success_rate 0.172, avg_steps 2.81
```

## Decisions made

- Sample only starts that are reachable under the modeled rival policy, so failure measures policy quality rather than impossible instances.
- Let LDT act as an exact finite-horizon planner rather than a learned model.
- Let TRM be a simple persistent heuristic baseline, not a trained neural model.
- Let hybrid override TRM only when the proposed action loses modeled reachability and a certified alternative exists.
- Keep VPD out of the benchmark and notes.

## Open questions

- Whether to add trained TRM and learned LDT variants after the symbolic policy baselines are fully summarized.
- Whether storyworld playing should later use the larger GPTStoryworld/SweepWeave harness.

## Recommended next step

Add an aggregate benchmark report across Sudoku, ARC-1, ARC-2, routing, and storyworld playing for paper-facing comparison.
