# Checkpoint: LSPG Architecture Tasks v1

## Completed

- Added a fixed-vocabulary, fixed-length single-answer contract shared by five task families.
- Added held-out-depth pointer, modular recurrence, and rewrite generators.
- Added uniqueness-checked procedural 4x4 Sudoku queries.
- Added deterministic, per-environment routing splits with source hashes and local prompt snapshots.
- Added macro metrics, Sudoku puzzle exactness, routing macro accuracy, paired bootstrap, and Holm thresholds.

## Integrity

Every example has a semantic fingerprint. Train, calibration, and evaluation fingerprints must be disjoint. Each JSONL shard is SHA-256 bound by the bundle manifest and is rechecked on load.

## Tests

```text
python -m pytest tests/test_lspg_architecture_tasks.py -q
3 passed
```

## Next Step

Materialize the full bundle, then implement the versioned planner and promotion funnel over the sealed schedule and task identities.
