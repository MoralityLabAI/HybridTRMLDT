# Checkpoint: Controller-Mesh Sheaf Forward Replication v2

Date: `2026-07-17`

## Completed

- froze a new task seed and 128-genome panel before outcome analysis;
- split 64 discovery and 64 held-out genomes by complete fallback families;
- preregistered spectral and categorical predictors as co-primary;
- sealed both held-out prediction vectors before held-out outcome reveal;
- evaluated a 2,000-sample paired bootstrap and 128 matched N0 predictors;
- retained the failed incremental gate without tuning;
- added receipt, leakage, reveal-order, and exact-result regression tests.

## Commands Run

```text
python -m research_gym.scripts.bench_controller_mesh_sheaf_replication
python -m pytest -q tests/test_controller_mesh_forward_replication.py
```

## Registered Result

- spectral held-out rho: `+0.5679`
- categorical held-out rho: `+0.5980`
- paired rho delta: `-0.0301`
- paired 95% interval: `[-0.3173, +0.2460]`
- matched N0 mean rho: `-0.1970`
- matched N0 one-sided p: `0.0078`
- spectral / categorical top-16 uplift: `+0.0928 / +0.1162`
- top-16 uplift delta: `-0.0234`
- registered incremental gate: failed

Spectral absolute prediction and the matched-null check pass. Increment over the categorical baseline does not:
rho delta, bootstrap lower bound, and top-set uplift delta all fail. This is the registered negative result.

## Held-Out Families

Correction infusion averages `1.0440` objective and `0.9325` utility versus `0.9011` and `0.8319` for safe LDT.
Both average `0.0265` unsafe rate. The correction arm uses exact synthetic calibration targets and retains its
oracle caveat.

## Files Changed

- `configs/controller_mesh_sheaf_forward_replication_v2.json`
- `research_gym/analysis/controller_mesh_forward_replication.py`
- `research_gym/scripts/bench_controller_mesh_sheaf_replication.py`
- `tests/test_controller_mesh_forward_replication.py`
- `data/benchmarks/controller_mesh_sheaf_forward_replication_v2.json`
- `experiments/controller_mesh_sheaf_forward_replication_v2/`
- `reports/controller_mesh_sheaf_forward_replication_v2.md`
- `docs/controller_mesh_sheaf_forward_replication.md`

## Receipts

- semantic config SHA-256: `6728719cdccf2f1dc425eca234e554230ed9abd8db2720356f58dfba159083ec`
- calibration geometry SHA-256: `31a0250dab855c42a5a4cb6e9939450b5d1003f78d2199a89178d46d87e3176e`
- discovery outcomes SHA-256: `f07804f5eb8e3d1fe566ba945cf19a0947c25574af17ffe12d5dedc803c56c8f`
- dual prediction SHA-256: `0111580842739741ab927613f05b7d4bc3d47d0cefe6f5d52b0d89a2c66dd1c9`
- held-out outcomes SHA-256: `ebb6ab29a783d4e1d58689a3341224f0693bf99c24c43e081ebcbc82fd940b6b`
- analysis receipt SHA-256: `adee88b22db004644e6e88a51604cf37bde4627a61d7c5dd6defa8f5108433ca`
- result-file SHA-256: `26fef973035249f081f83a3f6cde1bdf7df91179029ba5b17b3ab3bf11dfc1da`
- canonical/experiment mirror mismatch: `0`

## Tests

- focused replication suite: `4 passed`
- full suite: `157 passed`
- simulated install without neural extra: `122 passed, 5 skipped`
- dependency-free core import: passed with `torch_available=False`
- deterministic replay: byte-identical
- compileall: passed

## Decisions Made

- treat N0 separation as absolute spectral signal, not incremental value;
- reject an acquisition-function claim because the preregistered baseline is stronger;
- keep the v1 result as panel-specific evidence rather than retroactively invalidating it;
- preserve correction infusion as an oracle-backed controller arm, not learned improvement.

## Open Questions

- Does a measured neural stalk contribute typed residual geometry beyond policy labels?
- Can restriction maps be defined from independent evidence contracts rather than architecture-coded traces?

## Recommended Next Step

Only run a bridge study if it freezes a new mechanistic feature family and again requires positive increment over
the categorical baseline. Do not tune the current six spectral features against this held-out result.
