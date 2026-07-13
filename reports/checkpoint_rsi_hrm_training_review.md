# Checkpoint: RSITopology-Aware HRM Training Review

## Completed

- Ported the scalar RSITopology lineage and holonomy control-risk bound into a dependency-free review kernel.
- Added typed identity levels for engineering evidence, lineage certification, and holonomy-clean signed use.
- Separated ordinary, bundle-allocation, global-signed, and sectioned-signed training mechanisms.
- Added independent grouped utility, matched-Haar, KL, damage, provenance, checkpoint, timeout, and cleanup gates.
- Added Conductor-HRM routes for authorization, local sectioning, additional audit, and rejection.
- Saved a deterministic ten-case receipt-contract experiment and training notes.
- Added focused tests for false-authorization boundaries and mechanism-specific promotion behavior.

## Claim boundary

The control bound limits false authorization for registered signed coordinates under simultaneous one-sided
coverage. It does not imply general behavioral safety, model quality, or self-improvement. The experiment uses
synthetic sealed receipts and performs no model training or weight mutation.

## Frozen review policy

- signed operator-error budget: `0.5`
- simultaneous false-authorization delta: `0.05`
- minimum conservative edge retention: `0.8`
- minimum bundle occupancy margin: `0.05`
- maximum realized allocation KL: `0.05`
- minimum grouped replicates: `3`
- maximum held-out damage regression: `0.0`

## Saved outcomes

- clean signed model promotion: `authorize`, bound `0.095`
- high measured holonomy: `section`, bound `0.757`
- orientation reversal: `section`, bound `2.000`
- unmeasured loop: `audit`, bound `2.000`
- long path with strong local retention: `section`, bound `0.530`
- failed grouped utility: `reject` despite `holonomy_clean`
- ordinary high-holonomy promotion: `authorize` without signed-coordinate authority
- bundle high-holonomy promotion: `authorize` at `lineage_certified`
- incomplete checkpoint/cleanup receipts: `audit`
- aborted training run: `reject` for model promotion while preserving its structured evidence

## Commands

```powershell
python -m research_gym.scripts.review_model_training
python -m pytest tests\test_training_review.py -q
python -m pytest tests -q
git diff --check
```

The full suite completed with `94 passed`. `pdflatex` was not installed locally, so the updated Overleaf source
was checked structurally but not compiled to PDF in this workspace.

## Main files

- `research_gym/core/training_review.py`
- `research_gym/benchmarks/training_review_bench.py`
- `research_gym/scripts/review_model_training.py`
- `tests/test_training_review.py`
- `docs/rsi_hrm_training_review.md`
- `data/benchmarks/hrm_training_review_results.json`
- `experiments/hrm_training_review/results.json`
- `experiments/hrm_training_review/training_notes.md`
- `reports/hrm_training_review.md`

## Next native step

Wrap an existing capped HRM/TRM checkpoint run as a receipt producer, preregister the review policy before
training, and evaluate checkpoint-by-context lineage and held-out utility without changing the membrane or
authorization thresholds.
