# CCA-11A prospective CPU registration

Date: 2026-11-04. Parent theory: docs/cca11_recoverability_threshold.md.

## Aim

Exercise exact-rational scalar envelopes to test the proposed uniform robust
invariant and its sharpness in the equality-saturating worst-case recurrence.
Include an isolated above-uniform-threshold burst that remains viable, and a
rare-spike counterexample to interpreting expected drift as pathwise safety.

## Frozen parameters and predictions

- rho=1/2, gain=1, residual=1/10, margin=1, initial=1.
- Constant eta in {39/100, 2/5, 41/100, 13/20}, horizon 16.
- Predict eta_crit=2/5 exactly; eta<=2/5 remains viable from the boundary,
  while greater disturbances exceed the margin on step 1.
- Burst example: rho=1/2, gain=1, residual=0, margin=1, initial=0;
  eta=(3/4,0,0,0); no exit despite exceeding the 1/2 uniform threshold.
- Expected-only counterexample: P(spike=10)=1/100, mean error=1/10
  despite positive probability of violating margin=1.

These are illustrative mathematical fixtures; no neural, GPU, human-preference,
or semantic-monitor data have been collected.

## Reproduce

Run from repository root:
- python -m pytest -q tests/test_cca_recoverability_threshold.py
- python experiments/cca11_recoverability_threshold/run.py
