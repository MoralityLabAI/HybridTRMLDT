# Independently Induced AIRIS Rule Notes

## Design

- Rules are induced from calibration episode utility winners, not topology-plan sequence labels.
- Topology plans contribute route, orientation, protocol, and risk features only for authorization.
- Rule support is the calibration winner count; disagreement episodes are retained as counterexamples.
- Rule confidence is support divided by context calibration count.
- Integrity digests are frozen before held-out evaluation.
- The primary confidence threshold is fixed at 0.5; the remaining thresholds are sensitivity analysis.

## Results

- calibration episodes: `455`
- held-out episodes: `667`
- rules: `28`
- raw held-out oracle-label accuracy: `0.940030`
- raw proposal/control parity: `1.000000`
- raw macro utility: `0.900085`
- guarded acceptance: `1.000000`
- guarded accepted-label accuracy: `0.940030`
- intact-but-wrong held-out rules: `40`
- guarded harmful changes: `0`
- raw proposals below control: `0`
- fallback save rate: `0.000000`
- guarded macro utility delta vs control: `+0.000000`
- calibration-confidence Brier score: `0.044228`

## Boundary

AIRIS-style empirical context-rule induction from deterministic calibration outcomes; not causal identification, neural AIRIS training, or native distributed DAS performance.
