# Loop Schedule Algebra v0: gamma checkpoint

## Frozen protocol

- Config SHA-256: `86bc4bf4e7c1480722c8b377e0ec028958010225d331a748e8917b8d433ff7f1`
- Registered rounds: `2, 4, 8`; seeds: `101, 103, 107`; five power iterations.
- Fixed progress: 4,096 state-visit exposures per cell.
- Stop thresholds: across-seed kappa spread above 10x, or power-law `R^2 < 0.8`.

## Instrument correction

The first capped attempt was rejected before prediction sealing because it
double-counted `(beta/alpha)^2`. Its one-visible-visit value was 16 at
`beta=0.25`, contradicting the defining `kappa in [0,R_g]` bound. The complete
attempt is retained under
`experiments/loop_schedule_algebra_v0_invalid_double_normalization`; no P2
prediction was produced from it. The corrected coefficient measures
directional alignment, while residual scale is applied once in `B(A)`.

## Corrected gamma phase

| Regime | geometric mean kappa, R=2 | R=4 | R=8 | gamma | R^2 | P1 |
|---|---:|---:|---:|---:|---:|---|
| tied | 1.3580 | 1.8936 | 2.4407 | 0.4229 | 0.9940 | outside registered [0.7,1.0] |
| untied | 0.9755 | 1.0037 | 1.0115 | 0.0261 | 0.9026 | inside registered [0.0,0.3] |

The largest per-cell across-seed spread is 1.0923x, so the estimator-instability
stop does not fire. Both fits clear the registered `R^2 >= 0.8` criterion, so
the poor-power-law stop does not fire. P1 is therefore a mixed result: the
untied control confirms, while tied alignment grows materially more slowly
than preregistered.

P3 is monotone in all three seeds. At seeds 101/103/107 respectively, kappa is
`1.000/1.000/1.000` for one-step, `1.9068/1.9049/1.8890` for last-k, and
`2.3370/2.3318/2.3521` for full visibility.

## Integrity and resources

- Corrected records: 27 canonical JSONL rows.
- Records SHA-256: `d614837dff4dc7fab220c69fb49d5ccc05b19da9a33f26be9247217fa9c3482d`.
- Range audit: zero values outside `[0,R_g]`.
- Resource receipt: completed, peak RAM 890.016 MB, peak I/O 10.456 MB/s,
  no lingering owned process, cleanup passed.

No prediction about task accuracy, sample efficiency, or reasoning quality anywhere in the campaign; the algebra prognosticates trainability boundaries and cost only.
