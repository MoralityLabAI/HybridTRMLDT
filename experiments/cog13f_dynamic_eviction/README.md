# COG-13F: dynamic eviction-regret residual

Protocol registered before execution at [e2af4ee](https://github.com/MoralityLabAI/HybridTRMLDT/commit/e2af4ee0531ea7a7736f145abaca365940f385a0). The source is adapted from COG-13E at 8c5817d0079e3998f1b9704b662f37b58cc0739c. Native COG-13D results were unavailable at design time; these are synthetic cost/byte models, not native timing results.

## Label

Force each possible eviction of a candidate state, finish the current eviction using recent reuse density, then run the next 12 requests under that same policy. Record each candidate's realized future modeled work J_i. Dynamic regret is J_i minus the smallest J in that decision. Fit a root-balanced, ridge-regularized, within-decision centered residual of log(1+regret/10) relative to log(1+100 times recent density). This is a finite-horizon, fixed-continuation counterfactual, not the globally optimal dynamic cache policy. Future traces are restricted to training labels and the explicit oracle diagnostic.

## Executed evidence

152 generated roots: 64 train, 24 calibration, 32 ordinary, 32 deeper. Exact generated-world fingerprints are disjoint from the old COG-13E registry. Train/ordinary depth 4–10, deeper depth 11–16, 60 requests per root. Fixed 128/256/512 modeled-byte budgets. Training: 384 source rollouts, 23,040 source requests, 701,908 modeled work; 13,249 eviction decisions and 60,284 candidate labels/counterfactual rollouts, comprising 630,775 future simulated requests and 18,404,580 additional work units. Root-balanced 17-coordinate ridge, penalty .2. Within-decision shuffled regret control. No tuning using ordinary/deeper roots.

Calibration selected lambda **0**:

| Lambda | Raw work | Overhead-adjusted work |
|---:|---:|---:|
| 0 | 126529 | 142017 |
| 0.25 | 125500 | 175832.91 |
| 0.5 | 125321 | 175482.49 |
| 1 | 123857 | 174040.14 |

The residual saves some raw work on calibration, but does not pay for its synthetic scoring charge. The selected policy is exactly the analytic recent-density baseline and skips learned feature computation.

Deeper cohort, 32 roots, lower is better (work plus registered synthetic overhead):

| Byte cap | LRU | Recent | Coverage | Residual 1 | Shuffled 1 | Selected | Future oracle* |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 147334.85 | 134890.79 | 138809.08 | 147358.30 | 146250.55 | 134890.79 | 118273.88 |
| 256 | 134463.42 | 108677.49 | 113748.38 | 131498.85 | 133049.33 | 108677.49 | 89105.33 |
| 512 | 113732.68 | 82443.23 | 89390.28 | 115040.12 | 113957.73 | 82443.23 | 64308.07 |

*Future oracle is privileged, and its extra lookahead work is recorded separately rather than charged in this displayed diagnostic cost; it is not a deployable comparator or an optimality bound.

At 256 bytes, forced residual costs 21.0% more than recent density; at 512, 39.54% more. The forced residual only narrowly outperforms shuffled at 256, with a descriptive interval crossing zero, and loses at 128/512. Both registered development gates fail. The calibrated fallback avoids the learned overhead. Raw-work coverage sometimes improves over recent density, but the declared overhead reverses this. See results.json for all raw costs, counters and root-stratified descriptive bootstrap intervals.

1,824 calibration and evaluation rollouts, 109,440 audited request receipts. The auditor independently reconstructs prefix paths, work, admissibility, cap and cache transitions; it recomputes nonoracle scores with the shared scoring routine and oracle costs with the same continuation simulator. Four forged/missing receipt mutations rejected. This is not an independent implementation of the ridge or counterfactual simulation. Archived JSON includes all policies, roots and metrics; the runner exports the full replay and training groups with source and receipt hashes, verifying exact corpus nonoverlap and target arithmetic. No data has been ingested into DataGoblin and no neural LDT was trained.

## Next gate

COG-13D has since produced two native capped artifacts. The original CI failed from JSON property-order and error-format checks; [the independent retrospective replay](https://github.com/MoralityLabAI/HybridTRMLDT/blob/f6cea95d4ef02b99ce70efee416554af9eb20f0d/experiments/cog13d_native_state_cache/NATIVE_AUDIT_AMENDMENT.md) validates all 3,360 native requests and finds no memory pressure at either cap. The next stage should use native proof-state query timing, longer expensive branching traces, per-arm cgroup peaks and a pre-registered actual memory pressure test. Stop adding learned cache scores until a measured baseline demonstrates headroom. Preserve both synthetic negative results.
