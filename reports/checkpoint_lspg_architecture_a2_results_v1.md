# Checkpoint: LSPG Architecture Discovery A2 Results v1

## Integrity audit

Stage A2 completed all nine registered cells: two candidates and their shared
matched periodic control across seeds 401, 409, and 419. The stage receipt hash
is `48d9307c3871ee59de3719e5ae20cbb9231aba8cff325f0f2f9b23eda54c2f15`.
The post-run audit verified all result and prediction hashes, integrity and
cleanup flags, and one completed resource receipt per cell. All 27,648
prediction rows use only the calibration split.

Peak per-cell RAM was 1,042.469 MB. Peak single-sample I/O telemetry was 77.416
MB/s; no cell reached the registered three-consecutive-sample abort condition.
Completed-cell elapsed time totaled 837.726 seconds.

## Replication result

The A2 transition hash is
`a7f79b0407acb64550c9d34dd3ef4ba1031deb8650f3c4727425dccabd864a66`.
Neither candidate passed the registered requirement of a positive matched
macro-exact delta in at least two seeds.

| Candidate | Seed 401 | Seed 409 | Seed 419 | Mean delta | Positive seeds | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `LSAD-B1-K4L8-01` | -0.006510 | -0.010417 | +0.001302 | -0.005208 | 1 | Fail |
| `LSAD-B1-K4L8-02` | -0.002604 | -0.003906 | -0.002604 | -0.003038 | 0 | Fail |

Both candidates remained step-time matched on average and stayed within the
maximum per-task regression gate. The failure is specifically lack of a
replicated capability gain, not instability, resource failure, or excessive
task regression.

## Decision boundary

The A1 paired-block advantage did not replicate at four times the exposure.
The interleaved-return schedule was neutral at A1 and slightly negative in all
three A2 seeds. Neither architecture may be promoted to B or described as an
improvement over the periodic control.

The machine transition retains the generic action label `prepare_B` while its
`selected_candidates` list is empty. A B manifest with no candidates is not a
scientific experiment and must not be materialized. The frozen reserve trigger
is defined only after B, so opening it now would require an explicit
post-outcome protocol amendment; the reserve remains sealed and unused.
