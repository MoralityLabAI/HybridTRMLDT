# Checkpoint: LSAD A2 Results v1

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

## Resolution and effect bound

Primary macro exact averages three families of 256 examples, so its score
quantum is `1/768 = 0.0013020833`. The paired-block seed deltas are net changes
of -5, -8, and +1 correct examples relative to control; the interleaved-return
deltas are -2, -3, and -2. These are item-scale fluctuations, not evidence of a
persistent effect. The two A2-tested K4/L8 schedules therefore license the
bounded statement `|mean delta| <= 0.005209`, with maximum observed per-seed
absolute delta `0.010417`, at 5M parameters on this calibration task bundle.

That multi-seed bound does not extend to all twelve A1 candidates because ten
received only one seed. Their single-seed screen is instead evidence that a
full-budget one-seed selector can promote item-scale noise. A v2 should prefer
two screening seeds at approximately half exposure when compute is fixed, then
reserve longer runs for effects whose signs replicate.

The near-equality of candidates and matched controls also supports the absence
of a gross matching leak across parameters, visits, estimated FLOPs, exposure,
and gradient policy. It is supporting instrument evidence, not a formal
certification, because this campaign did not include an intentionally mismatched
positive control.

## Decision boundary

The A1 paired-block advantage did not replicate at four times the exposure.
The interleaved-return schedule was neutral at A1 and slightly negative in all
three A2 seeds. Neither architecture may be promoted to B or described as an
improvement over the periodic control.

The machine transition retains the generic action label `prepare_B` while its
`selected_candidates` list is empty. A B manifest with no candidates is not a
scientific experiment and must not be materialized. The frozen reserve clause
can be read as vacuously satisfied when zero candidates reach B, but it does not
specify a zero-candidate re-entry stage. A transparent post-outcome addendum
could close that procedural gap because reserve proposals were sealed before
A1; the resulting combined campaign could not be called wholly preregistered.

V1 closes instead as the registration's explicitly valid zero-winner result.
The reserve remains sealed and unused: another twelve words from the same
grammar would add breadth, while the open scientific question concerns scale
transfer beyond S0. No candidate is promoted to B, S1, or S2.
