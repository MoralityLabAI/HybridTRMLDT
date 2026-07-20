# Checkpoint: LSPG Architecture Discovery A1 Results v1

## Integrity audit

Stage A1 completed all 18 registered cells: twelve candidates and six matched
periodic controls at S0, seed 401, and 524,288 token-visit exposures. The stage
receipt hash is
`a4c397fb2299524f64e13de08e15400c4b1c9ff6376cca385fc0d8dc3591d800`.
The post-run audit verified all result hashes, prediction hashes, integrity and
cleanup flags, and exactly one completed resource receipt per cell. The 55,296
prediction rows use only the calibration split.

Peak per-cell RAM was 1,042.859 MB. Peak single-sample I/O telemetry was 68.456
MB/s during checkpointing; no cell reached the registered three-consecutive-
sample I/O abort condition. Completed-cell elapsed time totaled 1,569.746
seconds.

## Registered transition

The A1 transition hash is
`f759f244e4a32dac30ea1822cf0b8d386a4444e0455ca548a1aa03e0e1f70e9d`.
Exactly two candidates passed both the matched-step-time and maximum per-task
regression gates:

| Candidate | Schedule word | Macro exact delta | Maximum task regression | Step-time ratio |
| --- | --- | ---: | ---: | ---: |
| `LSAD-B1-K4L8-01` | `[0,0,1,1,2,2,3,3]` | +0.026042 | -0.027344 | 0.975646 |
| `LSAD-B1-K4L8-02` | `[0,1,2,3,1,0,2,3]` | +0.000000 | -0.027344 | 0.975908 |

Both use four physical modules and eight applications and are compared with
`LSAD-C-K4L8`. The first is a paired-block schedule with reversal symmetry 1.0
and transition spectral gap 0.5. The second is an interleaved-return schedule
with reversal symmetry 0.5 and transition spectral gap 1.0.

## Claim boundary

This is a deterministic screening decision, not evidence that either schedule
is superior. The second candidate is eligible but neutral in aggregate at A1.
Several earlier candidates failed the step-time gate while unrelated host/GPU
workloads were present; `LSAD-B1-K4L6-01` is the clearest timing outlier at a
14.48 matched-control ratio. A2 therefore tests replication under new seeds and
must not inherit a performance claim from the A1 ordering.
