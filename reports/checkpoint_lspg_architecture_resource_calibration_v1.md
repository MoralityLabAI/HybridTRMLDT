# Checkpoint: LSPG Resource Calibration v1

## Result

All three FP32 resource-only calibration cells completed 20 warm-up and 30
measured optimizer steps:

| Scale | Mean measured step | CUDA allocation peak | Peak RAM | Peak I/O sample |
| --- | ---: | ---: | ---: | ---: |
| S0, 5M | 0.9011 s | 162,363,392 B | 971.566 MB | 67.470 MB/s |
| S1, 12M | 1.7720 s | 293,166,592 B | 978.895 MB | 100.354 MB/s |
| S2, 30M | 4.7523 s | 684,327,936 B | 1,319.207 MB | 252.926 MB/s |

Every cell reports 30 measured steps, finite gradients, clean process cleanup,
and no task metrics. CUDA allocation and RAM remained below the 2,500 MB and
3,800 MB caps. The I/O peaks were isolated checkpoint-associated samples; none
met the registered three-consecutive-sample abort rule. Checkpoint writes were
paced at 40 MB/s.

## Profile Selection

The conservative mandatory-path forecasts, including a 1.2 safety factor, are:

| Profile | Forecast GPU-hours | Gate |
| --- | ---: | --- |
| full | 79.68 | reject |
| medium | 39.85 | reject |
| minimum | 19.95 | select |

The selected profile is `minimum`. Its mandatory-path forecast is 71,804.15
GPU-seconds, leaving the separately registered 43,200 GPU-seconds for the
single reserve batch. The profile selection hash is
`28324027699340a077508471b813f074e8798f844b944360b4221b10276bb4a4`.

## Claim Boundary

This checkpoint measures execution cost and finite training only. No task
accuracy, candidate ranking, or architecture winner was observed.

## Next Step

Commit the calibration corpus and profile receipt. Then materialize and commit
the sealed A1 manifest for 12 first-batch candidates and six deduplicated
periodic controls before launching any A1 cell.
