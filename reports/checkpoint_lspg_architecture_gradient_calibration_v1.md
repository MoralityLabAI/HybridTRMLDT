# Checkpoint: LSPG Gradient Calibration Failure v1

## Result

The first indexed-CUDA calibration pass constructed all three scale rungs but
stopped before optimizer step 1 under the absolute raw global-gradient threshold
of 100:

| Scale | Unique parameters | Raw global gradient norm | Optimizer steps |
| --- | ---: | ---: | ---: |
| S0 | 5,010,304 | 323.9803 | 0 |
| S1 | 11,949,344 | 525.2385 | 0 |
| S2 | 29,788,736 | 836.9609 | 0 |

All result and wrapper receipts report clean process cleanup. Peak I/O remained
below the 50 MB/s cap, and no task metric was emitted. These cells are failed
calibration evidence, not throughput measurements.

## Diagnosis

The threshold treated the expected size-dependent raw global norm at random
initialization as divergence. Because no optimizer update occurred at any
scale, it cannot discriminate finite training from an unstable trajectory and
would make the registered campaign unexecutable.

## Construction Correction

The value 100 is retained as a global-norm clipping bound. Receipts separately
record raw maximum norm, applied maximum norm, and clipped-step count. A
non-finite raw norm still stops immediately, and a raw norm above 1,000,000 is a
catastrophic construction stop. This correction is shared by every candidate
and matched control and will be frozen before task outcomes.

## Next Step

Commit these stopped cells and the clipping implementation, freeze an execution
v1.1 amendment, then retry all three resource-only calibration cells using new
attempt receipts.
