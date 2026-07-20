# Checkpoint: LSPG FP32 Precision Control v1

## Result

The five-step S0 FP32 control held all other settings fixed at their registered
values and completed:

| Measurement | Value |
| --- | ---: |
| finite optimizer steps | 5 / 5 |
| maximum raw gradient norm | 417.1418 |
| maximum applied gradient norm | 100.0 |
| mean step time | 1.0809 s |
| checkpoint bytes | 60,149,402 |
| peak wrapper RAM | 977.770 MB |
| peak wrapper I/O sample | 51.682 MB/s |

The I/O sample did not satisfy the registered three-consecutive-sample abort
condition. The paced checkpoint writer remained capped at 40 MB/s. The wrapper
reported clean process cleanup and no task metric was emitted.

## Finding

FP32 is finite at the original `3e-4` learning rate where AMP FP16 fails under
clip norms 100, 10, and 1 and under a near-zero update. Precision is the
discriminating variable.

## Instrumentation Correction

The result incorrectly reported zero PyTorch peak CUDA allocation because peak
tracking was conditioned on AMP rather than CUDA. Tracking now applies to every
CUDA precision. The next control is a five-step FP32 S2 run to measure the
high-scale memory envelope before amending campaign precision.
