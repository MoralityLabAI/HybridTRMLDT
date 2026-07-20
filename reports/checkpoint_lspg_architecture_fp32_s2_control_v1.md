# Checkpoint: LSPG FP32 S2 Control v1

## Result

The five-step 30M-parameter FP32 control completed under the frozen local caps:

| Measurement | Value |
| --- | ---: |
| finite optimizer steps | 5 / 5 |
| maximum raw gradient norm | 1,058.1274 |
| PyTorch peak CUDA allocation | 684,327,936 bytes |
| wrapper peak RAM | 1,337.137 MB |
| mean step time | 4.9785 s |
| checkpoint bytes | 357,490,586 |
| checkpoint paced-write time | 10.4858 s |

The CUDA allocation and process RAM are below the 2,500 MB and 3,800 MB caps.
The wrapper reported clean cleanup and no task metric was emitted.

## I/O Caveat

Wrapper telemetry observed a one-sample 246.767 MB/s peak during final checkpoint
handling. The registered abort requires three consecutive samples above 50
MB/s, so the cell remained valid. The checkpoint stream itself reports the
registered 40 MB/s writer cap. Both numbers are retained; this run does not
claim every OS-level sample stayed below 50 MB/s.

## Decision

FP32 is finite at both S0 and S2 and fits the complete executable ladder. AMP
FP16 deterministically fails by step 2 even under near-zero learning rate. Set
FP32 as the architecture campaign precision, require it during resource
calibration, and freeze this change before retrying calibration or opening task
outcomes.
