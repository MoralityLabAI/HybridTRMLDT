# Kappa State/Moment v0.2.3 Attempt 1 Interruption

Attempt 1 completed all four registered continuation cells, all 16 kappa measurements, and all four terminal checkpoints. Every model-weight and complete-AdamW-state gate passed tensor-exactly. The child wrote a complete result and exited with empty stderr.

The controlling tool session was interrupted before the PowerShell wrapper wrote `run.resource_receipt.json`. Peak resource telemetry and wrapper-level cleanup therefore cannot be independently reconstructed, so attempt 1 is **not sealed and will not be interpreted as the primary result**. An audit at `2026-07-21T04:21:40Z` found no live matching trainer, wrapper, or GPU process.

The recovery preserves attempt 1 byte-for-byte and reruns the unchanged frozen config in `experiments/loop_schedule_kappa_state_moment_v0_2_3_recovery1`. A conservative `620 s`, slightly above the child event interval, is charged through the wrapper's `ExternalPriorElapsedSeconds` input. The recovery is interpretable only if its resource receipt completes, cleanup passes, and its science records, result summary, and terminal checkpoints reproduce attempt 1 exactly.
