# Kappa Surface Recovery 2: Pacing Registration

Recovery 1's untied-gradient phase hit the unchanged 50 MB/s I/O abort because gradient-only cells wrote their 2048- and 4096-exposure checkpoints in rapid succession. This second recovery changes only write pacing.

All nine untied cells are rerun in a fresh output directory. No partial gradient record from the aborted phase is admitted. Model, task, seeds, exposures, optimizer, checkpoints, endpoint definitions, and hard caps remain unchanged. The runner sleeps two seconds after each checkpoint has been written and receipt-logged.

The completed tied `R=64` phase is reused only through hash-bound records and a passing resource receipt. The conservative aggregate clock begins at `1909.626 s`, leaving `1690.374 s` under the original one-GPU-hour cap.

Any final result remains a labeled post-partial recovery and retains the limitation that untied kappa equivalence is measured only at the sealed terminal point, not over four of five trajectory checkpoints.
