# Kappa Surface v1 Recovery 1: Post-Partial Registration

This recovery was registered after attempt 1 exposed all tied `R=16/R=32` trajectories and one incomplete tied `R=64`, seed-101 initialization measurement. It is not an as-preregistered completion of the original 90-measurement protocol.

## Admitted Evidence

Only the six cells with finite `cell_complete` events and all five original measurements are admitted from the hash-bound partial ledger. The incomplete `R=64` cell is excluded in full and all three `R=64` tied cells are rerun from initialization.

The reduced common surface uses exposures `0`, `1024`, `2048`, and `4096`. The missing `R=64` exposure-512 kappa probe is not imputed. Matched gradient intervals remain measured at `512`, `1024`, `2048`, and `4096`.

## New Work

1. Rerun three tied `R=64` cells with four kappa probes and four interval-gradient summaries.
2. Run all nine untied cells for interval-gradient summaries without intermediate kappa probes.
3. Reuse sealed terminal untied kappa only for a point-equivalence control, `|gamma| <= 0.1`.

The original five-checkpoint untied-kappa control remains unobserved and cannot be claimed.

## Decision

The reduced 12-cell surface retains the original separable, smooth-interaction, and `R=64` change-point alternatives. A model wins only with both a four-unit AICc advantage and a ten-percent blocked-CV RMSE improvement. Curvature onset can occur only at exposure 1024 or 2048 and requires `C(E) <= -0.2` with a paired-seed-bootstrap 95% upper bound below `-0.1`.

A post-partial depth-transition label additionally requires matched `R=64` gradient stress at least five times the untied control and greater than the matched `R=16/R=32` ratios, plus tied terminal replication and sealed untied terminal point equivalence.

## Resource Debit

Attempt 1 is conservatively charged `1204.1 s` against the unchanged 3600-second aggregate cap, leaving `2395.9 s`. Each recovery phase retains the 1800-second phase timeout and all original RAM, CPU, I/O, and VRAM hard caps.
