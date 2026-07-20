# Loop Schedule Kappa Surface v1: Registration Checkpoint

## Question

The prior campaign found that terminal visit alignment increased through `R=16`, then declined at `R=32` and `R=64`, while tied gradient maxima increased sharply. This follow-up asks when that depth-specific deviation appears during training and whether it co-localizes with optimization stress.

## Timing And Prior Knowledge

This protocol was written after all terminal `R=16`, `R=32`, and `R=64` outcomes were known, but before any nonterminal `R=32` or `R=64` trajectory was measured. The terminal cells are therefore replication and integrity checks. The untouched evidence is the common nonterminal trajectory at exposures `512`, `1024`, and `2048`, plus interval-level gradient summaries.

Exposure `256` is deliberately omitted. Under the frozen batch size of eight, one `R=64` optimizer step is 512 state-visit exposures; adding 256 would require changing batch statistics or interpolating a nonexistent checkpoint.

## Frozen Crossing

The study crosses:

- rounds: `16`, `32`, `64`
- regimes: tied, untied
- seeds: `101`, `103`, `107`
- measurement exposures: `0`, `512`, `1024`, `2048`, `4096`

Training, task generation, model width, residual scaling, optimizer, and stochastic probe direction remain identical to Loop Schedule Algebra v0.1. Model checkpoints are written at 2048 and 4096 state-visit exposures; the result trace records every measurement checkpoint.

## Registered Alternatives

Three models are compared on the tied geometric-mean surface:

1. A separable depth-plus-training surface.
2. A smooth depth-by-training interaction.
3. The smooth interaction plus an `R=64` depth-change term and its training interaction.

A winner requires both a four-unit AICc advantage and at least a ten-percent improvement in leave-one-exposure-block-out RMSE. Otherwise the model comparison is unresolved.

The direct transition diagnostic is depth curvature,

`C(E) = log(kappa64(E)) - 2 log(kappa32(E)) + log(kappa16(E))`.

The first preterminal checkpoint with `C(E) <= -0.2` and a seed-bootstrap 95% upper bound below `-0.1` is the registered onset. A mechanistic transition label additionally requires the tied-to-untied `R=64` interval-gradient stress ratio to be at least five and larger than the matched `R=16` and `R=32` ratios.

## Controls And Claim Boundary

Every terminal per-seed kappa must replicate the sealed predecessor within an absolute log ratio of 0.02. Untied depth gamma must remain practically equivalent to zero, `|gamma| <= 0.1`, at terminal exposure and at least four of five checkpoints. A failed terminal check yields `instrument_drift`; a failed untied check blocks a tied-specific mechanism claim.

This study can identify temporal co-localization in one frozen small residual-loop construction. It cannot establish gradient causality, general Transformer behavior, task performance, or a general stability law.

## Resource Contract

- RAM: 2048 MB hard process cap
- CPU: 50% hard Job Object cap
- I/O: abort after three consecutive samples above 50 MB/s
- VRAM: 1500 MB process cap and 0.35 Torch allocator fraction
- timeout: 1800 seconds per phase
- aggregate execution: one GPU-hour
- cleanup: PID-owned process cleanup and post-run RAM/GPU audit; foreign processes are observed but never terminated

No trajectory outcome existed when this registration checkpoint was authored.
