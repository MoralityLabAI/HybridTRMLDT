# Loop Schedule Algebra v0.1 protocol

## Question and scope

LSA v0 measured a tied visit-alignment exponent of `gamma=0.422909` with
`R^2=0.994025` at `R={2,4,8}`, while its visit-untied control remained near
zero. V0.1 asks whether that measurement survives a held-out loop count,
whether the exponent changes over training, whether the tied/untied separation
appears in a small attention+MLP loop on a second task, and whether the
left-censored stability result changes across learning rates.

The frozen claim is limited to direct visit-alignment and trainability at small
scale. The campaign makes no task-performance, sample-efficiency, general
Transformer, or general stability claim.

## Primary held-out test

The v0 tied fit is not refit before the holdout is scored:

```text
log(kappa) = 0.0259712741 + 0.4229090879 log(R)
kappa_hat(16) = 3.3152208460
```

The geometric-mean tied `R=16` measurement is divided by this prediction. A
ratio in `[0.8,1.25]` supports the three-point extrapolation. The ratio and log
residual are always reported; this binary interval is only a compact protocol
decision. The four-point exponent and a deterministic seed-bootstrap interval
are computed after the holdout comparison.

The visit-untied `R=16` cell is the matched negative control. It must be
reported even if the tied holdout misses.

## Alignment trajectory

The original ignored checkpoints are hashed and replayed as a provenance
check. They are not the primary `gamma(t)` evidence: the old step cadence gives
four snapshots at `R=2`, two at `R=4`, and only the terminal snapshot at `R=8`.
There is no common nonterminal exposure across the original fit.

Primary trajectories are therefore regenerated at common state-visit
exposures `{0,512,1024,2048,4096}` for `R={2,4,8,16}`. At each exposure, kappa
is measured on a fixed held-out batch and gamma is fit across loop counts.
Seed-bootstrap intervals resample the three paired seeds with replacement.

The result label is frozen before outcomes:

- `learned_growth`: final minus initial gamma is at least `0.1`, and at least
  three of four checkpoint transitions are nondecreasing.
- `invariant`: the range of checkpoint gamma values is at most `0.1`.
- `mixed`: neither rule holds.

These labels separate a mechanism claim from post-hoc visual interpretation.

## External-validity cell

A mini causal attention+MLP residual block is trained on deterministic prefix
context regression. Its outer residual uses the same explicit `alpha,beta`
parameterization as the MLP loop, while the branch contains manual multi-head
causal attention and a two-layer GELU MLP. Tied and visit-untied models are
measured at `R={2,4,8,16}` with the same estimator.

This is a small mechanistic replication, not a language-model result. It
supports transfer only if both fits are finite with `R^2>=0.8` and tied gamma
exceeds untied gamma by at least `0.1`.

## Censored-boundary ladder

The v0 `R={6,12}` cells were stable throughout `p={0.15,...,0.65}` at learning
rate `0.001`, yielding only `p*<=0.15`. V0.1 expands the aggressive edge to
`p={0,0.05,0.10,0.15}` and crosses it with learning rates
`{0.0003,0.001,0.003,0.01}`. Stability uses the unchanged finite-run,
gradient-norm, loss-ratio, and two-of-three seed rules.

At a learning rate where both boundaries are identified, the registered depth
ordering is `boundary_R12>=boundary_R6`. Censored comparisons are explicitly
non-identifying. If all cells remain stable, the result remains left-censored;
the protocol does not promote a grid endpoint to an observed boundary.

## Execution and integrity

Each phase runs under a Windows Job Object with 2,048 MB process memory and
50% CPU hard caps. Sustained I/O above 50 MB/s aborts after three samples;
observed VRAM above 1,500 MB aborts; PyTorch receives a 0.35 allocator fraction;
and each phase times out after 3,600 seconds. GPU preflight aborts on foreign
compute use and never kills foreign processes. Aggregate execution is capped
at 24 GPU-hours.

The config is hash-registered before outcome code is run. Canonical records,
checkpoint inventories, resource receipts, result receipts, and source hashes
are reverified before any claim is written.
