# Checkpoint: LSAD v1 Closeout

## Outcome

Loop Schedule Architecture Discovery v1 closes with zero qualifying
architectures. A1 screened twelve balanced primitive-period schedule words at
S0 and provisionally selected two K4/L8 schedules. A2 tested those schedules and
their shared periodic control across three seeds at four times A1 exposure.
Neither schedule met the registered two-positive-seed requirement.

This is the `valid_null_result` declared before proposals and outcomes in
`architecture_discovery_v1_registration.json`: zero or one qualifying
architectures. No B, S1, S2, or locked-evaluation stage is opened, and the
presealed reserve remains unused.

## Quantitative finding

The primary macro score has 768 equally weighted calibration examples at the
training depth, giving quantum `1/768 = 0.0013020833`. Relative to the matched
periodic control:

| Schedule | Seed deltas | Mean delta | Net-item deltas | Positive seeds |
| --- | --- | ---: | --- | ---: |
| Paired block `[0,0,1,1,2,2,3,3]` | `-0.006510, -0.010417, +0.001302` | -0.005208 | `-5, -8, +1` | 1/3 |
| Interleaved return `[0,1,2,3,1,0,2,3]` | `-0.002604, -0.003906, -0.002604` | -0.003038 | `-2, -3, -2` | 0/3 |

For these two A2-tested K4/L8 schedules at approximately 5M parameters, on the
registered pointer-chase, modular-recurrence, and rewrite-normalization bundle,
`|mean delta| <= 0.005209` and maximum per-seed absolute delta is `0.010417`.
This bound is not a statement about every word in the grammar, larger scales,
Sudoku, routing, or general language-model quality.

## Interpretation

A1's single-seed advantage was selection at the measurement floor. Under fixed
screening compute, a v2 should favor at least two shorter seeds over one longer
seed before promotion. The close candidate/control results support the fidelity
of matching on unique parameters, applications, estimated FLOPs, exposure, and
gradient policy, while stopping short of formal instrument certification.

Together with LSA v0, this is a second clean negative on a proposed loop-design
degree of freedom at toy scale: LSA v0 did not observe the predicted
residual-scaling boundary transfer, and LSAD v1 did not observe a replicated
dispatch-order benefit. This does not show that either axis is universally
flat. It shifts the next discriminating study toward scale transfer or a new
algebraic axis rather than additional same-scale words from the sealed reserve.

## Provenance

- Registration SHA-256: `2cec9cdd659181145df2de8145d393b0021218d323936f4af4bc7e13d870a0cb`.
- A1 stage receipt SHA-256: `cabb4ae40d8d18cf3b825bfb4bb1613169c4cca40c0b0560394ec1fba4cc88ce`.
- A1 transition SHA-256: `1b06ac4e33ea00e6e1778d9fcbd863b5fd9e0d0cf4c775f0f046e383eeef63e7`.
- A2 stage receipt SHA-256: `54cfe3f51b879699e01e931553a05a6e854dd4467d135a0e89057a286c543365`.
- A2 transition SHA-256: `c23296d4c19ff6327343184aa4e68a2ab56f37f908f85f3fb26d1ad1e1e761f2`.
- Reserve status: sealed, unopened, zero batches consumed.
