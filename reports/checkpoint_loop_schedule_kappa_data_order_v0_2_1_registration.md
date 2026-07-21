# Kappa Data-Order Intervention v0.2.1 Registration

## Question

Does the registered `R=64`, `E=2048` kappa trough persist when training-data order changes while model initialization, task transform, measurement batch, and probe direction remain fixed?

The construction seed is `103`. It was selected after parent outcomes because it has the clearest individual `R=64` trough, so every inference is conditional on that disclosed selection. Data-order seed `103` is a mandatory replay control; fresh order seeds are `211`, `223`, and `227`.

## Frozen Branches

Each order passes the trough rule when

```text
log(K2048/K1024) <= -0.1
and
log(K4096/K2048) >= 0.1.
```

- `persists_under_data_order_reseed`: at least two of three fresh orders pass.
- `data_order_sensitive`: at most one of three fresh orders passes.
- `instrument_drift`: the order-103 replay differs from the sealed parent by more than `1e-12` at any kappa checkpoint.
- `resource_or_training_failure`: a cell is nonfinite or incomplete.

The per-order suffix-sensitivity and gradient interference ratios at `E=2048` are diagnostics, not branch gates.

## Seed Isolation

The runner exposes five recorded channels:

| Channel | Frozen value |
|---|---:|
| Model initialization | 103 |
| Task permutation/signs | 103 |
| Measurement batch | 103 |
| Probe direction | 103 |
| Training-data order | 103, 211, 223, 227 |

Changing an ordinary legacy seed would alter all five and is inadmissible for this intervention.

## Execution Plan

- Four tied `R=64` cells, executed serially.
- State-visit exposure target `4096` per cell.
- Kappa checkpoints at `E={0,512,1024,2048,4096}`.
- Model and optimizer checkpoints at `E={2048,4096}` with two-second write pacing.
- Hard caps: `2048 MB` process RAM, `50%` CPU, `50 MB/s` sustained-I/O abort, `1500 MB` VRAM, `1800 s` phase timeout, and one aggregate GPU-hour.
- Every child is assigned to the Windows Job Object wrapper; foreign GPU processes are observed and cause preflight abort, never termination.
- Abort, cap breach, nonfinite training, or cleanup failure is retained as a first-class result.

## Frozen Identity

- Config: `configs/loop_schedule_kappa_data_order_v0_2_1.json`
- Config SHA-256: `5d02d12fe582b170284e29403afa9e90f8fea04ee60c0185dac0cb74c988ea11`
- Registration status: `frozen_before_any_seed_decoupled_training_outcome`
- Full repository verification before commit: `322 passed in 50.42s`

## Claim Boundary

This post-outcome-selected intervention can determine whether the seed-103 `R=64` trough persists when only training order changes. Persistence does not establish an intrinsic stability boundary; sensitivity does not prove a single anomalous batch. No Transformer, task-performance, asymptotic, or optimizer-causal claim is licensed.
