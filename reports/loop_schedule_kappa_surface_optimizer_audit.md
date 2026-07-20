# Kappa Surface v1: Optimizer-Schedule Confound Audit

## Verdict

The transient `R=64` alignment trough is not explained by an explicit learning-rate warmup, decay, or schedule knee. The frozen runner uses AdamW with a constant learning rate of `0.001`, and its execution path contains no scheduler or warmup operation.

The sealed combined records also contain the step-aligned cells omitted from the first audit figure. They reject a depth-generic step-4 trough: geometric-mean kappa rises at step 4 for `R=16` and `R=32`, while it falls only for `R=64`. This does not establish that the high-depth trough is intrinsic to the loop dynamics. An `R`-amplified early-Adam transient remains viable because AdamW's moment estimates and bias correction are based on only four updates at the `R=64` trough, and the experiment does not intervene on optimizer state independently.

![Constant learning rate and exposure-to-step mapping](figures/lsa_kappa_optimizer_schedule_audit.svg)

## Frozen Execution

The Recovery 2 config fixes:

- optimizer: `adamw`
- learning rate: `0.001`
- weight decay: `0.0`
- batch size: `8`
- state-visit target: `4096`
- loop depths: `R in {16,32,64}`

The runner constructs `torch.optim.AdamW` once, calls `optimizer.step()` once per batch, and never constructs or advances a scheduler. State-visit exposure advances by

`E = optimizer_step * batch_size * R`.

Therefore the registered checkpoints map to optimizer steps as follows:

| Exposure | R=16 | R=32 | R=64 |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1024 | 8 | 4 | 2 |
| 2048 | 16 | 8 | 4 |
| 4096 | 32 | 16 | 8 |

Exposure `2048` is not a common optimizer step. Conversely, optimizer step `4` occurs at exposures `512`, `1024`, and `2048` for `R=16`, `R=32`, and `R=64`, respectively.

## Step-Aligned Discriminator

The frozen combined records already include `R=16/E=512` and `R=32/E=1024`. No micro-run or post-registration measurement was needed. Aligning all depths at optimizer step 4 gives:

| R | Previous point | Step-4 point | Next point | Step-4 shape |
|---:|---:|---:|---:|---|
| 16 | E=0: 1.825 | E=512: 1.949 | E=1024: 2.148 | rising |
| 32 | E=512: 1.539 | E=1024: 1.586 | E=2048: 1.991 | rising |
| 64 | E=1024: 0.966 | E=2048: 0.824 | E=4096: 1.510 | trough and recovery |

The step-4 log changes from the previous measurement are `+0.066`, `+0.030`, and `-0.159` for `R=16`, `R=32`, and `R=64`. The excursion is therefore neither a shared exposure-2048 event nor a depth-generic step-4 event in the observed range.

The seed-level values are heterogeneous. Previous-to-step-4 signs are `(+,+,-)` at `R=16`, `(+,+,-)` at `R=32`, and `(+,-,-)` at `R=64`; all three `R=64` seeds then rise from step 4 to step 8. The registered paired depth-curvature interval at exposure `2048` excludes zero, but the small step-aligned ensemble cannot distinguish a depth-amplified optimizer transient from a loop-intrinsic stability episode.

## Stress Scaling

At the registered curvature onset, tied-to-untied interval-gradient stress is `6.541x`, `21.620x`, and `96.891x` for `R=16`, `R=32`, and `R=64`. The local two-point exponent from `R=32` to `R=64` is

`log2(96.891 / 21.620) = 2.164`.

Across `R=16` to `R=64`, the corresponding endpoint exponent is `1.944`. These are descriptive slopes over two or three depths, not an established stress scaling law. They motivate a held-out `R=128` transient-prognostic test but do not license it without a new frozen protocol.

## Held-Out Discriminator

At `R=128` with batch size 8:

- an exposure-pinned onset predicts the excursion at `E=2048`, optimizer step 2;
- a step-pinned onset predicts the excursion at `E=4096`, optimizer step 4;
- a measurement after step 4 tests whether the system recovers or crosses into persistent instability.

This is a divergent prediction on one new depth. It must be frozen before the `R=128` run, and its outcome must not be blended with the retrospective step-aligned check.

## Scope

The completed alignment v0.1 note remains unchanged. Its perimeter is learned alignment, bounded terminal depth dependence, and envelope rejection. The recovered excursion remains a separate v0.2 candidate until a preregistered transient model predicts onset, depth, and recovery on new data.
