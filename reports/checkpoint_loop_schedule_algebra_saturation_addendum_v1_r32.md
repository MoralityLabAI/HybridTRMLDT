# Loop Schedule Algebra saturation addendum v1: R32 prediction seal

## Registered fit result

The six `R=32` cells completed under the frozen resource envelope. Terminal
visit alignment was:

| Regime | Seed 101 | Seed 103 | Seed 107 | Geometric mean |
| --- | ---: | ---: | ---: | ---: |
| tied | 2.039178 | 2.609556 | 2.307900 | 2.307170 |
| untied | 1.068993 | 0.953478 | 0.997625 | 1.005582 |

The tied mean falls from `kappa(16)=2.550934` to `kappa(32)=2.307170`, giving
local effective `gamma(16->32)=-0.144901`. This is evidence against the earlier
power-law extrapolation, but it is not by itself a saturation result: the seed
spread is visible and a monotone saturating curve cannot represent the full
downturn.

## Untouched R64 predictions

The registered fits use only `R={2,4,8,16,32}`:

| Candidate | Fit R-squared | R64 prediction | 95% seed-bootstrap interval |
| --- | ---: | ---: | ---: |
| saturating exponential | 0.863095 | 2.551038 | [2.457716, 2.675424] |
| logarithmic | 0.683946 | 2.876785 | [2.673458, 3.096219] |

The fitted saturating parameters are `K=2.551044`, `A=1.589018`, and
`tau=5.108952`. The five-point legacy power-law diagnostic falls to
`gamma=0.195923`; it is excluded from the registered holdout discriminator.

The exact prediction artifact is
`experiments/loop_schedule_algebra_saturation_addendum_v1/r64_predictions.json`,
SHA-256 `910f4d994be6d24afb8bb370b9b6e42a2b907df7b4b82245516286295119ed42`.
It contains no `R=64` observation. The holdout runner must verify these bytes in
a commit on the remote branch before it can construct any `R=64` cell.

## Gate consequence

The saturating fit's `R^2=0.863095` is below the preregistered `0.9` support
threshold. Consequently, the holdout can favor the saturating predictor over
the logarithmic predictor, but this addendum cannot return the strict
`saturation_supported` classification. A saturating-model win would be reported
as `form_unresolved` with its prediction error, not promoted by changing the
gate after seeing `R=32`.

## Resource receipt

- elapsed: `119.130 s`
- peak RAM: `910.781 MB`
- peak I/O: `20.761 MB/s`
- peak VRAM observed by wrapper: `0 MB`
- maximum training gradient norm: `22.971289`
- finite cells: `6/6`
- cleanup: passed

The `R=64` holdout was not run at this checkpoint.
