# Loop Schedule Algebra saturation addendum v1: R64 holdout

## Holdout result

The untouched `R=64` tied geometric-mean alignment was `1.509898`, against the
two remotely sealed predictions:

| Candidate | Sealed prediction | Absolute error |
| --- | ---: | ---: |
| saturating exponential | 2.551038 | 1.041141 |
| logarithmic | 2.876785 | 1.366887 |

The saturating prediction is closer by `0.325747`, but its winner-to-loser
error ratio is `0.761687`, above the registered maximum `0.75`. The registered
discriminator therefore returns `unresolved`. Saturation also fails its
preregistered fit-quality and monotone-tail checks. The final classification is
`form_unresolved`, not `saturation_supported`.

The measured depth trajectory is nonmonotonic: tied kappa peaks at
`kappa(16)=2.550934`, falls to `2.307170` at `R=32`, and falls again to
`1.509898` at `R=64`. Local effective gamma is `-0.144901` from `16->32` and
`-0.611674` from `32->64`. A six-point power-law diagnostic has
`gamma=0.048106` and `R^2=0.055911`; it is descriptively inadequate.

## Optimization stress

All cells remained finite and all losses decreased, but tied `R=64` maximum
gradient norms were `149.466`, `165.824`, and `160.771`. The largest final to
initial loss ratio was `0.613923`. The matched untied cells had maximum gradient
norm at most `1.565090`.

This coincidence limits the interpretation. The result rejects the registered
smooth saturating and logarithmic extrapolations; it does not establish that a
stationary architectural ceiling caused the decline. The decline may be a
depth-by-training-state transition associated with the sharply higher tied
gradient scale. Bounded kappa still would not remove the explicit `R_g` factor
from `B(A)`, and these traces directly show that optimization stress does not
vanish at `R=64`.

## Untied control

The untied geometric mean at `R=64` was `1.097438`. Across
`R={2,4,8,16,32,64}`, its point exponent was `0.024572`, well inside the
registered practical magnitude bound `abs(gamma)<=0.1`. The bootstrap interval
was `[0.018400,0.033191]`, however, so it did not contain zero. The conjunction
registered for the flatness control therefore fails.

This is reported as a failed-as-registered control, not silently replaced. It
also shows why a future null-control gate should be an equivalence test against
a predeclared practical interval rather than requiring a confidence interval to
contain exactly zero: increasing precision can reject exact zero while the
effect remains operationally tiny. That future criterion was not used to score
this addendum.

## Integrity and resources

- prediction artifact SHA-256: `910f4d994be6d24afb8bb370b9b6e42a2b907df7b4b82245516286295119ed42`
- prediction commit: `d91b745007c25bc01403aa12b67de73acb969c77`
- holdout records SHA-256: `04840e2e4586578559a847b49d34e66c53092341bab0be4deeb6d6614f1241fe`
- holdout result SHA-256: `bb5a3223a7529b698692863bbcb44899b8220dcd44a202760934507abb3e9648`
- elapsed: `199.224 s`; aggregate training-phase elapsed: `318.354 s`
- peak RAM: `929.957 MB`; peak I/O: `41.910 MB/s`; peak VRAM observed: `0 MB`
- finite cells: `6/6`; cleanup: passed

The strict result is a registered negative: neither candidate functional form
predicted the high-loop regime within the frozen winner rule.
