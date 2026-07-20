# Loop Schedule Algebra v0.1: primary checkpoint

Primary attempt 2 is the first interpretable v0.1 outcome. Its fixed-direction
terminal probes reproduce the sealed v0 geometric-mean kappas exactly at
`R={2,4,8}`. This clears the comparability failure that invalidated attempt 1.

## Gamma over training

| State-visit exposures | tied gamma | R-squared | 95% seed-bootstrap interval |
|---:|---:|---:|---:|
| 0 | 0.1304 | 0.6116 | [0.1000, 0.1834] |
| 512 | 0.1570 | 0.6829 | [0.0913, 0.2176] |
| 1,024 | 0.2000 | 0.8214 | [0.1511, 0.2624] |
| 2,048 | 0.2406 | 0.8564 | [0.2062, 0.2794] |
| 4,096 | 0.3095 | 0.9140 | [0.2868, 0.3278] |

The preregistered label is `learned_growth`: final minus initial gamma is
`+0.1790`, and all four checkpoint transitions are nondecreasing. The poor
early `R^2` values matter. The result is not merely that a fixed power-law
coefficient increases; the power-law description itself becomes better formed
during training.

## Held-out R=16

The untouched v0 fit predicted `kappa(16)=3.315221`. The observed tied
geometric mean is `2.550934`, a ratio of `0.769461` and log residual
`-0.262065`. This misses the registered confirmation interval `[0.8,1.25]`.
After adding `R=16`, the tied exponent is `0.309485` with `R^2=0.914049`, not
the v0 three-point value `0.422909`.

The matched untied four-point estimate remains near zero at `0.017527`, but its
`R^2=0.796134` narrowly misses the registered `0.8` power-law quality gate. It
is therefore evidence of a flat negative control, not a licensed untied
power-law fit. The tied-minus-untied point-estimate contrast is `0.291958`.

The upper terminal tied bootstrap bound is `0.327830`, well below the
`gamma=1` aligned envelope. The licensed primary conclusion is thus two-part:
alignment is learned over this short training horizon, while the original
three-point exponent does not extrapolate to `R=16`.

## Integrity and resources

- records SHA-256:
  `1fc75ed2c26d110d9b5a3205a12683dbbb0c11cd1dee0a18b2866466b9ea090e`
- result SHA-256:
  `f3a1f5c81058c673c84a5ae49b71526758c936fe198f67bc0277bedf737ea40e`
- resource receipt SHA-256:
  `b844377c04cb5262e692635a54ec72bf0372485c24a7001f99ddf1fcbd47177e`

The phase completed in 77.671 seconds with 895.730 MB peak RAM and 14.728
MB/s peak observed I/O. Cleanup passed and no owned process lingered. As in the
short replay phase, one-second telemetry did not catch a nonzero VRAM sample;
the allocator cap still applied, but the receipt does not establish actual
zero CUDA allocation.
