# Visit Alignment Is Learned in Small Weight-Tied Residual Loops

## Abstract

Repeated use of one parameterized block changes how its update is accumulated
and read through an unrolled computation. DeepLoop formalizes this effect with
a visit-alignment coefficient `kappa_R` and analyzes decorrelated and aligned
envelopes, but leaves direct measurement of the coefficient as future work.
We directly estimate visit alignment in small tied residual loops by combining
per-visit suffix sensitivities with gradients from recomputations in which
exactly one application has live parameters.

An initial three-point measurement at `R={2,4,8}` gave
`kappa approximately R^0.423` with `R^2=0.994`, while a visit-untied control
gave `gamma=0.026`. A preregistered extension adds a held-out `R=16` point and
common exposure checkpoints. The held-out kappa is `2.551`, below the sealed
prediction `3.315` (ratio `0.769`), and the four-point exponent is
`gamma=0.309` with 95% seed-bootstrap interval `[0.287,0.328]`. More
importantly, gamma rises monotonically from `0.130` at initialization to
`0.309` after 4,096 state-visit exposures. Under the frozen rule, visit
alignment is classified as `learned_growth`, not an architectural constant.

A mini causal attention+MLP loop on prefix-context regression yields tied
`gamma=0.501` versus untied `gamma=0.018`. Its strict preregistered replication
gate does not pass because the nearly flat untied sequence has poor power-law
fit (`R^2=0.051`), despite a tied-minus-untied contrast of `0.483`. Finally, all
96 residual-scaling ladder cells remain stable down to `p=0` over learning
rates `0.0003` through `0.01`, leaving the stability boundary left-censored at
`p<=0` and the proposed depth ordering non-identifying.

The results reject using the fully aligned envelope as a descriptive model for
these small loops; they do not invalidate it as a conservative bound. They
also do not establish language-model performance, large-scale transfer, or a
general optimization threshold.

![Gamma grows during training](figures/lsa_v0_1_gamma_trajectory.svg)

## Measurement

For each gradient-visible visit `r`, let `U_r` be a top-direction JVP through
the suffix after that visit and `G_r` the parameter gradient from a recomputed
loss where only that application has live parameters. The scale-free
coefficient is

```text
kappa_g = ||sum_r U_r|| ||sum_r G_r||
          / (R_g max_r ||U_r|| max_r ||G_r||),    0 <= kappa_g <= R_g.
```

We fit

```text
log(kappa_R) = c + gamma log(R)
```

to geometric-mean kappas across seeds `{101,103,107}`. The schedule-visible
stability functional is

```text
B(A) = sum_j J_j R_g(j) kappa_g(j) (beta_j / alpha_j)^2.
```

Using `R_g` rather than forward visit count `R` matters when backpropagation is
truncated or parameter applications are frozen. Residual scale is excluded
from kappa and applied once in `B(A)`.

## Primary result

| Exposures | gamma | R-squared | 95% bootstrap interval |
|---:|---:|---:|---:|
| 0 | 0.1304 | 0.6116 | [0.1000, 0.1834] |
| 512 | 0.1570 | 0.6829 | [0.0913, 0.2176] |
| 1,024 | 0.2000 | 0.8214 | [0.1511, 0.2624] |
| 2,048 | 0.2406 | 0.8564 | [0.2062, 0.2794] |
| 4,096 | 0.3095 | 0.9140 | [0.2868, 0.3278] |

The exponent increases by `0.1790`, and all four transitions are
nondecreasing. Early power-law fits are weak, so the mechanism is more precise
than "gamma increases": a coherent scaling relation becomes better formed
during training.

The terminal `R={2,4,8}` kappas reproduce the sealed v0 values exactly under a
fixed probe direction. The new `R=16` point breaks the old extrapolation. The
four-point tied exponent falls to `0.3095`; the matched untied estimate remains
near zero at `0.0175`, though its `R^2=0.7961` narrowly misses the registered
fit-quality threshold.

![Kappa scaling across model families](figures/lsa_v0_1_kappa_scaling.svg)

## Attention+MLP cell

The second model family uses manual multi-head causal attention and a two-layer
GELU MLP inside one explicit outer residual. It is trained on deterministic
prefix-context regression at the same state-visit exposure budget.

| Regime | gamma | R-squared | Kappa at R=16 |
|---|---:|---:|---:|
| tied | 0.5012 | 0.8540 | 2.4021 |
| untied | 0.0179 | 0.0509 | 0.7426 |

The tied estimate has a good four-point fit and the contrast is large. The
strict composite gate nevertheless returns false because it required the
untied negative control to itself follow a high-quality power law. That was a
poor criterion for a flat null, but changing it after outcomes would erase the
distinction between preregistered confirmation and post-hoc interpretation.

## Stability ladder

The original boundary sweep was stable at every `p>=0.15`. The extension tests
`p={0,0.05,0.10,0.15}` across four learning rates and `R={6,12}`. All 96
seed-level runs remain stable. Maximum gradient norm is `66.07` against a
cutoff of 100; maximum final/initial loss ratio is `0.686` against a cutoff of
10. Every aggregate boundary is left-censored at `p<=0`.

![All ladder cells remain stable](figures/lsa_v0_1_boundary_ladder.svg)

## Interpretation

1. Weight tying creates measurable visit alignment. Both model families show
   a positive tied exponent and near-zero untied point estimate.
2. Alignment is learned in the primary loop. It is already positive at
   initialization but grows substantially as training proceeds.
3. The original `gamma=0.423` was not a transferable constant. `R=16` supplies
   the residual check that the three-point fit lacked.
4. The `gamma=1` aligned case is conservative rather than descriptive here.
   The terminal primary upper interval is `0.328`.
5. The tested stability threshold still does not bind. The LR ladder cannot
   identify the proposed ordering because no unstable cell was observed.

DeepLoop explicitly calls for direct measurement of `kappa_R` or cross-round
alignment. To our knowledge, this is the first direct estimate of its specific
visit-alignment coefficient, but that novelty statement is scoped to the
literature audit recorded in this repository rather than an exhaustive survey.

## Limits

The primary model is a two-linear-layer residual loop at hidden size 128. The
attention+MLP cell is small, uses deterministic regression rather than
language modeling, and changes task and block family together. Three seeds
support an interval but not a high-powered distributional claim. The stability
rule may be too permissive for this short horizon, and all ladder boundaries
remain censored. No result licenses a claim about downstream accuracy,
sample-efficiency, large-scale Transformers, or DeepLoop's empirical GPT
results.

## Reproducibility

- Frozen config SHA-256:
  `c6cd78a09909956962a2c6d585270807088a9b8b0dc5e1e69fc8b151d45882cc`
- Final receipt SHA-256:
  `3fa8c1e6f360035a8d1fc17932d18b2c6fe0249e6653c6e5aa9992a5cdf57414`
- Valid records: 42 replay, 72 primary, 24 attention+MLP, 96 ladder.
- Source replay: 18/18 terminal kappas exact, maximum absolute error `0.0`.
- Valid scientific-phase runtime: 336.169 seconds in aggregate.
- Invalid attempts are retained and excluded: drifting probe seed in primary
  attempt 1; pre-training exposure divisibility failure in ladder attempt 1.

Primary reference: Shuzhen Li, Yifan Zhang, Jiacheng Guo, Quanquan Gu, and
Mengdi Wang, [DeepLoop: Depth Scaling for Looped Transformers](https://arxiv.org/abs/2607.13491),
arXiv:2607.13491, 2026.
