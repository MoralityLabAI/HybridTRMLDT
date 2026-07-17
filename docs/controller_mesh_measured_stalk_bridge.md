# Measured Qwen Stalk Bridge

## Question

Does a target-blind measured model geometry improve controller-architecture ranking beyond both controller-only
spectra and categorical policy labels?

The whole-family replication rejected incremental value from the original six controller spectral summaries. This
successor changes the scientific question rather than tuning those features. It binds an independently measured
Qwen layer-23 restriction graph to each controller stability sheaf through a preregistered external product.

## Measured Source

The source is the fresh Qwen3.5-0.8B layer-23 Stage-B result in RSITopology. It contains base and naive-QLoRA
states, four context shards, and separate construction and geometry-validation halves. The source experiment
consumed no model answers or controller outcomes and performed no generation, gradients, or weight mutation.

The compact receipt binds:

- 16 activation chunks, each `288 x 1024 float32`;
- 18,878,784 total activation bytes;
- 10 rank-one restrictions over eight state/context nodes;
- every chunk, state-index, edge-receipt, stage-result, and repository-receipt hash;
- a coherent layer-23 graph with `beta_1=3` and trivial `w1` syndrome.

The restriction weight is fixed before controller outcomes:

```math
w_e=\min\{r_e^{\mathrm{construction}},r_e^{\mathrm{validation}}\}.
```

All measured transport signs are positive. The six context edges have weights from `0.9103` to `0.9615`; the four
checkpoint edges have weights from `0.9962` to `0.9968`.

This is graph-reachability geometry from a development model. It is not semantically aligned to the nine
controller applications. The bridge tests whether its measured restriction structure is a useful external factor,
not whether Qwen representations solve those tasks.

## External Product

For the measured graph, one coboundary row per edge is

```math
(\delta_Q x)_e=\sqrt{w_e}(x_{s(e)}-\sigma_e x_{t(e)}),
\qquad L_Q=\delta_Q^\top\delta_Q.
```

After maximum-eigenvalue normalization, the measured spectrum is

```text
0.0000, 0.1054, 0.3614, 0.3850, 0.4904, 0.6150, 0.7464, 1.0000.
```

For controller genome `g`, let `L_C(g)` be its calibration-only stability Laplacian over proposer, evidence,
gate, and executor interfaces. The bridge is the external-product sheaf with Kronecker-sum Laplacian

```math
L_B(g)=L_C(g)\otimes I_Q+I_C\otimes L_Q.
```

The implementation computes its eigenvalues exactly as all pairwise sums

```math
\lambda_{ij}^{B}=\lambda_i^C+\lambda_j^Q
```

and normalizes once by the product spectrum's maximum. No controller episode is paired with an unrelated Qwen
prompt; the two measured spaces meet only through the external product.

The six frozen bridge features are spectral gap, low-band fraction, slow-mode fraction, and normalized heat traces
at `t=1,4,16`:

```math
H_t(L_B)=\frac{1}{\dim L_B}\sum_k e^{-t\lambda_k^B}.
```

The combined predictor receives the original six controller features plus these six product features.

## Protocol

The task seed `20260917` is fresh. The benchmark contains 576 calibration and 576 evaluation tasks across nine
applications. As in the independent replication, 64 `identical`/`ldt` fallback genomes form discovery and 64
`safe_ldt`/`correction_infused` genomes form the complete-family holdout.

Three ridge predictors are fitted to the same discovery outcomes:

| Predictor | Inputs |
|---|---|
| Combined | six controller spectra plus six measured-product features |
| Controller | six controller spectra |
| Categorical | evidence/fallback one-hot labels plus threshold and threshold squared |

All three parameter sets and held-out predictions are sealed together before held-out outcomes.

N0 independently permutes measured restriction weights within the `context` and `checkpoint` edge types. It
preserves graph topology, endpoints, signs, rank, edge-type counts, and each type's exact weight multiset. Each of
128 N0 geometries receives a newly fitted combined predictor.

## Registered Result

The bridge gate fails three of seven required checks:

| Endpoint | Required | Observed | Passed |
|---|---:|---:|---:|
| Combined rho | at least +0.400 | +0.7523 | yes |
| Combined minus categorical rho | at least +0.050 | +0.1121 | yes |
| Combined minus controller rho | at least +0.050 | +0.1834 | yes |
| Bootstrap lower bound versus categorical | greater than 0.000 | -0.0855 | no |
| Bootstrap lower bound versus controller | greater than 0.000 | +0.0075 | yes |
| Typed measured-weight N0 p | at most 0.050 | 0.9380 | no |
| Top-16 uplift delta versus best baseline | at least +0.020 | -0.0112 | no |

The combined-minus-categorical 95% interval is `[-0.0855, +0.3121]`. The combined-minus-controller interval is
`[+0.0075, +0.3755]`.

Top-16 selection moves in the opposite direction from rank correlation:

| Predictor | Rho | Top-16 uplift | Actual-top overlap |
|---|---:|---:|---:|
| Combined | +0.7523 | +0.1037 | 7/16 |
| Controller | +0.5688 | +0.1093 | 9/16 |
| Categorical | +0.6402 | +0.1150 | 11/16 |

## Interpretation

The external-product feature map is useful relative to the six controller summaries: its rho gain is `+0.1834`
and the paired lower bound is positive. That supports richer spectral transforms as a modeling direction.

It does not support attribution to the measured Qwen geometry. Typed edge-weight shuffles average `rho=+0.7582`,
slightly above the observed `+0.7523`, and 120 of 128 null predictors meet or exceed the observed rho after the
finite-sample correction. The measured-weight N0 p-value is therefore `0.9380`. The combined predictor also fails
to improve top-set acquisition and does not separate reliably from the categorical baseline.

The defensible conclusion is:

> A preregistered external-product transform improves global rank prediction over the controller-only six-feature
> summary, but the exact measured Qwen restriction weights add no demonstrated value. The improvement is
> consistent with nonlinear expansion of the controller spectrum rather than source-specific neural geometry.

This is a qualified negative bridge result. It must not be reframed as validation of Qwen-informed controller
routing.

## Held-Out Controllers

Correction infusion averages `1.0012` objective and `0.9045` utility, versus `0.9071` and `0.8359` for safe LDT.
Both average `0.02794` unsafe rate. The exact-target oracle caveat remains unchanged.

## Receipts

- source config SHA-256: `5697d9acdc98e3238f81f29ab4378642bbc58ad9477019db7d71a19e2e0af22d`
- compact source receipt SHA-256: `61bf8342937b6e83d221aad86874af5f40028aac5ed7925ab1885e3047cb18d0`
- compact source file SHA-256: `8efbee2ef00b2c2424df474a4e7f5336bc680b1d5e91db998e697f781956da9b`
- bridge config SHA-256: `bbdeeb3fbdcd13cba087c4fb26d9d39488096743973416010615ff85444ac8b1`
- measured graph SHA-256: `0657e27150353572bc6b1cee9076b06cf2be6ab003be461f2afdb32c8db4704e`
- calibration geometry SHA-256: `338b2afeab2e43f9eb7d3d8d2a0c27be7cdff835ac98480ce73df93e1e1c349f`
- discovery outcomes SHA-256: `feebfeae6ce1ddb6d3bc0d90180d8b03825897062ccbec9fef76f4a02c4170ac`
- triple prediction SHA-256: `35c3babfae7b3949de009851f84bbe5782d69161fa4a24c0f2611969a27e787d`
- held-out outcomes SHA-256: `969fcc3b21fd9557f4300bcc0723041a70cfee34bb870a36498dc74246f79cb9`
- analysis receipt SHA-256: `b87888e0d62f40d18303897340e981a86b1023d32e51e3a37e668b5facaad355`
- result-file SHA-256: `f689608403ff920ee838b8b2034baafe2b5601e3e7784f3b582b52eccb3a2423`

## Resource Boundary

No new model capture or training was run. Source extraction streamed file hashes over 18.9 MB of already captured
activations. Bridge analysis used small CPU Laplacians and ridge regressions, no GPU, no model weights, no
generation, and no gradients. The upstream activation capture retains its own hard-cap and cleanup receipts.

## Next Test

Do not tune the frozen site, rank, coupling, heat times, or edge weights against this outcome. A separately
registered successor would need multiple independently measured stalks with task-family alignment and a
source-identity holdout. It should test whether the correct source beats both typed shuffles and matched synthetic
spectra. Without that source-specific discrimination, measured-stalk language adds provenance but not predictive
content.

## Reproduction

```powershell
python -m research_gym.scripts.extract_measured_qwen_stalk
python -m research_gym.scripts.bench_controller_mesh_stalk_bridge
python -m pytest tests/test_measured_stalk_bridge.py -q
```
