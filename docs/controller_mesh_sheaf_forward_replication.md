# Independent Whole-Family Spectral Replication

## Question

Does controller-mesh spectral geometry predict held-out controller utility beyond a simple categorical description
of the architecture?

The v1 forward study predicted 16 unseen combinations at `rho=+0.705`, but its categorical comparator was post-hoc
and nearly as strong. This replication freezes that comparator as co-primary, uses a new task seed, expands the
held-out panel to 64 genomes, and holds out complete fallback families rather than thresholds within known
families.

## Frozen Design

The registered cross contains 128 genomes:

| Axis | Values |
|---|---|
| Evidence authority | none, soft, exact, dual |
| Confidence threshold | 0.00, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00 |
| Rejection fallback | identical, LDT, safe LDT, correction-infused |

All 64 `identical` and `ldt` genomes form discovery. All 64 `safe_ldt` and `correction_infused` genomes are held
out. Evidence and threshold combinations are therefore exactly balanced across the split, while fallback families
do not overlap.

The task seed is `20260817`. Each of nine applications contributes 64 calibration and 64 evaluation tasks, for
576 tasks in each partition. The correction-infused fallback is fitted only on calibration hard cases using exact
safe-action utility. That target is available because this is a synthetic gym; it remains an oracle-backed control
arm rather than a deployment procedure.

## Reveal Discipline

The implementation uses this fixed order:

```text
generate independent calibration and evaluation tasks
fit the correction fallback on calibration hard cases
construct and seal calibration geometry for all 128 genomes
reveal evaluation outcomes for the 64 discovery genomes
fit spectral and categorical ridge predictors
seal both predictors and all 64 held-out predictions together
reveal held-out outcomes
score the paired bootstrap, N0 panel, and registered gate
```

The dual-prediction receipt binds the config hash, geometry receipt, both predictor parameter sets, and both
predictions for every held-out genome. Its material contains no outcome field. A regression test recomputes every
receipt and pins the negative result.

## Co-Primary Predictors

The spectral predictor uses the six preregistered intervention and stability features:

```math
x_S=(\lambda_I,r_I,s_I,\lambda_S,r_S,s_S).
```

Here `lambda` is spectral gap, `r` is low-band rank below 0.1, and `s` is slow-mode rank below 0.25. The categorical
predictor uses evidence and fallback one-hot indicators together with threshold `gamma` and `gamma^2`:

```math
x_C=(1_E,1_F,\gamma,\gamma^2).
```

Both use separately standardized ridge regression fitted on the same 64 discovery outcomes:

```math
\widehat\beta_j=
\arg\min_\beta\|y_D-X_{j,D}\beta\|_2^2+0.01\|\beta\|_2^2,
\qquad j\in\{S,C\}.
```

The primary increment is the paired held-out Spearman difference

```math
\Delta_\rho=\rho(\widehat y_S,y_H)-\rho(\widehat y_C,y_H).
```

Its percentile interval uses 2,000 paired genome bootstraps. The top-16 endpoint similarly compares each
predictor's selected-set uplift above the same held-out mean.

The matched N0 panel shuffles module transports while preserving topology, stalk rank, restriction singular
spectrum, and the constant section. It tests whether the observed spectral predictor is stronger than those
structure-preserving null geometries; it does not test superiority to the categorical predictor.

## Registered Gate

Every condition was required. Three of five failed:

| Endpoint | Required | Observed | Passed |
|---|---:|---:|---:|
| Spectral held-out rho | at least +0.400 | +0.568 | yes |
| Spectral minus categorical rho | at least +0.050 | -0.030 | no |
| Paired bootstrap lower bound | greater than 0.000 | -0.317 | no |
| Matched N0 p | at most 0.050 | 0.0078 | yes |
| Spectral minus categorical top-16 uplift | at least +0.020 | -0.0234 | no |

The paired interval for `Delta_rho` is `[-0.317, +0.246]`; its positive bootstrap share is `0.417`. Spectral
top-16 uplift is `+0.0928`, compared with `+0.1162` for the categorical baseline. Spectral top-16 overlap with the
actual top set is `8/16`; categorical overlap is `11/16`.

## Interpretation

The spectra carry nonrandom absolute signal: `rho=+0.568` and N0 `p=0.0078`. They do not provide the registered
incremental value. The categorical predictor is better by `0.030` in rank correlation and `0.0234` in top-set
uplift. The correct result is therefore a failed replication of unique spectral prediction, not a weakened
success claim.

This comparison is conservative in favor of the spectral model. Complete family holdout leaves the categorical
model with no fitted coefficient for either held-out fallback family; it can transfer only evidence and threshold
effects. Even that baseline outperforms the spectral geometry. The present spectra may largely re-encode coarse
policy structure without adding a useful architecture-acquisition signal.

The v1 absolute forward result remains valid for its panel, but its unique-information hypothesis does not survive
the preregistered comparator and whole-family split. The sheaf construction remains usable as a descriptive or
mechanistic coordinate system. It is not validated here as a controller-search objective.

## Held-Out Families

| Fallback family | Genomes | Mean objective | Mean utility | Mean unsafe rate |
|---|---:|---:|---:|---:|
| safe LDT | 32 | 0.9011 | 0.8319 | 0.0265 |
| correction-infused | 32 | 1.0440 | 0.9325 | 0.0265 |

Correction infusion transfers strongly to the new seed without changing mean unsafe rate. The gain is controller
correction supplied by an exact calibration target, not evidence that a proposer learned or that such targets are
available in deployment.

## Receipts

- semantic config SHA-256: `6728719cdccf2f1dc425eca234e554230ed9abd8db2720356f58dfba159083ec`
- task corpus SHA-256: `c8826ee423370edd60377aeebc3e5c8e5663167084f5c239e26626bb52cd418a`
- calibration geometry SHA-256: `31a0250dab855c42a5a4cb6e9939450b5d1003f78d2199a89178d46d87e3176e`
- discovery outcomes SHA-256: `f07804f5eb8e3d1fe566ba945cf19a0947c25574af17ffe12d5dedc803c56c8f`
- dual prediction SHA-256: `0111580842739741ab927613f05b7d4bc3d47d0cefe6f5d52b0d89a2c66dd1c9`
- held-out outcomes SHA-256: `ebb6ab29a783d4e1d58689a3341224f0693bf99c24c43e081ebcbc82fd940b6b`
- analysis receipt SHA-256: `adee88b22db004644e6e88a51604cf37bde4627a61d7c5dd6defa8f5108433ca`
- result-file SHA-256: `26fef973035249f081f83a3f6cde1bdf7df91179029ba5b17b3ab3bf11dfc1da`

## Boundary And Next Test

This is an independent-seed replication inside one deterministic proxy-task generator. It does not establish
population inference, causal spectral control, neural safety, or deployment utility. The negative gate must not be
tuned away.

Do not scale the same feature panel as an acquisition function. A defensible successor must change the scientific
question: bind a measured model stalk or typed restriction-map residual into the graph, freeze a mechanistic
hypothesis, and again require increment over the categorical baseline. A null result should retire that proposed
mechanism rather than trigger post-outcome feature selection.

## Reproduction

```powershell
python -m research_gym.scripts.bench_controller_mesh_sheaf_replication
python -m pytest tests/test_controller_mesh_forward_replication.py -q
```

Artifacts:

- `configs/controller_mesh_sheaf_forward_replication_v2.json`
- `data/benchmarks/controller_mesh_sheaf_forward_replication_v2.json`
- `experiments/controller_mesh_sheaf_forward_replication_v2/`
- `reports/controller_mesh_sheaf_forward_replication_v2.md`
