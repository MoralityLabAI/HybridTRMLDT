# Forward Spectral Prediction of Controller Genomes

## Question

Can calibration-only spectral features predict the held-out utility of controller architectures that were not
used to fit the predictor?

This is the forward successor to `controller_mesh_sheaf_retrodiction_v1`. It does not reuse final executor
outcomes as spectral inputs. It creates a new panel of 64 policy genomes over deterministic source-inspired
control tasks and holds out 16 evidence/threshold/fallback combinations.

## Genome Panel

The registered cross is:

| Axis | Values |
|---|---|
| Evidence authority | none, soft, exact, dual |
| Confidence threshold | 0.00, 0.10, 0.35, 0.75 |
| Rejection fallback | identical, LDT, safe LDT, correction-infused |

One minimum-SHA256 threshold is held out inside each evidence/fallback cell. This gives 48 discovery genomes and
16 held-out genomes, with every evidence and fallback family represented four times in the held-out panel.

`soft` promotes model- or experience-sound candidates into gate authority. `exact` checks only declared
environment constraints. `dual` requires both. The correction-infused fallback fits a bounded TRM/LDT blend per
skill on calibration hard cases and always selects within the exact safe set.

The correction target is available only because this synthetic environment exposes exact calibration utility.
It is an oracle-backed control arm, not a deployment recipe.

## Staged Reveal

The implementation order is fixed and regression-tested:

```text
generate calibration and evaluation tasks
fit correction fallback on calibration hard cases
construct and seal calibration geometry for all 64 genomes
reveal evaluation outcomes for 48 discovery genomes
fit the spectral predictor
seal predictions for 16 held-out genomes
reveal the 16 held-out outcomes
score the registered gate and N0 controls
```

The held-out prediction receipt binds the config hash, calibration-geometry receipt, predictor coefficients, and
predictions. It contains no held-out outcome field. Evaluation utilities exist in the deterministic task corpus,
but no spectral-construction function reads them. The correction model is the declared exception: its upstream
parameters are fitted on calibration targets before geometry is constructed.

## Typed Laplacians

Every controller has proposer, evidence, gate, fallback, and executor stalks. Calibration messages include TRM
scores, exact and soft masks, provenance type, gate decisions, fallback scores, and selected-action indicators.
Utility and optimal-action labels are forbidden from direct spectral construction.

The intervention subgraph is

```math
E\longrightarrow G\longrightarrow F\longrightarrow X,
```

with Laplacian

```math
\widetilde L_I=
\frac{\delta_I^\top\delta_I}
{\lambda_{\max}(\delta_I^\top\delta_I)}.
```

It measures modes associated with evidence-triggered fallback execution. The stability subgraph is

```math
P\longrightarrow E\longrightarrow G\longrightarrow X,
\qquad P\longrightarrow X,
```

with normalized Laplacian `L_S`. It measures compatibility along proposal, authorization, and direct execution.
The split prevents productive fallback residuals from being silently interpreted as compositional instability.

The six registered features are gap, low-band rank, and slow-mode rank for each Laplacian. A standardized ridge
model is fitted on the 48 discovery-genome evaluation objectives:

```math
\widehat\beta=
\arg\min_\beta
\|y_D-X_D\beta\|_2^2+0.01\|\beta\|_2^2.
```

The target combines utility, exact accuracy, unsafe rate, and controller cost. The primary endpoint is Spearman
rank correlation on the 16 held-out genomes.

## Matched Null

For each of 128 N0 replicates, module transports are independently shuffled while preserving each authority
subgraph, stalk rank, restriction singular spectrum, and constant section. A new predictor is fitted on each null
geometry using the same discovery outcomes, then scored on the same held-out outcomes.

This p-value is conditional on the fixed task and genome panel. It is not population inference over arbitrary
architectures.

## Registered Result

All three registered conditions pass:

| Endpoint | Required | Observed |
|---|---:|---:|
| Held-out Spearman rho | at least 0.400 | +0.705 |
| N0 one-sided p | at most 0.050 | 0.0078 |
| Predicted top-four uplift | nonnegative | +0.1476 |

The N0 mean correlation is `+0.276`. The spectral predictor identifies two of the actual top four genomes. Its
predicted top four average `1.0560` objective versus `0.9085` over all held-out genomes.

The actual best held-out genome is exact evidence at threshold 0.35 with correction infusion:

```text
e-exact__g-035__f-correction_infused
objective = 1.1531
utility   = 0.9703
unsafe    = 0.0000
```

Three of the actual top four use correction infusion, while the fourth uses exact evidence with safe LDT. This is
evidence that independently verified correction targets can outperform accepted-only or identical fallback in
this toy control panel. The result inherits the calibration-oracle caveat.

## Post-Hoc Diagnostics

These were computed only after the registered gate was known:

| Predictor | Held-out rho | Top-four uplift |
|---|---:|---:|
| Full dual spectrum | +0.705 | +0.1476 |
| Categorical genome baseline | +0.673 | +0.0950 |
| Intervention spectrum only | +0.084 | +0.0595 |
| Stability spectrum only | +0.567 | +0.0323 |

The full spectrum is directionally better than the additive evidence/fallback/threshold baseline, but the
difference in rho is only `0.032` on 16 cases. This run does not establish unique information beyond architecture
labels. A larger independent panel must test that claim directly.

The intervention Laplacian is weak alone. Stability carries most rank information, while combining both improves
top-genome selection. This supports keeping the two residual families separate, not interpreting all low modes as
one scalar constitutional bandwidth.

The four genomes nearest the registered 0.1 phase boundary average about `0.850` objective, below the full
held-out mean. Phase-boundary proximity is therefore not validated as a utility acquisition rule in this panel;
it remains an exploration heuristic only.

## Receipts

- config semantic SHA-256: `828f13f7eca5161327a765b10032f798f5d5a4d38c2dfc4eafe0f2cf2a96697a`
- calibration geometry SHA-256: `3e563b5623a017fc3d844223423a03fc2d353283c4e1f9098e5441b662b2c8bf`
- held-out prediction SHA-256: `863d2676c1dd4943f8243c40fb83d0140348eb748911edabb04fa46f8956b38e`
- held-out outcomes SHA-256: `1a3b0e19ccd6ee3dcb4b73fd17270b4229119539929e8438cb6ee8bb38615965`
- analysis receipt SHA-256: `fc8bc845de436dde0248c365429f47b095cf9c29b8e6f92fecb555fd8dfe2f8a`
- result-file SHA-256: `928f721ca44adc80f3eea3d616e5622a55986c001fefb89c730eca66961cba0d`

## Boundary

This is a forward prediction across 16 held-out combinations in one deterministic synthetic task generator. It
does not establish causal spectral control, generalization to a new task distribution, architecture-population
inference, neural safety, or deployment utility.

## Next Test

Freeze an independent replication before changing the predictor:

1. use a new task-generator seed and at least 64 held-out genomes;
2. preregister the categorical baseline as a co-primary comparator;
3. require spectral-minus-categorical uplift with bootstrap uncertainty;
4. cross complete evidence/fallback families rather than only unseen threshold combinations;
5. retain the failed phase-boundary result as a negative control.

After that replication, bind one measured Qwen/JSpace stalk into the same typed graph without changing the
registered outcome definitions.

## Reproduction

```powershell
python -m research_gym.scripts.bench_controller_mesh_sheaf_forward
python -m pytest tests/test_controller_mesh_forward.py -q
```

Artifacts:

- `configs/controller_mesh_sheaf_forward_v1.json`
- `data/benchmarks/controller_mesh_sheaf_forward_v1.json`
- `experiments/controller_mesh_sheaf_forward_v1/`
- `reports/controller_mesh_sheaf_forward_v1.md`
