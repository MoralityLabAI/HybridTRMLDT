# Sequencer Control-Math Benchmark

## Question

Does RSITopology-aware authorization improve a fixed library of TRM, LDT, and typed-hybrid skill sequences when
the same skills are replayed on known task families?

The treatment changes only the sequencer. It does not retrain, edit, or otherwise infuse neural weights. In this
study, "infused skill" means that each task context exposes the same three reusable skill sequences:

- `proposal_only`: TRM-style proposal/search.
- `deduction_only`: LDT-style propagation or certification.
- `typed_propose_certify`: proposal followed by typed certification.

## Registered Design

The frozen protocol in `configs/sequencer_control_v1.json` uses seed `20260713`, an error budget of `0.5`, and
separate router-model training, sequencer calibration, and evaluation partitions. Sequence utilities are fitted
only on calibration episodes. The primary endpoint is family-macro utility for `control_math` minus
`global_signed`.

The evaluation contains 667 paired episodes:

| Family | Episodes | Source |
|---|---:|---|
| Sudoku | 96 | Rule-preserving automorphisms of three local 4x4 puzzles |
| ARC-1 | 168 | Balanced procedural instances over seven primitive rules |
| ARC-2 | 160 | Balanced procedural instances over eight fixed rule pairs |
| Routing | 115 | Held-out normalized trajectories from the local Tesseract corpus |
| Storyworld secret | 64 | Disjoint finite-state starts under a hard ending predicate |
| Storyworld moral | 64 | Disjoint finite-state starts under a soft value objective |

These are known-task, local benchmark implementations. ARC-1 and ARC-2 are ARC-style procedural tasks, not
official ARC leaderboard evaluations. Routing measures the local Tesseract trajectory objective, not an official
INTELLECT-3 score.

ARC instances are retained only when all three candidate skill sequences solve them, deliberately isolating
sequencing efficiency from solver coverage. Consequently, ARC accuracy in this study is not an estimate over the
unfiltered procedural task distribution.

## Control Rule

For context `c`, two calibration folds produce independent transport estimates `T_c^(1)` and `T_c^(2)` over the
two-dimensional preference vector `(typed - proposal, typed - deduction)`. The controller records:

- lineage contraction `lambda_c`;
- loop/holonomy displacement `h_c` from disagreement between the two transport paths;
- orientation reversal `o_c`, detected by negative cosine alignment;
- simultaneous coverage and a matched-noise null receipt.

The conservative bound is

```text
R_hat(c) = min(2, lambda_c + h_c)
```

with orientation reversal forcing `R_hat(c)=2`. Global signed transfer is authorized only when the coverage and
noise receipts pass, orientation is preserved, and `R_hat(c) <= epsilon`. Otherwise the controller selects the
best calibration-fitted local section. `lineage_only` is the ablation that omits holonomy and orientation.

## Inference

- Families receive equal macro weight, preventing routing volume from dominating the endpoint.
- Confidence intervals use 5,000 paired, family-stratified hierarchical context-cluster bootstrap resamples.
- P-values use 10,000 context-cluster sign flips, so transformed episodes from one base context move together.
- Three registered control comparisons receive Holm correction.
- Cohen's `d_z` is computed over family-balanced context-cluster effects.
- Exact McNemar tests audit accuracy changes separately from utility.

## Results

| Sequencer | Macro utility | Macro accuracy | Macro cost | Violations |
|---|---:|---:|---:|---:|
| Global signed | 0.8931 | 0.9667 | 7.674 | 0.0000 |
| Lineage only | 0.8998 | 0.9667 | 6.453 | 0.0000 |
| Fixed typed | 0.8931 | 0.9667 | 7.674 | 0.0000 |
| Control math | **0.9001** | **0.9667** | **6.436** | **0.0000** |

Against global signed control, the paired macro-utility gain is `+0.006943` (95% clustered bootstrap CI
`[+0.004909,+0.009232]`, clustered sign-flip `p=0.000300`, Holm-adjusted `p=0.000600`, family-balanced cluster
`d_z=0.876`). Accuracy is unchanged (McNemar `p=1.0`), while macro cost falls by `1.238`.

Against lineage-only control, the gain is only `+0.000310` (95% CI `[0,+0.000864]`, `p=0.502350`,
Holm-adjusted `p=0.502350`, cluster `d_z=0.209`) and macro cost falls by `0.017`. The experiment therefore does
not establish incremental utility from holonomy/orientation checks beyond lineage-only localization on this
slice.

The registered controller invokes a local section for 100% of contexts, and its aggregate result equals the
`local_calibrated` reference. This is a conservative localization result, not evidence that global signed
transport is broadly safe. Orientation reversal is detected in 14.3% of contexts.

## Sensitivity

The post-registration epsilon sweep exposes the authorization tradeoff:

| Error budget | Local-section rate | Macro utility | Macro cost |
|---:|---:|---:|---:|
| 0.25 | 1.000 | 0.9001 | 6.436 |
| 0.50 | 1.000 | 0.9001 | 6.436 |
| 0.75 | 0.821 | 0.9001 | 6.436 |
| 1.00 | 0.714 | 0.8967 | 6.737 |
| 1.50 | 0.214 | 0.8949 | 6.885 |
| 2.00 | 0.143 | 0.8943 | 7.223 |

Relaxing the bound authorizes more global transfer but monotonically loses the measured utility advantage and
raises cost on this slice. Accuracy remains constant, showing that the effect is sequencer efficiency and
preference quality rather than additional solved episodes.

## Reproduction

```powershell
python -m research_gym.scripts.bench_sequencer_control
python -m pytest tests/test_sequencer_control_bench.py -q
```

The full result, per-episode receipts, and human-readable report are saved under `data/benchmarks/`,
`experiments/sequencer_control_math/`, and `reports/sequencer_control_bench.md`.
