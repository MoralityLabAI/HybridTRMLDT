# Late-iteration fragility, not factor specialization: a negative result on Huginn-0125

*MoralityLab, 2026-11-09. Exploratory local runs plus one registered confirmation (Silico experiment #3).
Receipts: `Huginn/results/exp3_r_confirmation/`.*

## Summary

We asked whether a recurrent-depth language model splits its work across iterations, with some
weight components used early in the loop and others late. On Huginn-0125, ablating rank-one factors
of the core block's MLP output looked like strong temporal specialization: no effect in iterations
1–8 and up to 0.77 nats in iterations 9–16. Under a registered test with activation-matched controls,
the specialization disappears. A random rotation that perturbs the activations by the same amount does
the same damage, late and not early. **Huginn's late iterations are fragile to any perturbation of that size,
and its early iterations absorb it. The factor's identity adds at most 0.0065 nats.** Per-iteration
component ablation without magnitude-matched controls reports this fragility as specialization.

## Setup

- **Model:** `tomg-group-umd/huginn-0125` at `bb6621b6`. It has a 2-layer prelude, a 4-layer core block
  applied `r` times, and a 2-layer coda, run in bf16. The recurrent state starts from noise and is reseeded
  per forward.
- **Task:** variable-binding programs (`a = 4`, `k = a`, …, `print(k)`) with distractor chains. The answer
  is read as a forced choice over the 10 digit tokens. The runbook pointer chase (`A->B` edges) is unusable on
  Huginn because its tokenizer fuses `>B` into one token. Even with spaced arrows, three-shot accuracy is
  17–25% at one hop and at chance from two hops.
- **Factors:** top singular directions of each core layer's `mlp.proj` (17920 → 5280). There is no full
  VPD fit for Huginn.

## Results

**1. Recurrence buys accuracy, but not hops.** Three-shot accuracy, 96 items per cell:

| r | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| 1 hop | 0.16 | 0.50 | 0.76 | 0.71 | 0.70 |
| 2 hops | 0.12 | 0.34 | 0.47 | 0.41 | 0.39 |
| 3–6 hops | 0.12–0.20 | 0.23–0.35 | 0.20–0.26 | 0.22–0.29 | 0.22–0.27 |

Gains stop at r = 16. From three hops, accuracy stays at the pick-a-digit-in-the-program level (about 0.33) at
every r. Huginn chases at most two bindings inside its loop on this task.

**2. The uncontrolled heat map shows apparent specialization.** At two hops, r = 16, with 64 clean-correct
items, we removed the 16 most-used factors per layer in iteration windows 1–4, 5–8, 9–12 and 13–16.
- No factor matters in iterations 1–8.
- Core layer 3's top direction costs 0.77 nats in iterations 13–16 (accuracy 0.94 → 0.42).
- Usage profiles at iteration 1 and iteration 16 correlate at only 0.07–0.24 in core layers 0–2.

The random controls were matched to a *random* factor's norm, not to the ablated factor's. The
top directions are the largest, so the controls were too small.

**3. With matched controls, factor specificity is near zero.** Silico experiment #3 tested a pre-registered
panel of eight factors:
- **Factors:** two each nominated early-only, late-only, large-effect and near-null.
- **Controls:** four input-read rotations per factor. They were selected from at least 512 candidates on
  a separate split and verified to match the factor's activation-change distribution (p10/p50/p90 within
  15%, p95 within 25%) in both windows on 120 fresh programs. 8/8 factors passed.
- **Confirmation set:** 240 sealed programs that are clean-correct under all three noise seeds.
- **Analysis:** program-clustered sign-flip tests with Holm correction across the eight factors.

| factor (nomination) | raw ablation, it 1–8 | raw ablation, it 9–16 | matched control, it 9–16 | specificity (late − early, factor − control) |
|---|---|---|---|---|
| core3 f1 (large effect) | −0.030 | **−0.366** | **−0.368** | +0.0007 [−0.003, +0.004] |
| core1 f26 (large effect) | −0.007 | −0.182 | −0.180 | −0.0014 [−0.005, +0.002] |
| core3 f17 (late only) | −0.022 | −0.145 | −0.139 | −0.0064 [−0.010, −0.003] (Holm) |
| core3 f60 (early only) | −0.040 | −0.127 | −0.127 | −0.0008 [−0.004, +0.002] |
| core1 f58 (late only) | −0.035 | −0.078 | −0.076 | −0.0007 [−0.004, +0.002] |
| core3 f44 (near null) | −0.022 | −0.030 | −0.034 | +0.0057 [+0.003, +0.009] (Holm) |
| core1 f19 (early only) | −0.011 | +0.016 | +0.010 | +0.0065 [+0.004, +0.010] (Holm) |
| core1 f31 (near null) | +0.006 | +0.019 | +0.020 | −0.0005 [−0.003, +0.002] |

All values are in nats, as the change in correct-digit log-probability.
- The raw late-window damage is real: core3 f1 reverses 31% of answers late against 10% early.
- A matched rotation that reads a *different* input direction does almost exactly the same damage.
- Three specificity terms reject zero after Holm, all below 0.007 nats.
- None of the four early-only or late-only nominations meets the registered criterion (active harm
  < −0.05 nats, with the inactive window equivalent to zero at ±0.05).
- The output-write control family also passed matching but was not causally tested.

## Interpretation

- **The temporal asymmetry is in the dynamics, not in the weights.** Perturbations in iterations 1–8 are
  repaired by the remaining iterations, and perturbations in 9–16 are not. This fits a model that settles
  toward a fixed point: what matters is how far the state is pushed and how many iterations remain to undo it.
  Which direction it is pushed in matters little.
- **Shared weights blur the component-to-role mapping.** Usage profiles differ by iteration, yet removing a
  well-used factor in its "active" window is no worse than removing a matched random direction. For this
  model and task, the components are not where the iteration-specific computation lives.
- **Methodological point.** A per-iteration ablation of a recurrent model needs controls matched to each
  ablated component's own activation change, measured per window. Norm-matching to an average component
  inflated a fragility effect into apparent specialization of up to 0.77 nats.

## Limits

- The factors are restricted SVD directions, not a trained VPD or SAE decomposition. A decomposition
  trained for faithfulness could still find iteration-specific components.
- One model, one task family, and two recurrence windows at r = 16. Huginn is weak at multi-hop binding,
  so there may be little iteration-specific computation to find.
- Inference is conditional on the 240 clean-correct programs. The four controls per factor are correlated
  draws, not independent trials.
- **Related:** Nanbeige4.2-3B (two shared passes) does not chase two or more hops, and was mapped at one
  hop only. Its VPD factors at layer 22 matter on pass 1 only (up to 0.1 nats), with no matched-control test.

## Consistent with published work

- *Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer*
  (arXiv 2507.02199): logit- and coda-lens probes of Huginn find little evidence of structured latent reasoning.
- *Loop, Think, & Generalize* (arXiv 2604.07822): finds the same on Huginn-3.5B.

We add a causal, control-matched test at the level of weight components.

## Provenance

- **Local scripts and uncontrolled heat map:** MoralityLabAI/LoopedTransformers `rmp-v0-phase2` (`bf694a0`).
  Mirrored publicly at HF `AlephFunk/nanbeige-vpd-shared-weights/rmp_local_2026-11-03` (`abc2973e`).
- **Registered confirmation:** this branch, `Huginn/` (PROTOCOL, gate, confirmation statistics). Factor
  archive sha256 `86f90a1c…4a50`; outcomes sha256 `213f6bcf…6eb8b6e48a7`; 17 CPU tests pass.
