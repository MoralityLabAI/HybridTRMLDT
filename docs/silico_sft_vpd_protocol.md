# Silico handoff: RIVET SFT + mechanistic VPD (protocol v0.1)

Status: **unlaunched experiment design** for Goodfire Silico. Related architecture: `docs/causal_receipt_recurrence_proposal.md`. Run config: `configs/silico_rivet_v1.json`. This protocol contains no completed training, VPD, or independent certification claims.

## Goal

Determine whether a small, shared-weight, receipt-aware reasoner learns reusable computation that generalizes across task depth, rule compositions and procedural families. Use **independent environment verification** to label repairs, then test whether identifiable components cause the intended intermediate behavior. Evaluate interpretability-guided SFT data reweighting as a subsequent, controlled phase.

The immediate practical route is a **supported Silico model** plus typed-state SFT and activation/parameter analysis. A custom 16M–70M tied transformer may require a separate architecture-compatible training backend; verify support through Silico's access-verification capability before allocating a GPU.

Platform methods documented at https://docs.goodfire.com/concepts/capabilities/ include training/SFT, LoRA, model-difference amplification, activation harvesting, supervised probes, causal mediation, data attribution, local parameter decomposition and interpretability in training. The existence of a method is not evidence that it works on every custom model or parameter scale.

## Inputs and provenance

- Repository: `https://github.com/MoralityLabAI/HybridTRMLDT`, a specifically pinned commit after this PR is reviewed.
- Paper catalogue: `papers/resolution-alignment/catalog.json`. Prioritize R2026-11, R2026-12, R2025-07, R2025-06, R2025-02, R2025-05, R2024-01.
- Prototype exporter: `python -m research_gym.scripts.export_receipt_sft --source data/generated --out-dir data/sft/rivet_stage1_v1 --holdout-family reachability`.
- Required output artifacts: dataset schema and group manifest; frozen preregistration; training run hashes; before/after model checkpoints or compatible adapters; paired baseline receipts; intervention traces; VPD decomposition diagnostics; short report with primary result and negative findings.

### Critical target-leak fix

The legacy `research_gym/scripts/export_sft_dataset.py` currently passes `Frame.to_jsonable()` to the model as input. That dictionary includes **`output_state`, `soundness_type` and `label`**, which are target columns. It may also contain witnesses in `metadata`.

Use `export_receipt_sft.py` for **stage-1** supervised data. It supplies only `family`, `input_state`, and `operation` in model input and puts gold labels exclusively in the output field. Its `authority` field explicitly says `supervised_prediction_only`.

The stage-1 exporter is one-step imitation and **does not fabricate verifier-tool trajectories**. For stage-2 repair, capture real rejected proposals under the independent exact checker and obtain a correction from exact mechanics, an admissible counterfactual oracle, or a separately labeled uncertain teacher. Encode the tool response as a later turn so future information cannot leak into an earlier proposal.

Proposed conversation sample structure for genuine stage-2 data:

```json
{
  "task": {"input_state": "...", "candidate_set": ["..."], "operation": "..."},
  "turns": [
    {"role": "assistant", "type": "proposal", "candidate": "a", "claimed_evidence": "heuristic"},
    {"role": "tool", "type": "verifier_receipt", "valid": false, "reason": "independently computed", "scope_digest": "..."},
    {"role": "assistant", "type": "repair", "candidate": "b", "rationale_type": "counterexample_addressed"}
  ],
  "source": {"environment_hash": "...", "verifier_hash": "...", "oracle_class": "exact"}
}
```

This is a **schema example**, not a recorded episode. A repair action can receive a truth label only when a mechanically valid correction exists. Retain `unknown` cases rather than converting them to hard negatives.

## Stage A: preflight, no full model training

1. Pull a pinned commit, run `python -m pytest tests/test_receipt_sft_export.py tests/test_sft_export.py -q` and locally regenerate the canonical frames.
2. Audit 100 randomized exported records for target leakage, duplicated public state across train/eval, family holdout and duplicate instruction templates. Fail fast if any final answer or counterfactual witness appears in a model input.
3. Measure the baseline for original GRU, a simple shallow/tied decoder and a deterministic LDT on the exact same procedures, depths and known cost unit.
4. Confirm that the chosen checkpoint and custom PyTorch forward hooks are supported for activation capture, variable-depth recurrence and parameter decomposition. Record unsupported models/operations and choose a compatible fallback before training.
5. Freeze synthetic generator definitions, exact verifier hash, seed list, group manifests, primary metric, control masks, all GPU caps and stop rules.

**Stop** on leakage, nonreproducible oracle labels, absent independent checker for an advertised env-sound claim, wrong adapter/architecture compatibility, or unresolved resource accounting.

## Stage B: minimal SFT comparison

Train at least three seeds for these same-example/same-compute arms:

- B1: normal decoder / one forward call, stage-1 SFT;
- B2: fixed tied recurrence with 1/2/4/8 steps, stage-1 SFT;
- B3: typed receipt conditioned recurrence with independently checked intermediate states, stage-1 SFT;
- B4: B3 + genuine counterfactual-repair SFT.

If Silico currently supports only a conventional decoder and LoRA, first run B1 versus **explicit tool-receipt SFT** as a smaller, hosted experiment; mark the tied-depth comparison as pending custom checkpoint support. Never silently call multiple generated text tokens a shared-weight hidden-state loop.

Evaluation: held-out task-family and generative rule-graph splits; unseen depths 9–12 (and further when performance permits); per-depth exact correctness, calibration, certificate violations, verifier interventions, compute/FLOPs and wall-time. Count the validator, branch visits and failed proposals in cost.

## Stage C: visit-aware VPD pilot

VPD (Goodfire parameter decomposition) can decompose parameter matrices into rank-one components, with input-conditional importance and causal reconstruction checks. The published validation is limited to relatively small language models; begin on a tractable compatible checkpoint. Code and discussion: https://github.com/goodfire-ai/param-decomp and https://github.com/goodfire-ai/param-decomp/blob/main/docs/handbook.md.

Start with **one checkpoint, two modules, two task families, and 128–512 paired examples**. Measure the same examples before and after SFT.

A tied matrix is physically reused at every recurrence visit. Fit its rank-one components and compute two intervention estimands:

```math
\Delta_c^{\mathrm{global}}=\mathbb E[Y\mid \operatorname{do}(c\leftarrow0,\text{all visits})]-\mathbb E[Y],
```

```math
\Delta_{c,t}^{\mathrm{visit}}=\mathbb E[Y\mid \operatorname{do}(c\leftarrow0,\text{visit }t)]-\mathbb E[Y].
```

`Y` must include **independently checked intermediate state progress** as well as final answer quality. A per-visit intervention on tied weights needs a custom gating hook that skips the component only on that call; report it as an experimental intervention, not a persistent change to shared parameter storage.

Fit conditions: full checkpoint, selected attention-output or MLP projections, component rank budget, sparsity, reconstruction fidelity and intervention storage. Before any attribution claim:

- Check unmasked model output parity and parameter reconstruction.
- Track faithfulness over several sparsity levels and verify the predicted effect by exact forward replays.
- Try adversarially chosen masks or equivalent mask-robustness checks.
- Compare norm/rank-matched random components, weight-shuffled decomposition sources and ordinary gradient attribution.
- Examine both all-visit and one-visit knockouts, conditional on whether the step is `INSPECT`, `PROPOSE` or `REPAIR`.
- Run a control with scrambled operation tags and a control with shuffled receipt chronology.
- Measure the full effect on performance, not just the internal probe's score.

A positive result is **causal task-specific importance** that generalizes to unseen problems, beats matched nulls and keeps reconstruction errors within pre-registered tolerances. Finding interpretable-looking components by itself remains exploratory.

### Budget rationale

The first VPD pilot targets ~16M–32M parameters and limited layers. A ~50M–70M model is a promotion target. A 100M–300M model and a complete Qwen 0.8B decomposition are stretch tests whose cost and feasibility require a measured pilot. Total model parameter count alone does not determine VPD expense: posterior sampling, mask optimization, repeated causal replays and unrolling across visits can dominate.

Use the attached budget gates in `configs/silico_rivet_v1.json`. Require a fresh budget approval if Silico projects an overrun. Record GPU type, actual GPU-hours, tokens, storage and RU consumption.

## Stage D: Resolution-style susceptibility and data patterning

Resolution's `Patterning: The Dual of Interpretability` motivates reweighting the training distribution to favor useful mechanisms, while the spectroscopy papers motivate measuring sensitivity and clusters. For small models, a local posterior sampler may produce a susceptibility matrix:

```math
\chi_{ci}=-\beta\operatorname{Cov}_{\pi_\beta}(O_c(\theta),\ell_i(\theta)),
```

where `O_c` is the predeclared observable for circuit/component c, `\ell_i` is per-example loss, and `\pi_\beta` is a specified local Gibbs posterior. This is **not equivalent** to a standard loss gradient, TracIn influence or VPD component score. A proper susceptibility claim requires a defined posterior, multiple chains and mixing/ESS diagnostics.

After a posterior pilot, define training data weights using calibration-set usefulness while keeping family support and realized sampling KL bounded:

```math
w_i \propto w_i^{(0)}\exp(\eta s_i),\qquad D_{\mathrm{KL}}(w\|w^{(0)})\le K.
```

For each frozen weighting policy compare (i) uniform, (ii) random matched KL, (iii) loss-only difficulty, (iv) simple gradient/influence, and (v) susceptibility or validated causal VPD score. Train from **the same initialization and SFT tokens**. Evaluate unseen-depth accuracy and causal component behavior, with all selection on calibration-only data. If local posterior sampling is too expensive, run the cheaper causal-ablation score and call it **intervention-guided reweighting**, without a posterior claim.

## Stage E: developmental analysis

Save checkpoints at fixed **token counts**, not chosen after peeking at the final result. For each stage:

- Measure when a stable component emerges, bifurcates or disappears under matched causal tests.
- Compute geometry-based similarity, component mass, causal mediation and probe transfer.
- Estimate local learning coefficient (LLC) only if the local posterior method and diagnostics justify it.
- Assess signed cross-checkpoint controls using the repo's RSITopology lineage and holonomy policy; otherwise use invariant subspace energy.
- Compare observed complexity trends to quantization, LoRA-coordinate changes and seed-aligned nulls.

This is a natural intersection of Resolution's developmental works, compactification theory and the HybridTRMLDT membrane.

## Costs and stopping

Spend zero paid compute for dataset audits and repo tests. Next, authorize only a **bounded small-model SFT and VPD smoke**. Limit the first GPU experiment to five percent of the available RU budget; after it passes, allow one additional phase with an explicit cap. The document proposes a maximum of fifteen percent of the starting RU balance for all pilot phases combined, with no automatic escalation.

Hard stop on an increase in unauthorized hard refinements, missing exact checker receipts, failure of any leakage sentinel, insufficient VPD reconstruction, matching or worse result than shuffled/magnitude controls, a dataset with contaminated family splits, unsupported forward hooks, or an actual/projected cost over cap.

Do not claim the hosted model has reached a scientific performance threshold until at least three seeds, all matched baselines, and held-out families have been completed. Preserve negative outcomes as primary artifacts.

## Deliverables and return-to-repo handoff

```text
experiments/rivet_sft_v1/
    preregistration.json
    dataset_manifest.json
    baseline_results.json
    trained_results.json
    paired_eval_records.jsonl
    costs.json
    checkpoints.json
experiments/rivet_vpd_v1/
    intervention_spec.json
    decomposition_diagnostics.json
    causal_effects.jsonl
    null_controls.json
    geometry_receipts.json
reports/rivet_silico_v1.md
```

Requested report sections: initial access/method audit; exact benchmark numbers with intervals; whether held-out depth improved; whether genuine repairs helped; how strong VPD faithfulness and causal specificity were; failure modes; resources; and best next use of compute.

### Copyable Silico thread instruction

> Work from the pinned `MoralityLabAI/HybridTRMLDT` RIVET branch/merge commit and execute the protocol in `docs/silico_sft_vpd_protocol.md` using `configs/silico_rivet_v1.json`. Begin with zero-GPU leakage and verifier audits. Use `research_gym/scripts/export_receipt_sft.py` for stage-1 frames. Inspect access to custom recurrence checkpoints and Goodfire parameter decomposition before proposing any paid run. Compare one-pass and tied-depth baselines with a genuine verifier-conditioned corrective SFT arm on grouped heldouts. Test causal VPD components across all visits and single visits with matched shuffles and reconstruction controls. Report negative results and absolute RU use. Do not launch large sweeps, alter the registered split, or interpret probe success as authority. Push implementation and results back as a draft PR only after tests and cost receipts pass.
