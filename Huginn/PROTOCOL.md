# HUG-CAL-02: preregistration-ready GPU protocol

**Status:** CPU prep only; new prospective registration and fresh calibration must precede any causal confirmation. Original closeout is permanently labeled *control-calibration failure / temporal claim unconfirmed*.

## Pinned evidence and assumptions

- Huginn model: `tomg-group-umd/huginn-0125`, revision `bb6621b65e90b6a4b9b29ef88dc83866d450470c`.
- Restricted original factors `factors.npz`, SHA256 `86f90a1c9403240e4f69fdfad1f9f88a2410a85edfb2cfef2302fc74959f4a50`. These factors arise from a restricted top-64 SVD surrogate, **not** full decomposition.
- Original registered program archive SHA256 `818312e22ac48361371fe1d553cd44f0a45acd2667859f0f325a91f3a931f165`.
- Original model paths: `core1` and `core3` recurrent MLP output, 16 iterations, early calls 1..8, late 9..16.
- Original frozen factors: `core1:19,58,26,31`; `core3:60,17,1,44`. Nominations early-only, late-only, large-effect and near-null respectively. SVD comparator indices from calibration only: `core1:15,0,29,2`; `core3:0,1,4,2`.
- Original primary output: ten leading-space single-digit token choices; correct digit log-probability.

## Mathematical hypothesis

For rank-one intervention `D = u v^T` and pre-projection state `x_t`, the immediate activation-change magnitude is `||D x_t|| = ||u|| |v^T x_t|`. Both `u` and `v` matter causally downstream; scalar norm matching by itself cannot separate read-gate and write-direction mechanisms. Different distributions of `x_t` across recurrence depth can make random rank-one controls appear temporally specific.

### Family W: same-gate, orthogonal-write (distinct estimand)

Construct four output vectors `u'_j` orthogonal to `u` and to one another, with `||u'_j|| = ||u||`; keep `v'_j=v`. The immediate injection norm is **identical on each input**, at every recurrent call and on every split. This tests whether the *output write direction* matters given the same input-read gate. As it retains `v`, it cannot establish read-gate specificity. Treat write controls as a separate family; show its causal results in their own table.

### Family R: covariance-aware, rotated-read (harder test)

Estimate input variances from **new selection-split state captures** for each core, pooling the early/late windows for the proposal distribution. Draw whitened random input gates, keep input cosine to the original `<=0.95`, and fit a single amplitude from selection only. Rank candidate matrices using worst normalized quantile discrepancy across **both** windows. Require four independent candidates selected and then passed on frozen verification; reuse the exact same control matrices in both early and late causal interventions. `control_repair.py` provides a deterministic diagonal-covariance proposal baseline. The worker may implement full covariance-aware candidates if instrument and budget permit, but must freeze the construction before verification and report any deviation.

### Remainder controls

Keep exact spectrum-preserving signed row/column transformations as a structural diagnostic. A globally amplitude-scaled remainder has a **different singular spectrum** and must be labeled `amplitude_matched_remainder`. Fit scalar from **selection only** across both windows. Report spectrum-preserving and amplitude-matched families separately. The earlier ~1.48-1.50 rescue after the archive was opened is exploratory.

## Stage 0: sealed data and instrument

1. Recover and SHA-check the previous model, factors, original program generator, old source hashes and prior 8-factor panel. Publish a manifest with Git revision, kernel version, source hashes, torch/CUDA/H100 configuration and exact 16-iteration invocation path.
2. Generate a **new** disjoint selection and verification corpus, each at least 120 independent source programs spanning the three original families, and three noise seeds per source. Group by exact program source hash; audit duplicates across previous pilot/calibration and prospective confirmation. Do not re-use the old 120-source verification outcomes for fitting or acceptance.
3. Preserve the old 480-program confirmation deck as sealed only if an independently checkable audit establishes no inference or tuning on those source hashes. Otherwise generate a fresh 480-program confirmation deck and freeze hashes before any new control validation.
4. Capture `x_t` at the *answer position* immediately before the original core MLP output projection at every recurrent call, with source hash, family, seed, batch shape and step index. Keep selection/verification data physically separated.
5. Check identical source order, same batch size/padding and same recurrent-noise seeds for the no-op, clean, factor and control intervention passes. Repeat zero-edit bitwise check on >=12 prompts across all three families. Audit original-weight hook in place; do not rebuild the full forward via low-rank factors. Treat original closeout's 0.034/0.060-nat rebuilt-forward and 0.083-nat batch/single drifts as reasons for paired execution.

## Stage 1: selection-only fitting

Construct and hash four W controls per each of eight frozen factors. Construct a >=512 candidate R pool per factor, fit only on new selection states, and choose up to four with diverse input gates. Store full candidate seed/index/u/v/scale, calibration loss, pairwise read/write cosines, singular norm, and target/control quantiles for early and late separately. Reject a candidate if any p10/p50/p90 relative absolute error exceeds 0.15 or p95 exceeds 0.25 in **either window**. Do not tune the gate after reading verification outcomes.

Expected reference W test: immediate norm error exactly zero within numerical precision; this is only an algebraic sanity condition and requires a GPU hook check before causal interpretation. R can validly fail. Include original spectrum-preserving remainder and optional amplitude controls as diagnostics, not in the 8-factor confirmatory null unless preregistered as such.

## Stage 2: untouched verification and acceptance

For each of eight original factors, each named family needs four controls whose stored control-matrix identities are unchanged between early and late windows, passing all four quantiles in both windows on independent verification. Require:

- pinned model/factor/panel/source hashes;
- SHA-disjoint new selection/verification/confirmation decks;
- the original-weight same-batch, same-seed zero-edit checks;
- four valid, diverse controls per factor, jointly matched on both windows;
- a `calibration_gate.json` receipt with status PASS/STOP and a per-family evidence ledger; compute it with `gate_evals.py` from stored split-state NPZ and single-matrix-per-control NPZ after source manifest freeze.

If W passes and R fails, the W-specific confirmation may proceed if it was separately preregistered, with claims limited to write-direction specificity. If neither family passes, STOP. The original broad temporal-role claim requires R confirmation as well. A `PASS` status issued by an agent without independent inspection of source checksums and runtime traces is insufficient.

## Stage 3: causal confirmation, ONLY after PASS

Run matched original-weight interventions subtracting factor/control at selected recurrent calls. Select clean-correct source programs prospectively from the clean baseline, before observing factor effects. All control IDs stay fixed across early and late. Compute for program `p`, seed `s`, window `w`:

`theta[p,s,w] = (logp_factor-logp_clean) - mean_j(logp_control_j-logp_clean)`.

For each factor, primary interaction is `theta_late-theta_early`. Bootstrap by **source program**, retaining all three seeds inside the sampled unit. Report 95% two-sided interval for active/interaction effect and 90% equivalence interval for inactive effect, with equivalence region `[-0.05,0.05]` nats. Harmful active effect must be `< -0.05` nats with the 95% interval excluding zero. Test interaction against zero with source-clustered sign-flips and Holm adjustment across all eight frozen factors in the same control family. Include conservative Bonferroni simultaneous CIs for context. Report both effect and uncertainty by family, core and seed. The code in `confirm_eval.py` will consume the staged output only after an externally validated PASS receipt.

## Compute ceilings and stop rules

Keep bounded stages: pilot <=0.5 H100-hours, new calibration <=1.0 H100-hours, confirmatory <=4.5 H100-hours, aggregate <=6.0 H100-hours. Instrument and source audits must pass before consuming the next stage; confirmation only after accepted control-family gate. Record GPU runtime and peak memory for each stage. If real capture/matching costs more, STOP at the budget ceiling and report the blocker. No H100 inference is contained in this preparation package.

## Required outputs

`registration.json`, `source-split-audit.json`, `instrument-receipt.json`, `control-matrices.npz` (or equivalent hash-pinned source), `calibration-selection.json`, `calibration-verification.json`, `calibration_gate.json`, and, if eligible, `confirmation-outcomes.npz`, `confirmation-analysis.json`, a human-readable failure/claim ledger and fresh figure data. Preserve the old closeout as a separate immutable evidence object.