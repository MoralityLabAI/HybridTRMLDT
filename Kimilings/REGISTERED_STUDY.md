# KIMILINGS-01: registered experiment design

## Research questions

**A. Instrument validity.** Can a faithful sparse parameter decomposition of one actual ~33M parameter Kimi expert survive held-out inputs and freshly restarted adversarial component masks? Report reconstruction quality on a sweep of active L0, including weight faithfulness and dense/all-off endpoints.

**B. Cross-family transfer.** Do components selected on logic tasks have measurable causal consequences on ARC-AGI-2 training/validation tasks, and vice versa? Compare loss changes against random rank-matched, magnitude, shuffled importance and architectural-unit controls at the same perturbation budget.

**C. Router utility.** Does a TRM recurrent proposal system with certified lattice constraints and VPD-conditioned soft features improve exact task solve rate per measured compute over a TRM-only selector and verified-LDT-only selector? Keep expert-bank sizes, token top-k, allowed steps, seeds, and inference hardware matched.

## Pre-registration

- Frozen slice commit: 387322684297ff5f014d772f4ac6549c6d94a77d (AlephFunk repo). Check each bundle hash and source tensor/scale reconstruction before decoding VPD targets.
- ARC-AGI-2 source commit: f3283f727488ad98fe575ea6a5ac981e4a188e49. Final 120 evaluation task IDs are sequestered. The existing synthetic ARC-2 benchmark is reported separately.
- Logic source: primeintellect/logic-env version 0.1.2; independently pin the complete environment source SHA and verifier implementation before measuring.
- Group split seed: 11, stable 65/15/20 VPD train/calibration/validation partitions; final only from official public evaluation.
- Run seeds: 11, 29, 47. Pre-registered expert-bank sizes: 1, 2, 4, 8, 16. Token top-k: 1, 2, 4, 8, restricted to <= bank size. A 16/16 bank with top-16 uses all indexed experts per token and cannot evaluate sparse expert routing.
- VPD: separate runs for sparsity strength, active importance thresholds [0.05, 0.10, 0.25], and fresh adversarial-mask step budgets [4, 16, 64]. Train to converged curves; save regular checkpoints and all seeds. Never optimize importance-as-mask diagnostic directly.
- Only calibration examples may select the routing coefficient, expert target, sparsity model, termination criteria, or score threshold. A reasoning target chosen using routed-frequency statistics counts as pre-selection and must be recorded before held-out testing.
- Baselines: dense/no decomposition, all-off, TRM, independently verified LDT, TRM+LDT, hybrid+VPD, matched-random route; plus decomposition controls of random rank-matched, magnitude, architectural-unit, shuffled importance.
- Do not treat all-on fidelity as sufficient. Monitor importance-mask error, weight-parameter error, fresh adversarial PGD error, active L0, total components, and training-curve plateau. Report adversarial source shape and initialization.
- Full task evaluations require executing each intervention in the shared backbone with its reference outputs. Expert-local MSE remains a separate endpoint.

## Statistical readouts

For routing, the primary task metrics are exact task success and compute-normalized success at equal bank size and token top-k. Report abstention, completion coverage, token/step count, latency, offload, CPU/GPU peak memory, energy if measurable, and actual routing traces. Match per-task random seeds and task families. For logic, use native task-specific exact verifier outcomes. For ARC-AGI-2, report task pass@2 over every test input and keep dimensions exact.

The primary VPD Pareto coordinate is (mean active subcomponent L0 per input, held-out fresh-adversarial reconstruction MSE). Also publish unmasked, importance-only, all-off error, parameter relative reconstruction error, source-specific calibration, and seed-wise curves. Reconstruction loss and whole-model task performance have separate scales.

For transfer, define the paired causal effect of a component group i on test family j as:

    T_ij = mean_x[loss(ablated group i, x) - loss(unmasked, x)].

Compare each observed intervention with a matched-cost control:

    Delta_ij = mean_x[loss(ablated group i, x) - loss(matched-control, x)].

The matrix cells are grouped by component family, evaluation family, mask budget and control method. Bootstrap by task lineage group, repeat across seeds, and control multiplicity (Holm) for pre-registered family comparisons. A lower reconstruction loss with fewer components is a candidate instrument result. An improved task score with measured costs is a model/runtime result.

## Execution stages and stop gates

### Gate 0: extraction and source pinning

Verify byte hashes for every bundle and MXFP4 decode against independent reference. Freeze the tokenizer, expert activation function, scale metadata, original Kimi checkpoint revision, and verifier versions. Current public Kimi slice release has been reported verified by Silico; local verification requires downloading bundles or an independently checkable receipt.

### Gate 1: real activation capture

Capture actual inputs to each routed expert, router decisions, and reference outputs from the original full model on tasks in the allowed partitions. Keep stage and family labels. Candidate selection requires sufficient routed examples from logic and ARC training partitions for the *same one-layer expert*. With no full-model service, Gate 1 is blocked. Synthetic activation tensors are suitable only for instrument tests.

### Gate 2: sparse-VPD optimization

Start with one selected 33M expert, one H100, explicit JAX versions and Goodfire param-decomp commit. In the previous 12-step synthetic canary, peak JAX allocation was 2.638 GB and measured post-compilation throughput was 37.71 steps/second at batch 4. Masked reconstruction error (0.01106) was nearly identical to all-off, despite all-on MSE ~7.488e-5. These figures are an instrument receipt, not extrapolated training time or a successful decomposition.

Train across sparsity coefficients until convergence, then evaluate with newly initialized adversaries at 4/16/64 steps. Report a full Pareto front. Require independent checkpoint reload and representative held-out data. Abort expensive scaling if masking retains almost no function or optimization drifts.

### Gate 3: reasoning transfer

Freeze a decomposition before extracting semantic component labels from calibration. Create matched intervention pairs, with random subcomponent rank/control and architectural unit controls. Require whole-model reference forwards to make any task-level claim. Repeatedly changing the candidate or labels on validation data invalidates the preregistration.

### Gate 4: TRM-LDT bank routing

The TRM ranks indices. The typed LDT state may hard-remove indices only after an environment-owned proof/verifier callback checks the proposed refinement; source model text and claimed provenance have no hard authority. VPD importance stays a soft feature unless separately certified. Execute bank plans with modified top-k, characterize coherence and interference, and measure actual task outputs. Since the shared backbone is approximately 114.8 GB in the original released representation, the locally extracted bank files alone do not supply a 5080 inference runtime.

### Gate 5: final benchmark

Freeze code, tasks, router coefficient, VPD checkpoint, baselines, seed plan and cost accounting. Run ARC-AGI-2 public evaluation only once with the final scoring ledger, alongside registered independent logic-verifier heldouts. Archive the exact input hashes, git SHAs, failure cases, stdout/stderr, latency distribution and hardware footprint. If a full Kimi service or trusted original checkpoint is missing, leave the report at the instrument/preparation stage.

## Source references

- ARC-AGI-2 official dataset and scoring: https://github.com/arcprize/ARC-AGI-2
- Prime Intellect environments and CLI: https://github.com/PrimeIntellect-ai/prime
- Goodfire parameter decomposition methodology and metrics: https://github.com/goodfire-ai/param-decomp/blob/main/docs/handbook.md
- HybridTRMLDT typed membrane: ../docs/hybrid_contract.md
- Existing Prime logic adapter: ../research_gym/adapters/intellect_logic.py
- Existing synthetic ARC-2: ../research_gym/benchmarks/arc_bench.py
