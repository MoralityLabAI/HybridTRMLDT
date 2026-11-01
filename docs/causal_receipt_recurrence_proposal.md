# RIVET: Receipt-Indexed Verifiable Execution Transformer (proposal v0.1)

Status: **design proposal, not an implemented or measured model**. Reviewed against HybridTRMLDT `master` on 2026-11-02. Scientific claims below are hypotheses until prospective experiments run.

## Decision

Build one shared-weight recurrent transformer whose **reasoning steps have typed interfaces and causally testable micro-operations**. Keep the existing independent LDT verifier and Conductor-HRM. Train a small model to make useful proposals and to use rejected proposals as corrective training cases. Let a 0.8B–4B teacher write candidate explanations if needed; the small model's authoritative objects are executable transitions and external receipts.

Name: **RIVET**, Receipt-Indexed Verifiable Execution Transformer. It can run the existing toy gym before requiring a large decoder.

## Current evidence that constrains the design

- `research_gym/neural/trm.py`: 256-wide GRU proposer, six tied recurrence steps. A viable control, with action, mode, conflict and provenance heads; evidence for a six-step toy proposer rather than learned cross-family computation.
- `reports/benchmark_comparison.md`: Sudoku hybrid solves as often as TRM while reducing guesses from 31 to 6; storyworld uses mechanically checkable reachability, while routing requires soft token evidence and the hard filter degrades accuracy from 0.815 to 0.584.
- `docs/gaming_vs_improvement_v2.md`: accepted-only self-imitation degrades all 30 above-majority adaptation trajectories. A separately computed corrective action is the missing teaching signal.
- `docs/airis_induction.md`: 28 induced rules all choose the same `control_math` fallback, so high nominal induction precision yields no change in objective. RIVET must use **state-specific** branching and nonidentical fallbacks.
- `docs/controller_mesh_sheaf_forward_replication.md`: rho 0.568 for spectra versus 0.598 for categorical baseline. Spectral geometry must demonstrate incremental value against matched ordinary predictors.
- `docs/controller_mesh_measured_stalk_bridge.md`: rho 0.752 for the measured-Qwen product, but typed shuffles match it (p=0.938). Any mechanistic predictor must beat weight-shuffled source controls.
- `MoralityLabAI/jev-qwen/reports/xbench/summary.md`: compact 459,776-param LOOP-T solves trained depth-1..8 pointer tasks extremely well; on depth-9..12 generalization, emitted reasoning reaches 0.717 and LOOP-T-ds 0.308. Training a loop on an unrelated decision mixture did not add algorithmic depth.
- `MoralityLabAI/jinn-or-beast-paper/.../SRT_PROTOCOL.md`: Spirographic Recursive Transformer (SGRT) is a prospectively registered **exploratory** itinerary, not an established result. Preserve its exact `G[5:2:0.8]^12.G[7:3:0.65]^12.S.C^6` schedule and same-point phase-shuffle control.

## Architecture: recurrent local workspace with external receipts

At microstep t, let x be the public task input, a_t the typed candidate lattice, B_t the append-only receipt ledger, z_t the private token workspace and c_t the remaining compute budget.

```text
task x, lattice a_t, receipt ledger B_t
         |             |
         v             v
    tiny controller: choose micro-op e_t, compute budget / halt
         |
         v
  tied transformer F_theta + micro-op adapter A_e
         |
         v
 proposal head: candidate action, predicted refinement, uncertainty
         |
         v
 EXACT INDEPENDENT CHECKER (only when scoped mechanics exist)
      /                 \
 accepted receipt       rejected/counterexample receipt
      \                 /
       update typed a_t and append B_{t+1}
                 |
      next microstep or bounded halt
```

A formal contract, with `e_t` in a small frozen instruction vocabulary:

```math
e_t \sim \pi_\phi(e\mid \operatorname{summary}(B_t),a_t,z_t,c_t),
\qquad
z_{t+1}=z_t+g_t F_\theta(z_t;\operatorname{enc}(x,a_t,B_t),A_{e_t}),
```

```math
q_t=Q_{e_t}(z_{t+1},a_t),\quad
r_t=V_{\rm exact}(q_t,a_t,\mathcal M),\quad
a_{t+1}=\begin{cases} a_t\wedge r_t.\mathrm{state},&
r_t.\mathrm{certificate}=\mathrm{env\_sound}\land r_t.\mathrm{valid}\\
a_t,&\mathrm{otherwise.}\end{cases}
```

where `V_exact` is owned by the environment and can return **unknown**. The model's own provenance prediction is telemetry with no authority. `B_t` records the verifier's independently computed outcome, source hash, candidate, problem state digest, cost and validity window. Private `z_t` never crosses the Conductor-HRM module membrane. An unsafe or ill-typed proposal can become a counterexample without becoming a committed lattice refinement.

Suggested micro-operations: `INSPECT`, `PROPOSE`, `TEST`, `REPAIR`, `COMPARE`, `HALT`. They are action vocabulary labels, not guarantees that the corresponding neural computation is causally separated. That separation is a **hypothesis** for intervention testing.

Each micro-operation has a compact rank-r adapter on the shared transformer block. Weights are tied across microsteps. Per-step learned gates make the cost of an extra visit explicit. The controller may terminate early when a checkable fixpoint is reached or when a registered cost budget expires.

**Variants:** (A) fixed shared block with typed receipts; (B) add micro-op adapters and learned router; (C) add explicit counterfactual repair supervision; (D) add causal-patterning data weights; (E) add optional search itinerary. Change one factor at a time.

### Why the recurrence has better prospects

A plain repeat of the same decoder may learn task-specific extra hops and then fail at unseen depth. RIVET's recurrent action is indexed by **what changed in the external state**. The small model can learn reusable operations such as propose, test and repair. A fixed verifier supplies independent evidence of progress. The architecture should be evaluated with unseen algorithmic depths and changed rule compositions, so a successful continuation demonstrates more than memorized visit numbers.

`INSPECT` and `REPAIR` make the computation visible as causal hypotheses, through a stable intervention API even if the hidden representation itself rotates. Verified state changes create traces to align between runs without demanding interpretability from natural-language CoT.

## How to keep and improve Spirograph

The actual SGRT candidate is the registered `G[5:2:0.8]^12.G[7:3:0.65]^12.S.C^6` schedule. Its value is an ordered bounded search over candidates. It offers a possible structured exploratory prior; the gear geometry currently has no demonstrated mechanistic relation to the task.

A defensible successor keeps the same 24 gear visits but lets a target-blind controller decide **when to branch** from public residual, candidate-set size, receipt entropy and budget. Train the visit selector on calibration examples, freeze before holdout, and compare with:

1. the exact original SGRT itinerary;
2. same points, phase-shuffled itinerary;
3. matched-visit deterministic low-discrepancy traversal;
4. matched-budget learned beam / best-first search;
5. RIVET with SGRT disabled.

A useful Spirograph gain must exceed phase-shuffle and matched-budget traversal on unseen task families. The next study should report novelty of surviving candidates, probability of certified progress and total verified cost. Use the gear pattern as a regularizer or candidate **coverage** policy only if it clears those tests.

## Small-model and larger-teacher tracks

| Track | Purpose | Starting configuration | Boundary |
|---|---|---|---|
| Toy | Debug proof and visit-level interventions | Existing 256-wide GRU + a tiny shared transformer baseline | CPU/GPU local, exact toy mechanics |
| RIVET-S | Show depth transfer and trace causality | Approx. 16M or 32M params, limited vocabulary, 1/2/4/8/12 visits | Count actual params in implementation |
| RIVET-M | Primary VPD candidate | Approx. 50M–70M params, 4–8 tied blocks with micro-op adapters | Only after small-pilot smoke |
| Transfer | See whether external receipts survive a strong prior | LoRA SFT on a Silico-supported 0.8B-class Qwen | Native VPD on full model is a separate cost/validation study |

A new custom architecture may need to be trained or exported outside Silico if the hosted SFT interface accepts only supported model families. The Silico-compatible immediate pilot is a supported decoder with **typed receipt SFT**. The custom shared-weight recurrence is a separately versioned checkpoint.

## RIVET training curriculum

**Stage 0: sealed data.** Split at procedural generator seed + rule-graph + world-state group. Reserve unseen rule compositions, depths and environments. Never train or select on evaluation results.

**Stage 1: proposal imitation.** Each sample's input contains only state, instruction, allowed operations and preceding receipts. Targets are proposals, predicted uncertainty and candidate-specific evidence. The proposal is explicitly noncertifying.

**Stage 2: counterfactual corrective SFT.** When a proposed action is rejected or a superior action is externally demonstrated, write a second sample conditioned on the verifier receipt. Train a distinct `REPAIR` proposal. Use mechanically validated corrective labels or trustworthy environment rollouts, with provenance. Compare this with accepted-only imitation from `gaming_vs_improvement_v2`.

**Stage 3: variable-depth unrolling.** Teacher-force the state/receipt boundaries for short tasks, gradually train closed-loop execution with external checks. Mix 1, 2, 4, 8 and 12 visits; hold out higher visit counts and new rule compositions. Add budgeted halting and penalties for repeated receipt-equivalent loops.

**Stage 4: constrained structure-directed reweighting.** Use small-model measured causal component importance, or genuine posterior susceptibility when sampling diagnostics pass. Freeze the reweighting rule on a calibration split and compare against matched random, loss-only and gradient-influence weights. Track realized sampling-policy KL and task-family coverage.

**Stage 5: cross-family and quantization stress.** Transfer trained operator adapters to routing, Sudoku, ARC-2 and native game/tool contracts where the exact mechanics scope is known. Quantize only after collecting the unquantized intervention baseline.

Illustrative loss:
```math
\mathcal L=\mathcal L_{\mathrm{proposal}}
+\lambda_r\mathcal L_{\mathrm{repair}}
+\lambda_s\mathcal L_{\mathrm{stop}}
+\lambda_c\mathcal L_{\mathrm{cost}}
+\lambda_k\mathcal L_{\mathrm{calibration}}
+\lambda_d\mathcal L_{\mathrm{distill}},
```
with independent structural authorization implemented in code rather than optimized as a soft penalty.

## Evaluation: one registered primary and prespecified controls

Core conditions should use a paired generator, equal unique training examples, three or more seeds, and **equal executed FLOPs or measured wall/GPU budget**, including verification and exploration calls. Report both accuracy at a fixed cost and minimum compute needed to reach each success threshold. The primary metric is family-macro held-out exact task accuracy for unseen rule graphs and depths 9–12, with unauthorized hard updates required to be zero.

| Arm | Recurrence | Typed receipt | Corrective SFT | Patterning / search |
|---|---|---|---|---|
| B0 decoder | no | no | no | no |
| B1 fixed loop | shared block | no | no | no |
| B2 typed loop | shared block | yes | no | no |
| B3 RIVET base | routed micro-op | yes | yes | no |
| B4 RIVET + patterning | routed micro-op | yes | yes | causal weights |
| B5 RIVET + SGRT | routed micro-op | yes | yes | frozen itinerary |
| B6 RIVET + selected search | routed micro-op | yes | yes | learned budgeted traversal |

Any arm that cannot make an environment-sound certification still gets credit for useful **proposals**; the independent verifier determines whether those proposals execute.

Suggested promotion gate, registered **before** outcomes: at least +8 percentage points unseen-depth accuracy over the strongest cost-matched baseline, **or** a 25% compute reduction at matched accuracy; at least 3 independent seeds; zero unauthorized hard updates; and a measurable causal feature intervention advantage over matched randomized controls. These values are proposed engineering targets, not observed results.

### Mechanism-level falsification

- Ablate a VPD component on **all** tied visits; then ablate it at precisely one visit using an experimental mask/gate. Record ability to repair, remaining effective depth, and off-target damage.
- Swap activation states between task pairs that share syntactic features but require different rules. Interventions should move the predicted suboperation and downstream certified behavior, not merely the label or output style.
- Compare rank-matched random, magnitude-matched and **weight-shuffled** sources. Require source-specific predictive gains beyond the shuffle effect that defeated the previous Qwen bridge.
- Compare checkpoint-aligned subspaces using projectors/CKA and RSITopology holonomy. Refuse signed cross-run steering wherever the transport and orientation checks fail.
- Run a masked-budget test: a causal component should remain useful when the final answer is hidden and only intermediate public state progress is scored.
- Vary LoRA parameterization and dequantization before interpreting individual weight components. Track whether the claim survives coordinate changes.

## Build order

1. Land a leak-free SFT exporter and a reproducible 28-paper catalogue.
2. Implement a tiny tied transformer and operational receipt interface with deterministic unit tests.
3. Run B0/B1/B2/B3 with 3 seeds on native Sudoku, ARC-2 and pointer-chasing depth splits.
4. Measure causal interventions and train a VPD pilot on the smallest compatible checkpoint.
5. Add patterning / posterior sampling only after baseline and mechanism gates pass.
6. Promote to 50M–70M, then consider a larger hosted teacher and a 100M–300M stretch.

See `docs/silico_sft_vpd_protocol.md` for the Silico experiment handoff and `configs/silico_rivet_v1.json` for a machine-readable registration draft.
