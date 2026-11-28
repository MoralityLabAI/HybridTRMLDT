# MOLT-GOBLIN-03 results (2026-11-28)

Branch `research/molt-goblin-03-20261128`; registered design in [`PREREG.md`](PREREG.md); freeze
`0a2fd748` was written before any hidden-root run.

Everything ran on one Windows laptop CPU with no GPU and no paid inference; Inkling-2-Mini
was used on the free Tinker id. At the operator's request, the campaign ran as one
below-normal-priority process (2 torch threads) and the finals as two single-thread
below-normal-priority workers. Peak resident memory of the process tree was 359 MiB
(campaign) and 887 MiB (finals), under the 3 GiB guard; no breach was recorded.

| Evidence kind | What it covers |
| --- | --- |
| Implementation, tested | `python -m pytest -q research_gym/molt_goblin_03`, 87 tests pass |
| Audit and replication | `receipts/audit/audit.json` |
| Development (selection-exposed) | Calibration and the 160-run Inky campaign on DEV3 roots |
| Completed experiment | 1,088 frozen cells on fresh HIDDEN3 roots: 19 arms, 8 families, all 8 registered seeds; 256 paired roots per evaluation. Plus 9 task-conditional routing models |

All intervals below are seed-level paired t 95% CIs, with n = 8 training seeds.

## 1. Audit of MOLT-GOBLIN-02

* **Replication.** Re-running eight frozen MOLT-02 hidden-root cells with the MOLT-02 code
  reproduces exact success root for root. The cells are f2 (HRM) and prefix, on proof_deps
  and experiment_select, at seeds 11 and 23. The v3 factory also reproduces a MOLT-02 HRM
  training trajectory to 1e-5 (`test_factory3.test_training_parity_with_v2`).
* **Causality.** Perturbing token t never changes outputs before t for any MOLT-02 arm
  (maximum leak 0.0).
* **Parameters.** The HRM manager's GRUs are 15.9 k of B's 50.6 k parameters (31%). The
  four blocks hold 33.5 k.
* **Visits and FLOPs.** Soft HRM evaluates all four modules every loop: 16 visits and
  5.50 MFLOP per example. Of that, the manager costs 1.39 MFLOP; A costs 1.03 MFLOP.
* **Loop forcing.** Under `loss=all`, each loop's readout receives equal gradient (0.0132);
  under `loss=last` only the final readout does.

## 2. Where the mixture advantage comes from

Exact on hidden roots (mean of 8 seeds):

| arm | cex_repair | cumsum3 | exp_portfolio | local_map3 | proof_deps | proof_resolve | research_plan* | revision_retract | params | visits | MFLOP/ex | train CPU s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A_single | 0.427 | 0.676 | 0.436 | 0.896 | 0.520 | 0.464 | 0.010 | 0.733 | 10.3 k | 4 | 1.03 | 5.2 |
| B_hrm | 0.678 | 0.800 | 0.472 | 0.988 | 0.645 | 0.553 | 0.018 | 0.765 | 51.5 k | 16 | 5.50 | 21.8 |
| C_mix | 0.679 | 0.779 | 0.450 | 0.986 | 0.663 | 0.538 | 0.017 | 0.783 | 35.6 k | 16 | 4.11 | 18.0 |
| D_tied | 0.433 | 0.810 | 0.462 | 0.859 | 0.518 | 0.464 | 0.014 | 0.740 | 26.3 k | 16 | 5.50 | 20.1 |
| E_param | 0.443 | 0.563 | 0.460 | 0.995 | 0.535 | 0.485 | 0.040 | 0.861 | 51.3 k | 4 | 5.45 | 7.5 |
| F_visit | 0.426 | 0.972 | 0.445 | 0.973 | 0.569 | 0.500 | 0.013 | 0.738 | 10.3 k | 16 | 4.13 | 16.0 |
| G_flop | 0.432 | 0.983 | 0.461 | 0.987 | 0.559 | 0.535 | 0.042 | 0.835 | 26.9 k | 8 | 5.62 | 11.3 |
| H_sparse_st | 0.446 | 0.752 | 0.421 | 0.910 | 0.487 | 0.441 | 0.016 | 0.711 | 51.5 k | 4 | 3.11† | 19.9 |
| I_anneal | 0.249 | 0.699 | 0.368 | 0.045 | 0.487 | 0.465 | 0.003 | 0.273 | 51.5 k | 4 | 3.11† | 19.7 |
| J_sequential | 0.677 | 0.975 | 0.502 | 0.992 | 0.664 | 0.578 | 0.039 | 0.814 | 35.6 k | 16 | 4.11 | 15.3 |
| K_distill | 0.400 | 0.506 | 0.435 | 0.701 | 0.524 | 0.471 | 0.010 | 0.745 | 10.3 k | 4 | 1.03 | 11.8 |
| **A_cpu** | **0.760** | **0.980** | **0.597** | **1.000** | **0.988** | **0.987** | **0.427** | **0.949** | 10.3 k | 4 | 1.03 | 32.7 |
| inky_08d849 | 0.626 | 0.823 | 0.472 | 0.975 | 0.660 | 0.576 | 0.025 | 0.776 | 43.0 k | 12 | 4.47 | 16.4 |
| inky_139d58 | 0.704 | 0.776 | 0.489 | 0.988 | 0.627 | 0.553 | 0.042 | 0.843 | 165.9 k | 12 | 17.2 | 25.6 |

\* research_plan is the finals-only family. † Hard-dispatch inference path; FLOPs are
measured on proof_resolve.

**The headline: the four-module advantage holds only when the number of training updates is
fixed.** A_cpu is arm A given B's measured process-CPU budget (2,158 updates instead of
300, a ratio of 7.2). It beats B on every family:

| family | A_cpu − B |
| --- | --- |
| proof_resolve | +0.434 [+0.389, +0.479] |
| proof_deps | +0.343 [+0.260, +0.425] |
| research_plan (finals-only) | +0.409 [+0.268, +0.551] |
| revision_retract | +0.184 [+0.151, +0.216] |
| experiment_portfolio | +0.125 [+0.090, +0.161] |
| cex_repair | +0.082 [+0.044, +0.120] |

It also has the cheapest inference: 1.03 MFLOP and 8.9 ms per batch of 64, against 40.3 ms
for B. It sees 7.2× more fresh training examples, so this arm matches CPU, not data.
Under a fixed CPU budget, spending the budget on updates beats spending it on modules.

**At a fixed update count, the advantage comes from separate module weights evaluated in
the same loop, not from the manager.**

* **Manager (P3, B − C):** within ±0.021 on every family, and every CI includes 0. The
  uniform mixture C matches B with 15.9 k fewer parameters.
* **Separate weights (B − D):** sharing one block behind four type embeddings removes the
  gain. Against D, B leads on cex_repair (+0.245 [+0.215, +0.275]), proof_deps (+0.127),
  local_map3 (+0.129) and proof_resolve (+0.089). D's modules are near copies of each other
  (module CKA 0.90–0.95).
* **Parameters (B − E):** a parameter-matched single block (width 76) does not explain
  cex_repair (+0.234), proof_deps (+0.110), proof_resolve (+0.068) or cumsum3 (+0.236).
  E wins on revision_retract (−0.096 [−0.128, −0.064]), as in MOLT-02.
* **FLOPs and visits (B − G, B − F):** neither explains cex_repair (+0.246 and +0.252) or
  proof_deps (+0.086 and +0.076). Both beat B on cumsum3 (G −0.183, F −0.172), a running-sum
  task that rewards sequential depth.
* **Parallel vs sequential (B − J):** J applies the four modules in series within each loop
  and has no router. It is never worse than B by more than 0.001. It beats B on cumsum3
  (+0.175), revision_retract (+0.049), experiment_portfolio (+0.031) and research_plan
  (+0.021). J − A is positive on all eight families, and every CI excludes 0.
* **Best fixed-update arm overall: J.** It has the highest eight-family mean among
  fixed-update arms: J 0.655, inky_139d58 0.628, B 0.615, C 0.612, G 0.604, E 0.548,
  A 0.520. J is top on proof_deps, proof_resolve and experiment_portfolio. On other
  families other arms win: E on revision_retract, G and F on cumsum3, inky_139d58 on
  cex_repair.

Interpretation: the useful ingredient at a fixed data budget is more distinct parameterized
computation per loop (four untied blocks), whether summed in parallel or composed in
series. A learned router is not needed. Under a CPU budget, the cheapest way to buy
accuracy is more optimization of a single block.

## 3. Native sparse routing (P4)

Native sparse routing does not keep the mixture's gain.
* **H (Gumbel straight-through top-1, load balance 0.01):** matches A, not B. H − B is
  −0.231 on cex_repair, −0.158 on proof_deps and −0.112 on proof_resolve. H − A is within
  ±0.033 on every family except cumsum3 (+0.076).
* **I (soft-to-hard anneal):** collapses. I − B is −0.94 on local_map3 and −0.49 on
  revision_retract.
* **Cost:** hard dispatch on the CPU reference path cuts FLOPs only to 3.11 MFLOP (B 5.50,
  A 1.03), because every module still projects K/V and the manager runs every loop. Latency
  barely moves (39.5 vs 42.4 ms).

## 4. Specialization (seeds 11 and 23, hidden roots)

* **B's modules are complementary, not specialists.**
  - Gradient cosine between modules is 0.03–0.08.
  - Output CKA ranges from 0.07 (cex_repair) to 0.54 (proof_resolve).
  - Removing any single module costs up to 0.27 exact (proof_resolve 0.59 → 0.32–0.56).
  - Forcing all routing onto one module drops accuracy to 0.03–0.33.
  - Cyclically permuting modules under the fixed router collapses proof_resolve to 0.12.
  - Uniform weights cost 0.03–0.34 on B, but C is trained with uniform weights and does as
    well as B. So the router matters only to the model trained with it.
* **Routing does not depend on the loop.** Routing/loop MI is 0.007 nats for B and 0.001 for
  H, against a 1.386-nat maximum. B's router entropy is 0.94 nats.
* **Task-conditional routing.** Each model is trained on all four principal families:

  | arm | MI(family; module) | of 1.386 max |
  | --- | ---: | ---: |
  | H (sparse) | 0.69 | about half |
  | B (soft) | 0.078 | |
  | I (anneal) | 0.001 | |

  Hard routing does produce task-specialized modules, but each module is then only as good
  as one block.
* **Effective rank** of final states and loop updates is similar across arms (12–26). It
  does not separate winners from losers.
* **Shared-memory interventions.** Swapping in another example's memory (`stale_swap`) sends
  every mixture arm to 0.12–0.22 on proof_resolve, so memory content is used causally.
* **Not measured for J.** Single-module ablation and routing interventions do not apply to J:
  the sequential path ignores module masks. Its permutation result is reported (0.34–0.50).

## 5. Loop forcing (P5)

The effect at 300 updates is small and mostly negative for soft arms. All-loop minus
final-loop supervision:

| arm | proof_resolve | experiment_portfolio | cex_repair | revision_retract |
| --- | --- | --- | --- | --- |
| B | −0.045 [−0.088, −0.002] | −0.028 [−0.046, −0.009] | — | — |
| A | −0.027 [−0.050, −0.004] | — | — | — |
| C | −0.021 [−0.043, −0.000] | — | — | — |
| H (sparse) | — | +0.026 | +0.051 [+0.023, +0.078] | +0.053 |

Other cells are within ±0.02 with CIs spanning zero. This partly reverses MOLT-02's
recommendation to default to loop forcing.

## 6. Distillation (P6): retention target not met

K is a single 4-visit TRM trained with labels plus per-loop KL to the frozen B teacher of
the same family and seed. It retains **−0.26 [−0.39, −0.14]** of B's advantage pooled over
seven families. It does no better than A anywhere, and is worse on cumsum3 (−0.170) and
local_map3 (−0.195). K_win, distilled from the width-64 Inky finalist, retains −0.13
[−0.23, −0.03]. The registered criterion (pooled retention ≥ 0.5 with a CI lower bound
> 0) fails. In development, Inky's hidden-state distillation (+0.024) and logits_all into
a 3-module student (+0.207, with the student itself a mixture) did not change this.

## 7. Inky Mini campaign

The archive is in [`receipts/campaign/ARCHIVE.md`](receipts/campaign/ARCHIVE.md). The
campaign ran 20 proposal iterations: 19 trained, 1 rejected by validation. 3 adversarial
variants were verified and used. It took 160 development runs, 30 requests (all ok; 7 hit
the completion cap and were retried at low effort), 257 k prompt and 77 k completion
tokens, and $0.

Inky's development findings, each tested against its own control:
* tying module weights kills the gain (iterations 2 and 11);
* sharpening routing with entropy +0.5 drops quality from +0.166 to +0.014;
* 2–3 modules capture most of the gain at 8–12 visits;
* damped updates and a ramp loss help a 3-module HRM;
* widening that model to 64 gives the best development quality (+0.250).

On hidden roots its finalists tie B:
* inky_08d849 is within ±0.023 of B on every family except cex_repair (−0.051).
* inky_139d58 (166 k params, 17.2 MFLOP) beats B only on revision_retract (+0.078) and
  research_plan (+0.024).

Failures, as recorded:
* iteration 12 asked for functional module objectives on 3 modules, which the grammar
  rejects;
* iteration 17's hidden-state distillation between widths 64 and 32 failed on a shape
  mismatch;
* the functional deduce/repair/verify objectives (iterations 3 and 18) lowered quality.

Hypotheses, predictions, experiments, observations, revisions and decisions are exported
to `receipts/campaign/datagoblin/research_distillation.jsonl` in the DataGoblin
research-distillation format. That is 24 rows: 20 hypothesis cycles, 3 adversarial tasks
and 1 final-evidence row bound to the freeze.

## 8. Repaired and new families

* **local_map3** (window-3 lookup over 4 digits) replaces the saturated shallow_copy and
  local_map controls. Calibration: A 0.965 vs B 0.984; finals: A 0.896.
* **cumsum3** replaces the floor-bound cumsum (A 0.676).
* **research_plan** was never seen before the freeze. Every fixed-update arm stays at
  0.003–0.042 on it; only A_cpu reaches 0.427. It is a useful hard family for the next
  tournament.
* **No-recurrence control.** local_map3 was registered as needing no recurrence, yet it
  still shows B − A +0.092 at fixed updates. That fits the undertraining reading: A_cpu
  reaches 1.000 on it.

## 9. Recommendation for the Baby Goblin / DataGoblin architecture

1. **Default skill module:** one recurrent block with prefix memory and 4 loops, with
   compute spent on optimization. It is the most accurate arm here and has the cheapest
   inference.
2. **When updates or data are the binding limit:** four untied blocks composed sequentially
   within each loop (J), with no router. It has the best fixed-update mean (0.655) at
   35.6 k params and 4.1 MFLOP.
3. **Drop:** the HRM manager, native sparse routing, soft-to-hard annealing, functional
   module objectives, and logit or hidden-state distillation into a single block. None
   cleared its registered control.
4. **Next tournament (preregister):**
   - a data-matched vs CPU-matched factorial: A with more updates on a fixed dataset,
     against A_cpu with fresh data, to separate optimization from data exposure;
   - J vs A at equal process CPU;
   - research_plan with larger budgets;
   - distillation that starts from the teacher's blocks rather than from scratch.

## Limitations

* One model scale (width 32, 300 updates) and synthetic tasks.
* A_cpu changes data exposure as well as compute.
* The campaign stayed small (20 iterations) because it ran at below-normal priority on a
  shared machine.
* Routing statistics written before metrics version 2 summed over loops; `analysis3.correct_v1`
  rescales them exactly, and new records are normalised. The campaign prompts did not
  include those statistics.
* There is no multiplicity correction across the 7 contrast groups × 8 families.
