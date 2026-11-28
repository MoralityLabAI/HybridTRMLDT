# TRM-SKILL-GYM-02: full campaign and Baby Goblin skill engineering

Branch `research/trm-skill-gym-02-20261128` (PR #51).

- Preregistration: [PREREGISTRATION_FULL.md](PREREGISTRATION_FULL.md), with stage addenda committed before
  each stage launched.
- Deviations: [full/DEVIATIONS.md](full/DEVIATIONS.md).
- Machine-readable files: [summary.json](summary.json) (gates and per-stage results) and
  [receipts.json](receipts.json) (sha256 of every result file and source module, plus replay outcomes).
- Raw outputs live in [full/](full/).
- The final skill registry is [research_gym/baby_goblin/registry.json](../../research_gym/baby_goblin/registry.json).

Earlier GYM-01 and GYM-02 diagnostic results are unchanged. The GYM-02 CI diagnostics were re-run
locally into [ci_replay_local/](ci_replay_local/), because downloading the CI artifacts requires
authentication.

## Execution

- The campaign ran 11:49–19:49 MPST under an 8-hour cap, CPU only, with one training worker.
- Every stage ran as a child process under a hard 3 GiB process-tree RSS limit. The largest observed
  peak was 1.48 GiB (sequential research stage).
- At 12:20 the user asked for less CPU. From then on the whole campaign tree was pinned to 3 cores and
  used about 0.8 cores in practice, so stages ran slower than planned.
- The live Inky Mini (free Inkling-2-Mini) calls ran in a separate network-bound process pinned to one
  core.

## Established results (preregistered gate passed)

1. **Race (S1): matched-damping TRM is promoted for deduction and repair.**
   - Setup: seeds 10–13, 800 steps, 4096 training instances, fresh roots, and selection on calibration
     only by worst-stratum exact, then macro exact, then FLOPs, within the direct-TRM FLOP ceiling and
     a two-receipt limit.
   - On test (256 roots per stratum, 4 seeds):

     | skill | winner | macro (vs B) | worst stratum (vs B) | worst stratum is |
     |---|---|---|---|---|
     | deduction | E | 0.982 (0.778) | 0.889 (0.046) | held-out long_chain_14 |
     | repair | E | 0.821 (0.712) | 0.640 (0.473) | withheld dense_large_region |

   - The deduction strata include hint-free copies, five training topologies, and the held-out ladder
     and long_chain_14 topologies.
   - The GRU proxy scored higher still on both skills at test (deduction worst 0.91, repair 0.74). It
     was ineligible because its 3.44M FLOPs per instance exceed the 3.35M TRM ceiling.
2. **Coverage-aware LDT/TRM hybrid (S3): the gate passes on retrieval and deduction with zero
   certificate violations.**
   - Soundness: 492 exhaustively enumerated proof obligations (every realization of the uncertain
     edges) and every held-out certified decision.
   - Macro exact over noisy and incomplete test strata (2 seeds):

     | skill | hybrid | neural only | certificate + constant | threshold solver | hybrid + verifier |
     |---|---|---|---|---|---|
     | retrieval | 0.699 | 0.664 | 0.373 | 0.433 | 0.738 |
     | deduction | 0.682 | 0.635 | 0.122 | 0.371 | 0.722 |

   - The network runs only on unresolved instances: 60–64% of them; the rest are fully certified.
   - On the unresolved instances the hybrid scores 0.67 exact, against 0.09–0.32 for constant fill. So
     neural computation adds value specifically on the unresolved portion.
   - On complete observations the certificate route is exact (1.00), against 0.88–0.93 for neural only.
3. **Mixture of loops (S5): the gate passes, but symbolic experts carry it.**
   - Compositional episodes: noisy retrieval, then deduction over the retrieved set (typed handoff),
     then repair, stopping and research. 384 test episodes, metering only executed experts.

     | dispatch mode | utility | all stages correct | metered cost |
     |---|---|---|---|
     | fixed neural | 0.549 | 0.180 | 0.148 |
     | learned | 1.035 | 0.680 | 0.0027 |
     | confidence-gated | 0.667 | 0.055 | 0.032 |
     | fixed symbolic | 1.037 | 0.682 | 0.0017 |
     | oracle (not realizable) | 1.086 | 0.758 | 0.0033 |

   - Learned dispatch: +0.486 over fixed neural (95% CI 0.45–0.52) at 1/55 of the metered cost. In
     effect it learned to route to the exact solvers and lands level with fixed symbolic.
   - Confidence-gated dispatch: +0.118 (CI 0.097–0.140).
   - Selective cross-expert latent transfer through a learned ridge adapter changed utility by +0.0005
     (CI 0–0.002). Typed intermediate handoff remains the right interface.
4. **Symbolic solvers are exact on the GYM-01 task contracts.**
   - Parity BFS (repair), finite-horizon DP from the public stopping features, exact EIG from the
     public research features, and graph search on complete graphs are exact on every tested instance
     (`tests/test_baby_goblin.py`).
   - So the GYM-01 skills that trained modules "solved" at 0.80–0.99 have exact symbolic solutions at
     about 1e-3 of the cost.

## Failed hypotheses (gate evaluated, not passed)

1. **Race promotion for retrieval, curation and research.**
   - Retrieval selected an archived Inky configuration (`archive_3bb535d47495`). Macro +0.014 and
     worst stratum 0.603 vs 0.600, so it is not promoted.
   - Curation kept the feedforward head; research kept the existing variance heuristic (test macro
     0.842 vs at most 0.73 for any module).
   - On the held-out long_chain_14 retrieval stratum, every recurrent model collapses (GRU 0.02, TRMs
     0.00). Only the feedforward head and the archived configuration reach 0.60.
2. **BPD selective removal of harmful recurrent writes (S2).**
   - Setup: genuine BPD at MoralityLabAI/BPD ab93e4d (`TiedComponentLinear` shared factors, exact
     residual, pass-tied CI, mean-write ablation, pass-shared PGD adversary).
   - Repair (2 seeds): learned-component interventions (F) changed test exact by −0.007 against the
     unmodified model.
   - Curation (1 seed): F changed test exact by −0.003, and F was no better than the random (G) or
     shuffled-importance (H) controls.
   - BPD's transient/sustaining analysis classified the selected components as inert.
3. **Sequential research policies (S4).**
   - On the withheld deep_chain_h5 family, the best learned policy (FF 1.11) is below the variance
     heuristic (1.47) and greedy EIG (1.14). The exact planner scores 1.77.
   - On the development families, the learned Q-regression policies approach the planner: GRU 3.07
     against planner 2.91 realized and variance heuristic 2.49 on screen_gate_h4. They generalise
     worse.
   - RLT-TRM did not revise plans better after contradicting evidence: −0.33 utility vs TRM (2 seeds,
     SE 0.60).
4. **Distillation (S6).** The residual ridge student beats the previous ridge pilot on rung-1 MAE in
   every holdout, but it does not beat the incumbent-score baseline (leave-skill-out rung-1 MAE 1.13 vs
   1.19 pp, CI includes 0). Within-round ranking stays near chance (about 0.5), and the student was
   picked after seeing its holdout numbers.

## Exploratory findings

- **BPD at extrapolated depth (curation, 4x visits).** Removing recurrent state helps: F +0.091, but H
  matches it exactly and coordinate reset reaches +0.172. That is a depth-stabilisation effect of
  removing state energy, not a selective one.
- **Hint dependence.** The leakage audit flags only the GYM-01 retrieval lexical-overlap column (|r| ≈
  0.35–0.39) on the original families, and no hint-free or topology stratum. Hint-free accuracy is
  close to hinted accuracy for every trained retrieval model. The existing-skill heuristic does better
  without the hint (layered_shallow 0.41 hinted vs 1.00 hint-free), because the hint adds false
  positives.
- **Live Inky prompt comparison** (free Inkling-2-Mini, temperature 0, low effort, 16 fresh instances
  per cell). Compact prompts scored below the full SKILL.md at matched caps:

  | skill | cap | compact | original SKILL.md |
  |---|---|---|---|
  | stopping | 2048 | 0.19 | 0.50 |
  | stopping | 6000 | 0.44 | 0.56 |
  | research | 6000 | 0.06 | 0.19 |

  Research truncates in 13–16 of 16 cases at every cap. No malformed-JSON failures were recorded
  without a truncation.
- **Inky architecture engineering (S7).** Inky received development measurements only. Proposals were made against the D incumbent on repair,
  curation, research and retrieval (2 rounds), and measured at 300 steps on one seed against 256
  development roots. The stage hit its wall cap after 41 of 48 proposals; the remaining 7 are recorded
  as unmeasured.

  | arm | mean dev-exact change vs incumbent | improved | prediction MAE |
  |---|---|---|---|
  | certificate-coverage-informed | +0.097 | 4 of 7 | 0.157 |
  | BPD-informed | −0.002 | 1 of 6 | 0.383 |
  | RLT-constrained revision | −0.074 | 3 of 7 | 0.253 |
  | random mutation | −0.079 | 2 of 7 | no prediction |
  | direct proposal | −0.089 | 2 of 7 | 0.285 |
  | failed-research-informed | −0.092 | 2 of 7 | 0.239 |

  The arms produced 0 malformed, truncated or invalid proposals. The student-ranked arm was not run.
  With one seed and seven proposals per arm this is a pilot: only the coverage-informed arm improved on
  average, and nothing here is confirmatory.
- **Shuffled-supervision audit (S8).** This uses the corrected GYM-01 shuffle on research and stopping (300 steps). With verifier feedback
  off, shuffled supervision scores 0.094 on research (calibration-majority slot 0.125) and 0.492 on
  stopping (majority 0.492). Both are within the 0.05 leak criterion. Feedback on gives the same
  numbers. True supervision reaches 0.49 and 0.79, so no label path leaks in either mode.

## Unfinished experiments

- Race: the stopping skill (cut by the 3.0 h calibration cap), and two archived research
  configurations (excluded as incomplete).
- BPD: research skill (both seeds) and curation seed 1, cut by the stage deadline.
- Research environment: ff/gru seed 13 and trm/rlt seeds 12–13.
- The student-ranked Inky arm, because the student failed its gate and has no validated scoring
  interface.
- Inky fallback as a routed expert.
- Coverage-aware hybrid checkpoints were not saved, so the hybrid is not shipped as a learned skill.
- Larger graph sizes: the 16-slot contract is unchanged.

## Verification

- Tests: 114 passed (GYM-01, GYM-02 v2 diagnostics, v3 core/research/BPD/distillation, Baby
  Goblin), run on one pinned core in 547 s. BPD receipt replay: 6 sampled interventions each for one
  repair job and one curation job (both seed 0) recompute exactly (`full/bpd_replay.json`).
- Audits ([full/audit.json](full/audit.json)):
  - no train/test content collisions in sampled pools, and disjoint root ranges per stage;
  - every race skill was frozen before its first test file existed;
  - all stage peaks are under the 3 GiB limit;
  - 10/10 sampled race test receipts replayed exactly from checkpoints;
  - no credential appears in 712 scanned artifact files.
- BPD receipts carry `replay()` digests (see [full/bpd/](full/bpd/)).

## Recommendation: what Baby Goblin should inherit

1. **Primary: exact symbolic solvers and sound three-valued certificates.** They are exact or sound
   wherever the observation is complete, cost about 1e-3 per instance, and learned dispatch converges
   to them anyway (S5).
2. **Coverage-aware recurrent hybrid for incomplete or noisy graph observations.** It is the only
   setting where learned recurrence added measured utility over strong symbolic baselines (S3). Retrain
   and checkpoint it before shipping.
3. **Matched-damping TRM (E) for deduction and repair, as learned fallbacks** (promoted in S1;
   `registry.json` status `accepted`). Use them only where a solver cannot run. The GRU proxy is more
   accurate but above the FLOP ceiling; revisit if that ceiling can move.
4. **Research: the exact finite-horizon planner where tractable, otherwise the variance heuristic or
   reuse-first.** Learned research policies stay experimental.
5. **Do not inherit yet:**
   - BPD component interventions: no selective benefit at the trained depth.
   - Cross-expert latent sharing: no effect.
   - Inky text-only execution for numeric skills: it truncates.
   - The distillation student: it does not beat the incumbent score.
   - The RLT schedule for plan revision: no benefit measured.
   - Every skill that failed its gate stays `experimental` in the registry.
