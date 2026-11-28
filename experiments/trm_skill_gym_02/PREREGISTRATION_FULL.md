# TRM-SKILL-GYM-02 full campaign preregistration

Registered 2026-11-28, before any full-campaign result. GYM-01 and the GYM-02 diagnostic results
(`PROTOCOL.md`, commits d3d3151..ca6d1eb) stay as they are. Stage-specific parameters that depend on
implementation details get appended under "Stage addenda" before that stage launches, and each addendum
is committed with a timestamp that precedes the stage's first job. Numerical margins are fixed here.

## Budget and process controls

- CPU only, 8-hour campaign cap, one training worker, `torch.set_num_threads(1)`.
- Every training or evaluation stage runs as child processes under
  `research_gym/trm_skill_gym_v3/supervisor.py`. The supervisor kills the whole process tree when its
  RSS exceeds 3 GiB or its wall cap expires. Before starting a job it waits while system available RAM
  is below 1.5 GiB.
- The desktop was at 100% CPU on all 20 cores at launch, so concurrency stays at one training worker.
- Live Inky (free `thinkingmachines/Inkling-2-Mini` through `research_gym/trm_skill_gym/inky.py`) runs
  in a separate network-bound process. Context is 12k by default with a 16k ceiling; there is no paid
  fallback. The credential is read from the Desktop key file at call time.
- Every stage is resumable: job result files are written once and skipped on restart.

## Stage 1: fresh architecture race (`trm_skill_gym_v3/race.py`)

- Roster per skill: A existing skill, B feedforward, C GRU, D direct TRM, E matched damping, F RLT, plus
  the first 2 distinct archived Inky rung-1 configurations per skill in (arm, round, proposal) order.
  Random-mutation entries are excluded from the archive roster. Limiting the archive to 2 configurations
  per skill is a budget deviation from "all archived finalists".
- Seeds 10, 11, 12, 13. 800 gradient steps at batch 32 per job. 4096 training instances.
- Roots: train 4,000,000+, calibration 4,100,000+, test 4,200,000+. These are disjoint from every
  earlier range.
- Graph strata (retrieval, deduction), `trm_skill_gym_v3/graphs.py`:
  - Calibration: the original dev families, hint-free copies of the dev families (label-correlated hint
    columns set to 0.5: retrieval col 1, deduction col 0), long_chain_6, long_chain_10, branching,
    shared_subdeps, distractors, misleading.
  - Test adds the original withheld family, its hint-free copy, ladder (a held-out topology) and
    long_chain_14.
  - All topology instances permute candidate slots at random.
  - Graph size stays at the 16-slot module contract; larger graphs are not supported.
- Non-graph skills: calibration uses the original dev families; test uses dev plus withheld.
- 128 calibration and 256 test instances per stratum.
- Training covers the calibration families on training roots. The withheld family and withheld
  topologies are never trained on.
- Selection (calibration only, identical strata for every candidate): worst-stratum exact, then macro
  exact, then mean model FLOPs. Eligibility: model FLOPs at most the D ceiling and at most 2 verifier
  calls.
- A candidate missing any seed (incomplete exposure, memory or wall kill) is excluded. A missing fixed
  baseline blocks freezing for that skill.
- Frozen before test: `race/frozen.json`.
- Confirmatory contrast: selected vs the strongest eligible baseline among A, B and C, with per-stratum
  paired root bootstrap (2000 resamples) on seed-averaged outcomes.
- **Gate (promotion):** test macro delta >= +0.02, and selected worst-stratum >= baseline worst-stratum
  - 0.02. If the selected candidate is itself a baseline, "incumbent retained".
- Descriptive: every candidate's test exact per stratum and seed, hint vs hint-free gap,
  withheld-topology gap, train CPU, inference CPU, FLOPs, host tensor ops, verifier calls and visits.

## Stage 2: BPD for TRM skills (`trm_skill_gym_v3/bpd_skill.py`)

- Uses MoralityLabAI/BPD at commit ab93e4da986beef87897f8d1244c7fe7564b5bde
  (`TiedComponentLinear` shared factors, exact residual, pass-indexed importance, pass-conditional
  mean-write ablation, pass-shared adversary where compatible).
- Skills: repair, curation and research. Checkpoints: GYM-01 F (RLT) seeds 0 and 1. Fresh race F
  checkpoints are used if available in time.
- Roots: fit 2,300,000+, calibration and selection 2,400,000+, test 2,500,000+.
- Arms: A unmodified, B full latent reset, C selected coordinate reset, D behavioural orthogonal
  projector, E SVD components, F learned BPD components, G rank- and energy-matched random, H
  shuffled-importance BPD.
- **Gate (selective removal of harmful writes):** on at least one priority skill, F improves test exact
  over A by >= +0.02 with a paired root-bootstrap 95% interval above 0. F must also beat both G and H by
  >= 0.02.
- Descriptive: transient vs sustaining components, per-pass vs whole-trajectory effects, and transfer
  to 2x and 4x visits.

## Stage 3: coverage-aware LDT/TRM hybrid (`trm_skill_gym_v3/hybrid.py`)

- Per-candidate certificate status (in / out / unresolved), certificate depth, propagation cost and
  remaining uncertainty are fed to a trained module that predicts only unresolved slots. An adaptive
  policy chooses among more symbolic propagation, neural recurrence, verification and termination.
- Exact graph solvers are baselines and get direct routing when the observation is complete.
- Noisy and incomplete observation strata, where propagation cannot settle the answer: edges hidden or
  flipped with known rates.
- **Gate:** on the noisy and incomplete strata, the hybrid beats both symbolic-only (unresolved slots
  filled by the best constant) and neural-only by >= 0.02 exact on test, with zero observed certificate
  violations. All host operations are metered.

## Stage 4: sequential research environment (`trm_skill_gym_v3/research_env.py`)

- Policies: exact finite-horizon planner (where tractable), greedy EIG, variance heuristic, reuse-first,
  random valid, FF, GRU, TRM, RLT-TRM. Train 3,000,000+, calibration 3,100,000+, test 3,200,000+.
- **Gate:** the best learned policy beats both the variance heuristic and greedy EIG in full-rollout
  verified utility on the withheld family by >= 5% of the (exact planner - random) utility gap, with a
  paired bootstrap interval above 0.
- RLT question (exploratory): RLT-TRM vs TRM utility after the contradicting outcome in the surprise
  condition.

## Stage 5: mixture of loops (`trm_skill_gym_v3/router.py`)

- Experts: race or GYM-01 checkpoints and symbolic solvers, each behind a typed contract.
- Compositional episodes need retrieval, deduction, repair and stopping. Dispatch execution and cost are
  metered: only executed experts are counted.
- **Gate:** learned or confidence-gated dispatch reaches episode utility >= fixed dispatch + 0.02 at
  metered cost <= fixed dispatch. Oracle routing is reported as an upper bound only.

## Stage 6: GoblinBench distillation (`trm_skill_gym_v3/distill.py`)

- **Gate:** under leave-one-skill-out, the best student beats both the ridge pilot and the incumbent
  baseline on rung-1 MAE and within-round ranking accuracy. Missing outcomes stay missing.

## Stage 7: live Inky research engineering (`trm_skill_gym_v3/inky_lab.py`)

- Development measurements only. Arms: direct proposal, RLT-constrained revision, random mutation,
  student-ranked, and Inky informed by BPD evidence, certificate coverage or failed research
  trajectories. Short JSON replies.
- Numeric-skill prompt comparison: compact prompt vs original SKILL.md at matched completion caps and
  reasoning effort.
- Descriptive. With a budget of at most a few rounds per arm, no gate.

## Stage 8: Baby Goblin package

A skill ships as `accepted` only if its source stage passed its gate. Otherwise it ships as
`experimental`.

## Verification

- Full test suite and independent replay of sampled receipts.
- Shuffled supervision audited with verifier feedback off and on.
- Label-leakage, train/test overlap and selection audits.
- Resource-accounting check and credential scan.
- A protocol deviation log.

## Stage addenda

(appended before each stage launch)

### Addendum S1/S3/S4/S2 (2026-11-28 11:55 MPST, before any stage launch)

- **Race (S1).** Parameters are exactly the `race.py` defaults: seeds 10-13, 800 steps, 128 calibration
  and 256 test instances per stratum, 4096 training instances, `--max-archive 2`, 900 s per training
  job, and a 3.0 h stage cap. Skill order: retrieval, deduction, repair, curation, research, stopping.
  Skills not frozen by the cap are reported as unfinished.
- **Hybrid (S3).**
  - Strata are `noisy|incomplete|complete` × base families. Base families for retrieval:
    layered_shallow, layered_wide, branching, shared_subdeps, distractors, misleading, long_chain_10.
    For deduction: kernel_h4, kernel_h5 and the same topologies. Test adds layered_deep/kernel_h7 and
    ladder.
  - Noise: p_drop 0.3 and a spurious-edge fraction of 0.2. Detector scores are informative (noisy) or
    0.5 (incomplete). The hint columns are neutralised throughout.
  - Seeds 10 and 11, 800 steps, train 5,000,000+, calibration 5,100,000+, test 5,200,000+.
  - Chosen on calibration: the constant fill in {0, 1} and the adaptive tau in {0.5, 0.7, 0.9, 0.97}
    (score = exact - 0.002 × verifier calls).
  - The S3 gate uses the `hybrid` arm (not adaptive) vs `cert_const` and `neural_only`, macro over the
    noisy and incomplete test strata. `threshold_exact` is the strong symbolic baseline and is reported
    beside them.
- **Research (S4).**
  - Families: dev screen_gate_h4 and confounded_h5, withheld deep_chain_h5, plus the separate
    surprise_h4 condition. 300 training worlds and 60 test worlds per family. Seeds 10-13, 3500 steps.
  - Learned policies regress exact Q-values; calibration roots are used only for checkpoint selection.
  - The gate is as registered above, with a world-level paired bootstrap.
- **BPD (S2).**
  - `bpd_skill.py` defaults: 800 steps, fit 96, calibration 96 and test 128 per family, 5 random seeds ×
    8 draws, depth 2x and 4x. GYM-01 F checkpoints seeds 0 and 1; fresh race checkpoints are not used
    in this stage (deviation).
  - F selection: components in the top 50% by peak CI, ranked by calibration gain, with the best
    calibration prefix in {1, 2, 4, 8, 16, 24}. H uses permuted importance.
  - G: rank- and energy-matched random sets, best of 8 draws on calibration. This makes G a stronger
    control than a single random draw.

### Addendum S5/S6/S7 (2026-11-28 12:20 MPST, before these stages launch; no router, distillation or Inky-lab result had been observed beyond a 12-episode router smoke run on scratch roots)

- **Router (S5), `router.py`.**
  - Episodes: noisy retrieval, then deduction over the retrieved set (typed handoff; the label is the
    kernel over the true set), then repair, stopping and research.
  - Utility: 0.2 per correct stage plus a 0.2 all-correct bonus, minus 1e-8/FLOP, 1e-5/host op,
    0.01/verifier call and 0.05/Inky call.
  - Experts: symbolic, cert, neural (GYM-01 packages and the research B checkpoint), heuristic.
  - Calibration: 192 episodes from 7,100,000+, used for the ridge predictors, tau in
    {0.05, 0.2, 0.4} and the latent adapter. Test: 384 episodes from 7,200,000+.
  - The S5 gate compares `learned` and `confidence_gated` against `fixed_neural`, which is the
    learned-skill dispatch registered as "fixed dispatch". `fixed_symbolic` is reported as the strong
    symbolic baseline.
  - Latent transfer (`fixed_neural+latent_transfer`) vs isolated typed handoff is reported
    descriptively.
  - The Inky fallback expert is not part of S5 (cost and latency); it is reported as unfinished.
- **Distillation (S6), `distill.py`.** CLI defaults; gate as registered.
- **Inky lab (S7), `inky_lab.py`.**
  - Arms: direct, rlt_constrained, random_mutation, coverage_informed, failed_research_informed,
    bpd_informed and student_ranked, on skills repair, curation, research and retrieval, for 2 rounds.
  - Proposals are generated against the D incumbent. Measurement trains 300 steps on seed 0 with
    2048 training instances from 6,200,000+ and scores exact on 256 development roots from 6,100,000+.
  - Arms whose evidence does not exist at generation time record `evidence_unavailable`.
  - Descriptive only.
