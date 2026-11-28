# TRM-SKILL-GYM-02: paired skill audit and research policy

Frozen source: 8cad4a3a33a36765274aab227715988038dd9ff9 (GYM-01).
Initial archive audit: 108 proposals, 36 rung-1 rows, 34 distinct skill/config pairs; one pair measured on both dev families.
This is a follow-up protocol, not a promotion claim.

## CPU diagnostic (implemented)
- 02A: frozen A/B/C/D/F/H checkpoint replay on identical per-family calibration roots starting at 1,100,000; held-out roots at 1,200,000. Add shuffled-slot chain graph depths 4/8/12, hold out 14. No lexical/score label correlations on chains. Rank by worst-stratum accuracy, then macro accuracy, then mean measured model FLOPs. Test on same strata plus original withheld and depth 14; never tune on test. Original checkpoints have only two training seeds; report them separately.
- 02B: causal coordinate reset probe on the frozen F TRM, choose top quartile of coordinates by single-coordinate effect on calibration, compare keep, full reset and seeded rank-matched random reset on test. This is not a BPD decomposition; subsequent BPD experiment should learn an orthogonal subspace on calibration only, include random orthogonal controls, and reproduce across fresh training seeds.
- 02C: per-skill ridge router predicts each frozen expert's correctness from public candidate statistics, graph density and global features. Calibrate on shared roots, constrain measured model FLOPs to the direct TRM and in-loop+receipt calls to <=2. Compare static calibration-selected arm and a hindsight oracle. Report dense routing FLOPs and CPU per expert separately. No claim of actual end-to-end speedup: replay executes every expert offline to obtain counterfactual outcomes. Heuristic host work is not counted by the model FLOP counter.
- 02D: public bounded graph certificates at 1/2/4/8/16 rounds, report coverage, violations, edge examinations, and standalone symbolic exact success. Retrieval monotone reachability; deduction propagates forced kernel membership on DAGs. Full symbolic solver is an explicit baseline. The diagnostic measures certificates; a trained coverage-aware projection is still a subsequent arm. Count host work, separate it from model FLOPs.
- 02E: downstream two-step expected entropy reduction minus cost with binary conditionally independent observations, no repeated actions and optional stopping. Fit ridge on 64 independent synthetic worlds in smoke mode, compare one-step EIG, variance and exact two-step planning on disjoint worlds. All first choices receive the same optimal continuation for evaluation, so this tests first-action ranking, not full policy rollout. No paid API.
- 02F: export every proposal with input/target separated, source digest, hold out research and stopping by whole skill. Include rejects and missing rung measurements, never impute them as zeros. Distillation student training is deferred; this is a supervised corpus, not an independently validated prediction benchmark (history may contain incumbent results from earlier rounds).
- 02G: opt-in free Tinker compact prompt vs original skill prompt on identical fresh numeric cases. Equal 2048 token cap, low reasoning effort and temperature zero. Strict JSON, valid slots, unique integer choices, record truncation and format failures separately via finish reason/status. API key only via existing client, no credentials in CI. This comparison changes prompt length and instructions together.
- Leakage: opt-in true/shuffled supervision x 0/1 verifier feedback for research/stopping, distinct roots; compare empirical calibration majority slot. Use the corrected GYM-01 shuffle implementation. The exact/diagnostic host can read targets, but feedback-off should isolate the supervision path. This is a diagnostic, not a general proof.

## Follow-up full campaign gate (not yet run)
Before architecture search, freeze train/cal/test manifests and preregister 4 independent training seeds, all original dev families and hard graph strata. Race all proposals (direct Inky, matched mutation, meta-RLT) on identical strata at matched exposure and CPU/verifier budgets; use paired roots and worst-family score, prohibit any held-out tuning. Require per-family paired bootstrap intervals, strict matched inference budgets, failure accounting, and no promotion from saturated development scores alone. Final promotion requires positive paired utility or accuracy versus the strongest eligible incumbent and no material worst-stratum regression; choose numerical margins before launch. Use a budget controller with process isolation and a hard RSS limit before attempting the eight-hour run. The supplied replay is bounded by a cooperative deadline between skill jobs, not a hard process kill. Original source results remain immutable.

## Run
```sh
python -m pytest -q tests/test_trm_skill_gym_v2.py
python -m research_gym.trm_skill_gym_v2.runner --out /tmp/skill02 --count 24 --seeds 0 --research-worlds 64
python -m research_gym.trm_skill_gym_v2.runner --out experiments/trm_skill_gym_02/run --count 128 --seeds 0 1 --research-worlds 512 --shuffle-steps 300
# Local only, with existing free Tinker credential:
python -m research_gym.trm_skill_gym_v2.runner --out experiments/trm_skill_gym_02/live --count 24 --seeds 0 --live-inky 16
```

Outputs: results.json, distill.jsonl. The replay deliberately uses model checkpoints already committed in GYM-01; no new architecture training claim. Certificate checks are exhaustive for small DAGs (up to five nodes), supplemented by fresh graph instances. The research utility is synthetic and is not a measure of open-ended research ability.

## Archived-finalist re-race (added executable)
The separate race module trains every distinct valid archived final-rung configuration and fixed A/B/C/D controls from scratch, by default at 800 steps across seeds 10, 11, 12 and 13. Shared training includes graph chain depths 2/4/6/8. All candidates see identical calibration strata. A per-job wall cap and deadline apply; incomplete candidates are explicitly excluded, and a missing baseline stops the freeze. Select within the direct TRM model-FLOP ceiling and two-receipt limit. Freeze before test metrics; compare with the strongest eligible fixed baseline selected on calibration. Bootstrap paired root outcomes averaged across seeds and separately report seed deltas. This does not estimate uncertainty across architecture-training seeds. This is an archived finalist re-race, not a new live Inky search. Training time is measured, not equalized; matched gradient steps are the exposure control. No hard RSS kill beyond the inherited incremental guard. Inspect incomplete counts before interpreting.

```sh
python -m research_gym.trm_skill_gym_v2.race --out experiments/trm_skill_gym_02/race --hours 8
# Small CI-only fixture
python -m research_gym.trm_skill_gym_v2.race --out /tmp/skill02-race --skills deduction --seeds 10 --steps 2 --count 8 --train-count 64 --max-archive 1
```

## SKILL-02B/F continuation protocol

Freeze this revision before examining new pilot outcomes.

- Latent intervention: use checkpoint F (RLT), both original checkpoint seeds, repair/curation/research/stopping. Fit an orthogonal eigenbasis from failure-minus-success trace-normalized second moments of recurrent latent state around z0 on fit roots 1,900,000+. Calibrate individual component interventions on disjoint roots 2,000,000+; fix top quartile. Evaluate on 2,100,000+ plus original withheld family. Controls: keep, full reset, coordinate-selected rank match, top-energy PCA, four seeded Haar rank-matched projectors in CI. Every method uses the same test roots. Report removal energy per instance and paired root bootstrap per family; no seed-level CI claim. No norm-matched claim, and the representation is state space.
- Weight control: one physically tied output matrix (net_z.layers.0.b), exact full SVD shared across invocations, pass-indexed fit-mean input centering, preserve residual/bias. Select top quartile singular components by independent calibration interventions. Controls: largest singular components, rank-matched random component sets, keep and single-invocation versus all-invocation edits. Report removed write energy and reconstruction error. Inspired by BPD's physical-matrix sharing and per-pass mean ablation; does not reproduce its trained factors, trajectory reconstruction objective, sphere normalization or adversary.
- Student: use original 108 archive proposals with whole-skill holdout research/stopping, fixed ridge alpha 10, pre-measurement config/changes/incumbent score only. Separate rung0/rung1 labels, no imputation, compare training mean and incumbent. Report MAE, bias, RMSE and within-round pair accuracy when measurable. This is a scalar prediction pilot, not sequence-model instruction distillation.
- Fresh training CI pilot: two independent seeds (10,11), retrieval and deduction, fixed baselines plus up to three distinct archived finalists each, 120 steps, 512 training examples, 64 roots per stratum. This tests execution and hard graph adaptation at modest exposure. All four-seed 800-step production-scale claims remain untested. Source training and test tasks share generator families by design.
- Inference cost excludes host work for the model-FLOP gate, and the state/weight intervention work is additional. No promotion based on a single pilot.

```sh
python -m research_gym.trm_skill_gym_v2.advanced --out experiments/trm_skill_gym_02/advanced --count 128 --nulls 8
```
