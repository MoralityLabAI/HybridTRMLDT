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
