# CCA-10: Action-count bound pilot

Status: exploratory successor to CCA-09. Source state domain, outcome tapes and oracle results were already available before the derived abstraction; label comparisons retrospective.

Source: CCA-05 moral score, coupled-storyworld transitions, six unchanged CCA-05 pass-four logit tapes.

Intervention: replace every unevaluated action's global upper value of 1 with the CCA-10 action-count upper from its modeled successor and horizon-1; keep the action proposals, evaluator, verifier, order, utility, and budgets fixed. Secret-ending bounds remain 1.0. Bind all upper certificates to the exact environment/state/horizon/action and verify them independently.

First run the exact dominance test over all 1,050 environment states at horizons k=0..5 (6,300 comparisons) and a deterministic 80-state smoke test. Stop if any bound is smaller than the exact known optimal utility. Run the full 1,750-state six-tape comparison only after those gates pass.

Primary endpoint: change in exact epsilon=0 optimality certifications by action-evaluation budget K=1..4 on the 585 viable moral states per checkpoint. Secondary: change in evaluator calls before mathematical certification and extra bound-computation cost.

Expected: executed choices and true utility at fixed budgets remain identical to CCA-09; only certified upper gaps, optimality claims, and early termination may differ. A null result is valid and should be preserved. No new holdout or independent replication claim.
