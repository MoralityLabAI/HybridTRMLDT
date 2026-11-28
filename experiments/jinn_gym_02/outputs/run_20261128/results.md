# JINN-GYM-02 results

Model `thinkingmachines/Inkling-2-Mini:peft:65536:free`; Lean (version 4.35.0, x86_64-w64-windows-gnu, commit eb6cafb7290e8bd5f2937522d8d804f58f02bafc, Release). Phase REPORT; stop None; 3.23 h; calls 1156 ({'decision_neutral': 840, 'mutation_patch': 128, 'selfmodel_elicit': 8, 'route_inky': 180}); cap hits 8, empty 3; paid spend $0.

## Routing on held-out (system level, normalized verified utility)

Neutral Inky 0.804; uniform over modules 0.720; domain_best 0.782; inky_route 0.762; oracle 0.874; rule 0.741; static_best 0.733; trm_k1 0.772; trm_k4 0.766

| contrast | level | diff | 95% CI | clusters |
|---|---|---|---|---|
| trm_k4 - neutral | root | -0.038 | [-0.076, -0.001] | 180 |
| trm_k4 - neutral | template | -0.038 | [-0.077, -0.002] | 24 |
| trm_k4 - static_best | root | 0.032 | [-0.005, 0.069] | 180 |
| trm_k4 - static_best | template | 0.032 | [-0.008, 0.074] | 24 |
| trm_k4 - domain_best | root | -0.016 | [-0.046, 0.012] | 180 |
| trm_k4 - domain_best | template | -0.016 | [-0.055, 0.014] | 24 |
| trm_k4 - rule | root | 0.025 | [-0.021, 0.072] | 180 |
| trm_k4 - rule | template | 0.025 | [-0.030, 0.082] | 24 |
| trm_k4 - inky_route | root | 0.003 | [-0.036, 0.045] | 180 |
| trm_k4 - inky_route | template | 0.003 | [-0.038, 0.041] | 24 |
| trm_k4 - trm_k1 | root | -0.006 | [-0.038, 0.024] | 180 |
| trm_k4 - trm_k1 | template | -0.006 | [-0.043, 0.030] | 24 |
| trm_k4 - uniform | root | 0.046 | [0.017, 0.074] | 180 |
| trm_k4 - uniform | template | 0.046 | [0.009, 0.092] | 24 |
| static_best - neutral | root | -0.070 | [-0.111, -0.030] | 180 |
| static_best - neutral | template | -0.070 | [-0.132, -0.013] | 24 |
| domain_best - neutral | root | -0.022 | [-0.061, 0.017] | 180 |
| domain_best - neutral | template | -0.022 | [-0.073, 0.025] | 24 |
| inky_route - neutral | root | -0.042 | [-0.080, -0.005] | 180 |
| inky_route - neutral | template | -0.042 | [-0.089, 0.002] | 24 |
| oracle - neutral | root | 0.070 | [0.047, 0.096] | 180 |
| oracle - neutral | template | 0.070 | [0.042, 0.103] | 24 |
| trm_k1 - neutral | root | -0.032 | [-0.073, 0.010] | 180 |
| trm_k1 - neutral | template | -0.032 | [-0.081, 0.013] | 24 |

## Persona level on held-out

| arm | fidelity | norm. utility | abstain F1 | Brier | unauth. |
|---|---|---|---|---|---|
| A | 0.210 | 0.804 | 0.867 | 0.208 | 0 |
| ARM_abstention | 0.441 | 0.736 | 0.845 | 0.291 | 0 |
| ARM_heuristic | 0.440 | 0.722 | 0.820 | 0.296 | 0 |
| ARM_inky | 0.424 | 0.737 | 0.835 | 0.289 | 0 |
| ARM_learned | 0.413 | 0.739 | 0.830 | 0.295 | 0 |
| ARM_random | 0.454 | 0.724 | 0.830 | 0.301 | 0 |
| FINAL | 0.445 | 0.739 | 0.859 | 0.277 | 0 |
| FINAL_C | 0.478 | 0.682 | 0.815 | 0.346 | 0 |
| ROOT | 0.421 | 0.725 | 0.814 | 0.296 | 0 |

| contrast | metric | level | diff | 95% CI |
|---|---|---|---|---|
| FINAL - A | normalized_utility | root | -0.065 | [-0.093, -0.037] |
| FINAL - A | normalized_utility | template | -0.065 | [-0.105, -0.028] |
| FINAL - A | fidelity | root | 0.236 | [0.205, 0.267] |
| FINAL - A | fidelity | template | 0.236 | [0.208, 0.263] |
| FINAL - ROOT | normalized_utility | root | 0.014 | [0.005, 0.023] |
| FINAL - ROOT | normalized_utility | template | 0.014 | [0.002, 0.025] |
| FINAL - ROOT | fidelity | root | 0.024 | [0.009, 0.041] |
| FINAL - ROOT | fidelity | template | 0.024 | [0.008, 0.042] |
| ROOT - A | normalized_utility | root | -0.079 | [-0.107, -0.050] |
| ROOT - A | normalized_utility | template | -0.079 | [-0.119, -0.041] |
| ROOT - A | fidelity | root | 0.211 | [0.182, 0.242] |
| ROOT - A | fidelity | template | 0.211 | [0.193, 0.231] |
| FINAL - FINAL_C | fidelity | root | -0.032 | [-0.053, -0.012] |
| FINAL - FINAL_C | fidelity | template | -0.032 | [-0.060, -0.007] |
| ARM_abstention - ARM_random | fidelity | root | -0.012 | [-0.028, 0.003] |
| ARM_abstention - ARM_random | fidelity | template | -0.012 | [-0.029, 0.003] |
| ARM_abstention - ARM_random | normalized_utility | root | 0.012 | [0.002, 0.022] |
| ARM_abstention - ARM_random | normalized_utility | template | 0.012 | [-0.000, 0.025] |
| ARM_heuristic - ARM_random | fidelity | root | -0.013 | [-0.030, 0.003] |
| ARM_heuristic - ARM_random | fidelity | template | -0.013 | [-0.027, 0.000] |
| ARM_heuristic - ARM_random | normalized_utility | root | -0.002 | [-0.012, 0.009] |
| ARM_heuristic - ARM_random | normalized_utility | template | -0.002 | [-0.012, 0.011] |
| ARM_inky - ARM_random | fidelity | root | -0.030 | [-0.048, -0.012] |
| ARM_inky - ARM_random | fidelity | template | -0.030 | [-0.046, -0.018] |
| ARM_inky - ARM_random | normalized_utility | root | 0.014 | [0.002, 0.026] |
| ARM_inky - ARM_random | normalized_utility | template | 0.014 | [-0.003, 0.033] |
| ARM_learned - ARM_random | fidelity | root | -0.040 | [-0.057, -0.024] |
| ARM_learned - ARM_random | fidelity | template | -0.040 | [-0.068, -0.019] |
| ARM_learned - ARM_random | normalized_utility | root | 0.016 | [0.006, 0.025] |
| ARM_learned - ARM_random | normalized_utility | template | 0.016 | [0.006, 0.028] |
| ARM_random - A | normalized_utility | root | -0.080 | [-0.107, -0.053] |
| ARM_random - A | normalized_utility | template | -0.080 | [-0.119, -0.041] |
| ARM_abstention - A | normalized_utility | root | -0.068 | [-0.096, -0.039] |
| ARM_abstention - A | normalized_utility | template | -0.068 | [-0.107, -0.031] |
| ARM_heuristic - A | normalized_utility | root | -0.082 | [-0.112, -0.053] |
| ARM_heuristic - A | normalized_utility | template | -0.082 | [-0.123, -0.043] |
| ARM_inky - A | normalized_utility | root | -0.066 | [-0.093, -0.040] |
| ARM_inky - A | normalized_utility | template | -0.066 | [-0.102, -0.033] |
| ARM_learned - A | normalized_utility | root | -0.065 | [-0.092, -0.037] |
| ARM_learned - A | normalized_utility | template | -0.065 | [-0.099, -0.031] |

## Mutation arms

Rounds 16 of 16.

| arm | candidates | accepted | mean dF accepted | mean d(gap) accepted | reasons |
|---|---|---|---|---|---|
| random | 128 | 18 | 0.048 | 0.018 | {'fidelity_gain_feasible': 9, 'no_gain': 46, 'violates_noninferiority': 20, 'pareto_utility_gain': 4, 'parent_infeasible_no_restoration': 44, 'feasibility_restoration': 5} |
| abstention | 128 | 17 | 0.050 | 0.032 | {'fidelity_gain_feasible': 7, 'no_gain': 47, 'violates_noninferiority': 17, 'parent_infeasible_no_restoration': 44, 'feasibility_restoration': 7, 'no_candidate': 3, 'pareto_utility_gain': 3} |
| heuristic | 128 | 22 | 0.043 | 0.015 | {'no_gain': 49, 'fidelity_gain_feasible': 11, 'violates_noninferiority': 10, 'pareto_utility_gain': 6, 'parent_infeasible_no_restoration': 47, 'feasibility_restoration': 5} |
| inky | 128 | 16 | 0.031 | 0.027 | {'invalid': 6, 'no_gain': 40, 'violates_noninferiority': 23, 'fidelity_gain_feasible': 6, 'parent_infeasible_no_restoration': 41, 'feasibility_restoration': 7, 'pareto_utility_gain': 3, 'no_candidate': 2} |
| learned | 128 | 23 | 0.023 | 0.019 | {'no_gain': 45, 'fidelity_gain_feasible': 8, 'violates_noninferiority': 13, 'pareto_utility_gain': 5, 'parent_infeasible_no_restoration': 47, 'feasibility_restoration': 10} |

Abstention-repair rules fired: {'tighten:threshold_up': 88, 'relax:no_ranked_action_off': 2, 'recovery:add_orient_diagnose': 11, 'relax:threshold_down': 6, 'recovery:add_diagnose_test': 6, 'justified_or_balanced:no_op': 3, 'recovery:log_req_off': 12}

## Recombination

{
 "feature": {
  "n": 12,
  "transfer": 0.006634764736924206,
  "interference": 0.03973711873213532,
  "complementarity": -0.019153586107721655,
  "share_positive_complementarity": 0.4166666666666667
 },
 "ranking": {
  "n": 12,
  "transfer": 0.005013950611957256,
  "interference": 0.08169560966903157,
  "complementarity": -0.006180684087977377,
  "share_positive_complementarity": 0.3333333333333333
 },
 "params": {
  "n": 12,
  "transfer": 0.08102792279702578,
  "interference": 0.051097711811997515,
  "complementarity": -0.012683469812691445,
  "share_positive_complementarity": 0.4166666666666667
 },
 "invalid": 0
}

## Self-model, compactification, boundary

See results.json keys selfmodel_eval, compact, cca_audit_freeze.
Unauthorized executions (held-out persona rows and routing): 0.
