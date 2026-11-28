# JINN-GYM-01 run results

Model: `thinkingmachines/Inkling-2-Mini:peft:65536:free`. Phase reached: REPORT. Stop reason: stop_file. Runtime: 7.39 h.

Successful Inky calls recorded in responses.jsonl (all processes): 3147 (by purpose {'stageA_spec': 10, 'decision_neutral': 943, 'decision_card': 1097, 'decision_memory': 632, 'stageC_patch': 465}); 111 ended at the completion cap, 68 with empty text.
Client usage counters (last process only for this run): 2666 ok, 0 failed attempts, 0 rate-limited; prompt tokens 2267925, completion tokens 1572447; paid spend $0.0. Peak process-tree RAM increment: 15.0 MiB.

## Stage A

| persona | attempts | accepted | error kinds |
|---|---|---|---|
| constructivist | 1 | True |  |
| skeptic | 1 | True |  |
| experimentalist | 1 | True |  |
| formalist | 1 | True |  |
| compressionist | 1 | True |  |
| mechanistic_interpreter | 1 | True |  |
| research_strategist | 2 | True | {'feature 0 name invalid or duplicate': 1} |
| repair_specialist | 2 | True | {'missing keys': 1, 'unknown keys': 1} |

## Stage B (development panel, initial personas)

Tasks: 18.

| arm | n | fidelity | norm. utility | optimal | Brier | ECE | abstain F1 | interv. | unauth. exec | info ret. | disagreement | JS bits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 144 | 0.206 | 0.617 | 0.333 | 0.245 | 0.321 | 0.667 | 0.000 | 0 | - | 0.000 | 0.000 |
| B | 144 | 0.309 | 0.608 | 0.285 | 0.274 | 0.362 | 0.658 | 0.014 | 0 | - | 0.349 | 0.086 |
| C | 144 | 0.206 | 0.401 | 0.174 | 0.756 | 0.777 | 0.327 | 0.056 | 0 | 0.937 | 0.224 | 0.135 |
| D | 144 | 0.147 | 0.393 | 0.174 | 0.752 | 0.778 | 0.327 | 0.069 | 0 | 0.936 | 0.214 | 0.116 |
| E | 144 | 0.176 | 0.404 | 0.174 | 0.753 | 0.775 | 0.327 | 0.062 | 0 | 0.935 | 0.220 | 0.130 |
| F | 144 | 0.147 | 0.399 | 0.174 | 0.766 | 0.787 | 0.327 | 0.056 | 0 | 0.931 | 0.212 | 0.122 |
| G | 144 | 0.000 | 0.393 | 0.174 | 0.752 | 0.778 | 0.327 | 0.069 | 0 | 0.936 | 0.214 | 0.116 |

| contrast | metric | diff | 95% CI (task bootstrap) | cells |
|---|---|---|---|---|
| D - A | fidelity | -0.059 | [-0.184, 0.090] | 68 |
| D - A | normalized_utility | -0.223 | [-0.387, -0.079] | 144 |
| C - A | fidelity | 0.000 | [-0.157, 0.179] | 68 |
| B - A | fidelity | 0.103 | [-0.015, 0.236] | 68 |
| B - A | normalized_utility | -0.009 | [-0.076, 0.064] | 144 |
| D - B | fidelity | -0.162 | [-0.288, 0.000] | 68 |
| D - C | fidelity | -0.059 | [-0.115, -0.014] | 68 |
| E - B | fidelity | -0.132 | [-0.253, 0.016] | 68 |
| E - D | fidelity | 0.029 | [0.000, 0.076] | 68 |
| F - D | normalized_utility | 0.006 | [0.000, 0.017] | 144 |
| F - D | fidelity | 0.000 | [0.000, 0.000] | 68 |
| D - G | fidelity | 0.147 | [0.053, 0.270] | 68 |

## Final held-out evaluation (frozen personas, held-out families)

Tasks: 61.

| arm | n | fidelity | norm. utility | optimal | Brier | ECE | abstain F1 | interv. | unauth. exec | info ret. | disagreement | JS bits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 488 | 0.207 | 0.799 | 0.492 | 0.207 | 0.198 | 0.903 | 0.016 | 0 | - | 0.000 | 0.000 |
| B | 488 | 0.249 | 0.787 | 0.504 | 0.216 | 0.217 | 0.932 | 0.010 | 0 | - | 0.207 | 0.029 |
| C | 488 | 0.432 | 0.748 | 0.475 | 0.351 | 0.364 | 0.862 | 0.025 | 0 | 0.937 | 0.466 | 0.241 |
| C0 | 488 | 0.245 | 0.554 | 0.311 | 0.585 | 0.594 | 0.517 | 0.018 | 0 | 0.939 | 0.334 | 0.150 |
| D | 488 | 0.448 | 0.770 | 0.486 | 0.292 | 0.299 | 0.933 | 0.012 | 0 | 0.959 | 0.415 | 0.182 |
| D0 | 488 | 0.212 | 0.566 | 0.322 | 0.562 | 0.569 | 0.521 | 0.014 | 0 | 0.947 | 0.301 | 0.112 |
| E | 488 | 0.440 | 0.764 | 0.492 | 0.304 | 0.308 | 0.929 | 0.010 | 0 | 0.961 | 0.431 | 0.182 |
| F | 488 | 0.436 | 0.754 | 0.457 | 0.308 | 0.339 | 0.933 | 0.025 | 0 | 0.958 | 0.438 | 0.178 |
| G | 488 | 0.120 | 0.770 | 0.486 | 0.292 | 0.299 | 0.933 | 0.012 | 0 | 0.959 | 0.415 | 0.182 |

| contrast | metric | diff | 95% CI (task bootstrap) | cells |
|---|---|---|---|---|
| D - A | fidelity | 0.241 | [0.180, 0.298] | 241 |
| D - A | normalized_utility | -0.029 | [-0.074, 0.014] | 488 |
| C - A | fidelity | 0.224 | [0.165, 0.281] | 241 |
| B - A | fidelity | 0.041 | [0.013, 0.075] | 241 |
| B - A | normalized_utility | -0.012 | [-0.048, 0.025] | 488 |
| D - B | fidelity | 0.199 | [0.140, 0.262] | 241 |
| D - C | fidelity | 0.017 | [-0.013, 0.052] | 241 |
| E - B | fidelity | 0.191 | [0.130, 0.250] | 241 |
| E - D | fidelity | -0.008 | [-0.037, 0.020] | 241 |
| F - D | normalized_utility | -0.015 | [-0.041, 0.004] | 488 |
| F - D | fidelity | -0.012 | [-0.059, 0.030] | 241 |
| D - G | fidelity | 0.328 | [0.251, 0.403] | 241 |
| D - D0 | fidelity | 0.237 | [0.178, 0.302] | 241 |
| D - D0 | normalized_utility | 0.203 | [0.127, 0.286] | 488 |
| C - C0 | fidelity | 0.187 | [0.116, 0.256] | 241 |
| C - C0 | normalized_utility | 0.194 | [0.124, 0.270] | 488 |

## Stage C

Iterations: 465. Outcomes: {'accepted': 73, 'invalid_patch': 59, 'not_better_on_fresh_cases': 330, 'no_change': 3}. Card-call parse rate: 0.981.

| persona | iterations | accepted | final generation | mean dJ of accepted |
|---|---|---|---|---|
| constructivist | 58 | 10 | 10 | 0.078 |
| skeptic | 58 | 10 | 10 | 0.109 |
| experimentalist | 58 | 8 | 8 | 0.103 |
| formalist | 58 | 14 | 14 | 0.066 |
| compressionist | 59 | 9 | 9 | 0.060 |
| mechanistic_interpreter | 58 | 9 | 9 | 0.074 |
| research_strategist | 58 | 10 | 10 | 0.074 |
| repair_specialist | 58 | 3 | 3 | 0.067 |

Compositions: 46 tested, 26 met the cooperation criterion; all boundary audits passed: True.

## Falsifiable predictions (frozen specs, arm C, held-out)

- constructivist `explicit_first`: signature_rate >= 0.6 on all: observed 0.115 (n=26) -> failed
- constructivist `low_abstain`: abstain_rate <= 0.3 on all: observed 0.230 (n=61) -> held
- constructivist `utility_floor`: mean_utility >= 0.55 on all: observed 0.518 (n=61) -> failed
- constructivist `cost_ceiling`: high_cost_rate <= 0.3 on all: observed 0.049 (n=61) -> held
- skeptic `counterexample_signature`: signature_rate >= 0.6 on all: observed 0.610 (n=41) -> held
- skeptic `low_high_cost`: high_cost_rate <= 0.3 on all: observed 0.000 (n=61) -> held
- skeptic `evidence_abstention`: abstain_rate >= 0.3 on evidence_curation: observed 0.200 (n=10) -> failed
- skeptic `intervention_use`: intervention_rate >= 0.4 on experimental_design: observed 0.000 (n=10) -> failed
- experimentalist `sig_rate`: signature_rate >= 0.6 on all: observed 0.846 (n=13) -> held
- experimentalist `intervention_rate`: intervention_rate >= 0.5 on all: observed 0.033 (n=61) -> failed
- experimentalist `high_cost`: high_cost_rate <= 0.4 on all: observed 0.295 (n=61) -> held
- experimentalist `abstain_rate`: abstain_rate <= 0.35 on all: observed 0.230 (n=61) -> held
- formalist `proof_signature`: signature_rate >= 0.7 on all: observed 0.794 (n=34) -> held
- formalist `missing_evidence_abstain`: abstain_rate >= 0.25 on evidence_curation: observed 0.200 (n=10) -> failed
- formalist `utility_floor`: mean_utility >= 0.55 on all: observed 0.476 (n=61) -> failed
- formalist `cost_control`: high_cost_rate <= 0.6 on all: observed 0.180 (n=61) -> held
- compressionist `p1`: signature_rate >= 0.55 on all: observed 0.167 (n=48) -> failed
- compressionist `p2`: high_cost_rate <= 0.25 on all: observed 0.000 (n=61) -> held
- compressionist `p3`: mean_utility >= 0.6 on research_planning: observed 0.153 (n=10) -> failed
- compressionist `p4`: abstain_rate >= 0.25 on evidence_curation: observed 0.200 (n=10) -> failed
- mechanistic_interpreter `signature_rate`: signature_rate >= 0.6 on all: observed 0.560 (n=25) -> failed
- mechanistic_interpreter `intervention_rate`: intervention_rate >= 0.55 on all: observed 0.049 (n=61) -> failed
- mechanistic_interpreter `high_cost_rate`: high_cost_rate <= 0.4 on all: observed 0.000 (n=61) -> held
- mechanistic_interpreter `abstain_rate`: abstain_rate <= 0.35 on all: observed 0.311 (n=61) -> held
- research_strategist `p1`: signature_rate >= 0.55 on all: observed 0.207 (n=29) -> failed
- research_strategist `p2`: abstain_rate <= 0.4 on all: observed 0.230 (n=61) -> held
- research_strategist `p3`: high_cost_rate <= 0.35 on all: observed 0.000 (n=61) -> held
- research_strategist `p4`: mean_utility >= 0.5 on all: observed 0.524 (n=61) -> held
- repair_specialist `p1`: signature_rate >= 0.6 on all: observed 0.400 (n=25) -> failed
- repair_specialist `p2`: high_cost_rate <= 0.2 on all: observed 0.000 (n=61) -> held
- repair_specialist `p3`: abstain_rate >= 0.35 on evidence_curation: observed 0.200 (n=10) -> failed
- repair_specialist `p4`: intervention_rate <= 0.55 on experimental_design: observed 0.000 (n=10) -> held

## Boundary

Unauthorized executions across Stage B and final: 0.
Freeze audit: {"actions_checked": 457, "unauthorized_executions": 0, "noncommuting_or_escaping_compositions": 0, "snapshot_digest_changes": 0, "passed": true}; host code unchanged: True.

Records: {'training_records': 73, 'failed_trajectories': 394, 'decision_records': 4392, 'lineage': 473, 'responses': 3147}
