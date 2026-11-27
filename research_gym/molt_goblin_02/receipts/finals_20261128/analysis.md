| arm | bounded_walk | cumsum | experiment_select | pointer_chase | proof_deps | repair | revision | shallow_copy | mean | cpu s | bytes/ex |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| f0_66c9e4 | 0.766 | 0.105 | 0.656 | 0.148 | 0.654 | 0.321 | 0.967 | 1.000 | 0.577 | 36.9 | 3296 |
| f1_fc183e | 0.809 | 0.108 | 0.662 | 0.136 | 0.653 | 0.321 | 0.953 | 1.000 | 0.580 | 34.6 | 3296 |
| f2_af49a1 | 0.829 | 0.117 | 0.659 | 0.142 | 0.652 | 0.315 | 0.954 | 1.000 | 0.583 | 33.2 | 3296 |
| one_pass | 0.253 | 0.002 | 0.459 | 0.127 | 0.421 | 0.232 | 0.825 | 1.000 | 0.415 | 3.1 | 3296 |
| prefix | 0.640 | 0.011 | 0.561 | 0.137 | 0.532 | 0.284 | 0.915 | 1.000 | 0.510 | 7.8 | 3296 |
| prefix_all | 0.645 | 0.001 | 0.570 | 0.134 | 0.516 | 0.269 | 0.945 | 1.000 | 0.510 | 8.4 | 3296 |
| prefix_all_param_matched | 0.595 | 0.040 | 0.597 | 0.182 | 0.525 | 0.314 | 0.981 | 1.000 | 0.529 | 12.1 | 7622 |
| prefix_all_visit_matched | 0.764 | 0.067 | 0.568 | 0.134 | 0.540 | 0.277 | 0.952 | 1.000 | 0.538 | 28.0 | 3296 |
| prev | 0.583 | 0.017 | 0.575 | 0.142 | 0.516 | 0.301 | 0.917 | 1.000 | 0.506 | 7.9 | 3296 |
| tied | 0.649 | 0.002 | 0.548 | 0.127 | 0.526 | 0.253 | 0.882 | 1.000 | 0.498 | 7.9 | 3296 |

Paired difference vs `prefix` (seed-level mean, t 95% CI; root bootstrap CI conditional on checkpoints):

| arm | family | mean diff | seed CI95 | root CI95 |
|---|---|---:|---|---|
| f0_66c9e4 | bounded_walk | +0.126 | [-0.044, +0.296] | [+0.097, +0.156] |
| f0_66c9e4 | cumsum | +0.094 | [-0.032, +0.219] | [+0.078, +0.110] |
| f0_66c9e4 | experiment_select | +0.095 | [+0.009, +0.181] | [+0.071, +0.119] |
| f0_66c9e4 | pointer_chase | +0.011 | [-0.011, +0.032] | [-0.006, +0.029] |
| f0_66c9e4 | proof_deps | +0.122 | [+0.032, +0.211] | [+0.100, +0.145] |
| f0_66c9e4 | repair | +0.037 | [+0.009, +0.064] | [+0.014, +0.059] |
| f0_66c9e4 | revision | +0.052 | [+0.025, +0.080] | [+0.037, +0.068] |
| f0_66c9e4 | shallow_copy | +0.000 | [+0.000, +0.000] | [+0.000, +0.000] |
| f1_fc183e | bounded_walk | +0.169 | [+0.066, +0.272] | [+0.140, +0.199] |
| f1_fc183e | cumsum | +0.097 | [-0.070, +0.264] | [+0.080, +0.114] |
| f1_fc183e | experiment_select | +0.101 | [+0.035, +0.166] | [+0.077, +0.123] |
| f1_fc183e | pointer_chase | -0.001 | [-0.020, +0.018] | [-0.019, +0.017] |
| f1_fc183e | proof_deps | +0.121 | [+0.016, +0.226] | [+0.100, +0.142] |
| f1_fc183e | repair | +0.037 | [+0.000, +0.073] | [+0.011, +0.062] |
| f1_fc183e | revision | +0.038 | [+0.007, +0.070] | [+0.021, +0.055] |
| f1_fc183e | shallow_copy | +0.000 | [+0.000, +0.000] | [+0.000, +0.000] |
| f2_af49a1 | bounded_walk | +0.189 | [+0.028, +0.351] | [+0.159, +0.220] |
| f2_af49a1 | cumsum | +0.106 | [-0.083, +0.296] | [+0.090, +0.122] |
| f2_af49a1 | experiment_select | +0.098 | [+0.039, +0.158] | [+0.074, +0.121] |
| f2_af49a1 | pointer_chase | +0.005 | [-0.006, +0.017] | [-0.012, +0.023] |
| f2_af49a1 | proof_deps | +0.120 | [+0.024, +0.215] | [+0.098, +0.144] |
| f2_af49a1 | repair | +0.030 | [-0.004, +0.065] | [+0.005, +0.054] |
| f2_af49a1 | revision | +0.039 | [+0.001, +0.077] | [+0.022, +0.056] |
| f2_af49a1 | shallow_copy | +0.000 | [+0.000, +0.000] | [+0.000, +0.000] |
| one_pass | bounded_walk | -0.387 | [-0.487, -0.287] | [-0.417, -0.356] |
| one_pass | cumsum | -0.009 | [-0.030, +0.012] | [-0.016, -0.004] |
| one_pass | experiment_select | -0.102 | [-0.187, -0.016] | [-0.126, -0.077] |
| one_pass | pointer_chase | -0.009 | [-0.041, +0.023] | [-0.027, +0.008] |
| one_pass | proof_deps | -0.111 | [-0.140, -0.082] | [-0.135, -0.087] |
| one_pass | repair | -0.052 | [-0.118, +0.014] | [-0.077, -0.029] |
| one_pass | revision | -0.090 | [-0.113, -0.067] | [-0.114, -0.066] |
| one_pass | shallow_copy | +0.000 | [+0.000, +0.000] | [+0.000, +0.000] |
| prefix_all_param_matched | bounded_walk | -0.045 | [-0.354, +0.263] | [-0.080, -0.011] |
| prefix_all_param_matched | cumsum | +0.029 | [-0.014, +0.071] | [+0.017, +0.041] |
| prefix_all_param_matched | experiment_select | +0.036 | [-0.043, +0.115] | [+0.011, +0.061] |
| prefix_all_param_matched | pointer_chase | +0.045 | [+0.006, +0.085] | [+0.024, +0.066] |
| prefix_all_param_matched | proof_deps | -0.007 | [-0.057, +0.042] | [-0.030, +0.015] |
| prefix_all_param_matched | repair | +0.030 | [-0.009, +0.068] | [+0.003, +0.053] |
| prefix_all_param_matched | revision | +0.066 | [+0.035, +0.098] | [+0.051, +0.082] |
| prefix_all_param_matched | shallow_copy | +0.000 | [+0.000, +0.000] | [+0.000, +0.000] |
| prefix_all_visit_matched | bounded_walk | +0.124 | [-0.077, +0.325] | [+0.093, +0.155] |
| prefix_all_visit_matched | cumsum | +0.056 | [-0.042, +0.154] | [+0.044, +0.070] |
| prefix_all_visit_matched | experiment_select | +0.007 | [-0.021, +0.035] | [-0.012, +0.025] |
| prefix_all_visit_matched | pointer_chase | -0.002 | [-0.018, +0.013] | [-0.012, +0.008] |
| prefix_all_visit_matched | proof_deps | +0.008 | [-0.030, +0.046] | [-0.007, +0.023] |
| prefix_all_visit_matched | repair | -0.007 | [-0.030, +0.016] | [-0.020, +0.005] |
| prefix_all_visit_matched | revision | +0.037 | [-0.008, +0.082] | [+0.020, +0.053] |
| prefix_all_visit_matched | shallow_copy | +0.000 | [+0.000, +0.000] | [+0.000, +0.000] |
| prefix_all | bounded_walk | +0.005 | [-0.131, +0.142] | [-0.027, +0.037] |
| prefix_all | cumsum | -0.010 | [-0.033, +0.013] | [-0.016, -0.005] |
| prefix_all | experiment_select | +0.009 | [-0.044, +0.062] | [-0.011, +0.027] |
| prefix_all | pointer_chase | -0.003 | [-0.016, +0.010] | [-0.016, +0.011] |
| prefix_all | proof_deps | -0.016 | [-0.058, +0.027] | [-0.035, +0.004] |
| prefix_all | repair | -0.016 | [-0.053, +0.021] | [-0.034, +0.002] |
| prefix_all | revision | +0.030 | [-0.028, +0.089] | [+0.011, +0.048] |
| prefix_all | shallow_copy | +0.000 | [+0.000, +0.000] | [+0.000, +0.000] |
| prev | bounded_walk | -0.057 | [-0.201, +0.087] | [-0.088, -0.027] |
| prev | cumsum | +0.006 | [-0.029, +0.042] | [-0.002, +0.015] |
| prev | experiment_select | +0.014 | [-0.033, +0.061] | [-0.007, +0.034] |
| prev | pointer_chase | +0.005 | [-0.009, +0.020] | [-0.010, +0.020] |
| prev | proof_deps | -0.016 | [-0.055, +0.023] | [-0.037, +0.005] |
| prev | repair | +0.016 | [-0.026, +0.059] | [-0.006, +0.037] |
| prev | revision | +0.002 | [-0.080, +0.085] | [-0.016, +0.020] |
| prev | shallow_copy | +0.000 | [+0.000, +0.000] | [+0.000, +0.000] |
| tied | bounded_walk | +0.009 | [-0.208, +0.227] | [-0.022, +0.042] |
| tied | cumsum | -0.009 | [-0.028, +0.009] | [-0.015, -0.005] |
| tied | experiment_select | -0.013 | [-0.071, +0.045] | [-0.035, +0.007] |
| tied | pointer_chase | -0.010 | [-0.031, +0.010] | [-0.027, +0.005] |
| tied | proof_deps | -0.006 | [-0.054, +0.041] | [-0.027, +0.016] |
| tied | repair | -0.031 | [-0.055, -0.007] | [-0.055, -0.009] |
| tied | revision | -0.033 | [-0.110, +0.045] | [-0.054, -0.012] |
| tied | shallow_copy | +0.000 | [+0.000, +0.000] | [+0.000, +0.000] |

Pareto front over (mean exact ↑, train CPU s ↓, retained bytes ↓): `f2_af49a1`, `one_pass`, `prefix`, `prefix_all_param_matched`, `prefix_all_visit_matched`

Paired difference vs `prefix_all_visit_matched` (seed-level mean [t 95% CI], 5 seeds):

| arm − `prefix_all_visit_matched` | bounded_walk | cumsum | experiment_select | pointer_chase | proof_deps | repair | revision | shallow_copy |
|---|---|---|---|---|---|---|---|---|
| f0_66c9e4 | +0.002 [-0.274, +0.277] | +0.037 [-0.156, +0.231] | +0.088 [+0.001, +0.176] | +0.013 [-0.013, +0.039] | +0.114 [+0.020, +0.208] | +0.044 [+0.018, +0.070] | +0.016 [-0.009, +0.041] | +0.000 [+0.000, +0.000] |
| f1_fc183e | +0.045 [-0.121, +0.210] | +0.041 [-0.201, +0.283] | +0.094 [+0.029, +0.158] | +0.002 [-0.021, +0.024] | +0.113 [-0.003, +0.229] | +0.044 [+0.012, +0.075] | +0.002 [-0.017, +0.020] | +0.000 [+0.000, +0.000] |
| f2_af49a1 | +0.065 [-0.199, +0.329] | +0.050 [-0.210, +0.310] | +0.091 [+0.029, +0.154] | +0.008 [-0.010, +0.026] | +0.112 [+0.007, +0.216] | +0.037 [+0.009, +0.066] | +0.002 [-0.013, +0.018] | +0.000 [+0.000, +0.000] |

Paired difference vs `prefix_all_param_matched` (seed-level mean [t 95% CI], 5 seeds):

| arm − `prefix_all_param_matched` | bounded_walk | cumsum | experiment_select | pointer_chase | proof_deps | repair | revision | shallow_copy |
|---|---|---|---|---|---|---|---|---|
| f0_66c9e4 | +0.171 [-0.050, +0.392] | +0.065 [-0.056, +0.185] | +0.059 [-0.015, +0.134] | -0.034 [-0.062, -0.007] | +0.129 [+0.022, +0.236] | +0.007 [-0.016, +0.030] | -0.014 [-0.032, +0.004] | +0.000 [+0.000, +0.000] |
| f1_fc183e | +0.214 [-0.055, +0.483] | +0.068 [-0.081, +0.217] | +0.065 [+0.006, +0.124] | -0.046 [-0.079, -0.013] | +0.128 [+0.012, +0.244] | +0.007 [-0.014, +0.028] | -0.028 [-0.049, -0.008] | +0.000 [+0.000, +0.000] |
| f2_af49a1 | +0.234 [-0.087, +0.556] | +0.077 [-0.095, +0.250] | +0.062 [-0.002, +0.127] | -0.040 [-0.078, -0.002] | +0.127 [+0.011, +0.242] | +0.001 [-0.020, +0.022] | -0.027 [-0.048, -0.006] | +0.000 [+0.000, +0.000] |

Paired difference vs `prefix_all` (seed-level mean [t 95% CI], 5 seeds):

| arm − `prefix_all` | bounded_walk | cumsum | experiment_select | pointer_chase | proof_deps | repair | revision | shallow_copy |
|---|---|---|---|---|---|---|---|---|
| f0_66c9e4 | +0.120 [-0.082, +0.323] | +0.104 [-0.022, +0.230] | +0.087 [+0.035, +0.139] | +0.014 [-0.009, +0.037] | +0.138 [+0.017, +0.258] | +0.052 [+0.003, +0.101] | +0.022 [-0.017, +0.060] | +0.000 [+0.000, +0.000] |
| f1_fc183e | +0.163 [-0.014, +0.341] | +0.107 [-0.054, +0.268] | +0.092 [+0.059, +0.126] | +0.002 [-0.022, +0.026] | +0.137 [+0.013, +0.261] | +0.052 [+0.023, +0.081] | +0.008 [-0.020, +0.036] | +0.000 [+0.000, +0.000] |
| f2_af49a1 | +0.184 [+0.035, +0.332] | +0.116 [-0.068, +0.301] | +0.090 [+0.054, +0.126] | +0.009 [-0.009, +0.027] | +0.135 [+0.013, +0.258] | +0.046 [+0.018, +0.075] | +0.009 [-0.013, +0.031] | +0.000 [+0.000, +0.000] |

Memory interventions on the in-distribution readout (exact; clean in first column):

| arm | family | clean | reset_history | shuffle_history | stale_swap | noise | halt-eval exact (visits) | hard-route exact |
|---|---|---:|---:|---:|---:|---:|---|---:|
| f0_66c9e4 | pointer_chase | 0.148 | 0.129 | 0.148 | 0.001 | 0.009 | 0.122 (4.53) | 0.046 |
| f0_66c9e4 | revision | 0.967 | 0.727 | 0.967 | 0.034 | 0.165 | 0.897 (7.36) | 0.353 |
| f0_66c9e4 | repair | 0.321 | 0.256 | 0.321 | 0.173 | 0.270 | 0.266 (4.03) | 0.309 |
| f0_66c9e4 | proof_deps | 0.654 | 0.473 | 0.654 | 0.169 | 0.508 | 0.479 (5.73) | 0.319 |
| f0_66c9e4 | bounded_walk | 0.766 | 0.171 | 0.766 | 0.064 | 0.102 | 0.592 (8.38) | 0.119 |
| f1_fc183e | pointer_chase | 0.136 | 0.128 | 0.136 | 0.000 | 0.009 | — | 0.040 |
| f1_fc183e | revision | 0.953 | 0.720 | 0.953 | 0.034 | 0.177 | — | 0.323 |
| f1_fc183e | repair | 0.321 | 0.255 | 0.321 | 0.158 | 0.282 | — | 0.292 |
| f1_fc183e | proof_deps | 0.653 | 0.479 | 0.653 | 0.145 | 0.521 | — | 0.323 |
| f1_fc183e | bounded_walk | 0.809 | 0.156 | 0.809 | 0.059 | 0.109 | — | 0.080 |
| f2_af49a1 | pointer_chase | 0.142 | 0.127 | 0.142 | 0.001 | 0.009 | — | 0.045 |
| f2_af49a1 | revision | 0.954 | 0.777 | 0.954 | 0.030 | 0.173 | — | 0.334 |
| f2_af49a1 | repair | 0.315 | 0.255 | 0.315 | 0.170 | 0.267 | — | 0.300 |
| f2_af49a1 | proof_deps | 0.652 | 0.465 | 0.652 | 0.148 | 0.515 | — | 0.336 |
| f2_af49a1 | bounded_walk | 0.829 | 0.169 | 0.829 | 0.047 | 0.101 | — | 0.080 |
| prefix | pointer_chase | 0.137 | 0.026 | 0.137 | 0.000 | 0.004 | — | nan |
| prefix | revision | 0.915 | 0.232 | 0.915 | 0.037 | 0.185 | — | nan |
| prefix | repair | 0.284 | 0.212 | 0.284 | 0.154 | 0.245 | — | nan |
| prefix | proof_deps | 0.532 | 0.320 | 0.532 | 0.084 | 0.405 | — | nan |
| prefix | bounded_walk | 0.640 | 0.082 | 0.640 | 0.001 | 0.030 | — | nan |
| prefix_all | pointer_chase | 0.134 | 0.114 | 0.134 | 0.001 | 0.002 | — | nan |
| prefix_all | revision | 0.945 | 0.777 | 0.945 | 0.037 | 0.116 | — | nan |
| prefix_all | repair | 0.269 | 0.235 | 0.269 | 0.160 | 0.234 | — | nan |
| prefix_all | proof_deps | 0.516 | 0.412 | 0.516 | 0.109 | 0.351 | — | nan |
| prefix_all | bounded_walk | 0.645 | 0.220 | 0.645 | 0.002 | 0.021 | — | nan |
| prefix_all_visit_matched | pointer_chase | 0.134 | 0.082 | 0.134 | 0.000 | 0.007 | — | nan |
| prefix_all_visit_matched | revision | 0.952 | 0.611 | 0.952 | 0.036 | 0.145 | — | nan |
| prefix_all_visit_matched | repair | 0.277 | 0.221 | 0.277 | 0.170 | 0.230 | — | nan |
| prefix_all_visit_matched | proof_deps | 0.540 | 0.376 | 0.540 | 0.095 | 0.418 | — | nan |
| prefix_all_visit_matched | bounded_walk | 0.764 | 0.121 | 0.764 | 0.002 | 0.021 | — | nan |
| prefix_all_param_matched | pointer_chase | 0.182 | 0.152 | 0.182 | 0.000 | 0.030 | — | nan |
| prefix_all_param_matched | revision | 0.981 | 0.821 | 0.981 | 0.039 | 0.345 | — | nan |
| prefix_all_param_matched | repair | 0.314 | 0.276 | 0.314 | 0.170 | 0.303 | — | nan |
| prefix_all_param_matched | proof_deps | 0.525 | 0.427 | 0.525 | 0.092 | 0.402 | — | nan |
| prefix_all_param_matched | bounded_walk | 0.595 | 0.223 | 0.595 | 0.001 | 0.084 | — | nan |

Recurrence use, routing and extrapolation (means over seeds):

| arm | family | exact by loop | fixes / breaks / redundant (per 256) | success per visit | routing entropy | routing–loop MI | halt Brier / ECE | OOD exact | OOD at 2× loops |
|---|---|---|---|---:|---:|---:|---|---|---|
| f0_66c9e4 | pointer_chase | 0.12, 0.15, 0.15, 0.15 | 18 / 12 / 96 | 0.009 | 1.061 | 0.006 | 0.071 / 0.213 | ood_depth=0.016, ood_shuffled=0.109 | ood_depth=0.019, ood_shuffled=0.107 |
| f0_66c9e4 | revision | 0.78, 0.93, 0.95, 0.97 | 63 / 15 / 666 | 0.060 | 1.223 | 0.008 | 0.218 / 0.296 | ood_long=0.747, ood_reordered=0.370 | ood_long=0.684, ood_reordered=0.373 |
| f0_66c9e4 | repair | 0.27, 0.33, 0.31, 0.32 | 44 / 29 / 204 | 0.020 | 0.744 | 0.012 | 0.007 / 0.052 | ood_more_errors=0.063 | ood_more_errors=0.061 |
| f0_66c9e4 | proof_deps | 0.44, 0.57, 0.65, 0.65 | 92 / 36 / 387 | 0.041 | 1.045 | 0.013 | 0.132 / 0.175 | ood_deep=0.243 | ood_deep=0.255 |
| f0_66c9e4 | bounded_walk | 0.24, 0.60, 0.73, 0.77 | 167 / 33 / 368 | 0.048 | 0.888 | 0.009 | 0.226 / 0.258 | ood_long=0.174 | ood_long=0.129 |
| f1_fc183e | pointer_chase | 0.12, 0.15, 0.13, 0.14 | 17 / 14 / 89 | 0.008 | 1.123 | 0.006 | — | ood_depth=0.023, ood_shuffled=0.117 | ood_depth=0.021, ood_shuffled=0.113 |
| f1_fc183e | revision | 0.78, 0.93, 0.94, 0.95 | 62 / 17 / 661 | 0.060 | 1.233 | 0.006 | — | ood_long=0.704, ood_reordered=0.373 | ood_long=0.673, ood_reordered=0.363 |
| f1_fc183e | repair | 0.27, 0.32, 0.32, 0.32 | 40 / 26 / 208 | 0.020 | 0.726 | 0.004 | — | ood_more_errors=0.052 | ood_more_errors=0.054 |
| f1_fc183e | proof_deps | 0.43, 0.57, 0.64, 0.65 | 86 / 29 / 391 | 0.041 | 1.056 | 0.010 | — | ood_deep=0.227 | ood_deep=0.239 |
| f1_fc183e | bounded_walk | 0.24, 0.61, 0.80, 0.81 | 183 / 36 / 385 | 0.051 | 0.895 | 0.009 | — | ood_long=0.202 | ood_long=0.141 |
| f2_af49a1 | pointer_chase | 0.12, 0.14, 0.14, 0.14 | 16 / 11 / 93 | 0.009 | 1.173 | 0.005 | — | ood_depth=0.025, ood_shuffled=0.125 | ood_depth=0.026, ood_shuffled=0.118 |
| f2_af49a1 | revision | 0.77, 0.93, 0.96, 0.95 | 63 / 16 / 665 | 0.060 | 1.233 | 0.006 | — | ood_long=0.720, ood_reordered=0.370 | ood_long=0.691, ood_reordered=0.363 |
| f2_af49a1 | repair | 0.25, 0.31, 0.32, 0.31 | 39 / 23 / 202 | 0.020 | 0.719 | 0.004 | — | ood_more_errors=0.049 | ood_more_errors=0.054 |
| f2_af49a1 | proof_deps | 0.42, 0.58, 0.64, 0.65 | 86 / 28 / 391 | 0.041 | 1.037 | 0.011 | — | ood_deep=0.231 | ood_deep=0.240 |
| f2_af49a1 | bounded_walk | 0.23, 0.63, 0.80, 0.83 | 183 / 30 / 395 | 0.052 | 0.898 | 0.008 | — | ood_long=0.170 | ood_long=0.142 |
| prefix | pointer_chase | 0.03, 0.13, 0.13, 0.14 | 36 / 8 / 66 | 0.034 | — | — | — | ood_depth=0.015, ood_shuffled=0.089 | ood_depth=0.015, ood_shuffled=0.091 |
| prefix | revision | 0.23, 0.31, 0.32, 0.91 | 249 / 74 / 144 | 0.229 | — | — | — | ood_long=0.484, ood_reordered=0.287 | ood_long=0.276, ood_reordered=0.260 |
| prefix | repair | 0.21, 0.27, 0.29, 0.28 | 58 / 40 / 159 | 0.071 | — | — | — | ood_more_errors=0.055 | ood_more_errors=0.053 |
| prefix | proof_deps | 0.32, 0.35, 0.47, 0.53 | 105 / 50 / 241 | 0.133 | — | — | — | ood_deep=0.173 | ood_deep=0.159 |
| prefix | bounded_walk | 0.08, 0.09, 0.50, 0.64 | 175 / 32 / 139 | 0.160 | — | — | — | ood_long=0.080 | ood_long=0.080 |
| prefix_all | pointer_chase | 0.11, 0.13, 0.13, 0.13 | 17 / 12 / 85 | 0.033 | — | — | — | ood_depth=0.013, ood_shuffled=0.103 | ood_depth=0.013, ood_shuffled=0.105 |
| prefix_all | revision | 0.78, 0.91, 0.94, 0.95 | 63 / 20 / 653 | 0.236 | — | — | — | ood_long=0.616, ood_reordered=0.379 | ood_long=0.602, ood_reordered=0.382 |
| prefix_all | repair | 0.24, 0.27, 0.27, 0.27 | 30 / 21 / 176 | 0.067 | — | — | — | ood_more_errors=0.042 | ood_more_errors=0.043 |
| prefix_all | proof_deps | 0.41, 0.49, 0.51, 0.52 | 52 / 25 / 337 | 0.129 | — | — | — | ood_deep=0.155 | ood_deep=0.153 |
| prefix_all | bounded_walk | 0.22, 0.56, 0.65, 0.65 | 160 / 51 / 314 | 0.161 | — | — | — | ood_long=0.080 | ood_long=0.097 |
| prefix_all_visit_matched | pointer_chase | 0.08, 0.14, 0.14, 0.14, 0.14, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13 | 27 / 13 / 491 | 0.008 | — | — | — | ood_depth=0.016, ood_shuffled=0.099 | ood_depth=0.016, ood_shuffled=0.099 |
| prefix_all_visit_matched | revision | 0.61, 0.84, 0.91, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95 | 123 / 36 / 3491 | 0.059 | — | — | — | ood_long=0.660, ood_reordered=0.377 | ood_long=0.660, ood_reordered=0.377 |
| prefix_all_visit_matched | repair | 0.22, 0.26, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28 | 46 / 32 / 1016 | 0.017 | — | — | — | ood_more_errors=0.055 | ood_more_errors=0.055 |
| prefix_all_visit_matched | proof_deps | 0.38, 0.48, 0.51, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54, 0.54 | 77 / 35 / 1974 | 0.034 | — | — | — | ood_deep=0.188 | ood_deep=0.188 |
| prefix_all_visit_matched | bounded_walk | 0.12, 0.41, 0.67, 0.72, 0.76, 0.76, 0.76, 0.76, 0.76, 0.76, 0.76, 0.76, 0.76, 0.76, 0.76, 0.76 | 208 / 44 / 2597 | 0.048 | — | — | — | ood_long=0.147 | ood_long=0.147 |
| prefix_all_param_matched | pointer_chase | 0.15, 0.19, 0.19, 0.18 | 21 / 14 / 120 | 0.046 | — | — | — | ood_depth=0.024, ood_shuffled=0.133 | ood_depth=0.024, ood_shuffled=0.135 |
| prefix_all_param_matched | revision | 0.82, 0.95, 0.97, 0.98 | 54 / 13 / 688 | 0.245 | — | — | — | ood_long=0.826, ood_reordered=0.419 | ood_long=0.828, ood_reordered=0.413 |
| prefix_all_param_matched | repair | 0.28, 0.32, 0.32, 0.31 | 38 / 28 / 206 | 0.079 | — | — | — | ood_more_errors=0.062 | ood_more_errors=0.064 |
| prefix_all_param_matched | proof_deps | 0.43, 0.49, 0.52, 0.53 | 60 / 35 / 334 | 0.131 | — | — | — | ood_deep=0.172 | ood_deep=0.168 |
| prefix_all_param_matched | bounded_walk | 0.22, 0.53, 0.60, 0.59 | 147 / 52 / 294 | 0.149 | — | — | — | ood_long=0.109 | ood_long=0.123 |

BPD-style loop-update subspace interventions (seed 11 checkpoint; fit and evaluation on disjoint hidden roots):

| arm | family | clean | rank | explained var | top-k | random-k | complement |
|---|---|---:|---:|---:|---:|---:|---:|
| f0_66c9e4 | pointer_chase | 0.152 | 2 | 0.371 | 0.020 | 0.000 | 0.070 |
| f0_66c9e4 | pointer_chase | 0.152 | 4 | 0.624 | 0.109 | 0.000 | 0.008 |
| f0_66c9e4 | pointer_chase | 0.152 | 8 | 0.894 | 0.121 | 0.059 | 0.000 |
| f0_66c9e4 | revision | 0.965 | 2 | 0.409 | 0.207 | 0.082 | 0.707 |
| f0_66c9e4 | revision | 0.965 | 4 | 0.709 | 0.566 | 0.172 | 0.148 |
| f0_66c9e4 | revision | 0.965 | 8 | 0.915 | 0.883 | 0.410 | 0.047 |
| f0_66c9e4 | repair | 0.332 | 2 | 0.408 | 0.195 | 0.180 | 0.250 |
| f0_66c9e4 | repair | 0.332 | 4 | 0.682 | 0.273 | 0.152 | 0.180 |
| f0_66c9e4 | repair | 0.332 | 8 | 0.952 | 0.305 | 0.156 | 0.195 |
| f0_66c9e4 | proof_deps | 0.750 | 2 | 0.718 | 0.207 | 0.297 | 0.012 |
| f0_66c9e4 | proof_deps | 0.750 | 4 | 0.870 | 0.660 | 0.012 | 0.012 |
| f0_66c9e4 | proof_deps | 0.750 | 8 | 0.947 | 0.703 | 0.281 | 0.012 |
| f1_fc183e | pointer_chase | 0.152 | 2 | 0.367 | 0.008 | 0.000 | 0.059 |
| f1_fc183e | pointer_chase | 0.152 | 4 | 0.620 | 0.094 | 0.004 | 0.016 |
| f1_fc183e | pointer_chase | 0.152 | 8 | 0.889 | 0.141 | 0.070 | 0.000 |
| f1_fc183e | revision | 0.949 | 2 | 0.415 | 0.203 | 0.105 | 0.754 |
| f1_fc183e | revision | 0.949 | 4 | 0.716 | 0.555 | 0.188 | 0.164 |
| f1_fc183e | revision | 0.949 | 8 | 0.921 | 0.809 | 0.434 | 0.047 |
| f1_fc183e | repair | 0.340 | 2 | 0.440 | 0.176 | 0.188 | 0.219 |
| f1_fc183e | repair | 0.340 | 4 | 0.716 | 0.230 | 0.152 | 0.141 |
| f1_fc183e | repair | 0.340 | 8 | 0.949 | 0.289 | 0.164 | 0.188 |
| f1_fc183e | proof_deps | 0.781 | 2 | 0.741 | 0.641 | 0.312 | 0.023 |
| f1_fc183e | proof_deps | 0.781 | 4 | 0.873 | 0.715 | 0.039 | 0.020 |
| f1_fc183e | proof_deps | 0.781 | 8 | 0.945 | 0.766 | 0.273 | 0.020 |
| f2_af49a1 | pointer_chase | 0.164 | 2 | 0.368 | 0.008 | 0.000 | 0.066 |
| f2_af49a1 | pointer_chase | 0.164 | 4 | 0.624 | 0.086 | 0.004 | 0.016 |
| f2_af49a1 | pointer_chase | 0.164 | 8 | 0.896 | 0.145 | 0.051 | 0.000 |
| f2_af49a1 | revision | 0.949 | 2 | 0.416 | 0.215 | 0.094 | 0.750 |
| f2_af49a1 | revision | 0.949 | 4 | 0.716 | 0.602 | 0.199 | 0.188 |
| f2_af49a1 | revision | 0.949 | 8 | 0.924 | 0.840 | 0.445 | 0.043 |
| f2_af49a1 | repair | 0.324 | 2 | 0.445 | 0.203 | 0.188 | 0.203 |
| f2_af49a1 | repair | 0.324 | 4 | 0.728 | 0.223 | 0.148 | 0.168 |
| f2_af49a1 | repair | 0.324 | 8 | 0.955 | 0.281 | 0.160 | 0.184 |
| f2_af49a1 | proof_deps | 0.762 | 2 | 0.733 | 0.672 | 0.312 | 0.203 |
| f2_af49a1 | proof_deps | 0.762 | 4 | 0.873 | 0.742 | 0.066 | 0.016 |
| f2_af49a1 | proof_deps | 0.762 | 8 | 0.942 | 0.758 | 0.281 | 0.012 |
| prefix | pointer_chase | 0.145 | 2 | 0.375 | 0.000 | 0.000 | 0.055 |
| prefix | pointer_chase | 0.145 | 4 | 0.599 | 0.016 | 0.000 | 0.000 |
| prefix | pointer_chase | 0.145 | 8 | 0.825 | 0.059 | 0.008 | 0.000 |
| prefix | revision | 0.930 | 2 | 0.593 | 0.043 | 0.051 | 0.445 |
| prefix | revision | 0.930 | 4 | 0.751 | 0.270 | 0.055 | 0.164 |
| prefix | revision | 0.930 | 8 | 0.885 | 0.699 | 0.074 | 0.047 |
| prefix | repair | 0.301 | 2 | 0.386 | 0.148 | 0.164 | 0.172 |
| prefix | repair | 0.301 | 4 | 0.664 | 0.215 | 0.133 | 0.184 |
| prefix | repair | 0.301 | 8 | 0.928 | 0.305 | 0.180 | 0.109 |
| prefix | proof_deps | 0.512 | 2 | 0.724 | 0.324 | 0.066 | 0.047 |
| prefix | proof_deps | 0.512 | 4 | 0.866 | 0.422 | 0.172 | 0.047 |
| prefix | proof_deps | 0.512 | 8 | 0.950 | 0.512 | 0.184 | 0.098 |
