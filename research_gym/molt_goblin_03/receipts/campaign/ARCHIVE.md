Campaign: 20 proposal iterations (19 trained, 1 recorded failures), 3 adversarial task requests (3 verified and used), 160 development training runs, 5.62 h, resumed 2x. Inky Mini: 30 requests, 30 ok, 257,305 prompt / 76,554 completion tokens, 7 truncated; paid spend $0. Peak process-tree RSS 359 MiB.

### Hypothesis archive (development roots, rung 0 = seed 11)

| it | mechanism | base | candidate change | quality | score | visits | outcome |
|---:|---|---|---|---:|---:|---:|---|
| 0 | aux_loss | C_mix | `{"diversity":0.5}` | +0.123 | +0.063 | 16 | cf65cd1 |
| 1 | distillation | A_single | `{"distill":"hidden","distill_coef":1.0,"distill_temp":2.0}` | +0.024 | +0.024 | 4 | cdbbe6f |
| 2 | sharing | B_hrm | `{"tie_modules":true}` | +0.016 | -0.044 | 16 | c09459b |
| 3 | module_objective | B_hrm | `{"functional":true,"func_coef":0.5}` | +0.118 | +0.058 | 16 | c8a8055 |
| 4 | capacity | B_hrm | `{"n_types":2}` | +0.135 | +0.105 | 8 | c34a15d |
| 5 | capacity | c34a15d | `{"n_types":3}` | +0.166 | +0.118 | 12 | cd17a81 |
| 6 | composition | cd17a81 | `{"routing":"sequential","n_types":3}` | +0.151 | +0.103 | 12 | cec17e1 |
| 7 | loop_forcing | cd17a81 | `{"loss":"ramp"}` | +0.155 | +0.107 | 12 | c08d849 |
| 8 | routing | cd17a81 | `{"route_entropy":0.5}` | +0.014 | -0.033 | 12 | ce97ec3 |
| 9 | routing | cd17a81 | `{"route_entropy":-0.5}` | +0.099 | +0.051 | 12 | c6819f1 |
| 10 | distillation | cd17a81 | `{"distill":"logits_all","distill_coef":1.0}` | +0.128 | +0.081 | 12 | c9c3704 |
| 11 | sharing | cd17a81 | `{"tie_modules":true}` | +0.031 | -0.016 | 12 | ce2e348 |
| 12 | module_objective | cd17a81 | `{"functional":true,"func_coef":0.5}` | — | — | — | invalid proposal: ValueError: functional objectives need 4 modules |
| 13 | loop_forcing | c08d849 | `{"update":"damped"}` | +0.174 | +0.127 | 12 | c318560 |
| 14 | capacity | c34a15d | `{"update":"damped"}` | +0.125 | +0.095 | 8 | c81d1cd |
| 15 | capacity | c318560 | `{"width":64}` | +0.250 | +0.202 | 12 | c139d58 |
| 16 | distillation | c139d58 | `{"distill":"logits_all","distill_coef":1.0}` | +0.207 | +0.160 | 12 | c15418a |
| 17 | distillation | c139d58 | `{"distill":"hidden","distill_coef":1.0}` | — | — | — | c4631c1 |
| 18 | module_objective | B_hrm | `{"loops":2,"functional":true,"func_coef":0.5}` | +0.085 | +0.055 | 8 | c826501 |
| 19 | capacity | c139d58 | `{"n_types":2}` | +0.191 | +0.161 | 8 | ca59a0b |

Hypotheses as stated by Inky:

* it0: A diversity penalty decorrelates the four mix modules and improves exact/ood.
* it1: hidden-state distillation transfers B_hrm's internal module-averaged computation into a single-block 4-visit student better than final-logit distillation.
* it2: Tying the four HRM modules to one shared block (only type embeddings differ) preserves most of B_hrm's quality, showing specialization comes from routing/type embeddings rather than separate weights.
* it3: Functional module objectives (deduce/repair/verify) will push the four modules toward complementary skills and improve repair-like and proof-like families.
* it4: A 2-module soft HRM (8 visits) retains most of B_hrm quality, implying the gain is mostly module capacity; if it collapses, routing/order/4-module specialization is the real mechanism.
* it5: n_types=3 (12 visits) recovers most of the cex_repair gap while keeping most of the cheap-architecture savings of n_types=2.
* it6: At fixed n_types=3, sequential module order beats soft HRM mixture on repair-like tasks, implying ordered composition is the hidden advantage.
* it7: Ramping loop-forcing loss (down-weighting early loops) concentrates supervision on late refinement, lifting repair-like families without hurting proof/portfolio.
* it8: Positive route_entropy sharpening will concentrate HRM routing enough to reduce cross-module interference and improve revision_retract/cex_repair while leaving proof_resolve roughly flat.
* it9: Negative route_entropy (spreading soft routing) at n_types=3 recovers cex_repair/revision_retract without losing proof_resolve.
* it10: Distilling frozen B_hrm logits into the 3-module HRM student transfers cex_repair/portfolio knowledge, closing most of the 12-vs-16-visit gap.
* it11: Tying the 3 modules (shared block, type embeddings only) will lose quality versus untied 3-module HRM, especially on repair/proof-like families.
* it12: Functional module objectives (deduce/repair/verify) at n_types=3 HRM re-specialize modules and improve repair-like families without extra visits.
* it13: Damped slow/fast GRU updates reduce manager oscillation and improve repair-like families without hurting proof/portfolio.
* it14: Damped updates at n_types=2 (8 visits) will recover most of the n_types=2 direct gap on repair-like families, making the cheaper architecture competitive.
* it15: Doubling width (32->64) at n_types=3 damped+ramp lifts proof_resolve/repair-like exact without extra visits.
* it16: Distilling frozen B_hrm logits into the wide 3-module HRM student improves repair/proof transfer without extra visits.
* it17: Hidden (intermediate-state) distillation from frozen B_hrm preserves proof/repair exact without the logits_all penalty, improving revision_retract/cex_repair over the no-distill control at equal 12 visits.
* it18: At fixed 4 modules and 8 visits, functional objectives (deduce/repair/verify) will specialize modules enough to improve repair/proof-like exact versus the identical 8-visit HRM without them.
* it19: At width 64 with damped+ramp 3-module HRM, dropping to n_types=2 (8 visits) retains most repair-like exact while losing proof/portfolio modestly.

### Architecture archive (best rung)

| id | origin | canonical diff | rung | quality | score | seeds |
|---|---|---|---:|---:|---:|---|
| c139d58 | candidate | `{"width":64,"routing":"hrm","n_types":3,"update":"damped","loss":"ramp"}` | 2 | +0.187 | +0.139 | [11, 23, 37] |
| c08d849 | candidate | `{"routing":"hrm","n_types":3,"loss":"ramp"}` | 2 | +0.152 | +0.104 | [11, 23, 37] |
| c34a15d | candidate | `{"routing":"hrm","n_types":2}` | 2 | +0.113 | +0.083 | [11, 23, 37] |
| cf65cd1 | candidate | `{"routing":"mix","n_types":4,"diversity":0.5}` | 2 | +0.098 | +0.038 | [11, 23, 37] |
| ca59a0b | candidate | `{"width":64,"routing":"hrm","n_types":2,"update":"damped","loss":"ramp"}` | 1 | +0.174 | +0.144 | [11, 23] |
| cec17e1 | candidate | `{"routing":"sequential","n_types":3}` | 1 | +0.161 | +0.114 | [11, 23] |
| c318560 | candidate | `{"routing":"hrm","n_types":3,"update":"damped","loss":"ramp"}` | 1 | +0.153 | +0.105 | [11, 23] |
| cd17a81 | candidate | `{"routing":"hrm","n_types":3}` | 1 | +0.152 | +0.104 | [11, 23] |
| c15418a | candidate | `{"width":64,"routing":"hrm","n_types":3,"update":"damped","loss":"ramp","distill":"logits_` | 0 | +0.207 | +0.160 | [11] |
| B_hrm | registered_arm | `{"routing":"hrm","n_types":4}` | 0 | +0.182 | +0.122 | [11] |
| C_mix | registered_arm | `{"routing":"mix","n_types":4}` | 0 | +0.181 | +0.121 | [11] |
| c1dc771 | control | `{"loops":2,"routing":"hrm","n_types":4}` | 0 | +0.127 | +0.097 | [11] |
| c81d1cd | candidate | `{"routing":"hrm","n_types":2,"update":"damped"}` | 0 | +0.125 | +0.095 | [11] |
| c9c3704 | candidate | `{"routing":"hrm","n_types":3,"distill":"logits_all","distill_coef":1.0}` | 0 | +0.128 | +0.081 | [11] |
| c8a8055 | candidate | `{"routing":"hrm","n_types":4,"functional":true,"func_coef":0.5}` | 0 | +0.118 | +0.058 | [11] |
| c826501 | candidate | `{"loops":2,"routing":"hrm","n_types":4,"functional":true,"func_coef":0.5}` | 0 | +0.085 | +0.055 | [11] |
| c6819f1 | candidate | `{"routing":"hrm","n_types":3,"route_entropy":-0.5}` | 0 | +0.099 | +0.051 | [11] |
| H_sparse_st | registered_arm | `{"routing":"sparse_st","n_types":4,"load_balance":0.01}` | 0 | +0.032 | +0.032 | [11] |
| cdbbe6f | candidate | `{"distill":"hidden","distill_coef":1.0}` | 0 | +0.024 | +0.024 | [11] |
| A_single | registered_arm | `{}` | 0 | +0.000 | +0.000 | [11] |
| ce2e348 | candidate | `{"routing":"hrm","n_types":3,"tie_modules":true}` | 0 | +0.031 | -0.016 | [11] |
| K_distill | registered_arm | `{"distill":"logits_all","distill_coef":1.0}` | 0 | -0.021 | -0.021 | [11] |
| ce97ec3 | candidate | `{"routing":"hrm","n_types":3,"route_entropy":0.5}` | 0 | +0.014 | -0.033 | [11] |
| c09459b | candidate | `{"routing":"hrm","n_types":4,"tie_modules":true}` | 0 | +0.016 | -0.044 | [11] |
| I_anneal | registered_arm | `{"routing":"anneal","n_types":4}` | 0 | -0.216 | -0.216 | [11] |
| c4631c1 | candidate | `{"width":64,"routing":"hrm","n_types":3,"update":"damped","loss":"ramp","distill":"hidden"` | 0 | — | — | None |

### Adversarial task variants

| it | spec | accepted |
|---:|---|---|
| 5 | `{"family":"proof_resolve","length":24,"depth":4,"distract":0,"variant":"clean"}` | a0 |
| 11 | `{"family":"proof_resolve","length":28,"depth":8,"distract":4,"variant":"clean"}` | a1 |
| 17 | `{"family":"cex_repair","length":4,"depth":3,"distract":0,"variant":"clean"}` | a2 |
