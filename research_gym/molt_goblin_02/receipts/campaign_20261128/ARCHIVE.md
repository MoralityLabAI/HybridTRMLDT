Campaign `campaign_20261128`: 63 proposal iterations, 55 trained, 8 recorded failures (parse, duplicate, invalid or training errors), 11 adversarial task requests (10 verified and used, 1 rejected), 400 CPU training runs, 4.33 h wall clock, proposer `inky`, resumed 2×.
Inky Mini: 93 requests (93 ok, 0 failed after retries, 21 truncated by the completion cap), 900,955 prompt and 244,495 completion tokens, paid spend $0.

### Hypothesis archive

| it | mech | candidate change (from base) | base | score | Δ vs prefix by family (pc/rev/rep/pd) | Δ vs control | predicted→observed matches | outcome |
|---:|---|---|---|---:|---|---|---|---|
| 0 | 2 | `{"memory":"topk","topk":8}` | prefix_all | -0.086 | +0.00/-0.05/-0.18/-0.14 | -0.04/-0.09/-0.05/-0.05 | 0/4 | cc0d732 |
| 1 | 5 | `{"conf_bias":true}` | prefix_all | -0.007 | +0.03/-0.01/-0.03/-0.08 | -0.02/-0.04/+0.02/-0.04 | 2/4 | c86671e |
| 2 | 12 | `{"update_rank":4}` | prefix_all | -0.091 | -0.04/-0.10/-0.12/-0.07 | -0.08/-0.12/-0.07/-0.04 | 0/4 | c9eab24 |
| 3 | — | `—` | — | — | — | — | — | no JSON object in reply |
| 4 | 3 | `{"memory":"retention"}` | c4d2769 | -0.006 | +0.04/+0.04/-0.12/-0.10 | -0.00/+0.01/+0.00/-0.00 | 1/4 | c953be5 |
| 5 | 11 | `{"update":"damped"}` | prefix_all | -0.020 | +0.03/+0.01/-0.03/-0.10 | -0.02/-0.01/+0.02/-0.06 | 1/4 | c910d5d |
| 6 | 9 | `{"ldt_mask":true}` | prefix_all | +0.041 | -0.03/-0.02/+0.02/-0.03 | -0.08/-0.04/+0.07/+0.01 | 2/4 | cdf7388 |
| 7 | 7 | `{"routing":"cycle","n_types":4}` | cdf7388 | +0.061 | +0.00/+0.03/-0.01/-0.00 | +0.04/+0.05/-0.03/+0.02 | 1/4 | c1840a9 |
| 8 | 10 | `{"halt":"utility"}` | prefix_all | +0.006 | +0.03/-0.02/+0.00/-0.05 | -0.02/-0.05/+0.05/-0.01 | 2/4 | cef516c |
| 9 | 8 | `{"routing":"hrm","n_types":4}` | prefix_all | +0.131 | +0.04/+0.08/-0.04/+0.19 | -0.00/+0.05/+0.01/+0.23 | 2/4 | caf49a1 |
| 10 | 8 | `{"route_hard_eval":true}` | caf49a1 | -0.297 | -0.09/-0.63/-0.07/-0.25 | -0.13/-0.71/-0.03/-0.44 | 0/4 | c33f968 |
| 11 | 2 | `{"memory":"topk","topk":4}` | caf49a1 | -0.025 | +0.02/-0.00/-0.13/-0.05 | -0.03/-0.08/-0.09/-0.24 | 1/4 | cf28ceb |
| 12 | 12 | `{"update_rank":4}` | caf49a1 | -0.007 | +0.00/-0.02/-0.08/+0.02 | -0.04/-0.10/-0.04/-0.18 | 2/4 | c6f249b |
| 13 | 11 | `{"update":"damped"}` | caf49a1 | +0.103 | +0.03/+0.06/-0.09/+0.23 | -0.01/-0.02/-0.05/+0.04 | 2/4 | c60f599 |
| 14 | 6 | `{"error_feedback":true}` | tied | +0.005 | +0.01/-0.02/+0.00/-0.03 | +0.02/+0.00/+0.04/-0.05 | 2/4 | c593ab9 |
| 15 | 1 | `{"memory":"full"}` | c60f599 | -0.001 | +0.05/+0.03/-0.12/-0.04 | +0.02/-0.03/-0.03/-0.27 | 1/4 | cf9e318 |
| 16 | 3 | `{"memory":"retention"}` | c60f599 | -0.000 | +0.03/+0.01/-0.10/-0.02 | -0.00/-0.05/-0.01/-0.25 | 1/4 | c1b73ac |
| 17 | 11 | `{"update":"rlt"}` | caf49a1 | +0.109 | +0.01/+0.08/+0.04/+0.10 | -0.02/+0.02/+0.13/-0.13 | 3/4 | c68a02e |
| 18 | 4 | `{"memory":"lowrank","rank":4}` | c68a02e | -0.091 | -0.09/-0.23/-0.14/+0.10 | -0.10/-0.30/-0.18/+0.00 | 1/4 | c7bdda7 |
| 19 | 2 | `{"memory":"topk","topk":4}` | c68a02e | +0.018 | +0.03/-0.05/-0.02/+0.07 | +0.02/-0.12/-0.06/-0.04 | 2/4 | c5e3d44 |
| 20 | 12 | `{"update_rank":4}` | c68a02e | -0.088 | +0.02/-0.11/-0.12/-0.08 | +0.01/-0.18/-0.16/-0.18 | 1/4 | c46ea5b |
| 21 | 5 | `{"conf_bias":true}` | c68a02e | +0.100 | +0.00/+0.09/+0.04/+0.07 | -0.01/+0.01/+0.00/-0.03 | 3/4 | cd49cc3 |
| 22 | 7 | `{"routing":"cycle","n_types":4}` | c68a02e | +0.035 | +0.03/+0.04/-0.00/-0.04 | +0.02/-0.04/-0.04/-0.14 | 3/4 | cf58735 |
| 23 | 1 | `{"memory":"full"}` | c68a02e | +0.035 | +0.04/+0.03/-0.04/+0.02 | +0.03/-0.05/-0.07/-0.08 | 2/4 | c30018f |
| 24 | 3 | `{"memory":"retention"}` | c68a02e | +0.053 | +0.05/+0.08/-0.01/-0.04 | +0.04/+0.00/-0.05/-0.14 | 2/4 | cb1b64c |
| 25 | 8 | `{"slow_period":1}` | cd49cc3 | +0.101 | +0.01/+0.09/+0.04/+0.07 | +0.01/-0.00/+0.01/-0.00 | 3/4 | ce5985b |
| 26 | 10 | `{"halt":"utility"}` | c68a02e | +0.100 | +0.02/+0.08/+0.04/+0.06 | +0.01/+0.00/+0.00/-0.04 | 1/4 | c0c0912 |
| 27 | 2 | `{"memory":"topk","topk":4}` | c68a02e | — | — | — | — | duplicate of already-tested c5e3d44 |
| 28 | 4 | `{"memory":"lowrank","rank":8}` | ce5985b | -0.022 | -0.03/-0.05/-0.12/+0.06 | -0.04/-0.14/-0.16/-0.01 | 0/4 | c5792bd |
| 29 | 12 | `{"update_rank":4}` | ce5985b | -0.086 | +0.04/-0.11/-0.11/-0.08 | +0.03/-0.20/-0.15/-0.15 | 0/4 | c6384c5 |
| 30 | 6 | `{"error_feedback":true}` | tied | — | — | — | — | duplicate of already-tested c593ab9 |
| 31 | 8 | `{"route_hard_eval":true}` | ce5985b | -0.242 | -0.10/-0.67/+0.03/-0.11 | -0.11/-0.76/-0.02/-0.18 | 3/4 | c1cf32d |
| 32 | 7 | `{"routing":"cycle","n_types":4}` | cd49cc3 | +0.024 | +0.03/+0.04/+0.00/-0.07 | -0.00/-0.01/-0.01/+0.02 | 1/4 | c55be3b |
| 33 | 3 | `{"memory":"retention"}` | caf49a1 | +0.035 | +0.04/+0.06/-0.09/-0.01 | +0.00/-0.02/-0.04/-0.20 | 1/4 | cd1c2e2 |
| 34 | 5 | `{"conf_bias":true}` | caf49a1 | +0.128 | +0.05/+0.06/-0.05/+0.21 | +0.00/-0.02/-0.01/+0.02 | 1/4 | cfc183e |
| 35 | 9 | `{"ldt_mask":true}` | caf49a1 | +0.108 | -0.02/+0.06/+0.00/+0.10 | -0.06/-0.02/+0.04/-0.09 | 1/4 | c7fdd76 |
| 36 | 2 | `{"memory":"topk","topk":4}` | cfc183e | -0.013 | +0.00/+0.04/-0.11/-0.05 | -0.05/-0.02/-0.06/-0.25 | 0/4 | cecfcbd |
| 37 | 4 | `{"memory":"lowrank","rank":8}` | cfc183e | -0.045 | +0.01/-0.18/-0.14/+0.10 | -0.04/-0.23/-0.09/-0.11 | 2/4 | c4e477a |
| 38 | 12 | `{"update_rank":8}` | cfc183e | -0.033 | +0.04/-0.06/-0.04/-0.06 | -0.01/-0.12/+0.02/-0.27 | 0/4 | c113bc2 |
| 39 | 1 | `{"memory":"full"}` | cfc183e | +0.032 | +0.05/+0.06/-0.09/+0.00 | +0.01/+0.00/-0.04/-0.20 | 1/4 | ca709b8 |
| 40 | 10 | `{"halt":"utility"}` | cfc183e | +0.126 | +0.06/+0.07/-0.02/+0.18 | +0.02/+0.01/+0.03/-0.03 | 1/4 | c66c9e4 |
| 41 | 3 | `{"memory":"retention"}` | cfc183e | +0.032 | +0.06/+0.05/-0.09/+0.00 | +0.02/-0.01/-0.04/-0.21 | 1/4 | c8c8f2f |
| 42 | 6 | `{"memory":"full","error_feedback":true}` | cfc183e | +0.026 | +0.05/+0.04/-0.09/+0.01 | +0.00/-0.02/+0.00/+0.00 | 2/4 | ca58f26 |
| 43 | 9 | `{"ldt_mask":true}` | cfc183e | +0.095 | +0.00/+0.01/+0.01/+0.09 | -0.05/-0.05/+0.06/-0.11 | 2/4 | ccc200d |
| 44 | 2 | `{"memory":"topk","topk":8}` | cfc183e | +0.017 | +0.04/+0.04/-0.07/-0.03 | -0.01/-0.02/+0.02/-0.03 | 2/4 | c7a4a13 |
| 45 | 5 | `{"conf_bias":true}` | prefix | -0.013 | +0.02/-0.05/+0.01/-0.03 | +0.02/-0.05/+0.01/-0.03 | 2/4 | cfa78e0 |
| 46 | 12 | `{"update_rank":4}` | prefix | -0.095 | -0.03/-0.01/-0.13/-0.14 | -0.03/-0.01/-0.13/-0.14 | 0/4 | cdba9c3 |
| 47 | 4 | `{"memory":"lowrank","rank":8}` | prefix | -0.073 | -0.02/-0.10/-0.11/-0.09 | -0.02/-0.10/-0.11/-0.09 | 1/4 | c26a07a |
| 48 | 9 | `{"ldt_mask":true}` | c66c9e4 | +0.111 | +0.00/+0.08/+0.00/+0.05 | -0.06/+0.02/+0.03/-0.12 | 1/4 | cfc85df |
| 49 | 3 | `{"memory":"retention"}` | c66c9e4 | +0.042 | +0.04/+0.04/-0.02/-0.01 | -0.02/-0.02/+0.00/-0.18 | 2/4 | c3ff0c0 |
| 50 | 1 | `{"memory":"full"}` | c66c9e4 | +0.029 | +0.03/+0.06/-0.09/+0.00 | -0.04/-0.01/-0.06/-0.18 | 2/4 | c38f576 |
| 51 | 7 | `{"routing":"cycle"}` | c66c9e4 | +0.043 | +0.02/+0.07/+0.00/-0.00 | -0.04/+0.00/+0.02/-0.18 | 1/4 | c8fdeba |
| 52 | 2 | `{"memory":"topk","topk":4}` | c66c9e4 | -0.026 | +0.01/+0.02/-0.09/-0.05 | -0.02/-0.04/-0.01/-0.05 | 1/4 | c627854 |
| 53 | 11 | `{"update":"damped"}` | c66c9e4 | +0.077 | +0.04/+0.05/-0.04/+0.11 | -0.02/-0.01/-0.02/-0.06 | 2/4 | cca3ae4 |
| 54 | 12 | `{"update_rank":4}` | c66c9e4 | -0.022 | +0.02/-0.03/-0.06/+0.00 | -0.04/-0.10/-0.04/-0.18 | 2/4 | cdc1f2e |
| 55 | 4 | `{"memory":"lowrank","rank":8}` | c66c9e4 | -0.049 | +0.00/-0.16/-0.14/+0.07 | -0.06/-0.23/-0.12/-0.10 | 2/4 | cca4844 |
| 56 | 8 | `{"route_hard_eval":true}` | c66c9e4 | -0.271 | -0.09/-0.64/-0.01/-0.18 | -0.16/-0.71/+0.01/-0.36 | 1/4 | c717e02 |
| 57 | 3 | `{"memory":"retention"}` | c66c9e4 | — | — | — | — | duplicate of already-tested c3ff0c0 |
| 58 | 6 | `{"error_feedback":true}` | tied | — | — | — | — | duplicate of already-tested c593ab9 |
| 59 | 9 | `{"ldt_mask":true}` | caf49a1 | — | — | — | — | duplicate of already-tested c7fdd76 |
| 60 | 1 | `{"memory":"full"}` | c66c9e4 | — | — | — | — | duplicate of already-tested c38f576 |
| 61 | 7 | `{"routing":"cycle"}` | c66c9e4 | — | — | — | — | duplicate of already-tested c8fdeba |
| 62 | 7 | `{"routing":"mix"}` | c66c9e4 | +0.079 | +0.04/+0.07/-0.03/+0.10 | -0.02/+0.01/-0.01/-0.07 | 1/4 | c0d997b |

### Architecture archive (best rung reached)

| id | origin | diff from BASE | rung | score at rung | mean exact | seeds |
|---|---|---|---:|---:|---:|---|
| c66c9e4 | candidate it40 | `{"conf_bias":true,"routing":"hrm","n_types":4,"halt":"utility","loss":"all"}` | 2 | +0.108 | +0.539 | [11, 23, 37] |
| cfc183e | candidate it34 | `{"conf_bias":true,"routing":"hrm","n_types":4,"loss":"all"}` | 2 | +0.098 | +0.529 | [11, 23, 37] |
| caf49a1 | candidate it9 | `{"routing":"hrm","n_types":4,"loss":"all"}` | 2 | +0.093 | +0.528 | [11, 23, 37] |
| ccc200d | candidate it43 | `{"conf_bias":true,"routing":"hrm","n_types":4,"loss":"all","ldt_mask":true}` | 2 | +0.091 | +0.505 | [11, 23, 37] |
| c68a02e | candidate it17 | `{"routing":"hrm","n_types":4,"update":"rlt","loss":"all"}` | 2 | +0.083 | +0.521 | [11, 23, 37] |
| cfc85df | candidate it48 | `{"conf_bias":true,"routing":"hrm","n_types":4,"halt":"utility","loss":"all","ldt_mask":tru` | 2 | +0.083 | +0.501 | [11, 23, 37] |
| cd49cc3 | candidate it21 | `{"conf_bias":true,"routing":"hrm","n_types":4,"update":"rlt","loss":"all"}` | 2 | +0.078 | +0.517 | [11, 23, 37] |
| ce5985b | candidate it25 | `{"conf_bias":true,"routing":"hrm","n_types":4,"slow_period":1,"update":"rlt","loss":"all"}` | 2 | +0.076 | +0.515 | [11, 23, 37] |
| c0c0912 | candidate it26 | `{"routing":"hrm","n_types":4,"update":"rlt","halt":"utility","loss":"all"}` | 2 | +0.065 | +0.507 | [11, 23, 37] |
| c60f599 | candidate it13 | `{"routing":"hrm","n_types":4,"update":"damped","loss":"all"}` | 2 | +0.059 | +0.508 | [11, 23, 37] |
| cca3ae4 | candidate it53 | `{"conf_bias":true,"routing":"hrm","n_types":4,"update":"damped","halt":"utility","loss":"a` | 2 | +0.051 | +0.507 | [11, 23, 37] |
| cf58735 | candidate it22 | `{"routing":"cycle","n_types":4,"update":"rlt","loss":"all"}` | 2 | +0.031 | +0.486 | [11, 23, 37] |
| c86671e | candidate it1 | `{"conf_bias":true,"loss":"all"}` | 2 | -0.005 | +0.459 | [11, 23, 37] |
| cdf7388 | candidate it6 | `{"loss":"all","ldt_mask":true}` | 2 | -0.015 | +0.427 | [11, 23, 37] |
| cb1b64c | candidate it24 | `{"memory":"retention","routing":"hrm","n_types":4,"update":"rlt","loss":"all"}` | 1 | +0.027 | +0.467 | [11, 23] |
| ca709b8 | candidate it39 | `{"memory":"full","conf_bias":true,"routing":"hrm","n_types":4,"loss":"all"}` | 1 | +0.015 | +0.462 | [11, 23] |
| c30018f | candidate it23 | `{"memory":"full","routing":"hrm","n_types":4,"update":"rlt","loss":"all"}` | 1 | +0.007 | +0.462 | [11, 23] |
| c1840a9 | candidate it7 | `{"routing":"cycle","n_types":4,"loss":"all","ldt_mask":true}` | 1 | +0.001 | +0.435 | [11, 23] |
| cef516c | candidate it8 | `{"halt":"utility","loss":"all"}` | 1 | +0.001 | +0.451 | [11, 23] |
| c593ab9 | candidate it14 | `{"memory":"tied","error_feedback":true}` | 1 | -0.002 | +0.451 | [11, 23] |
| c5e3d44 | candidate it19 | `{"memory":"topk","topk":4,"routing":"hrm","n_types":4,"update":"rlt","loss":"all"}` | 1 | -0.031 | +0.435 | [11, 23] |
| c717e02 | candidate it56 | `{"conf_bias":true,"routing":"hrm","n_types":4,"route_hard_eval":true,"halt":"utility","los` | 1 | -0.206 | +0.289 | [11, 23] |
| c7fdd76 | candidate it35 | `{"routing":"hrm","n_types":4,"loss":"all","ldt_mask":true}` | 0 | +0.108 | +0.518 | [11] |
| c0d997b | candidate it62 | `{"conf_bias":true,"routing":"mix","n_types":4,"halt":"utility","loss":"all"}` | 0 | +0.079 | +0.528 | [11] |
| c8fdeba | candidate it51 | `{"conf_bias":true,"routing":"cycle","n_types":4,"halt":"utility","loss":"all"}` | 0 | +0.043 | +0.503 | [11] |
| c3ff0c0 | candidate it49 | `{"memory":"retention","conf_bias":true,"routing":"hrm","n_types":4,"halt":"utility","loss"` | 0 | +0.042 | +0.496 | [11] |
| cd1c2e2 | candidate it33 | `{"memory":"retention","routing":"hrm","n_types":4,"loss":"all"}` | 0 | +0.035 | +0.484 | [11] |
| c8c8f2f | candidate it41 | `{"memory":"retention","conf_bias":true,"routing":"hrm","n_types":4,"loss":"all"}` | 0 | +0.032 | +0.486 | [11] |
| c38f576 | candidate it50 | `{"memory":"full","conf_bias":true,"routing":"hrm","n_types":4,"halt":"utility","loss":"all` | 0 | +0.029 | +0.482 | [11] |
| cca1870 | control it32 | `{"conf_bias":true,"update":"rlt","loss":"all"}` | 0 | +0.027 | +0.483 | [11] |
| ca58f26 | candidate it42 | `{"memory":"full","conf_bias":true,"error_feedback":true,"routing":"hrm","n_types":4,"loss"` | 0 | +0.026 | +0.485 | [11] |
| c55be3b | candidate it32 | `{"conf_bias":true,"routing":"cycle","n_types":4,"update":"rlt","loss":"all"}` | 0 | +0.024 | +0.481 | [11] |
| c7a4a13 | candidate it44 | `{"memory":"topk","conf_bias":true,"routing":"hrm","n_types":4,"loss":"all"}` | 0 | +0.017 | +0.479 | [11] |
| prefix_all | baseline | `{"loss":"all"}` | 0 | +0.016 | +0.478 | [11] |
| tied | baseline | `{"memory":"tied"}` | 0 | +0.005 | +0.471 | [11] |
| prefix | baseline | `{}` | 0 | +0.000 | +0.482 | [11] |
| c1b73ac | candidate it16 | `{"memory":"retention","routing":"hrm","n_types":4,"update":"damped","loss":"all"}` | 0 | -0.000 | +0.461 | [11] |
| cf9e318 | candidate it15 | `{"memory":"full","routing":"hrm","n_types":4,"update":"damped","loss":"all"}` | 0 | -0.000 | +0.462 | [11] |
| c953be5 | candidate it4 | `{"memory":"retention","loss":"all"}` | 0 | -0.006 | +0.445 | [11] |
| c4d2769 | control it0 | `{"memory":"full","loss":"all"}` | 0 | -0.007 | +0.445 | [11] |
| c6f249b | candidate it12 | `{"routing":"hrm","n_types":4,"update_rank":4,"loss":"all"}` | 0 | -0.007 | +0.461 | [11] |
| cfa78e0 | candidate it45 | `{"conf_bias":true}` | 0 | -0.013 | +0.470 | [11] |
| cecfcbd | candidate it36 | `{"memory":"topk","topk":4,"conf_bias":true,"routing":"hrm","n_types":4,"loss":"all"}` | 0 | -0.014 | +0.453 | [11] |
| c910d5d | candidate it5 | `{"update":"damped","loss":"all"}` | 0 | -0.020 | +0.461 | [11] |
| cdc1f2e | candidate it54 | `{"conf_bias":true,"routing":"hrm","n_types":4,"update_rank":4,"halt":"utility","loss":"all` | 0 | -0.022 | +0.464 | [11] |
| prev | baseline | `{"memory":"prev"}` | 0 | -0.022 | +0.458 | [11] |
| c5792bd | candidate it28 | `{"memory":"lowrank","conf_bias":true,"routing":"hrm","n_types":4,"slow_period":1,"update":` | 0 | -0.022 | +0.447 | [11] |
| cf28ceb | candidate it11 | `{"memory":"topk","topk":4,"routing":"hrm","n_types":4,"loss":"all"}` | 0 | -0.025 | +0.439 | [11] |
| c627854 | candidate it52 | `{"memory":"topk","topk":4,"conf_bias":true,"routing":"hrm","n_types":4,"halt":"utility","l` | 0 | -0.026 | +0.455 | [11] |
| c113bc2 | candidate it38 | `{"conf_bias":true,"routing":"hrm","n_types":4,"update_rank":8,"loss":"all"}` | 0 | -0.033 | +0.453 | [11] |
| c4e477a | candidate it37 | `{"memory":"lowrank","conf_bias":true,"routing":"hrm","n_types":4,"loss":"all"}` | 0 | -0.045 | +0.430 | [11] |
| cca4844 | candidate it55 | `{"memory":"lowrank","conf_bias":true,"routing":"hrm","n_types":4,"halt":"utility","loss":"` | 0 | -0.049 | +0.425 | [11] |
| c26a07a | candidate it47 | `{"memory":"lowrank"}` | 0 | -0.073 | +0.403 | [11] |
| one_pass | baseline | `{"loops":1,"memory":"tied"}` | 0 | -0.086 | +0.410 | [11] |
| cc0d732 | candidate it0 | `{"memory":"topk","loss":"all"}` | 0 | -0.086 | +0.388 | [11] |
| c6384c5 | candidate it29 | `{"conf_bias":true,"routing":"hrm","n_types":4,"slow_period":1,"update":"rlt","update_rank"` | 0 | -0.086 | +0.416 | [11] |
| c46ea5b | candidate it20 | `{"routing":"hrm","n_types":4,"update":"rlt","update_rank":4,"loss":"all"}` | 0 | -0.088 | +0.409 | [11] |
| c9eab24 | candidate it2 | `{"update_rank":4,"loss":"all"}` | 0 | -0.091 | +0.399 | [11] |
| c7bdda7 | candidate it18 | `{"memory":"lowrank","rank":4,"routing":"hrm","n_types":4,"update":"rlt","loss":"all"}` | 0 | -0.091 | +0.394 | [11] |
| cdba9c3 | candidate it46 | `{"update_rank":4}` | 0 | -0.095 | +0.404 | [11] |
| c1cf32d | candidate it31 | `{"conf_bias":true,"routing":"hrm","n_types":4,"route_hard_eval":true,"slow_period":1,"upda` | 0 | -0.242 | +0.271 | [11] |
| c33f968 | candidate it10 | `{"routing":"hrm","n_types":4,"route_hard_eval":true,"loss":"all"}` | 0 | -0.297 | +0.224 | [11] |

### Adversarial task variants

| it | spec | accepted | explanations it targets |
|---:|---|---|---|
| 5 | `null` | no: KeyError: 'family' |  |
| 5 | `{"family":"revision","length":28,"depth":2,"distract":6,"variant":"reordered"}` | a0 | Uniform loop forcing (loss=all) is the active gain; extra memory/low-rank/top-k  vs Loop memory helps only revision-like long-range state overwrite/tracking; it is  |
| 11 | `{"family":"repair","length":8,"depth":2,"distract":0,"variant":"flagged"}` | a1 | Typed/hierarchical HRM routing specializes loops and drives revision/proof_deps  vs Prefix recurrent memory with loss-all loop forcing is the baseline; memory/full/ |
| 17 | `{"family":"revision","length":16,"depth":1,"distract":0,"variant":"reordered"}` | a2 | typed HRM routing with all-loop forcing/damped refinement is the active gain mec vs richer loop memory (full/retention/top-k) is the active gain mechanism |
| 23 | `{"family":"repair","length":8,"depth":2,"distract":0,"variant":"flagged"}` | a3 | adaptive typed/HRM loop routing is the main driver of gains on compositional fam vs confidence-weighted memory attention is a separate refinement/error-correction a |
| 29 | `{"family":"revision","length":16,"depth":2,"distract":4,"variant":"reordered"}` | a4 | Typed HRM routing with frequent manager updates (slow_period=1) reallocates dedu vs Answer-state accumulation plus all-loop forcing (rlt+loss=all) is the active ing |
| 35 | `{"family":"repair","length":6,"depth":1,"distract":0,"variant":"unflagged"}` | a5 | typed/HRM routing + loop forcing improves iterative refinement by role specializ vs conf_bias/confidence-weighted memory attention concentrates reads on reliable lo |
| 41 | `{"family":"revision","length":16,"depth":1,"distract":2,"variant":"reordered"}` | a6 | Typed hrm routing with loss=all is the main gain driver and should remain useful vs Confidence/history-weighted memory is the main gain driver and should degrade wh |
| 47 | `{"family":"revision","length":16,"depth":1,"distract":0,"variant":"reordered"}` | a7 | HRM/typed loop routing (n_types=4) improves structured state tracking, especiall vs Uniform all-loop supervision (loss=all) is a general training-signal boost rathe |
| 53 | `{"family":"repair","length":8,"depth":2,"distract":0,"variant":"unflagged"}` | a8 | Typed HRM routing + utility halting allocates compute to repair/verify stages, w vs Prefix memory with all-loop loss already captures the needed signal; extra routi |
| 59 | `{"family":"revision","length":16,"depth":2,"distract":4,"variant":"reordered"}` | a9 | Typed HRM routing with all-loop loss and confidence-biased memory gives order-ro vs The gain is mostly generic effective depth/all-loop forcing/confidence bias, not |
