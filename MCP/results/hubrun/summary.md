| run | episodes | mean score | solved (95% CI) | episodes with errors | prompt tok/episode | completion tok/episode | peak prompt | compacted calls | context overflows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| docedit-goblin | 8 | 0.33 | 1 (0.02-0.47) | 0 | 60187.0 | 11687.6 | 6650 | 87 | 0 |
| docedit-raw | 8 | 0.6994 | 5 (0.31-0.86) | 0 | 48087.9 | 7616.6 | 9468 | 0 | 0 |
| er-t1-goblin | 5 | 0.0 | 0 (0.00-0.43) | 0 | 294898.4 | 6419.4 | 32248 | 185 | 0 |
| er-t1-late | 5 | 0.0 | 0 (0.00-0.43) | 5 | 210853.0 | 4310.2 | 32119 | 39 | 5 |
| er-t1-raw | 5 | 0.0 | 0 (0.00-0.43) | 5 | 159908.2 | 5517.0 | 31066 | 0 | 5 |
| feishu-expert-goblin | 7 | 0.8571 | 6 (0.49-0.97) | 0 | 19536.6 | 3111.0 | 5240 | 0 | 0 |
| feishu-expert-raw | 7 | 1.0 | 7 (0.65-1.00) | 0 | 22045.1 | 3338.4 | 5659 | 0 | 0 |
| feishu-hard-goblin | 7 | 1.0 | 7 (0.65-1.00) | 0 | 21991.9 | 2377.1 | 5402 | 0 | 0 |
| feishu-hard-raw | 7 | 1.0 | 7 (0.65-1.00) | 0 | 24819.9 | 3145.4 | 6612 | 0 | 0 |
| jbm-constructs-dev | 4 | 0.8805 | 0 (0.00-0.49) | 0 | 278.8 | 903.5 | 287 | 0 | 0 |
| jbm-cross-dev80 | 80 | 0.7993 | 0 (0.00-0.05) | 0 | 457.8 | 1131.2 | 577 | 0 | 0 |
| jbm-mesh2-dev | 48 | 0.9293 | 0 (0.00-0.07) | 0 | 2086.1 | 1909.8 | 1369 | 0 | 0 |
| jbm-reasoner-dev | 16 | 0.8686 | 0 (0.00-0.19) | 0 | 315.6 | 810.2 | 331 | 0 | 0 |
| notebook-hardened | 8 | 0.9479 | 6 (0.41-0.93) | 0 | 22043.5 | 2740.1 | 1751 | 0 | 0 |
| sqlite-all | 57 | 0.386 | 22 (0.27-0.52) | 0 | 14198.8 | 3145.6 | 8916 | 0 | 0 |
| sw-constitution-eval | 15 | 0.0 | 0 (0.00-0.20) | 0 | 1306.4 | 1085.7 | 1889 | 0 | 0 |

| pair | tasks | raw mean | goblin mean | goblin better / raw better | prompt tokens goblin/raw | peak raw / goblin | context overflows raw / goblin | episodes with errors raw / goblin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| docedit | 8 | 0.6994 | 0.33 | 0 / 6 | 1.252 | 9468 / 6650 | 0 / 0 | 0 / 0 |
| er-t1 | 5 | 0.0 | 0.0 | 0 / 0 | 1.844 | 31066 / 32248 | 5 / 0 | 5 / 0 |
| er-t1-late | 5 | 0.0 | 0.0 | 0 / 0 | 1.319 | 31066 / 32119 | 5 / 5 | 5 / 5 |
| feishu-expert | 7 | 1.0 | 0.8571 | 0 / 1 | 0.886 | 5659 / 5240 | 0 / 0 | 0 / 0 |
| feishu-hard | 7 | 1.0 | 1.0 | 0 / 0 | 0.886 | 6612 / 5402 | 0 / 0 | 0 / 0 |

Skipped `er-t2-goblin`: skipped: event-recon tier 1 scored 0 in both arms; tier 2 dropped to save GPU time

Skipped `er-t2-raw`: skipped: event-recon tier 1 scored 0 in both arms; tier 2 dropped to save GPU time
