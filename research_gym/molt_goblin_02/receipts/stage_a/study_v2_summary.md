Upstream Study v2, torch 2.15.0+cpu, device cpu, threads 2, train loops 4, status development_only.

| arm | panel | loss | seeds | params | updates | fwd block calls | len8 seq acc @train loops | len16 seq acc @16 loops | train CPU s | append-only parity max abs |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| one_pass | fixed_data | last | 3 | 8807 | 256 | 256 | 0.000 (tok 0.311) | 0.000 (tok 0.215) | 5.4 | nan |
| one_pass | reinvest_forward_calls | last | 3 | 8807 | 1024 | 1024 | 0.031 (tok 0.561) | 0.000 (tok 0.356) | 31.1 | nan |
| prefix_kv | fixed_data | all | 3 | 8807 | 256 | 1024 | 0.000 (tok 0.303) | 0.000 (tok 0.223) | 18.0 | 0.00e+00 |
| prefix_kv | fixed_data | last | 3 | 8807 | 256 | 1024 | 0.000 (tok 0.286) | 0.000 (tok 0.214) | 20.7 | 0.00e+00 |
| residual | fixed_data | all | 3 | 8807 | 256 | 1024 | 0.000 (tok 0.336) | 0.000 (tok 0.165) | 15.5 | nan |
| residual | fixed_data | last | 3 | 8807 | 256 | 1024 | 0.016 (tok 0.469) | 0.000 (tok 0.163) | 15.8 | nan |
