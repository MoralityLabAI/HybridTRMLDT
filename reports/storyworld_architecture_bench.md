# Storyworld Architecture Benchmark

Episodes per scenario/policy: 64
Horizon: 6
Confidence gamma: 2.00

## moral_optimization

| Policy | Success Rate | Avg Score | Overrides | Avg Margin |
|---|---:|---:|---:|---:|
| `confidence_arbitration` | 1.000 | 11.06 | 134 | 0.23 |
| `hard_gate` | 1.000 | 10.80 | 0 | 0.00 |
| `trm` | 0.188 | 3.45 | 0 | 0.00 |
| `typed_membrane` | 1.000 | 9.95 | 124 | 0.00 |

## secret_ending

| Policy | Success Rate | Avg Score | Overrides | Avg Margin |
|---|---:|---:|---:|---:|
| `confidence_arbitration` | 0.938 | 0.94 | 134 | 0.23 |
| `hard_gate` | 1.000 | 1.00 | 0 | 0.00 |
| `trm` | 0.172 | 0.17 | 0 | 0.00 |
| `typed_membrane` | 1.000 | 1.00 | 124 | 0.00 |
