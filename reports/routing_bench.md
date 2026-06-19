# Env Pointer Routing Benchmark

Train examples: 402
Test examples: 173

| Router | Accuracy | Correct | Abstained | Avg Candidates |
|---|---:|---:|---:|---:|
| `ldt` | 0.474 | 82/173 | 15 | 0.00 |
| `trm` | 0.815 | 141/173 | 0 | 0.00 |
| `hybrid` | 0.815 | 141/173 | 0 | 2.09 |

## Ablations

| Router | Accuracy | Correct | Abstained | Avg Candidates |
|---|---:|---:|---:|---:|
| `hybrid_hard_filter` | 0.584 | 101/173 | 0 | 2.09 |
| `hybrid_confidence_arbitration` | 0.815 | 141/173 | 0 | 2.09 |

## Architecture Variants

Confidence arbitration gamma: `0.00`

| Variant | Accuracy | Correct | Abstained | Avg Candidates |
|---|---:|---:|---:|---:|
| `typed_membrane` | 0.815 | 141/173 | 0 | 2.09 |
| `hard_gate` | 0.584 | 101/173 | 0 | 2.09 |
| `confidence_arbitration` | 0.815 | 141/173 | 0 | 2.09 |
