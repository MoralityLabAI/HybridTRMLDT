# AIRIS/DAS Forecast Integrity Resilience

Episodes: `667`
Paired scenarios per episode: `10`
Total trials: `6670`

| Controller | Clean acceptance | Negative-control acceptance | Negative-control fallback | Changed from control | Max regret |
|---|---:|---:|---:|---:|---:|
| `topology_only` | 1.000 | 0.444 | 0.556 | 0.111 | 1.0037 |
| `integrity_sealed` | 1.000 | 0.000 | 1.000 | 0.000 | 0.0000 |

## Scenario Breakdown

| Scenario | Topology-only accept | Integrity-sealed accept | Integrity fallback |
|---|---:|---:|---:|
| `clean` | 1.000 | 1.000 | 0.000 |
| `protocol_shift` | 0.000 | 0.000 | 1.000 |
| `unseen_context` | 0.000 | 0.000 | 1.000 |
| `sequence_substitution` | 1.000 | 0.000 | 1.000 |
| `confidence_inflation` | 1.000 | 0.000 | 1.000 |
| `support_inflation` | 1.000 | 0.000 | 1.000 |
| `route_override` | 0.000 | 0.000 | 1.000 |
| `authority_override` | 0.000 | 0.000 | 1.000 |
| `unknown_rule_id` | 1.000 | 0.000 | 1.000 |
| `retrieval_outage` | 0.000 | 0.000 | 1.000 |

Claim boundary: Deterministic paired fault-injection over sealed AIRIS/DAS replay receipts. It measures authorization resilience, not adversarial robustness of a learned AIRIS model or native distributed DAS.
