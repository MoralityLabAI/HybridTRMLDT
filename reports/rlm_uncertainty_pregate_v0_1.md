# RLM Uncertainty Pre-Gate v0.1

## Result

| Architecture | Raw | Evidence net | All-in primary | Calls | Tokens | Wall s | Unsafe | Contract |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gated_deterministic_voi` | 0.6341 | 0.6124 | 0.6124 | 0 | 0 | 0.00 | 0 | - |
| `ungated_rlm` | 0.6149 | 0.6040 | 0.5965 | 24 | 153,691 | 54.55 | 0 | 1.0000 |
| `gated_rlm` | 0.5974 | 0.5945 | 0.5916 | 9 | 58,074 | 22.30 | 0 | 1.0000 |
| `fixed_no_query` | 0.5907 | 0.5907 | 0.5907 | 0 | 0 | 0.00 | 0 | - |
| `gated_forced_failure` | 0.5907 | 0.5907 | 0.5907 | 0 | 0 | 0.00 | 0 | 0.0000 |
| `gated_always_exact` | 0.5823 | 0.5673 | 0.5673 | 0 | 0 | 0.00 | 0 | - |

## Registered Contrasts

- `gated_rlm` vs `ungated_rlm`: raw -0.0175, evidence net -0.0096, primary all-in -0.0049, calls -15, tokens -95617, wall -32.25s.
- `gated_rlm` vs `fixed_no_query`: raw +0.0067, evidence net +0.0037, primary all-in +0.0009, calls +9, tokens +58074, wall +22.30s.
- `gated_rlm` vs `gated_deterministic_voi`: raw -0.0367, evidence net -0.0179, primary all-in -0.0208, calls +9, tokens +58074, wall +22.30s.

The deterministic gate registered `9` gated calls and observed
`9`. Closed-gate call violations: `0`.
Unsafe executions: `0`. Forced-failure identity:
`true`.

## Cost Definition

Primary all-in utility subtracts evidence cost, `0.001` utility per 1,000 provider tokens, and `0.0005` utility
per provider wall second. These are registered benchmark-local exchange rates, not actual dollar prices.

## Boundary

This fresh synthetic confirmatory pilot tests a deterministic three-way-disagreement invocation gate around one official RLM evidence controller under registered benchmark-local token and latency charges. It does not price actual dollars, train a model, test official TinyRecursiveModels, establish general safety, or establish uniform task improvement.

## Integrity

- Result SHA-256: `3b80252b5cde431a3bc4f5f9649615f74d23e7194f85f9594e1670f9a5a53a69`
- Records SHA-256: `da0770d32aede6a95ec9da93a1d36181252102db9645f73cc9928ee4eb3eb5b0`
- Trajectory SHA-256: `b60d55dcae2577d87d8a82b4d75669f126528127d08e49e9290537acff9ab50f`
- Config SHA-256: `365c2a2e30d2cc51b676a1aaaaf8d8917774362a27bc0be34604f1e46ac9d719`
- Registration SHA-256: `1eb9988df3cdcb9437ff3695af58ea164be6a930c4809dd2fb5ccd55a5e631c9`
