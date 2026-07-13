# TheySing Native Hybrid Enforcement Benchmark

This benchmark runs the compiled TheySing headless engine directly.
The enforcement modes are native game mechanics, not source-inspired proxy task cards.
The three game scenarios use native heuristic policies; `bilateral_probe` is a matched forced-pressure turn.

Scenarios: `high_pressure_detente`, `diplomacy_ladder`, `babel_compact`, `bilateral_probe`
Seeds: 400, 401
Requested turns per run: `2`

| Mode | Runs | Order Acceptance | Breach Attempts | Blocked | Executed | Sanctions | Prevention | Mean Trust |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `soft` | 8 | 1.000 | 4 | 0 | 4 | 4 | 0.000 | 47.69 |
| `hard` | 8 | 0.992 | 4 | 4 | 0 | 4 | 1.000 | 47.69 |
| `graduated` | 8 | 0.996 | 4 | 2 | 2 | 4 | 0.500 | 47.69 |

## Scenario splits

### babel_compact

| Mode | Acceptance | Attempts | Blocked | Executed | Mean TAS | Mean Kessler |
|---|---:|---:|---:|---:|---:|---:|
| `soft` | 1.000 | 0 | 0 | 0 | 34.31 | 8.00 |
| `hard` | 1.000 | 0 | 0 | 0 | 34.31 | 8.00 |
| `graduated` | 1.000 | 0 | 0 | 0 | 34.31 | 8.00 |

### bilateral_probe

| Mode | Acceptance | Attempts | Blocked | Executed | Mean TAS | Mean Kessler |
|---|---:|---:|---:|---:|---:|---:|
| `soft` | 1.000 | 2 | 0 | 2 | 39.50 | 58.00 |
| `hard` | 0.000 | 2 | 2 | 0 | 39.50 | 58.00 |
| `graduated` | 1.000 | 2 | 0 | 2 | 39.50 | 58.00 |

### diplomacy_ladder

| Mode | Acceptance | Attempts | Blocked | Executed | Mean TAS | Mean Kessler |
|---|---:|---:|---:|---:|---:|---:|
| `soft` | 1.000 | 0 | 0 | 0 | 35.13 | 5.00 |
| `hard` | 1.000 | 0 | 0 | 0 | 35.13 | 5.00 |
| `graduated` | 1.000 | 0 | 0 | 0 | 35.13 | 5.00 |

### high_pressure_detente

| Mode | Acceptance | Attempts | Blocked | Executed | Mean TAS | Mean Kessler |
|---|---:|---:|---:|---:|---:|---:|
| `soft` | 1.000 | 2 | 0 | 2 | 51.06 | 44.00 |
| `hard` | 0.987 | 2 | 2 | 0 | 51.06 | 44.00 |
| `graduated` | 0.987 | 2 | 2 | 0 | 51.06 | 44.00 |

## Control interpretation

- The institutional breach tests destructive action under a cislunar common-carrier pact.
- The bilateral probe tests the same action channel under a non-aggression pact.
- Graduated enforcement should block the institutional breach but execute and sanction the bilateral breach.
- Hard enforcement should block both; soft enforcement should execute and sanction both.

## Notes

- Game-scenario agents used TheySing's deterministic native heuristic policy.
- `bilateral_probe` used one matched native manual turn to force a non-aggression conflict.
- Seeds, scenarios, and requested turns are matched across enforcement modes.
- Raw native JSONL traces are retained losslessly as `*.jsonl.gz` under `native_logs/`.
- Safety and usefulness remain separate; prevention is not treated as free utility.
