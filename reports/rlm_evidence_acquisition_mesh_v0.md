# RLM Adaptive Evidence-Acquisition Mesh v0

## Construction

The official RLM sees an action-opaque canonical JSON snapshot and may select at most two universal evidence
queries. It cannot select actions, candidates, thresholds, or schedules. Typed receipts feed a deterministic
arbiter; the exact executor commits a safe action or executes fixed consensus. Provider and parser failures stop
acquisition without erasing baseline utility.

## Results

| Architecture | Net utility | Raw utility | Oracle regret | Unsafe | Contract | Queries | Query cost | 2nd query | Changes | Provider errors | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `rlm_adaptive_two_query` | 0.6642 | 0.6801 | 0.0427 | 0 | 1.0000 | 0.792 | 0.0158 | 0.3750 | 0.1250 | 0 | 165,273 |
| `mesh_fixed_no_query` | 0.6396 | 0.6396 | 0.0674 | 0 | - | 0.000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| `rlm_forced_failure` | 0.6396 | 0.6396 | 0.0674 | 0 | 0.0000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| `mesh_always_exact` | 0.6388 | 0.6788 | 0.0681 | 0 | - | 1.000 | 0.0400 | 0.0000 | 0.2500 | 0 | 0 |
| `mesh_deterministic_voi` | 0.6311 | 0.6878 | 0.0759 | 0 | - | 1.917 | 0.0567 | 0.9167 | 0.2917 | 0 | 0 |
| `mesh_exact_then_rollout` | 0.6211 | 0.7011 | 0.0858 | 0 | - | 2.000 | 0.0800 | 1.0000 | 0.2917 | 0 | 0 |
| `mesh_random_two_query` | 0.6063 | 0.6600 | 0.1007 | 0 | - | 2.000 | 0.0537 | 1.0000 | 0.1667 | 0 | 0 |

Typed unsafe executions: `0`. Forced-RLM-failure identity:
`true`.

## Matched Contrasts

- `rlm_adaptive_two_query` vs `mesh_fixed_no_query`: net +0.0246, raw +0.0405.
- `rlm_adaptive_two_query` vs `mesh_random_two_query`: net +0.0580, raw +0.0200.
- `rlm_adaptive_two_query` vs `mesh_deterministic_voi`: net +0.0331, raw -0.0077.
- `rlm_adaptive_two_query` vs `mesh_exact_then_rollout`: net +0.0431, raw -0.0211.

## Boundary

This fresh synthetic mixed-control pilot tests bounded active evidence selection by one official RLM runtime around a deterministic typed mesh. It does not test official TinyRecursiveModels, proposer adaptation, general orchestrator safety, neural alignment, model superiority, or a sheaf-spectral claim.

## Integrity

- Result SHA-256: `7b44018a412527274b1078242650cc42a55293a8ebc4e27c11f0a5759975f067`
- Records SHA-256: `6ccf296ca56640b7a557f576315ee39b64541ab8858f84ae71edb00f69ff0d69`
- Trajectory SHA-256: `a4b98b15dae68b491c3712f104515609c4fa53310ab6764e5a4f05b5837d02c2`
- Config SHA-256: `6ecc3ffabbd31ccedd8831068eca6736a06200ef09d52058cb1c98b8f0b9b877`
- Acquisition contract SHA-256: `26452606d5e110e5b92a8ce6cf3b8da2c046760085550ded6d4e691b5304655e`
