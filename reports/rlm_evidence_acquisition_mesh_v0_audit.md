# RLM Evidence-Acquisition Mesh v0: Independent Audit

This is a **post-hoc descriptive audit**, not a registered endpoint. It recomputes the sealed records without
changing the canonical result or receipt.

## Finding

The adaptive RLM ranked first on macro net utility (`0.6642`).
Against fixed no-query consensus, its mean net delta was `+0.0246` and raw-utility
delta was `+0.0405`. The effect was concentrated: task-level net deltas were positive
on `2/24`, zero on `14/24`, and negative on
`8/24`. Two large gains outweighed eight net losses, mostly query costs; this is not a
uniform improvement result.

| Family | Net delta vs fixed |
|---|---:|
| `latest_rule_action` | +0.0769 |
| `multi_hop_reachability` | -0.0050 |
| `provenance_gate` | +0.0550 |
| `storyworld_control` | -0.0283 |

## Control Role

The RLM skipped acquisition on `14/24` tasks, queried on `10/24`, and
used a second query on `9/24`. Of queried tasks, `7`
did not change the action, `2` improved raw utility, and
`1` reduced it. It never selected `counterfactual_rollout`; its measured role was a
sparse evidence buyer and stopping policy, not a comprehensive evidence planner.

Its exact query sequence matched the enumerated sequence oracle on `10/24` tasks,
all through `STOP`: it skipped `4` tasks where the oracle acquired evidence
and queried `7` tasks where the oracle stopped. The run therefore demonstrates
useful cost-sensitive sparsity, not successful sequence-oracle recovery.

Against deterministic VOI, the RLM gained `+0.0331` net utility while losing
`-0.0077` raw utility. The net advantage therefore comes from lower query cost,
not higher decision quality. The richer fixed exact-plus-rollout arm likewise had higher raw utility but lower net
utility at the registered costs.

## Integrity

- 168 records and all 24 task shards replayed.
- Evidence receipt failures: `0`.
- Unsafe executions: `0`.
- Forced-failure identity: `true`.
- RLM contract rate: `1.0000`; provider errors: `0`.
- Records SHA-256: `6ccf296ca56640b7a557f576315ee39b64541ab8858f84ae71edb00f69ff0d69`.

## Boundary

The positive mean net-utility result is concentrated in two tasks and is heterogeneous across families. It supports sparse evidence acquisition as a candidate mesh role, not uniform policy superiority, general safety, or a neural architecture claim.
