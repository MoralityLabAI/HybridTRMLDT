# RLM Uncertainty Pre-Gate v0.1: Independent Audit

This audit is post-hoc. The registered operational contrast remains the controlling result.

## Operational Result

The pre-gate reduced provider calls from `24` to `9`
(`62.5%`), saving `95,617` tokens and
`32.25` wall seconds. It nevertheless changed macro utility versus ungated by:

- Raw utility: `-0.0175`
- Evidence-only net utility: `-0.0096`
- Low-cost all-in: `-0.0073`
- Primary all-in: `-0.0049`
- High-cost all-in: `-0.0003`

The gate therefore failed its primary comparison. Registered provider costs would need to be approximately
`2.06x` the primary coefficients for the operational mean to
break even.

## Instability Decomposition

Both RLM arms were called independently on `9` gate-open tasks. Only
`2/9` query sequences and
`7/9` executed actions matched, despite
temperature zero. Their gate-open raw-utility sum differed by
`-0.300` in favor of ungated.

One gate-closed provenance task produced an ungated raw gain of `+0.120`, so the rule also made a genuine miss.
In a post-hoc coupled replay that uses the same ungated sample on gate-open tasks, the gate loses
`-0.0050` raw utility but gains
`+0.0061` primary all-in utility. This shows that the
cost-saving mechanism works conditionally; it does not overturn the registered operational failure.

## Winner

`gated_deterministic_voi` ranked first at
`0.6124` primary all-in utility. In this panel, a deterministic local
value-of-information policy dominated both RLM arms.

## Integrity

- Records replayed: `144`
- Task shards replayed: `24`
- Evidence receipt failures: `0`
- Unsafe executions: `0`
- Closed-gate provider calls: `0`
- Records SHA-256: `da0770d32aede6a95ec9da93a1d36181252102db9645f73cc9928ee4eb3eb5b0`

## Conclusion

The registered operational gate failed to preserve RLM quality and did not beat the ungated arm. A coupled replay shows that cost savings would be sufficient if controller outcomes were shared, but low gate-open repeatability makes that a mechanism diagnostic rather than a positive endpoint.
