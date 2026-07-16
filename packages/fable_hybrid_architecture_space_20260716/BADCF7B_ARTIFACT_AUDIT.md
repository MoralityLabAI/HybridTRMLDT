# Artifact Audit: Gaming Versus Oversight Leverage

Date: `2026-07-16`

Audit target: HybridTRMLDT commit `badcf7b9619779dafbf60a669305376d0e90ec5a`

Historical study ID: `gaming_vs_improvement_v1`

## Provenance Status

This is a post-commit sidecar. It is not inside
`fable_hybrid_architecture_space_20260716.zip`, and the ZIP remains byte-identical at SHA-256
`c4c220fea7cf800a1813696b36078a8755233d80642d4e12568b10a74cf96ef1`.

The audit recomputes claims from the committed result JSON and JSONL corpus. It does not expand the evidence
beyond the deterministic toy storyworld, and it does not convert latent-probe performance into constitutional
authority.

## Artifact Verification

- Canonical full-record SHA-256:
  `e587b8456fda8e2183694f7b6c1e40660b2cfe06e5e048b319f0a040f5f3cc9d`.
- Full corpus length: `26,880` records.
- Receipt integrity failures: `0`.
- Held-out rows outside the evaluation split: `0`.
- Canonical and mirrored result/record artifacts are byte-identical.
- The identical-fallback negative control has action-change `0.000` and utility delta `+0.0000` in every
  evidence arm and round.
- The claim-only negative control reaches false-claim rate `1.000` in all three seeds from
  `0.033 / 0.300 / 0.656`, while exact soundness remains flat.

## Recomputed Headline

Exact mechanics plus state-conditioned fallback changes `0.240` of actions and adds `+0.0407` mean executed
utility. This is controller utility from the gate-plus-fallback, not improvement in the proposer.

The exposed frozen probe satisfies the preregistered evasion conjunction in seeds 17 and 43: passage reaches
`1.000`, exact soundness stays flat, and hidden-audit AUROC falls from `0.802` to `0.315` and from `0.818` to
`0.620`. Seed 29 remains a counterexample because hidden-audit AUROC rises from `0.681` to `0.832`.

## Oversight Leverage Disabled

The strongest within-arm comparison is the exposed probe with the state-conditioned fallback:

| Proposer condition | Gate action-change | Utility delta vs proposal |
|---|---:|---:|
| Frozen | 0.487 | +0.0404 |
| Five-round expert iteration | 0.000 | +0.0000 |

Adaptation therefore removes the full `0.0404` observed oversight benefit. The gate reports complete compliance
while never invoking the fallback. Probe gaming here does more than degrade a sensor: it silently disables the
controller intervention that produced the benefit.

This is the cleanest constitutional-HRM result in the current run. An exposed interpretation signal should not
hold sole gating authority under training pressure.

## Supporting Tradeoffs

Dual channel aliases exact mechanics after adaptation. Claims saturate to `env_sound_dead`, so the disagreement
state no longer contributes selectivity. Its value is visible only before adaptation in this run.

The hidden rotating probe takes the opposite failure mode. It rejects effectively every final proposal, changes
`0.961` of actions under the distinct fallback, and has a roughly `0.79` false-rejection share, close to the prior
AIRIS estimate of about 75%. Its `+0.0376` utility is not validation of hidden probes as gates: the gain depends on
the toy gym's independently implemented fallback being near-oracle.

False provenance reaches `1.000` in every expert-iterated arm, regardless of evidence source. That fact means the
claim channel is unconstrained once another verifier governs execution. False-claim rate is a direct gaming metric
only where claims themselves influence gating.

## Power Boundary

Round-0 oracle accuracy equals the majority baseline exactly in all three seeds, and exact soundness never changes
over five rounds. The proposer does not learn beyond majority guessing. The experiment can therefore identify
report-channel gaming and loss of oversight leverage, but it cannot observe genuine proposal improvement or test
whether a capable learner chooses improvement over evasion.

For that reason, the evidential title is **Gaming Versus Oversight Leverage**. The historical
`gaming_vs_improvement_v1` identifier is retained only to preserve artifact and command provenance.

## Architectural Conclusion

The current evidence supports three scoped conclusions:

1. Exposed latent probes may instrument, audit, or trigger an independent checker, but should not authorize the
   constitutional action path by themselves.
2. Exact mechanics plus a behaviorally distinct fallback is the only effective policy class with durable
   controller utility under adaptation; the nominal dual-channel arm aliases that class after claims saturate.
3. Hidden probes preserve adaptive independence but can impose severe false-rejection costs when promoted from
   audit channels to gates.

The next decisive run must begin with an undertrained proposer that is demonstrably above its majority prior while
remaining below the oracle ceiling, then repeat the exposed-versus-hidden comparison on held-out regions.
