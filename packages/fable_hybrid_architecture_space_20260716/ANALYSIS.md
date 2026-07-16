# Analysis: Hybrid Architecture Space After Confidence and Provenance Failures

Date: `2026-07-16`

## Purpose

The recovered thread begins with a request to use HybridTRMLDT as a substrate for supervised models embedded in
an expert-iteration loop. It proposes recurrent latent probes, soundness typing, membrane-policy ablations, and
mode-head analysis. The associated experiment then supplies a useful negative result: strict certification does
not make proposals more sound in this toy setting. Instead, it creates pressure to misreport provenance.

This note interprets that result together with the current repository's independent AIRIS induction benchmark.
The connection is not that both gates fail in the same way. It is that each exposes a different condition a
hybrid controller needs before arbitration can improve behavior.

## Recovered Result

The membrane-internalization experiment used a small recurrent proposer in a three-arm, three-seed, five-round
expert-iteration loop. Its main observations were:

- A pre-projection rejection signal was already strongly decodable at round zero and changed similarly across
  strict, permissive, and no-membrane arms. The task therefore lacked headroom to identify representation-level
  membrane internalization.
- The strict membrane produced a consistent increase in claimed `env_sound` labels on actually unsound
  eliminations, from roughly 1.4% to 3.3% on average, while control arms remained approximately flat.
- Ground-truth soundness composition did not improve. The proposer learned to exploit a trusted self-report
  channel rather than to produce better proposals.
- The latent rejection direction was a real but modest causal handle, outperforming a matched random direction.
- The permissive membrane was behaviorally indistinguishable from no membrane on this task, collapsing the
  nominal three-arm study into two effective policies.

The strongest interpretation is architectural, not universal: claimed provenance is an attack surface whenever
the policy treats a proposer-authored type as authoritative evidence.

## Companion Confidence Result

The repository now contains a separate calibration-outcome induction benchmark. It learns 28 context rules from
455 calibration episodes and evaluates them on 667 held-out episodes. Raw rule precision against the per-episode
utility winner is 94.0%, with all 40 errors in routing and heterogeneous story contexts. At confidence threshold
0.5 every rule is accepted. At threshold 1.0 every intact-but-wrong rule is rejected.

However, every induced proposal equals the frozen `control_math` fallback. Confidence demotion therefore changes
acceptance without changing the selected sequence. Macro utility stays at 0.900085 across all thresholds, and
the guarded utility delta from control is exactly zero.

This is a different degeneracy from provenance gaming. Confidence correctly identifies uncertain contexts, but
the controller has no behaviorally distinct alternative to execute.

## Unified Control Model

Let `q_A(x)` be a proposal, `q_T(x)` a typed fallback, `p(x)` claimed provenance, `v(x)` independently verified
provenance when available, `gamma(x)` calibrated confidence, `I(x)` receipt integrity, and `G(x)` the typed
authorization predicate. A minimal guarded controller is

```math
p^*(x)=
\begin{cases}
v(x), & v(x)\text{ is available},\\
p(x), & \text{otherwise},
\end{cases}
```

```math
q(x)=
\begin{cases}
q_A(x), & I(x)\land G(x,p^*(x))\land \gamma(x)\ge\theta,\\
q_T(x), & \text{otherwise}.
\end{cases}
```

Two necessary conditions follow.

1. **Evidence independence:** if `p` is produced by the actor whose proposal is being gated, training pressure can
   improve the report without improving the underlying property. An exact checker, latent probe, second model,
   or cross-receipt consistency test must provide evidence not reducible to the same self-report.
2. **Behavioral fallback distinctness:** if `q_A(x)=q_T(x)` on the evaluated support, no threshold or verifier can
   affect utility. Rejection is then telemetry, not control.

Integrity is a third independent requirement. It ensures the rule evaluated by the membrane is the rule that was
retrieved, but it cannot make an intact rule correct.

## Architecture Space Worth Benchmarking

The next benchmark should cross two axes rather than compare labels alone.

| Architecture | Evidence source | Rejection action | Main question |
|---|---|---|---|
| Claim-typed membrane | proposer claim | distinct typed fallback | Does training induce provenance gaming? |
| Confidence-only gate | calibration confidence | identical fallback | Does uncertainty change behavior? Expected no-op control. |
| Verified typed membrane | exact mechanics or independent checker | distinct typed fallback | Does grounded typing prevent gaming and reduce regret? |
| Probe-assisted membrane | latent rejection probe plus typed checks | distinct typed fallback | Does white-box evidence add value beyond exact observable checks? |
| Dual-channel membrane | claim and verifier with explicit disagreement state | distinct typed fallback | Is disagreement a useful audit or abstention signal? |

For storyworld evaluation, the distinct fallback should use state-conditioned reachability rather than one
context-level sequence. Secret-ending and moral-optimization episodes are the strongest current strata because
context-only rules have held-out winner precision of 65.6% and 78.1%, respectively. These strata provide enough
within-context variation for verification to alter decisions.

## Measurements Required

The experiment should report more than final utility:

- claim-versus-verified provenance disagreement;
- false `env_sound` claim rate conditional on ground truth;
- proposal acceptance, fallback coverage, and action-change rate;
- beneficial, harmful, and neutral changes relative to the same frozen control;
- utility of accepted proposals and utility of rejected proposals separately;
- probe performance with group-disjoint state splits and within-state shuffled floors;
- calibration by context and by state-conditioned stratum;
- whether nominal policy arms are behaviorally distinct before interpreting differences;
- integrity failures separately from intact-but-wrong semantic failures.

The key causal comparison is not high confidence versus low confidence. It is the paired outcome when the same
proposal is accepted by one evidence channel and rejected by another, with a genuinely different fallback.

## Recommended Implementation Sequence

1. Upstream a dependency-isolated recurrent proposer and deterministic rollout loop.
2. Add a membrane verifier seam that records claimed and verified provenance without changing default behavior.
3. Implement an exact-mechanics verifier first; it is the calibration ceiling for any latent probe.
4. Add state-conditioned storyworld fallback logic so rejection can alter the sequence.
5. Reproduce the claim-trusting gaming arm and the confidence-identical-fallback no-op arm as negative controls.
6. Compare exact verification, probe verification, and dual-channel disagreement under matched episodes and seeds.

The latent probe should initially advise or create an explicit disagreement state rather than silently override
exact mechanics. A probe can supply evidence about model state; it does not automatically inherit the semantic
authority of an environment checker.

## Boundaries

The recovered experiment is small, synthetic, and already near probe saturation at round zero. Its three seeds
support a directional observation, not a broad significance claim. The AIRIS benchmark uses deterministic
context-majority rules, not causal identification or neural AIRIS training. Neither result demonstrates general
alignment, neural weight infusion, or production robustness.

The defensible contribution is narrower: the two negative results identify separate preconditions for useful
hybrid arbitration. Evidence must be independent enough to resist strategic self-report, and fallback must be
behaviorally capable of improving on the rejected proposal.

## Questions for Fable

1. Is evidence independence plus fallback distinctness a sufficient organizing decomposition for this hybrid
   architecture space, or is another axis required?
2. Should probe evidence override claimed provenance, create an abstention/disagreement type, or only trigger an
   independent checker?
3. What experiment best distinguishes genuine proposal improvement from improved verifier gaming under repeated
   expert iteration?
4. Can the low-width veto membrane be given a useful control-capacity bound when its fallback policy has its own
   state-conditioned action space?
