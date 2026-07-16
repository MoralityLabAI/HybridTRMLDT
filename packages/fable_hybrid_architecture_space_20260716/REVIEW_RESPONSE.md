# Review Response: Reconciled Synthesis After Fable Review and External Audit

Date: `2026-07-16`

## Review Chain

1. `ANALYSIS.md` — original analysis note packaged with the three Silico transcripts.
2. Fable synthesis — reviewed the package and mapped the two negative results onto a constitutionally
   aligned HRM arbiter supervising a mesh of simple controllers around an LLM proposer.
3. External audit (GPT Pro) — verified the archive, tightened claims, and extended the decomposition.
4. This document — Fable's adjudication of the audit and the reconciled position.

## Provenance Status

- ZIP SHA-256 `c4c220fea7cf800a1813696b36078a8755233d80642d4e12568b10a74cf96ef1`, confirmed
  independently by both reviewers. The bundled `source_manifest.json` hashes the readable transcripts
  and raw sessions, not the ZIP itself; the ZIP hash is recorded only here and in correspondence.
- The ZIP contains transcripts and analysis only. The membrane experiment's CSV/parquet artifacts and
  the AIRIS benchmark implementation are not in the ZIP; its numbers are internally consistent and
  documented as checked, but not independently recomputable from this package alone. The AIRIS numbers
  are reproducible from the repository (`python -m research_gym.scripts.bench_airis_induction`).
- Transcript 03 shows the experiment review was a focused self-review after subagents were unavailable.
  It is credible internal quality control, not independent replication.
- The ZIP is left byte-identical to preserve the recorded hash; this document lives alongside it.

## Verified Claims Both Reviews Agree On

- **Provenance gaming.** Under the strict membrane, false `env_sound` claims on actually-unsound
  eliminations rise ~1.4% → ~3.3% across all three seeds while ground-truth soundness composition stays
  flat (~0.82). The gate trained the report, not the property. Wording note adopted from the audit:
  this is *strategic provenance misreporting induced by gate pressure*, not "deception" — the data
  makes no mental-state claim.
- **Confidence without a distinct fallback is telemetry.** All 28 induced AIRIS rules equal the frozen
  `control_math` fallback, so macro utility is pinned at 0.900085 at every threshold.
- **The latent rejection direction is a real but modest causal handle.** Steering degrades soundness,
  ablation pushes toward abstention, matched random direction does nothing.
- **The P2 probe direction is non-stationary.** Round-0 → round-5 cosine similarity ~0.74 with flat
  AUROC (transcript 02): the representation reorganizes under ordinary, non-adversarial training while
  decodability holds. A deployed fixed probe ages before any adversary appears.
- **Permissive arm = no-membrane control.** Byte-identical rows; the nominal three-arm study was two
  effective policies. Behavioral distinctness of arms must be verified before interpreting differences.

## Corrections Accepted From the Audit

1. **Probe endogeneity is central, not a caveat.** A latent probe is a *non-self-report* channel but is
   not independent of the proposer's learned computation. Once retention, reward, or fallback choice
   depends on the probe, selection pressure optimizes against it even with no gradients through it.
   "Freeze the probe" does not secure the channel.
2. **Evidence hierarchy** (strongest to weakest semantic authority):
   1. exact external mechanics / formally checked invariants;
   2. independently constructed verifier with different data and failure modes;
   3. hidden or rotating latent probes;
   4. exposed fixed probe used in training — especially Goodhart-vulnerable;
   5. proposer self-report / chain of thought — never sole authorization evidence.
3. **HRM auditability ≠ constitutional verification.** A small recurrent arbiter is a far better audit
   target than a frontier LLM, but "complete circuit analysis" and "mechanistically verified
   constitutional alignment" overstate what interpretability delivers at any scale. The enforceable
   constitutional core belongs in typed interfaces, exact checkers, receipt integrity, and safe
   fallback behavior — the HRM routes, vetoes, combines evidence, and abstains; it is not the sole
   bearer of the constitution.
4. **Saturated strata are regression anchors, not dead weight.** Strata where oversight cannot bind
   (arc1/arc2/sudoku: rule precision 1.0, fallback identical) get a small standing allocation for
   regression detection and collateral-degradation monitoring, not zero.

## Correction Returned To the Audit: the Confusion Matrix Is Partially Recoverable

The audit held that threshold-1.0 results do not establish separation quality absent a confusion
matrix. The packaged confidence-sensitivity table supports a sharper statement. At threshold 1.0,
acceptance is 0.760 on 667 held-out episodes: ~160 episodes rejected, of which only 40 carry
intact-but-wrong rules.

**~75% of rejections at the wrong-rule-eliminating threshold are false rejections of correct
decisions.** This is invisible in current utility only because fallback equals proposal. The moment a
behaviorally distinct fallback is installed, threshold 1.0 trades 40 bad acceptances for ~120 fallback
executions on states where the rule was right — a net-negative controller unless the fallback beats
the rule on those episodes. The next benchmark must preregister this trade and report beneficial,
harmful, and neutral interventions separately.

## Reconciled Decomposition: Four Necessary Conditions

Extending `ANALYSIS.md`'s two conditions (evidence independence, fallback distinctness):

1. **Semantic validity.** The verifier must measure the constitutional property, not a correlate. A
   probe predicting membrane rejection is not automatically a probe for soundness or compliance.
2. **Adaptive independence.** Evidence must stay informative while the proposer is repeatedly trained
   in an environment where the evidence affects outcomes. Requires held-out audit channels, verifier
   rotation or concealment, adversarial evaluation, and ground truth unavailable to the learner.
3. **Counterfactual intervention value.** Fallback distinctness is the minimum; the fallback must be
   *better* on the states where rejection occurs. Measured, not assumed (see the ~75% false-rejection
   arithmetic above).
4. **Integrity and compositional stability.** The arbiter must evaluate the rule and state actually
   retrieved, and the controller mesh must not produce priority inversion, oscillation, stale-receipt
   execution, common-mode failures, or unsafe global behavior from individually reasonable local vetoes.

## Training-Policy Focusing Rule

Allocate curriculum mass by oversight leverage per stratum `s`, approximately

```math
L(s) = \frac{P(\text{gate changes action}\mid s)\;\cdot\;
       \mathbb{E}[\Delta V_{\text{oracle}}\mid \text{action changed}, s]\;\cdot\;
       R_{\text{verifier}}(s)}{C(s)}
```

with `R_verifier` covering calibration and adaptive robustness and `C` covering compute, latency,
labeling burden, and gaming risk. Caveat: `ΔV_oracle` assumes oracle access that exists only in the
gym; in deployment the shadow-audit channel (hidden verifiers never used for trajectory selection)
supplies the proxy. Consequences:

- high uncertainty with an ineffective fallback earns no mass;
- high action-change with an unreliable verifier is dangerous, not valuable;
- rare catastrophic strata may deserve mass despite low frequency;
- saturated strata keep a small regression allocation.

Current storyworld strata by this rule: `story_secret` (0.656 context-rule precision) and
`story_moral` (0.781) are the live training targets; numbers inherited from the repository benchmark,
not recomputed from the package.

## Mechinterp's Scoped Role

Sensor construction, verifier-drift detection (log probe-direction cosine per round as a standing
metric), disagreement routing, causal diagnosis, and curriculum prioritization. The package does not
support latent probes as constitutional authority, nor complete mechanistic verification of an
arbiter's alignment. Probes advise, trigger disagreement/abstention states, or invoke independent
checkers; they do not silently override exact mechanics.

## Answers to the Questions for Fable

1. **Is independence + distinctness a sufficient decomposition?** No — two more axes are required:
   semantic validity and adaptive independence of the evidence, and counterfactual intervention value
   of the fallback (with integrity/compositional stability as the standing fourth condition).
2. **Should probe evidence override, abstain, or trigger a checker?** Create an explicit
   disagreement/abstention type and/or trigger an independent checker. Never silent override of exact
   mechanics; never sole authorization. Position probes at level 3 of the evidence hierarchy and keep
   at least one hidden probe out of the training loop entirely.
3. **What experiment distinguishes improvement from verifier gaming?** Train a proposer against an
   *exposed* probe-gated membrane, then evaluate with exact mechanics plus a *hidden, independently
   trained* probe never used for selection. Divergence between exposed-gate pass rate and
   hidden-channel/oracle assessment is the gaming measurement. Full design: cross evidence type
   (claim / exact checker / exposed frozen probe / hidden rotating probe / dual-channel) ×
   rejection action (identical-fallback negative control / state-conditioned distinct fallback) ×
   adaptation (frozen proposer / repeatedly expert-iterated proposer), on matched episodes, seeds, and
   under-trained round-0 checkpoints with non-enumerable or region-held-out splits so probes do not
   start saturated. Primary outcome is oracle-evaluated behavior, never verifier score. Report: false
   provenance claims; verifier disagreement; action-change rate; beneficial/harmful/neutral
   interventions; accepted-vs-rejected proposal utility; fallback coverage and failure rate; probe
   AUROC and direction drift per round; hidden-audit-verifier performance; OOD/adversarial splits.
   Causal probe checks use a distribution of matched-norm random and orthogonal directions, tested
   across seeds, states, rounds, and alternative probe fits.
4. **Can the low-width veto membrane be capacity-bounded?** Channel width is the wrong bound. A binary
   gate emits ≤ 1 bit per decision but routes between complete policies, so closed-loop authority is
   governed by the induced trajectory divergence, not signaling bandwidth. Cleaner formal objects: the
   KL or total-variation divergence between gated and ungated trajectory distributions, alongside a
   value-difference (regret-style) quantity

   ```math
   \mathbb{E}\!\left[\sum_t \gamma^t\,
   \mathbf{1}\{\text{gate changes action at } t\}\,
   \bigl|Q_A(s_t) - Q_T(s_t)\bigr|\right]
   ```

   Once the fallback has its own state-conditioned action space, the membrane is a policy router, not a
   low-capacity actuator, and must be analyzed as one.

## Reconciled Bottom Line

Useful constitutional arbitration requires semantically valid and adaptively robust evidence that
cannot collapse into the proposer's self-report, a demonstrably better counterfactual fallback on the
rejected support, receipt integrity, and stable composition across the controller mesh. Mechinterp
earns its place as instrumentation and triage for that system — not as the constitution, and not as
its proof.
