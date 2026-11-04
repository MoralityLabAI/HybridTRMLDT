# Dual-LDT-01: Corrigible Beast / preference-sorting Jinn

**Status:** formal core, frozen-head synthetic tag-menu diagnostic. Mathematical contracts are conditional on host-owned authority, independent evidence, and enforced execution. The seed-0 heads are actual fixed paper checkpoints; the menu reuses the paper's in-distribution route-tag patterns. No new language-model behavior or independently sealed test is claimed.

## Beast = trusted interior operator

For action universe A and fixed host-certified set S, define

\[
\mathcal{B}_S(X)=X\cap S.
\]

Its four laws are contractiveness, idempotence, monotonicity in candidate sets, and preservation of meet. For an authorized policy tightening S' subset S,

\[
\mathcal B_{S'}(\mathcal B_S(X))=\mathcal B_{S'}(X).
\]

These are proved in formal/CCA/DualLDT.lean, which also states that an action selected from this filtered set belongs to S.

The reference runtime uses BeastSnapshot with a host-supplied epoch, independently reported authority evidence, a set of possible semantic interpretations for each action, and exact forbidden tags. Four states are tracked separately:

- AUTHORIZED: positive authority, no negative authority, nonempty semantic interpretations and every interpretation excludes prohibited tags.
- HOLD: no positive authority, conflicting authority evidence, unknown semantics, or mixed safe/forbidden semantic interpretations.
- REJECT: explicit negative authority without conflicting positive authority, or every represented interpretation contains a forbidden tag.
- HALTED: a host-controlled intervention making the executable set empty.

The live snapshot digest binds a proposed action to policy, evidence and epoch. At execution the host must recompute the authorization and independently check the receipt. Snapshot hashes are version bindings, not signatures or proof that the caller is authorized. Production must authenticate the host channel and control external side effects.

Corrigibility extends beyond this local contract: future ability to intervene and shutdown reachability require an environment game, and action meanings require independently trustworthy extraction. The local result is formal responsiveness to changes in a host-owned authorization snapshot.

## Jinn = learned preference distribution conditional on authorization

Load an original train/hold/reject additive LDT head as a scoring function. For audited tags z_a define

\[
u_\theta(a)=\log\Pr_\theta(\mathrm{train}\mid z_a),
\quad
p_{\theta,\lambda}(a)\propto q(a)e^{\lambda u_\theta(a)}.
\]

The log probability uses all three learned lane logits with numerically stable log-sum-exp. This is a declared preference interpretation of the old head; the original scorer was trained for lane classification, not a calibrated pairwise moral utility.

For nonempty certified set S, define Jinn sorting by

\[
\pi_{\theta,\lambda}^S(a)=\mathbf 1[a\in S]\,p_{\theta,\lambda}(a)/p_{\theta,\lambda}(S).
\]

This is the KL minimizer over distributions supported on S (the I-projection in D_KL(mu||p)). It has two useful properties:

1. Tightening S to S' subset S preserves the odds between any two actions still allowed, provided head, reference q and observed tags remain fixed.
2. The constitutional preference tax is -log p(S), the exact minimum KL distortion caused by restricting support. Conditioning on fixed S gives nondecreasing expected u_theta with increased lambda.

Multiple safe semantic interpretations are scored with the smallest log P(train), an explicit conservative sorting convention. Semantic ambiguity including a forbidden interpretation is held by the Beast independently of learned score.

## Historical LDT comparison and limitations

The original Phase B character heads kept an action if its top lane was not reject; the Beast used a pinned v2 policy and exact hard critical tags. The Dual-LDT diagnostic holds the hard five-tag boundary fixed, compares original binary character membership with a soft distribution over every certified action, and shows the cost of converting a lane classifier into a sorter.

Four actual seed-0 AdditiveCandidateHead checkpoints copied from MoralityLabAI/jinn-or-beast-paper/experiments/character_ldt_heads_v1/outputs/canonical/models/ are included as JSON fixtures. Original Git blob SHAs are asserted in the runner before use. The test menu uses eight fixed T/P/R/L a/b route tag bundles from individuated_jinn_v1/src/lanes.py, plus one action bearing fabricate_tool_result. It is a synthetic in-distribution diagnostic. The historical multi-agent council, v2 soft Beast scorer, held-out outcomes, generated text and interpreter trust failures are outside this test.

The experiment measures allowed-action equality across personas, hard-critical survivors, character preference tax and choice distribution, differential exclusion by the historical-style binary filter, policy-correction odds invariance and halting. It records pairwise total variation between sorted persona policies, but cannot establish human-recognizable persona differentiation.

A useful next stage is a prospective task family with independently audited action semantics and human updates to the constitution. Evaluate false positive cost, preference loss, and intervention compliance separately.

Implementation: research_gym/core/dual_ldt.py.
Tests: tests/test_dual_ldt.py.
Registration: experiments/dual_ldt_01/REGISTRATION.md.
Runner: python experiments/dual_ldt_01/src/run.py.
Lean: lean formal/CCA/DualLDT.lean.
