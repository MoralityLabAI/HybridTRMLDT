# CCA-01: proof-carrying constitutional control

Status: **reference implementation**, pending integration tests in the full research gym. Exact guarantees are conditional on independently supplied action interpretations and mechanical checks. The historical hybrid membrane still permits self-attested environment soundness for benchmark compatibility.

Implementation: research_gym/core/constitutional_control.py
Tests: tests/test_constitutional_control.py

## Authorization algebra

For an action universe U, independent hard-safe set S, and two persona-acceptable sets J and B, the four allowed sets are:

\[
A_{\mathrm{beast}}=S\cap B,\quad
A_{\mathrm{jinn}}=S\cap J,\quad
A_{\mathrm{meet}}=S\cap J\cap B,\quad
A_{\mathrm{parallax}}=S\cap(J\cup B).
\]

**CCA-A (floor preservation).** For each operation c, A_c is a subset of S. Each expression is an intersection with S. Property tests enumerate all triples of subsets of a four-element universe.

## Independent action interpretation

Given observed action description o(a), an environment-owned reader returns a nonempty uncertainty set Gamma(o) of possible tag sets. For forbidden tags K, authorization requires every possible tag set to be disjoint from K.

**CCA-B (conditional soundness).** If the true tags are in Gamma(o), then authorization implies that no true forbidden tag is present. An empty interpretation set fails closed. Reader correctness, coverage, and resilience to an adaptive attacker remain independent empirical obligations. The proposing model cannot supply its own trusted verifier or reader.

## Exact three-lane LDT condition

For binary input x, train/hold/reject linear scores are g_k=b_k+w_k.x; the tie rule prefers train, then hold, then reject. If critical feature j is set, the maximum margin of lane k over reject is

\[
M_{kj}=b_k-b_R+(w_{kj}-w_{Rj})
+\sum_{i\ne j}\max(0,w_{ki}-w_{Ri}).
\]

**CCA-C (exact veto).** The head rejects every vector containing any critical tag if and only if M_kj is strictly negative for each critical j and k in {train,hold}. Proof: maximize the affine difference by choosing every unconstrained remaining bit independently. The module uses rational arithmetic; a floating-point deployment needs an independent numerical error bound or a structural hard veto. Constrained or correlated tag domains require a different maximization, e.g. SMT.

## Persona preference distribution

Given positive baseline q and utility u on a fixed nonempty certified set A, set pi_lambda(a) proportional to q(a) exp(lambda u(a)), for lambda >= 0. This uniquely solves KL-regularized utility optimization. Its expected utility derivative is Var_pi(u) >= 0. This proves monotonic expected persona utility under the stated conditions, not monotonic individual decisions or downstream village histories. Empty certified sets require abstention.

## Minimal repair

Choose the nearest certified action under a deterministic tie rule. For a metric, repair is safe relative to the certified set, identity-preserving on safe actions, and idempotent; when the set is empty, abstain. The distance metric must have a justified task-level meaning.

## Finite lattice termination

On a finite pointwise powerset lattice C with rank rho(C)=sum_j |C_j|, requiring each successful nonterminal refinement to strictly shrink rho bounds the number of such refinements by rho(C_0). This is a guarantee about committed certified updates, not about arbitrary neural recurrence.

## Next implementation steps

1. Add a strict, independently verified membrane adapter. Leave the historical call behavior unchanged and make the strict path explicit for new control experiments.
2. Add a finite-state viability/shield kernel for multistep actions, with an environment-owned transition model.
3. Verify certificates on the paper's real LDT heads and on the six TRM checkpoints, including adversarial self-attested provenance.
4. Compare fixed-certification LDT, TRM with repair, and the composed controller on identical held-out tasks. Report safety, utility, abstention and compute.

The mathematical development is CPU-only. Silico should test whether an independent semantic reader covers true action interpretations under attack.


## CCA-02: strict adapter and temporal shield

Implementation now includes the optional strict entry point in research_gym/core/hybrid.py, named certify_and_apply_verified. It requires a host-owned provenance_verifier returning env_sound_dead for the exact proposal and context. Missing, unknown, or model-only verification fails closed. It checks a strict rank decrease (unless explicitly disabled), and delegates the existing lattice checks. The historical certify_and_apply path is unchanged for comparability with the published benchmarks. Security-sensitive callers should use the strict entry point and must retain custody of the verifier implementation and context.

For finite safety games, viability_kernel computes the greatest fixed point of the controlled predecessor of safe states. If a safe state offers no action whose nonempty successor set stays inside the candidate winning region, the state is removed. shield_actions returns exactly the actions whose possible successors stay within the resulting region. Missing transitions fail closed. Given a sound and complete transition relation for the scope of the environment, repeatedly choosing such an action preserves safety indefinitely. For incomplete or adversarially manipulated transition relations, this is only a conditional model-based guarantee.

The first reference benchmark is a two-step trap: a tempting first action leads to a state with only a dangerous continuation. The viability kernel eliminates that first action, even though its immediate successor is labeled safe. This addresses the *structural form* of the red-team split-step attack; a semantic action-to-state model is still required to make it applicable to real outputs.

Regression: nine local CCA kernel tests passed before commit; the new strict-path tests and historical membrane tests are included in a dedicated GitHub Actions workflow.
