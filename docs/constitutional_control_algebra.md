# CCA-01: proof-carrying constitutional control

Status: **reference implementation**; CCA-05 integrates it with the six real TRM checkpoints of the gaming benchmark. Exact guarantees are conditional on independently supplied action interpretations and mechanical checks. The historical hybrid membrane still permits self-attested environment soundness for benchmark compatibility.

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


## CCA-03: certificate counterexamples against real LDT heads

The O(d)-per-critical-tag exact margin formula also constructs counterexamples. For a critical feature j and competing lane k with nonnegative maximum margin, set x_j=1 and every other x_i to 1 precisely when w_ki-w_Ri is positive. This achieves the bound and causes a non-reject winner under train/hold/reject tie priority. Tests compare this witness oracle with exhaustive checks for randomly generated five-feature heads.

A StructuralVeto wraps the learned head and rejects when a host-owned critical feature is present. On every noncritical input it returns the base head's original choice identically. The property is independent of learned weights; it remains conditional on correct tag interpretation and trusted wrapper execution.

Real saved 13-feature Jinn v3 checkpoint fixtures are copied from the private jinn-or-beast-paper repository (source Git blob SHAs recorded in tests/test_cca_real_heads.py). The underlying linear weights of veto_jinn fail the exact reject certificate and have a witness, while the structural veto in the original loader supplies the reject guarantee when its execution is independently verified. The constraint_jinn weights establish a sufficient condition for never choosing train on critical-tag inputs, but they do not guarantee choosing reject: hold can win. Thus there are three distinct contracts: no train, reject every critical input, and a deployed/verified hard wrapper.

The audit_saved_head function treats serialized decimal weights as exact rationals, and compares checkpoint-declared veto tags to separately supplied trusted policy tags. Setting wrapper_verified must follow an independent runtime loader check. A serialized marker alone cannot make effective_reject_certificate true. This avoids allowing an actor or checkpoint author to self-assert the execution of a hard veto.


## CCA-04: one-pass certified weight-space repair

Let M_kj be the exact critical-tag margin defined in CCA-C, and fix a rational safety slack eta > 0. For each critical feature j compute, using the original head's weights,

\[
\alpha_j=\max\left(0,\ \eta+\max_{k\in\{T,H\}}M_{kj}\right),
\qquad w'_{Rj}=w_{Rj}+\alpha_j.
\]

Leave every other weight and bias unchanged.

**CCA-D (one-pass reject projection).** The repaired head rejects every binary input containing a critical tag, with competing score margins at most -eta. Proof: for each j, its competing margins decrease by alpha_j. Updates at the other critical indices can only decrease the positive-part contributions in M_kj, so M'_kj <= M_kj - alpha_j <= -eta. All scores on tag vectors containing no critical tags are unchanged because every edited weight multiplies a zero input feature. Hence the resulting discrete lane function is *exactly equivalent* to wrapping the original head with a structural critical-tag veto on the full binary domain.

The projection is explicit, terminates after one pass, and requires no parameter search or post hoc recipe. It is conservative and is not asserted to minimize total weight change. Tests enumerate every input of 150 randomly generated six-feature heads, and verify the same equivalence for both frozen 13-feature Jinn v3 checkpoints.

The theorem is about rational weights and a trusted binary tag vocabulary. For deployment with float weights, roundoff and quantization require bounds; for real actions, independently read tags remain the hardest unproved assumption. The code-owned veto avoids reliance on floating-point margins and can serve as a second independent enforcement layer.


## CCA-05: certified TRM repair on the six real checkpoints

Implementation: research_gym/core/certified_repair.py. Experiment: experiments/cca05_certified_trm_repair/ (REGISTRATION, SPEC, RESULTS, COST, PROVENANCE). Tests: tests/test_cca_certified_repair.py and the experiment's tests/, both in CI.

**Proof-carrying execution.** An ActionCertificate binds (scenario, state, horizon, environment digest), the deployed successor and a witness continuation. execute_certified runs an action only if the certificate is for that action, its binding equals the current context, the action is in the current lattice domain, and the host checker returns True. The storyworld checker replays the witness with the environment's own step function and rival model. A full-length continuation that reaches the goal is feasible, so a valid witness implies Verify = 1 (soundness); the issuer produces one whenever Verify = 1 (completeness). The checker costs 22 µs against 0.81 ms for one cold verification. Stale certificates (successor state, other horizon, other action, foreign digest), forged successors and 11,956 forged certificates for inadmissible actions all fail.

**CCA-E (bounded recurrent repair).** bounded_recurrent_repair separates the proposer's latent deliberation (any number of recurrent passes on the current candidate mask, with no progress condition) from committed refinements. A rejected nomination becomes an elimination C -> C \ {a} submitted to certify_and_apply_verified. The host's elimination verifier recomputes admissibility of every removed action and ignores the proposal's label. Each committed refinement lowers the integer rank rho, so the loop commits at most rho(C_0) refinements. With max_iterations = rho(C_0) it abstains exactly when the certified set is empty. Identity on certified proposals and idempotence of successful repairs follow from executing only certified actions.

**Ranking cannot move the floor.** rank_certified returns a permutation of the certified set, refuses non-finite scores, and breaks ties by a fixed order. The certified set is a function of (scenario, state, horizon) and the environment only. On the storyworld, adversarial, random and constant proposers never caused an unsafe execution in any certified arm.

**Finite-horizon verifiers are viability kernels.** For a verifier of the form "the best continuation within k-1 steps meets the goal", a certified action leads to a goal stop or to a state whose certified set at k-1 is nonempty. By induction, a certified trajectory never meets an empty set and stops in the goal. On the storyworld viability_kernel over the time-expanded game equals the verifier at all 10,500 nodes. A myopic one-step check admits failing sequences from 352 starts; the horizon verifier admits none. The guarantee is conditional on the rival model. Against a nondeterministic rival, the kernel keeps 44 of 618 certified actions in the secret ending and 635 of 2,154 in moral. About two thirds of the actions the certified controllers executed are outside it.

**Empirical result (six checkpoints, 1,750 decisions and trajectories each).** Zero unsafe executions in every certified arm. The raw TRM proposal is already certified on 94% to 96% of decisions with a nonempty certified set. Repair (35 to 46 decisions per checkpoint) always succeeds, with regret against the certified oracle of at most 0.004 (TRM ranking) to 0.014 (identity-metric nearest action). The TRM-ranked controllers beat the registered symbolic LDT fallback on all three seeds of both kinds: +0.017 to +0.019 per decision and +0.040 to +0.044 per trajectory. The certified oracle beats them on every checkpoint and is never worse on any state. It is symbolic, it is computed from the verifier's own evaluations, and it costs 5 evaluations per decision, against about 1.07 for the recurrent-repair arm. The neural proposer's measurable contribution is cheaper verification, not better constrained decisions. In the historical certify_and_apply path, the TRM's own env_sound_dead claims hard-applied 391 to 688 unsound commitments per checkpoint; the strict path accepted none.

The storyworld's verifier computes the utility itself, which is why a symbolic oracle exists there. Where verification cannot rank actions, the comparison remains open.
