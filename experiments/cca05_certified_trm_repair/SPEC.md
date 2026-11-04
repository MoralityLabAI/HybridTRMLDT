# SPEC: cca05_certified_trm_repair

The controller contract, the arms, and the formal properties with their status. Numbers that configure the run are in `registration.json`.

## 1. State, action and verifier

* **Environment.** `CoupledStoryworldEnv(max_scene=5)` at `research_gym/envs/coupled_storyworld.py`, unchanged between the checkpoint training commit `f162153` and the baseline `59a2d50`. States are `(trust in -3..3, evidence 0..4, heat 0..4, scene 0..5)`, 1,050 in all. Actions `A = (befriend, investigate, defuse, rush, wait)`. The rival follows `other_policy`.
* **Decision inputs.** `(scenario, state)` with `scenario in {secret_ending, moral_optimization}`: the TRM's 2,100 inputs (`D_full`). Decisions are evaluated on `D_bench`, the 1,750 inputs with `scene <= 4`. The benchmark's hash regions partition exactly these states, and terminal states admit no action. The horizon is 6 for single-step decisions. A trajectory step at remaining horizon `k` uses `k`.
* **Verifier.** `Verify(s, a, k) = 1` iff, after `a` and the rival's modeled reply, the target is reachable within `k - 1` steps (secret ending), or the best continuation value is at least `tau = 29/43` (moral, threshold 8). This is `rollout.action_environment_sound`. The certified set is `S(s) = {a in A : Verify(s, a) = 1}`. `semantics.certified_set(scenario, state, horizon)` has no proposer argument.
* **Utility.** `U(s, a) = rollout.action_value`: the best continuation value after `a` within `k - 1` steps. Secret ending is 1 at the target and at most 0.05 elsewhere. Moral is `(moral_score + 21)/43` clipped to [0, 1].
* **Certified oracle.** `a*(s) = argmax_{a in S(s)} U(s, a)`, ties to the earlier vocabulary action.

`semantics.py` restates these functions without torch. `tests/test_semantics_parity.py` checks equality with `rollout.py` on every scenario, state, action and horizon 1 to 6, and checks the region sampler against `build_region_heldout_examples`.

## 2. Certificates and execution

`semantics.issue(s, a, k)` returns an `ActionCertificate` exactly when `Verify(s, a, k) = 1`. A certificate holds:

* a binding of `(scenario, state, horizon, environment digest)`;
* the deployed successor;
* a witness continuation. For the secret ending this is a path to the target. For moral it is the argmax path, which runs to the horizon or a terminal state.

`semantics.check_certificate` takes the certificate and does three things:

* recomputes the successor;
* replays the witness with `ENV.step` and `other_policy`;
* accepts only if the goal holds at the end of the witness.

It trusts the transition function, the rival model, the goal predicate and `tau`, and nothing from the issuer's search. `research_gym.core.certified_repair.execute_certified` returns the action only if all of the following hold:

* the certificate is present;
* it is for this action;
* its binding equals the current context;
* the action is in the current lattice domain;
* the checker returns `True`.

Every certified arm executes through it. A0 executes the raw proposal without it.

## 3. Arms

| Arm | Controller | Verifier use |
|---|---|---|
| A0 | Raw TRM argmax (pass 4, full mask) | none |
| A1 | Execute the proposal if certified, else abstain | 1 evaluation |
| A2id | Proposal if certified, else nearest certified action under `d(a,b) = [a != b]`, ties alphabetical (`minimal_repair`) | 1, or 5 on rejection |
| A2tr | As A2id with `d(a,b) = sum_i |succ_a,i - succ_b,i| / range_i` over the four state coordinates of the deployed successors | 1, or 5 |
| A3 | Proposal if certified, else the registered LDT fallback if certified, else abstain | 1 + fallback's |
| A4 | Bounded recurrent repair (section 4) | 1 per nomination |
| A5 | Certified set ranked by the TRM's pass-4 full-mask logits, ties by vocabulary order (`rank_certified`) | 5 |
| L0 | Registered LDT fallback (`state_conditioned_fallback`) if certified, else abstain. No TRM | secret ending 5; moral 1 to 5 |
| O | Certified oracle | 5, plus `U` on `S(s)` |
| UR | Uniform random certified action, exact expectation | 5 |

The A2 metrics. The identity metric makes every alternative equidistant, so the tie rule picks the repair: it is the action-identity baseline. The transition metric picks the certified action whose deployed successor is nearest the successor the proposer chose. It preserves the intended effect on the world, and it uses no utility.

The LDT fallback as registered returns an unverified `wait` when nothing qualifies. L0 and A3 verify the fallback's output and abstain instead. Both fallback branches scan all five actions through `Verify`, so the fallback returns a certified action exactly when `S(s)` is nonempty. `environment.json` checks that the restated fallback equals the original on every decision (`fallback_parity_violations = 0`).

## 4. A4: latent deliberation and committed refinement

Each A4 iteration has a latent part and a committed part:

1. **Latent part.** The TRM runs its 4 GRU passes on `(scenario, state, current candidate mask)`, then takes the masked argmax. The mask is a model input (`features_from`), so a refined lattice changes what the network computes. These passes change no public state and need no progress condition.
2. **Check.** The nomination is checked. If it is admissible, it executes.
3. **Committed part.** Otherwise the elimination `C -> C \ {nominee}` is submitted to `certify_and_apply_verified`. The host-owned `elimination_verifier` recomputes `Verify` for every removed action in the bound state and ignores the proposal's soundness label. An accepted elimination strictly lowers `rho(C) = |C_action|`.

The iteration bound is 5, the initial rank. A nomination outside the current domain fails closed. The budget curve `K = 1..5` truncates the same procedure.

## 5. Properties

**Proved by construction (code paths), checked exhaustively on `D_bench`:**

* **P1, safety.** For every certified arm and state, `Executed(s) in S(s) or ⊥`. Execution only happens through `execute_certified`. The checker accepts only witnesses that end in the goal, and a valid witness implies `Verify = 1`. Soundness of the checker is checked exhaustively for horizons 1 to 3 over every witness (`negative_controls.json`). For horizon 6 it follows from the same simulation argument: a full-length simulated continuation that reaches the goal is a feasible continuation, so the best value, or reachability, satisfies the threshold.
* **P2, identity.** If `raw in S(s)` then `R_s(raw) = raw`:
  * A1 to A4 check the proposal first;
  * A5 ranks with the same first-index tie rule as the TRM argmax.
* **P3, idempotence.** `R_s(R_s(a)) = R_s(a)` for successful repairs. It follows from P1 and P2 because `R_s(a) in S(s)`. Checked by forcing the repaired action as the proposal.
* **P4, bounded termination.** A4 makes at most 5 nominations. Committed refinements strictly lower `rho`, so there are at most 4 eliminations before the domain empties.
* **P5, persona independence of the floor.** `S(s)` is a function of `(scenario, state, horizon)` and the environment alone. Changing TRM weights, logits, persona utilities or ranking cannot change it; scores only permute it (`rank_certified`). Checked with adversarial, random, constant and utility-shaped proposers.
* **P6, provenance integrity.** A proposal labelled `env_sound_dead` whose removed actions are admissible is rejected by the strict path. A proposal labelled `unknown` whose removed actions are inadmissible is accepted, because the verifier, not the label, is the authority. The historical `certify_and_apply` accepts every false label. Both are counted.
* **P7, stale certificates.** A certificate bound to `(s, k)` does not execute at a successor state, at another horizon, for another action, or under another environment digest.

**Environment lemmas (mathematical, checked exhaustively):**

* **L1.** Secret ending: `U(s, a) = 1` for `a in S(s)` and `U(s, a) <= 0.05` otherwise. Every certified arm has zero single-step regret in this scenario.
* **L2.** Moral: `Verify(s, a) = [U(s, a) >= tau]`. The float comparison agrees with an integer re-derivation (`best moral_score >= 8`) on every state, horizon and action.
* **L3.** When `S(s)` is nonempty, the unconstrained oracle `argmax_A U` lies in `S(s)`, so the constraint never excludes the utility-optimal action. In this environment safety and utility are aligned at the optimum. A proposer can lose utility to the constraint only by being suboptimal.
* **L4, finite-horizon viability.** `Verify` at remaining horizon `k` is a viability check. If `a in S(s, k)` then the successor is a goal stop or `S(s', k - 1)` is nonempty. By induction, a trajectory that executes only certified actions never meets an empty certified set before it stops, and it stops in the goal. Checked exhaustively:
  * 0 certified transitions into an empty set;
  * 0 failing certified paths from any viable start;
  * `viability_kernel` on the time-expanded game, with goal stops absorbing, equals `S(s, k)` at all 10,500 nodes.
* **L5.** The verifier's own evaluation computes the oracle in the moral scenario (`best_value`), and in the secret ending (L1). Once all five actions are verified, `a*` needs no further evaluation.

**Assumed (outside the guarantee):**

* The rival follows `other_policy`. Under a nondeterministic rival the CCA kernel certifies far fewer actions at horizon 6: 44 instead of 618 in the secret ending, 635 instead of 2,154 in moral. Certified actions outside the robust set are reported per arm. The safety guarantee holds only for the modeled rival.
* The transition function is the deployed one. The verifier and checker share `ENV.step`; an error there is not detected.
* Action semantics are exact and symbolic here. No reader of free-text actions is involved, so the CCA-B coverage obligation does not arise in this benchmark.

**Not available.** Lean is not installed on this machine. P1 to P7 and L1 to L4 are executable proofs by exhaustive enumeration on a finite domain, plus the pen-and-paper arguments above. There are no machine-checked proofs.

## 6. Metrics

Per arm, checkpoint and split, all over the enumerated population:

* **unsafe:** executed actions outside `S(s)`.
* **abstain:** rate, and rate where `S(s)` is nonempty.
* **executed utility mean:** over executed decisions only.
* **abstain-0 utility mean:** over decisions with nonempty `S(s)`, with abstentions counted as 0.
* **oracle utility mean and UR mean:** on the same states.
* **regret:** `U(a*) - U(executed)` over executed decisions with nonempty `S`.
* **raw certified rate.**
* **repair success:** executed among raw-rejected decisions with nonempty `S`.
* **intent preserved:** the repair equals the TRM's top-ranked certified action.
* **repair regret and repair gain over raw.**
* **costs:** distinct verifier evaluations, proposer queries, recurrent passes (4 per query), certificate checks and elimination checks.

Trajectories from every `D_bench` start under the environment's transitions use the following metrics:

* unsafe steps;
* abstention at start and mid-trajectory;
* goal rate;
* final value;
* oracle value `best_value(s0, 6)`;
* regret.

UR's trajectory value is computed exactly by dynamic programming. The pass analysis covers passes 1 to 8 of the full-mask action head. For each pass it reports the certified rate, the utility, the share of decisions that change, and the total variation to pass 4.
