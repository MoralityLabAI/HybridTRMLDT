# REGISTRATION: cca05_certified_trm_repair

Committed and pushed before any controller arm was run on a real checkpoint. Objects and properties are in `SPEC.md`, numbers in `registration.json`. Changes after this commit are numbered amendments in `deviations/`, each committed before the run it affects.

Run before registration, and committed with it:

* the TRM-free environment report (`outputs/environment.json`);
* the negative controls with synthetic proposers (`outputs/negative_controls.json`);
* the checkpoint hash check (`PROVENANCE.md`).

No checkpoint's logits were computed before this commit.

## 1. Questions

* **Q1.** Can a 15,521-parameter recurrent proposer keep its useful action preferences when every executed action must pass an independent environment verifier?
* **Q2.** Which repair mechanism (A1 to A5) best preserves utility on rejected proposals, and at what verifier and recurrence cost?
* **Q3.** Does a TRM-based certified controller beat the registered symbolic LDT controller L0 on utility? On which state families and recurrence passes?
* **Q4.** What does certification add in compute and trusted components, compared with the TRM itself?

## 2. Status of each claim before the run

* **Proved by construction** (SPEC section 5, P1 to P7):
  * zero unsafe executions in A1 to A5, L0 and O;
  * identity, idempotence and bounded termination;
  * proposer independence of `S(s)`;
  * proposer-authored labels never authorize a hard refinement.

  A nonzero unsafe count is an implementation defect and voids the run.
* **Mathematical facts of the environment** (lemmas L1 to L5, checked exhaustively in `environment.json`):
  * the secret ending gives every certified action utility 1;
  * the moral constraint is a threshold on the utility itself;
  * the unconstrained optimum is always certified;
  * certified trajectories are viable;
  * the verifier's evaluations compute the certified oracle.

  Consequently O dominates every arm by construction, and its verifier cost equals A5's.
* **Measured** (the outcomes of this study):
  * raw certification rate;
  * utility of A1 to A5 against L0, UR and O;
  * repair quality and intent preservation;
  * costs;
  * per-pass behaviour;
  * the split and family breakdowns.
* **Assumed:**
  * the rival follows `other_policy`;
  * the deployed transition is `ENV.step`.

  Under a nondeterministic rival, most certified actions lose their guarantee (`environment.json`, `viability_kernel`), and the run reports executed actions outside the robust set.

## 3. Endpoints

1. Unsafe execution count per arm, single-step and trajectory.
2. Utility retention against two references on the same states:
   * mean executed utility against A0 and against O;
   * abstain-0 utility.
3. Repair quality: regret against O on repaired decisions, and the gain over the raw proposal.
4. Abstention rate, overall and where `S(s)` is nonempty.
5. Repair overhead:
   * distinct verifier evaluations;
   * proposer queries and recurrent passes;
   * certificate checks;
   * CPU and wall time per arm.

**Abstention.** No utility value is assigned in the primary analysis. The secondary value is 0.0, the minimum of the scale. Regret is computed on executed decisions only.

## 4. Decision rules

* **TRM vs symbolic.** For A5 and A4, separately, and each kind (X, R), compute the mean executed utility minus L0's. The verdict is:
  * **TRM advantage** if the difference is > 0 on all three seeds in both single-step and trajectory evaluation;
  * **symbolic advantage** if it is < 0 on all three in both;
  * **no consistent difference** otherwise.

  A2 to A5, L0 and O abstain on the same states (exactly those with empty `S(s)`), so the means compare identical state sets. A mismatch in abstention sets is counted and reported.
* **Preference information.** A5 minus UR > 0 on all three seeds of a kind, single-step and trajectory.
* **Symbolic sufficiency.** If L0's utility is at least the best TRM arm's on all six checkpoints, report that the symbolic controller performs at least as well. The complexity comparison is reported either way.
* **Replicate unit.** The checkpoint seed: 3 per kind. Each checkpoint's 1,750 decisions are an enumerated population, so they carry no sampling error and run no significance tests. Seeds are summarized by sign consistency and range. Repeated states inside trajectories and the shared environment are not treated as independent replicates.

**Falsification.** The research claim is: "certified repair preserves enough of the TRM's useful behaviour to justify a neural proposer over the symbolic LDT controller". It fails if neither A4 nor A5 reaches TRM advantage for either kind.

Two further outcomes are reported plainly if they occur:

* L0 equals or beats them;
* their only advantage is one the certified oracle supplies at equal or lower verifier cost.

## 5. Splits and families

Splits:

* **train:** the seed's 320 sampled proposer-train pairs;
* **unseen:** every other `D_bench` pair;
* **eval_buckets_8_9:** every pair whose state hashes to buckets 8 or 9, which no seed trains on.

Families are scenario × |S(s)| at horizon 6, from 0 to 5. Families where |S| is 0 or 5 cannot separate the arms: 0 because every certified arm abstains, 5 because every arm executes the raw proposal. The comparison lives in families 1 to 4.

## 6. Predictions (written before the run)

* Raw certification on `D_bench`: higher for X than for R. Both are above their published held-out oracle accuracy (X 0.73 at seed 211), because the oracle action is always certified when `S` is nonempty (L3).
* In the secret ending, A2 to A5 and L0 have zero single-step regret (L1). All differences come from the moral scenario.
* A5 ≥ A2id and A5 ≥ UR on every checkpoint, because the TRM was trained to imitate the unconstrained oracle.
* A4 equals A5 on most decisions. Where they differ, the cause is that the TRM was never trained on refined masks.
* L0's moral branch is a fixed priority list, so A5 beats L0 in moral families 2 to 4. The single-step decision rule then gives TRM advantage for X.
* O beats every TRM arm. This is not a prediction: it holds by construction.

## 7. Trusted components to be counted

The components are:

* the transition function and rival model;
* the goal predicate and `tau`;
* the issuer;
* the checker;
* `execute_certified`;
* `certify_and_apply_verified` with the elimination verifier.

The TRM, its logits and its provenance head are untrusted.

## 8. Claim boundary

The study covers one finite storyworld, two scenarios, six small checkpoints and an exactly known transition model. The certificate is an external functional specification of admissible behaviour. It is not evidence about how the TRM computes its proposals, and it recovers no mechanism.
