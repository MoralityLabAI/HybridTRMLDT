# Results: cca05_certified_trm_repair

Registration: commit `9d3cd7c`, pushed before any arm ran on a real checkpoint. Amendment 001 (`6b94f21`) was committed before the rerun it governs. It fixed the A0 repair-count definition, and it added the nondeterministic-rival count that SPEC registered but the first run did not compute. No decision-rule input changed: the summary of the first run and of the rerun are identical.

All six registered checkpoints (TRM X and R, seeds 211, 223, 227) were loaded with both hashes equal to the registration. Each reproduces the published exact final-pass action tree of `complete_accounts_v1` on all 2,100 inputs (agreement 1.0, `published_tree_parity`).

The population is the 1,750 decisions of `D_bench` at horizon 6, and 1,750 trajectories from the same starts. 957 of the 1,750 inputs have an empty certified set (secret ending 667, moral 290). There every certified arm abstains and no executing controller can be safe. The 793 inputs with a nonempty set are the comparison domain. Ranges below run over the six checkpoints. L0, O and UR use no TRM, so they have one value.

## Verdicts under the registered rules

| Rule | X | R |
|---|---|---|
| Safety: unsafe executions, A1 to A5, L0, O | 0 (all six checkpoints, single-step and trajectory) | 0 |
| A5 vs L0 | TRM advantage (single-step +0.0174, +0.0177, +0.0174; trajectory +0.0396, +0.0417, +0.0409) | TRM advantage (+0.0187, +0.0192, +0.0175; +0.0429, +0.0439, +0.0428) |
| A4 vs L0 | TRM advantage (+0.0174, +0.0177, +0.0172; +0.0396, +0.0416, +0.0408) | TRM advantage (+0.0184, +0.0187, +0.0175; +0.0430, +0.0436, +0.0428) |
| Preference information, A5 vs UR | informative (+0.0170 to +0.0173; +0.0408 to +0.0429) | informative (+0.0171 to +0.0187; +0.0440 to +0.0451) |
| Symbolic sufficiency | L0 below the best TRM arm on all six; O at least the best TRM arm on all six | same |

The registered research claim survives against L0. The falsification clause is not met: A4 and A5 reach TRM advantage for both kinds. The registration flagged one reportable condition in advance, and it is present: the certified oracle O beats every TRM arm, using the same verifier evaluations as A5.

## Single-step utility (793 decisions with nonempty S)

| Arm | Executed utility | Regret vs O | Verifier evaluations per executed decision |
|---|---:|---:|---:|
| A0 raw TRM (executes uncertified actions) | 0.808 to 0.820 | — | 0 |
| A1 reject and abstain (747 to 758 executed) | 0.8417 to 0.8435 | 0.0015 to 0.0033 | 1.00 |
| A2id nearest certified, identity metric | 0.8431 to 0.8446 | 0.0020 to 0.0035 | 1.18 to 1.23 |
| A2tr nearest certified, transition metric | 0.8432 to 0.8446 | 0.0020 to 0.0034 | 1.18 to 1.23 |
| A3 TRM, then certified LDT fallback | 0.8432 to 0.8447 | 0.0019 to 0.0034 | 1.13 to 1.19 |
| A4 bounded recurrent repair | 0.8432 to 0.8446 | 0.0020 to 0.0034 | 1.06 to 1.09 |
| A5 TRM-ranked certified set | 0.8433 to 0.8451 | 0.0015 to 0.0033 | 5 |
| L0 symbolic LDT fallback | 0.8259 | 0.0206 | 2.32 |
| UR uniform certified (exact expectation) | 0.8264 | 0.0202 | 5 |
| O certified oracle | 0.8466 | 0 | 5 |

The A0 row is the raw proposal's utility on the same 793 decisions. It also executes uncertified actions on 992 to 1,003 of the 1,750, including all 957 with an empty set. Certification therefore raises utility over the raw TRM on the decisions where it can act: utility retention exceeds 1.0.

A5 reaches 99.6% to 99.8% of O. Measured as gap closure, (A5 − L0)/(O − L0), it closes 84% to 93% of L0's gap to the oracle. L0 is no better than a uniform random choice among certified actions (0.8259 against 0.8264).

## Trajectories (793 viable starts, horizon 6, environment transitions)

| Arm | Final value (completed) | Goal met | Verifier evaluations per viable trajectory |
|---|---:|---:|---:|
| A0 | 0.684 to 0.738 | 0.80 to 0.86; 1,065 to 1,112 of 1,750 trajectories with an unsafe step | 0 |
| A1 | 0.815 to 0.828; abstains mid-trajectory on 64 to 109 viable starts (0.656 to 0.711 with abstention = 0) | 1.0 | |
| A2id, A2tr, A3 | 0.8337 to 0.8378 | 1.0 | |
| A4 | 0.8338 to 0.8378 | 1.0 | 3.9 to 4.2 |
| A5 | 0.8338 to 0.8381 | 1.0 | 17.4 to 18.0 |
| L0 | 0.7942 | 1.0 | 8.7 |
| UR | 0.7930 | | |
| O | 0.8466 | 1.0 | 17.8 |

Every certified arm met the goal on every viable start that it did not abstain on. This is lemma L4, observed with zero violations. On trajectories A5 beats L0 on 319 to 339 of the 793 starts and loses on 4 to 26. O beats A5 on 109 to 161 starts and is never worse.

A1's mid-trajectory abstentions show the TRM proposing more uncertified actions at the shorter remaining horizons it was never trained on (training used horizon 6 only). Repair arms absorb these without abstaining.

## Repair on the constrained states

The raw TRM proposal is certified on 94.2% to 95.6% of the 793 decisions. Repair is needed on 35 to 46 decisions per checkpoint. They lie in the tightly constrained families:

* secret ending with |S| = 1 to 3;
* moral with |S| = 1 to 4, most with |S| = 1.

On these decisions:

| Arm | Repair success | Regret vs O | Same action as the TRM's top-ranked certified choice |
|---|---:|---:|---:|
| A2id | 1.0 | 0.0047 to 0.0137 | 0.74 to 0.89 |
| A2tr | 1.0 | 0.0047 to 0.0101 | 0.58 to 0.77 |
| A3 | 1.0 | 0.0047 to 0.0106 | 0.70 to 0.86 |
| A4 | 1.0 | 0 to 0.0093 | 0.82 to 0.98 |
| A5 | 1.0 | 0 to 0.0042 | 1.0 (by construction) |

Every repair raised utility over the rejected proposal by 0.54 to 0.65 on average. The rejected proposals were mostly actions with no continuation to the goal.

The transition metric (A2tr) did not beat the identity metric. It preserved the TRM's preference less often and had regret within the identity metric's range. The registered semantic metric bought nothing here.

**A4 budget.** The fixed budget K matters only on the repair decisions:

| K | Abstentions with nonempty S |
|---|---:|
| 1 | 35 to 46 (A1's behaviour) |
| 2 | 6 to 15 |
| 3 | 2 to 8 |
| 4 | 0 to 3 |
| 5 | 0 |

Committed ranks decreased strictly in every A4 run. A4 and A5 executed different actions on 1 to 8 of the 1,750 decisions. The TRM re-run on a refined mask, which it was never trained on, usually keeps its original ordering.

## Where the TRM's advantage over L0 comes from

The advantage comes from the moral scenario. In the secret ending, every certified arm ties L0 on every decision and trajectory. Every certified action has utility 1 there (lemma L1). Single-step A5 − L0 by moral family (range over the six checkpoints; decision counts at X211):

| Moral family | Mean difference | Decisions better / worse |
|---|---:|---:|
| \|S\| = 1 | 0 | 0 / 0 |
| \|S\| = 2 | +0.009 | 7 / 1 |
| \|S\| = 3 | +0.031 to +0.036 | 49 / 4 |
| \|S\| = 4 | +0.033 to +0.037 | 69 / 4 |
| \|S\| = 5 | +0.021 to +0.026 | 91 / 7 |

On trajectories the |S| = 5 family carries the largest share (+0.089 at X211). The constraint does not bind there at all. The TRM's advantage is therefore a better utility preference among already-safe actions. L0's moral branch is a fixed priority list (`defuse, befriend, investigate, rush, wait`), and repair contributes little: the 35 to 46 repaired decisions are a small share of the 793.

The advantage holds on held-out states:

| Split | A5 − L0 |
|---|---:|
| Buckets 8 and 9 (408 decisions, 158 with nonempty S; never trained on) | +0.0188 to +0.0210 |
| All 1,430 untrained decisions | +0.0162 to +0.0180 |
| Training pairs | +0.0203 to +0.0254 |

## Recurrence passes

The full-mask action head at each pass, on `D_bench`:

* **Pass 1 to 3.** Agreement with the oracle where S is nonempty rises from 0.842–0.889 at pass 1 to 0.899–0.927 at pass 3. The certified share of the 1,750 rises from 0.408–0.426 to 0.428–0.431; the ceiling is 793/1,750 = 0.453.
* **Pass 4.** Pass 3 to 4 is flat or slightly worse in five of six checkpoints (oracle agreement 0.894–0.919 at pass 4). The registered proposer uses pass 4.
* **Changes between passes.** Decisions change on 10% to 22% of inputs between passes 1 and 2, and on 3% to 5% between passes 3 and 4.
* **Passes 5 to 8.** These iterate the GRU past its trained depth. They change 0.3% to 3.5% of decisions per pass and drift slowly away from the oracle (0.868–0.903 at pass 8).

Later passes do not generate more admissible proposals after pass 3. The recurrence adds most of its admissibility and oracle agreement in passes 1 to 3.

## Safety, provenance and the assumption boundary

* **Zero unsafe executions** in A1 to A5, L0 and O on all six checkpoints, in 1,750 single-step decisions and 1,750 trajectories each. A0 executed 992 to 1,003 uncertified actions single-step, and 3,326 to 3,596 uncertified trajectory steps.
* **The TRM's own provenance claims.**

  | Kind | Threshold | Inputs claiming `env_sound_dead` | False (proposal uncertified) |
  |---|---|---:|---:|
  | X | default 0.5 | 420 to 761 | 29 to 92 |
  | R | reconstructed: 0.056, 0.127, 0.062 | 753 to 853 | 126 to 139 |

  Each claim was submitted as a hard refinement that commits to the proposal alone:
  * the historical `certify_and_apply` hard-applied every claimed commitment, 391 to 688 of them unsound because they eliminated admissible alternatives;
  * the strict path with the environment's elimination verifier accepted 1,037 to 1,047 commitments, 0 unsound, and it accepted them regardless of label.
* **TRM-free negative controls** (`negative_controls.json`). Over `D_bench`, the following were accepted or executed zero times:
  * false `env_sound_dead` eliminations (0 of 2,772; the historical path accepted 2,772 of 2,772);
  * eliminations with a missing or inconclusive verifier;
  * widening and swapped refinements;
  * stale-state replays (2,538) and stale-horizon replays;
  * certificates for another action or a foreign environment digest;
  * forged successors;
  * 11,956 forged certificates for inadmissible actions;
  * actions outside the current mask;
  * invalid repair nominations;
  * empty domains.

  Exhaustive witness soundness held on 332,500 witnesses at horizons 1 to 3. Adversarial, random and constant proposers produced zero unsafe executions in every certified arm.
* **Split-step control.** A myopic one-step check admits failing action sequences from 262 of 394 secret-ending starts and from 90 of 273 moral starts. Driven by the TRM's ranking, it executed 1,285 inadmissible steps on 252 trajectories. The horizon verifier admits no failing sequence from any viable start (environment lemma L4). The CCA viability kernel on the time-expanded game equals the verifier at all 10,500 nodes.
* **Rival assumption.** The guarantee is conditional on the rival following `other_policy`. Of the actions the certified arms executed, 498 to 555 per arm and checkpoint (66% to 70% of executed decisions) fall outside the nondeterministic-rival kernel. Against an arbitrary rival, the certificate says nothing about those actions. At horizon 6 the robust kernel certifies:
  * 44 of 618 model-certified actions in the secret ending;
  * 635 of 2,154 in moral.

## Interpretability and size

From `complexity.json` and `COST.md`.

| | Certified controller | TRM |
|---|---|---|
| Trusted code on the execution path | 126 lines: transition, rival model, goal and threshold, checker, executor (plus `moral_score`, 1 line) | none (untrusted) |
| Issuer (untrusted, its output is re-checked) | 70 lines | |
| Repair membrane (A4) | 201 lines including the historical `certify_and_apply` that the strict path reuses | |
| Verified constraints per decision | one, `Verify(s, a, k)` for the executed action, with two parameters (horizon 6, threshold 8) | |
| Certificate | 238 to 305 bytes; witness 0 to 5 actions | |
| Cost of one check | 22 µs (replays at most 5 steps), against 0.81 ms for one cold verification | |
| Parameters | 0 | 15,521; 4 passes per query |
| Exact description of the function | `S(s)` as an exact tree: 278 leaves (81 to 107 per action) | 677 to 834 leaves (X), 371 to 588 (R) for the three final-pass outputs; 229 to 590-state exact automaton (`complete_accounts_v1`) |

The certificate is a functional specification of admissible execution, which a small program checks. It says nothing about how the TRM computes its proposals and is not evidence of a recovered mechanism. Even the exact certified-set table is not a readable rule (278 leaves). What is small is the program that decides it, not the table.

## Predictions against outcomes

| Prediction | Outcome |
|---|---|
| Raw certification higher for X than for R | **Failed.** Equal within 0.014 (0.942–0.945 X, 0.943–0.956 R). Above the published held-out oracle accuracy: held. |
| Zero single-step regret in the secret ending | Held. |
| A5 ≥ A2id and A5 ≥ UR on every checkpoint | Held (A5 − A2id from +0.00006 to +0.0005). |
| A4 = A5 on most decisions | Held (1 to 8 differ). |
| A5 > L0 in moral families 2 to 4; TRM advantage for X | Held, and for R as well. |
| O beats every TRM arm | Held (by construction). |

## Decision question

**Does certified repair preserve enough of the TRM's behaviour to justify a neural proposer over an entirely symbolic LDT controller?**

* **Against the registered LDT controller L0: yes.** TRM advantage holds on all three seeds of both kinds. Certified TRM controllers (A4, A5) gain +0.017 to +0.019 utility per decision and +0.040 to +0.044 per trajectory on the [0, 1] scale. Safety is identical: zero unsafe executions, and the same abstention set.
* **The advantage is preference quality, not repair.** It lives in moral states where L0's fixed priority list chooses a lower-value safe action, including states where the constraint does not bind. The 35 to 46 repaired decisions are repaired at near-oracle quality, and they do not drive the comparison.
* **Against the best symbolic controller: no.** The certified oracle O is symbolic and computable from the verifier's own evaluations (lemma L5). It is better than every TRM arm on every checkpoint: +0.0015 to +0.0033 single-step, +0.0085 to +0.0128 per trajectory. It is never worse on a single state.
* **The TRM's remaining role is cheaper verification.** A4 executes after 1.06 to 1.09 verifier evaluations per decision and about 4 per viable trajectory. O needs 5 per decision and 17.8 per trajectory, L0 2.32 and 8.7. A4 also adds 4 recurrent passes per evaluation (0.37 ms per forward on this CPU). Here one cold verification costs 0.81 ms, so the saving is a few milliseconds per trajectory. The trade is worth it only where verification is much more expensive than a forward pass.

Plainly: in this benchmark a symbolic controller with the verifier's own utility information performs better than the neural proposer at a comparable compute cost. The registered LDT fallback is weaker than both, because it is utility-blind.

## Limitations

* One finite storyworld, two scenarios and an exactly known transition model. The verifier is complete and computes the utility itself (lemmas L1, L2, L5); that is why the symbolic oracle exists. In a domain where verification cannot rank actions, the comparison could differ.
* Six checkpoints, trained only at horizon 6 with the full candidate mask. The replicate unit is the seed (n = 3 per kind), with no significance tests.
* The safety guarantee depends on the modeled rival and on `ENV.step` being the deployed transition. The verifier and checker share that code.
* No machine-checked proofs: Lean is not installed. Properties are exhaustive executable checks on the finite domain plus the arguments in SPEC.
* R's provenance threshold is reconstructed, not recorded (PROVENANCE).
* Utility differences are small in absolute terms: hundredths on a [0, 1] scale.

## Reproduction

```bash
python -I experiments/cca05_certified_trm_repair/src/run_cca05.py environment --out experiments/cca05_certified_trm_repair/outputs
python -I experiments/cca05_certified_trm_repair/src/run_cca05.py controls --out experiments/cca05_certified_trm_repair/outputs
python -I experiments/cca05_certified_trm_repair/src/run_cca05.py live --out experiments/cca05_certified_trm_repair/outputs   # torch, checkpoints
python -I experiments/cca05_certified_trm_repair/src/run_cca05.py replay --out experiments/cca05_certified_trm_repair/outputs # no checkpoints
python -I experiments/cca05_certified_trm_repair/src/summarize.py
python -I experiments/cca05_certified_trm_repair/src/complexity.py
python -m pytest -q tests experiments/cca05_certified_trm_repair/tests
```

Machine-readable outputs:

* `outputs/arms/{tag}.json`: all metrics, splits, families, passes and the budget curve.
* `outputs/receipts/{tag}.json.gz`: every decision and trajectory.
* `outputs/tapes/{tag}.json.gz`: every logit consumed.
* `outputs/summary.json`, `outputs/environment.json`, `outputs/negative_controls.json`, `outputs/complexity.json`, `outputs/replay_check.json`, `outputs/manifest_live.json`.
