# CCA-09: Certified anytime search with advisory neural proposals

A host-owned evaluator issues verified feasibility verdicts and exact utilities. A second checker verifies each verdict, including negative verdicts. Independently checked upper-bound certificates cover every unevaluated action. The TRM supplies action **ordering only**.

For E_k, the evaluated actions, and S, the actual admissible actions, the incumbent satisfies

  L_k = max { U(a) : a in E_k intersection S }.

For each action outside E_k let B(a) be a valid upper bound on its potential utility. When an incumbent exists, define

  H_k = max(L_k, max { B(a) : a outside E_k }).

**CCA-L (bounded regret).** With exact checked utilities for evaluated feasible actions, independently checked negative verdicts for evaluated infeasible actions, and sound bounds for unchecked actions,

  L_k <= U* <= H_k, hence U* - U(incumbent) <= H_k - L_k.

**CCA-M (anytime monotonicity).** Increasing the checked set can only improve the incumbent utility and reduce the certified upper bound. The bound gap H_k-L_k cannot increase. A candidate is epsilon-optimal when H_k-L_k <= epsilon. Empty feasible domains require checking all negative verdicts before claiming they are empty.

An untrusted neural proposer may repeat suggestions, so the implementation deduplicates its list and appends all missing actions in a deterministic fallback order. This preserves completeness when the evaluation budget covers the domain. Proposals outside the action universe fail closed.

The Lean file formal/CCA/AnytimeBounds.lean proves a general preorder version of the sound-upper-bound theorem. The Python tests enumerate all feasibility assignments, three-value utility assignments, orderings, and budgets for three-action domains, exercising the emitted regret certificates.

The retrospective CCA-05 experiment replays real TRM pass-four logits from the existing six compressed tapes, comparing the learned order with a fixed action priority. Both share the exact CCA-05 verifier and its ActionCertificate checker. The only baseline upper bound is 1.0, which follows from the environment's clipped terminal values and max recursion. This deliberately weak upper bound can force exhaustive evaluation for moral states even when an early action is nearly best.

This experiment establishes how efficiently learned ordering *discovers a high-utility certified incumbent* and how often it achieves a mathematical stopping certificate. The two quantities can differ. Search forward time and host bound-checking overhead must be included in any wall-clock efficiency claim.

**Protocol:** experiments/cca09_anytime_certified_search/REGISTRATION.md
**Runner:** experiments/cca09_anytime_certified_search/src/run.py
**Tests:** tests/test_cca_anytime_search.py

No new model training or sealed confirmation is claimed. The CCA-05 tapes and benchmark outcomes have already been inspected.
