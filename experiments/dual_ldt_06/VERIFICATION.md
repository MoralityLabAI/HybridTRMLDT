# Dual-LDT-06 completed verification receipt

Scientific implementation and completed-result commit: `8e8574ffaedf332c5002ba753819d7215287a239`. The results-preservation commit leaves the registered generator, training recipe and learned weights unchanged. This document records completed checks; it initiates no training or additional report scoring.

## Completed workflow runs

- Training and original report: [41868351947](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/41868351947), source `3c1734574af530ba85eedddf728e0500b598ab04`. Contract tests, original trajectory Lean proofs, six-model CPU training, development evaluation, report evaluation and artifact export all succeeded.
- Outcome-cover verification and independent replay: [41871203966](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/41871203966), source `8e8574ffaedf332c5002ba753819d7215287a239`. Both jobs `exact-cover` and `artifact-audit` completed successfully.

The exact-cover job passed **14 tests** (nine trajectory contracts and five outcome-cover tests). Lean 4.16.0 kernel-checked `dominance_preserves_all_personas` and `coverage_preserves_optimum`. The printed axiom dependencies were `[propext, Quot.sound]` and `[propext]` respectively; there is no `sorryAx` dependency. The first theorem concerns exact integer outcome numerators and nonnegative integer weights. The finite Python algorithm and rational-weight checks are separately tested; no Lean implementation-equivalence claim is made.

The artifact-audit job downloaded the original immutable report artifact and reran a separate Fraction-based evaluator. It checked:

- 21,952 report world/plan combinations, of which 7,875 were feasible;
- 31,500 feasible world/persona/plan upper-bound inequalities;
- 10,240 recorded selected-plan utility and regret assertions;
- 2,048 anytime sequences, split disjointness, source/weight hashes and report-summary reconciliation.

Violations: **zero**. All computed outcome covers retained exact optimum values on the tested 81 nonnegative weight combinations per world and supplied a verified dominating witness for each discarded plan. Mean feasible plans: 123.046875; mean unique outcome/cost vectors: 12.46875; mean nondominated vectors: 4.96875 (range 2 to 13).

## Retained artifacts

Original reproduction artifact: `dual-ldt-06-reproduction`, ID `13343636356`, ZIP SHA-256 `7e07701d8f7541ae5ec6d9162bd87ba7a92f52b044d3aba3be31dade2e62b077`.

Independent audit artifact: `dual-ldt-06-independent-audit`, ID `13344083669`, ZIP SHA-256 `a5b617ae9bc611d7fd75f0924d39d6a8cee0758acff96550048542a8fbe1b759`. It contains `INDEPENDENT_AUDIT.json` with per-world cover sizes and the original report, summary and weight-freeze hashes. The downloaded ZIP digest and contained counts were also checked in the local analysis container.

The original report remains a negative result for neural ordering relative to the strong symbolic count-relaxation comparator. Compression was introduced after the registered comparison and is labeled post hoc. It requires evaluating the input plans, is valid for a fixed certified domain with nonnegative linear preferences, and does not establish a runtime or neural-generalization advantage. Safety assertions remain conditional on the declared synthetic transitions and trusted host authority.
