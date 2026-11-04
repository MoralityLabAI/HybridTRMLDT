# CCA-06 finite-regression results

Status: deterministic CPU checks of mathematics; not model-backed evaluation.

Reproduce: python -m pytest -q tests/test_cca_compositional.py

- Enumerated all 41 deterministic retractions on four actions and all 1,681 ordered pairs. Exactly 601 pairs commute, and all 601 compositions retract to the intersection.
- A three-action counterexample cycles 0 -> 1 -> 0 despite common allowed action 2.
- With u(a,b,c)=(0,1,2) and R(c)=a, expected deployed utility falls from 0.333333 at strength 0 to 0.117310 at strength 2, while expected proposed utility rises. The analytic covariance derivative agrees with finite differences.
- The Fisher identity I_before = I_after + I_lost was verified numerically, including nonnegative loss quadratic forms and an exact-zero-information-loss example.
- Exhaustive tests across all triples of four-action permission sets check sound authorization and refinement monotonicity.
- Repair mappings are copied to immutable views so callers cannot mutate a previously verified map.

Ten local Python tests passed. Lean proof checking is a separate CI job. The analytic persona and Fisher proofs are written in docs/cca06_compositional_persona.md and remain unformalized in Lean. No claims about neural behavior are made.
