# CCA-07 finite abstraction receipts

Status: mathematical reference implementation, no neural outcome.

Run: python -m pytest -q tests/test_cca_abstraction_refiner.py

- A target write action on an eight-state domain was soundly authorized after exactly two environment-owned predicate refinements; both steps excluded concrete counterexamples and preserved the true target state.
- All 16 permission assignments on a four-state domain were evaluated against all four targets, with zero incorrect authorizations. With state-identity predicates available, every admissible target was eventually authorized.
- Each refinement strictly shrank the possible-state class and monotonically expanded its robust action set; all traces finished in at most |X|-1 steps.
- An inseparable counterexample correctly produced unresolved_predicates without admitting the requested action.
- Invalid, missing and mutable predicate tables were rejected or snapshotted immutably.

Five CCA-07 tests passed locally along with ten existing CCA-06 tests. The proof is conditional on a correct host-owned finite state/permission model and predicate values.
