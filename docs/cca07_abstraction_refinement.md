# CCA-07: Counterexample-guided refinement of constitutional observations

**Status:** finite-state reference implementation with exhaustive small-domain tests. This is a mathematical control-abstraction experiment, not a demonstration of semantic monitors on language-model activations.

Inputs: finite concrete state set X, host-owned exact permission sets S(x), and a finite bank of environment-evaluated predicates p_j(x). An abstract observation is a list of chosen predicates; its possible-state class Gamma consists of states agreeing with the actual target on their values.

For a desired action a that is permitted at actual state x*, the algorithm searches for counterexamples y in Gamma where a is inadmissible. It selects a trusted predicate that separates x* from at least one counterexample, breaking ties deterministically and recording the excluded states as a receipt. It stops when every remaining possible state permits a, or when no available predicate can separate the counterexamples.

**CCA-I (soundness).** Every successful authorization belongs to the intersection of S(y) across the final possible states. Since x* remains in that set, the desired action is permitted at x*. Proof: every refinement preserves the target and only deletes possible states; the final action is authorized exclusively through the intersection.

**CCA-J (monotonicity).** Every successful refinement shrinks Gamma and hence can only increase the soundly authorized action set. It cannot revoke a previously certified action.

**CCA-K (termination).** Every chosen predicate excludes at least one state from the current class, so there are at most |X|-1 successful refinements. When the bank separates the target from all counterexamples, the desired action is eventually authorized; otherwise the result is explicitly unresolved. The greedy selection criterion maximizes the number of currently excluded unsafe states; it is not a certificate of minimal predicate count.

A reference example has eight states encoded by three binary predicates. A write action is allowed only when the lowest two bits are both set. Starting from no predicates, the algorithm excludes the unsafe counterexamples in two refinement steps and certifies the write action. Exhaustive tests cover all 16 permission functions on a four-state universe. Incomplete predicate truth tables and unknown states fail closed.

**Scope:** the proof assumes that the environment permission table is correct and that the truth values supplied by the host correspond to the concrete state. A false or attacker-controlled tag can violate these assumptions. Refining a representation does not repair an omitted true state. A live SAE or residual monitor needs its own empirical coverage and provenance audit.

Paths: research_gym/core/cca_abstraction_refiner.py, tests/test_cca_abstraction_refiner.py.
