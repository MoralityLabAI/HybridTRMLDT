# CCA-06: Compositional persona repair and abstraction

Status: mathematical derivations, finite CPU regression, and Lean proofs under CI verification. Persona differentiation and Fisher information formulas remain pen-and-paper derivations with numerical regression rather than machine-checked calculus.

Code: research_gym/core/cca_compositional.py
Tests: tests/test_cca_compositional.py
Formal proofs: formal/CCA/RepairComposition.lean

## CCA-F: Commuting retractions compose

For C,D subsets of action universe A, let R_C map every action into C and fix every element of C. Let R_D do likewise for D. If R_C and R_D commute, the composition R_C o R_D is a retraction onto C intersection D, because it lands in C and, by commutation, in D, and it fixes every point in their intersection. Idempotence follows. The Lean file states the theorem without finiteness assumptions.

Local correctness alone permits cycles: with A={0,1,2}, C={0,2}, D={1,2}, R_C(1)=0 and R_D(0)=1, alternating repair cycles 0 -> 1 -> 0 even though C intersection D is {2}. The commuter condition is sufficient, not necessary; joint repair into the intersection can avoid cycles.

## Persona dose after repair

With a positive baseline q, scalar persona utility u, and parameter lambda, define p_lambda(a) proportional to q(a) exp(lambda*u(a)). For a fixed deterministic repair R, Y=R(A), and differentiation of finite sums yields:

d/dlambda E[u(Y)] = Cov_p(u(R(A)), u(A)).

The covariance can be negative. With actions a,b,c, utilities 0,1,2, uniform baseline, certified actions a,b, and R(c)=a, the expected deployed utility is exp(lambda)/(1+exp(lambda)+exp(2lambda)). It drops from 1/3 at lambda=0 to about 0.117310 at lambda=2, despite a safe idempotent repair. Titration on a fixed certified support, with the same utility used to measure expectation, has nondecreasing expected utility. This does not guarantee monotone individual decisions.

## Fisher information after repair

For vector persona feature f and p_theta(a) proportional to q(a) exp(theta dot f(a)), the input score is f(A)-E[f(A)]. After deterministic theta-independent repair Y, the output score is E[f(A)|Y]-E[f(A)]. The total covariance law gives:

I_before - I_after = E[Cov(f(A)|Y)], positive semidefinite.

Information loss equals zero exactly when f is constant within every repair fiber of positive mass. This is information about the declared persona parameters, not a claim of recovered neural mechanism or moral identity.

## CCA-G/H: Sound abstraction and monotonic refinement

If an observation z represents possible concrete states Gamma(z), and S(x) is the environment-owned set of admissible actions in x, the maximally permissive sound action set is the intersection of S(x) across x in Gamma(z).

If actual x belongs to Gamma(z), every authorized action is admissible in x. If a refined abstraction represents a subset of the coarse possible states, the robustly admissible set can only expand. Both propositions have Lean proofs. The executable reference fails closed when the set of possible states is empty or unresolved.

The monitor must still establish that the actual state belongs to Gamma(z) under the deployment threat model. The proof has no power over false semantic descriptions.

## Next

CCA-07: counterexample-guided abstraction refinement, using environment-owned predicates.
CCA-08: persona-preserving repair optimization over the jointly certified set.
CCA-09: verified anytime search and recurrent certificate-discovery efficiency.
CCA-10: adaptive semantic-monitor coverage and risk accounting on selected actions.
