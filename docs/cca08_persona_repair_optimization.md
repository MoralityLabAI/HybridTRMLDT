# CCA-08: globally optimized repair on a certified finite action set

Status: exhaustive finite optimization with regression tests. Safety depends on the host-owned certified action set.

Given finite action universe A and nonempty certified set S, a repair is a retraction R:A->S: it maps every action into S and fixes S pointwise. There are exactly |S|^(|A|-|S|) such repairs. The algorithm enumerates them up to an explicit cap, with no held-out tuning.

For fixed persona distribution p_theta, scalar persona feature f, nonnegative intervention cost d, and task utility U, evaluate:

D(R) = E_p[d(A,R(A))]: expected intervention cost.
F(R) = E_p[Var(f(A)|R(A))]: Fisher information lost about the scalar exponential-family parameter.
G(R) = E_p[max_{b in S}U(b)-U(R(A))]: regret against the best certified action.

The Pareto frontier preserves nondominated choices. For declared weights beta,gamma>=0, exhaustive argmin of D+beta*F+gamma*G is globally optimal within this enumerated finite family. The floating-point evaluation establishes an approximate numeric objective; interval bounds or rational scoring would be needed for exact-arithmetic optimality.

A three-action example has persona utilities 0,1,2 and S={a,b}. Repairing unsafe c to a costs 1/3 intervention and loses 2/3 Fisher information; repairing c to b costs 1 and loses 1/6. Both are Pareto-optimal. beta=0 chooses a; beta=4 chooses b.

Direct repair into the certified intersection avoids potential cycles created by alternating independent repair maps. Every candidate preserves identity inside S and idempotence by construction.

Combinatorial cost grows as |S|^(|A|-|S|). This supplies an exact finite reference for small control heads and a baseline for scalable approximations, with no assertion about neural semantic-monitor soundness.

Code: research_gym/core/cca_persona_repair_optimizer.py
Tests: tests/test_cca_persona_repair_optimizer.py
