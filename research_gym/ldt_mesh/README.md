# ldt_mesh

Port of DCF-15 (LDT mesh routing):
https://github.com/MoralityLabAI/Compactification-Theory/tree/research/dcf01-dense-compute-frontier-20261125/research/dcf15_ldt_mesh_routing

16 latent bits form four 4-bit units. Each block of units holds an exact log-belief
table; noisy sensor factors that span blocks stay pending as dependencies. This is the
explicit checkable state `a_t` of the repo thesis: a router reads typed unit state and
chooses an operator, and nothing a learned model proposes bypasses that state.

## Contents

- `units.py`: `MeshUnit` exposes exactly seven fields (`local_state`, `dependencies`,
  `uncertainty`, `operators`, `cost_estimates`, `verification`, `info_lost`) and
  `MeshUnit.check()` validates them. `Mesh` absorbs evidence, merges and splits
  blocks, and provides the operators `local`, `bp` (cluster BP), `exact`, and
  abstain. `exact` enumerates only the units of the query's relevant component, never
  a 2^16 table. Pure Python, no dependencies.
- `router.py`: `objective_J` computes J(a) = E[R_hat(a)] + lambda*C(a) + mu*KL_hat(a)
  + gamma*F(a) from exposed fields and a `CostModel`. `Router` is the deterministic
  structural router (argmin J within the exact budget). `RegretModel` is the hook
  for a learned regret predictor.
- `tests/test_ldt_mesh.py`: exact equals brute force over all 16 bits on random
  episodes; local equals exact without cross factors; the seven-field interface; the
  router picks exact near the 0.8/0.2 thresholds when the budget allows.

## Not claimed

- No trained neural router. `RegretModel` is an interface only; no training code or
  learned weights are included.
- No hardware results. Costs are work counts times a configurable per-operation
  coefficient, not measured kernel times.
- No reproduction of the DCF-15 receipts or its held-out routing comparison.
