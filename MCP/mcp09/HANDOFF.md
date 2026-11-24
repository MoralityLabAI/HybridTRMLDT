# Continue MCP-09

1. Run dependency-free tests and both smoke capacities. Fix any import or integration errors; never use registered namespaces. Record command, environment, hashes and per-world receipts. Audit strict witness validity; if failures occur, diagnose generator vs strict verifier without deleting cases. Audit final counterfactual replay and lock invariants. Add regression tests for real milestone failures, record-cache mutation and budget limits. Compare legacy MCP-08 with strict replay; instrument actual routes in a new copy of the solver, never overwrite MCP-08.

2. Build hardened MCP-08 with no blind commits, route-sensitive cache or proven safe equivalence, and exact budget guards, including charged final replay; add restoration stack parity and a speculative state snapshot/restore test. Integrate depth-2 deterministic lookahead into that solver. Compare fixed-medium reset-state controls to persistent slow state. Inspect per-bucket and first-lock traces. Keep the old MCP-08 results immutable.

3. Implement the 24–32 latent shared action-conditioned transition head, a parameter-matched static MLP control and a separate retrieval router. Generate training traces in isolated 16K namespaces; distinguish inaccessible and unknown branches from verified dead. Run development ablations and cost accounting before registering. No oracle reads in policy; instrument them explicitly if used for offline labels.

4. Freeze manifest, code, seeds, checkpoints, dependencies, budgets and statistical tests in a new commit. Run only then on fresh sealed namespaces; write the report, machine-readable summary, negative findings and evidence hashes. Never retrofit thresholds or merge automatically. Stop if smoke or audit fails and report exact blocker.

Tail prompt: Continue MCP-09 from the development branch. Run and repair the native smoke and integrity audit, implement the hardened matched-budget comparator, then develop the prospective head and retrieval dispatch in isolated namespaces. Freeze only after passing preflight. Preserve MCP-08 and all negative findings.
