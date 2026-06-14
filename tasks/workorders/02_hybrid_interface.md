# Hybrid Interface Workorder

Goal: extend `research_gym/core/hybrid.py` without adding neural dependencies.

Tasks:

1. Add a protocol or stub class for a proposal source that consumes `(problem, a_t, h_t)` and emits `LatticeProposal`.
2. Add a tiny deterministic proposal source for tests.
3. Add one CLI that reads candidate states from JSONL and writes accepted/rejected membrane results.
4. Keep soft provenance rejected by default.

Deliverable: tests plus `docs/hybrid_contract.md` update.
