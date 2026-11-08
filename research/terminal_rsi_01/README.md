# TRSI-CLOCK-01 and TRSI-REWARD-01

Prepared 2026-11-09 for MoralityLabAI/HybridTRMLDT.

Status: standalone CPU arithmetic contracts executed. Native gym integration and learned-controller comparisons are unrun. The reference uses only Python's standard library.

This folder owns event-clock GAE arithmetic and reward-objective checks. Preserve the existing lattice and environment-owned provenance path in `research_gym/core/hybrid.py`; use `certify_and_apply_verified` for new native experiments as directed by the root README. The proposed adapter only records trusted host event boundaries and exports trajectories. It must not change `MembranePolicy` or promote model-claimed soundness.

```bash
cd research/terminal_rsi_01
python -m unittest discover -s . -p 'test_*.py' -v
python run_cpu.py
```

Read `MATH.md` for assumptions, then `PROTOCOL.md`. `CPU_RESULTS.json` and `TEST_LOG.txt` record the executed finite checks. No GPU/model run, new CI workflow, or host integration is included.
