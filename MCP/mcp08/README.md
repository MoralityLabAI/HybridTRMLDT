# MCP-08: Verifier-Guided Recursive Memory Search and Error Recovery

Parent baseline: `gym/mcp07-adaptive-geometry` (`e0bf57f760890856dbc87c26cefc5065a70f18de`).
All MCP-07 checkpoints, registered receipts, manifests, and negative findings remain untouched and are loaded directly as the `mcp07_orig` baseline.

See [`SPEC.md`](SPEC.md) for the formal specification and [`manifest.json`](manifest.json) for the frozen registered protocol.

## Quickstart

From the repository root (CPU-only, parallel single-thread workers, `< 2 GiB` peak RSS):

```bash
python -m pip install -r MCP/mcp08/requirements.txt
python -m unittest MCP.mcp08.test_mcp08 -v
python -m MCP.mcp08.gym --smoke --threads 4 --out MCP/results/mcp08/smoke
python -m MCP.mcp08.gym --threads 6 --out MCP/results/mcp08/registered
```

## Module Overview

- `memory256k.py`: Extended shared physical memory (`16K`–`256K` / `262,144` physical slots) and bounded LRU `TranspositionCache` with explicit lookup and storage charging.
- `storyworld_bench.py`: Independent native Storyworld benchmark (`5–50` sequential decisions) on MCP-05's two-qubit Hilbert algebra with branching DAGs, distractor traps, irreversible transitions, noncommuting counterfactuals, and strict witness withholding.
- `domains.py`: Verifier-backed Canvas layout search and Lean-style proof-state tactic search sharing the external-memory & persistent-slow-state interface.
- `models.py`: `~54K`-parameter `MCP08RecursiveController` with persistent slow state, calibrated uncertainty head, hierarchical compute tier allocator (`{4, 8, 16}`), post-failure projection switching, and recursive meta-action head.
- `train.py`: 16K-only two-stage trainer (compact-span teacher + 16K on-policy DAgger calibration) and 5 trained-from-scratch ablations.
- `search.py`: Verifier-guided recursive search engine with bounded backtracking, irreversible transition barriers, transposition caching, and all 7 comparison baselines (`mcp07_orig`, `det_router_greedy`, `uniform_retries`, `greedy_planning`, `beam_search`, `bounded_symbolic`, `mcp08_full`).
- `mechanistic.py`: `3 × 3` zero-shot cross-domain linear probe transfer (Storyworld, Canvas, Lean), 6-pair cross-domain slow-state activation patching, and 7 paired inference-time lesions.
- `gym.py`: End-to-end CLI orchestrator writing sealed receipts, failure logs, SHA-256 manifests, and gate verdicts.
