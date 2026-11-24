# MCP-07: Geometry-Adaptive Recurrent Memory Routing

Parent baseline: `gym/mcp06-spatial-memory` (`3c03ccc6c4a0cac059c670011a55ea7ad9666de9`).

MCP-07 tests whether a compact two-timescale recurrent metacontroller (`~54K` active parameters, scaled variant capped `<= 128K`) can learn which memory addressing geometry (`row`, `col`, `hilbert`, `morton`) and search stride resolution to use—switching projections mid-episode when query geometry changes or stale replicas are encountered, and halting adaptively to save charged physical record reads.

See [`SPEC.md`](SPEC.md) for the complete formal specification and [`manifest.json`](manifest.json) for the frozen registered protocol.

## Quickstart

From the repository root (CPU-first, 4–8 threads, no GPU required):

```bash
python -m pip install -r MCP/mcp07/requirements.txt
python -m unittest MCP.mcp07.test_mcp07 -v
python -m MCP.mcp07.gym --smoke --threads 4 --out MCP/results/mcp07/smoke
python -m MCP.mcp07.gym --threads 6 --out MCP/results/mcp07/registered
```

## Module Layout

- `memory.py`: Shared physical memory (`16K`–`128K` slots) with 6 address projections (`row`, `col`, `hilbert`, `morton`, `rand_coloc`, `rand_full`), explicit index storage/preprocessing/lookup/switch cost accounting, and shared physical slot ID deduplication.
- `curriculum.py`: 8 distinct query families (`horizontal`, `vertical`, `diagonal`, `isotropic`, `sparse_long_range`, `multihop`, `regime_shift`, `corrupted_replica`) plus 80/20 → 20/80 distribution-shift mixtures.
- `models.py`: Two-timescale coupled GRU controller (`54K` matched scale and `116K` scaled variant `<= 128K`), projection-specific state memory with optional cross-projection communication, learned projection router, factored resolution + 13-stride head, learned adaptive stop head, and optimal-action-set loss.
- `train.py`: Compact-span shortest-path verified trace teacher and 16K-only training loop (`action_set` vs `single_best` supervision).
- `eval_harness.py`: Vectorized batched rollout for all experimental arms (A–H), paired bootstrap 95% confidence intervals, and Section 7 intervention hooks.
- `native_storyworld.py`: Separate native Storyworld integration track (MCP-05 archived witness replay + witness-withheld independent route discovery with 5-point native verification).
- `mechanistic.py`: Router probability & state trajectory extraction, slow/fast linear probes (geometry, progress, uncertainty), and 5 paired causal interventions.
- `gym.py`: End-to-end CLI entrypoint writing receipts, SHA-256 hashes, failed episodes, and gate evaluations.
