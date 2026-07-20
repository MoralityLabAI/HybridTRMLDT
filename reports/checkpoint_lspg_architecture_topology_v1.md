# Checkpoint: LSPG Architecture Topology v1

## Completed

- Added canonical first-appearance module identities and bounded primitive-word enumeration.
- Added deterministic schedule descriptors, fixed-periodic controls, balanced nulls, and disjoint discovery/reserve selection.
- Added explicit module-word execution and variable inference depth to `LoopedDecoderLM`.
- Preserved the legacy v0 construction and parameter-count path.

## Tests

```text
python -m pytest tests/test_lspg_architecture_topology.py tests/test_lspg_architecture_decoder.py tests/test_looped_decoder.py tests/test_lsa_schedule_executor.py -q
11 passed
```

The branch baseline has one unrelated pre-existing failure: the committed attested-provenance registration references a results hash present only in the dirty shared worktree. Frozen LSA and LSPG v0 artifact hashes pass after byte-preserving worktree checkout.

## Decisions

- Architecture candidates vary only the physical-module schedule word.
- Residual, gradient, state-retention, supervision, and carry policies remain fixed for causal attribution.
- Runtime depth is a repeated prefix of the registered macro word.

## Next Step

Implement deterministic synthetic, Sudoku, and routing task bundles with sealed split manifests.
