# MOLT-HYBRID-02: exact epoch adapter and incremental scalar bank

This fixture connects the native strict verified membrane to a finite premise environment.
A public-state snapshot binds the root, frame, epoch, candidate state and current premise
versions. A proposer can request a refinement only against that snapshot. The environment
computes its own exact admissible state, then uses certify_and_apply_verified to reject
unsupported eliminations, widening and no-ops. Replacing a premise is a trusted environment
operation: a new frame starts with the exact state under current premises, restoring candidates
whose previous exclusion expired. Earlier snapshots are frozen.

ScalarBank implements exact topological incremental evaluation for the existing mod-17
scalar recurrence. Local edits recompute affected nodes and descendants in the old/new
graph union. A changed context digest or graph size forces a complete rebuild.
Cache records report reused and recomputed node counts, not elapsed neural compute savings.
The context binding must be supplied by the host; this fixture has no authenticated external
receipt store or concurrency support. It is not a native K/V cache.

```sh
python -m unittest discover -s research_gym/benchmarks/molt_hybrid -p 'test_*.py' -v
```

Tests include stale proposals, candidate restoration, local invalidation, and all 4,096 ordered
pairs of strict-prefix graphs on four nodes. The last test checks incremental results against
complete evaluation for one fixed vector per graph pair. This is exact finite fixture evidence,
not a claim about all learned systems or measured training performance.

The DataGoblin MOLT-GOBLIN-02 episode host uses analogous versioned receipt semantics.
A shared live adapter remains future work: the fixture does not read or score DataGoblin
episodes directly.
