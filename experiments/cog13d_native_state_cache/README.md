# COG-13D — exact native tactic-state accretion

Stacked on [COG-13C #37](https://github.com/MoralityLabAI/HybridTRMLDT/pull/37). COG-13C's RAM15 probe always restores the theorem root, even when the process is resident. This experiment changes that boundary using actual Lean Tactic.saveState/SavedState.restore. The environment retains snapshots local to one theorem process, keyed by the exact tactic string prefix. It restores the longest matching cached prefix and evaluates only the remaining actions. Root state is always retained and is additional to the cache slot count.

## Fixed development design

Eight synthetic Mathlib-backed roots (two each of Nat order, rational ring identities, Finset membership, and propositional conjunctions), conjunction depth 4 or 5. Two prewritten valid scripts per theorem (constructor or equivalent refine at the first conjunction); no model is trained or evaluated in this experiment. Each fixed trace interleaves both branches, revisits earlier states, submits intentionally invalid final commands, and closes both scripts. All arms receive the identical tape of 336 requests over eight roots. Exact strings are the cache keys; printed goals are compared against the full-root native reference and are not used as identity keys.

Lean 4.35.0, Mathlib commit 56a005855ea17e0b0958f73ec801ae140b0a1679. Five cache arms: 0 extra snapshots (root replay), LRU 1, LRU 4, value 4, LRU 16. The value rule retains high depth times (hit count + 1), breaking ties by recency. It is a deliberately cheap heuristic; hit count alone does not optimize avoided work. Cache entries are committed only at successfully completed nonterminal request boundaries. A rejected request may touch a previous valid entry, but never publishes a new snapshot. The live working state rolls back after rejection. A non-backtracked IO.Ref counts attempted tactic strings including the failing one. This counts commands submitted to parser/evaluator, not CPU instructions or internal Lean tactic steps.

The independent JS checker reconstructs the expected cache transitions and rejects mismatches, checks every accepted goal display against full-root native replay, verifies failed-request rollback through later replay, and ensures exact command conservation. Only the *native* backend validates Lean behavior; the fixture uses the same JS transition function for its mock producer and is a software/accounting check. Full proof source for each predetermined script is compiled twice separately without importing the speculative cache server. Both compilations must pass with matching source hash and allowed axiom inventory (propext, Classical.choice, Quot.sound, or empty); a separate invalid-identifier certificate must be rejected. The speculative server's internal shutdown sorry is quarantined and cannot certify results.

## Executed software fixture

1,680 request receipts (336 × five). All four negative mutations rejected: forged cache hit, forged resulting goal, forged cache state, missing request.

| Policy | Executed tactic strings | Avoided via prefix reuse | Cache hits | Peak extra entries |
|---|---:|---:|---:|---:|
| Root replay | 1504 | 0 | 0 | 0 |
| LRU 1 | 1128 | 376 | 112 | 1 |
| LRU 4 | 364 | 1140 | 280 | 4 |
| Value 4 | 704 | 800 | 320 | 4 |
| LRU 16 | 240 | 1264 | 320 | 16 |

LRU 4 predicts 4.13× fewer attempted commands and LRU 16 predicts 6.27×. These are deterministic accounting outcomes conditional on the fixture's expected accepted/rejected replies. No native tactic execution, CPU-time speedup, or memory saving has been established by the fixture. Value 4 has more hits than LRU 4 yet avoids fewer commands; this suggests prioritizing saved recomputation rather than raw hit frequency.

## Native measurement

The workflow first runs the fixture and compares its frozen receipt. Two separate Linux CI jobs run the identical native benchmark under cgroup v2 MemoryMax 2 GiB and 4 GiB with swap disabled. Mathlib preparation/compilation is outside those caps; the measured workload contains the Node host, Lean subprocesses and certificate compilation. The runner verifies its effective memory.max and memory.swap.max before scoring. Failure to create the scope or an OOM is an infrastructure/cap result, never substituted with an uncapped run. Per-arm native request wall time, whole-session wall time, GNU time user/system CPU seconds, peak RSS, exact submitted/executed/restored tactic counts and raw transcripts are retained. cgroup memory peak/events and CPU counters are recorded at whole-tier level; separate GitHub hosts and one repetition per tier do not support a causal claim about RAM capacity or CPU speed. RSS sampling can miss spikes; incomplete CPU records are counted, not treated as zero. Snapshot count is not a byte cap; actual memory is enforced at process-group level by the cgroup.

Run fixture:
`node experiments/cog13d_native_state_cache/run.mjs --fixture /tmp/cog13d-fixture.json`

The native workflow compiles StateCacheServer.lean and starts the capped job. Locally (Linux, prepared Mathlib and LEAN_PATH with compiled StateCacheServer.olean), the native CLI accepts output dir, Lean binary, Mathlib cwd and required cgroup cap in bytes (0 explicitly permits an uncapped developer run). The result file retains every incomplete arm and validates all successful ones.

## Scope and follow-on

This is a deterministic, prewritten search-trace microbenchmark. It isolates checkpoint costs from learned proposal quality. It does not measure theorem discovery, true cross-process serialization of MetaM, eviction under a dynamic byte-specific allocation policy, or neural LDT/BPD integration. Full native results are pending. The next experiment should integrate this server into COG-13C, fit a retained-value estimator using measured replay cost and observed future reuse on training traces, compare LRU/LFU and matched byte/cgroup controls on fresh roots, and measure certified theorems per charged CPU second. Never allow lossy BPD messages to stand in for verified Lean states.
