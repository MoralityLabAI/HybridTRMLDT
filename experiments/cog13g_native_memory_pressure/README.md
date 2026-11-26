# COG-13G: native proof-memory pressure

Protocol: [registered before execution](https://github.com/MoralityLabAI/HybridTRMLDT/commit/db54fd0788b982094586941cca091734a0fbc92d). Based on COG-13D's corrected native snapshot server and the negative COG-13E/F retention studies. The existing COG-13D native archive reports peak cgroup memory below 1 GiB; both previous physical caps were nonbinding.

This stage increases the number of genuinely retained Lean states and resident sessions without allocating unrelated memory. See PROTOCOL.md for frozen roots, schedule, cgroup constraints, certification and interpretation.

## Implementation

- 32 generated statements, four proof families, 6–9 conjunctions; two registered proof scripts/root, 8 requests/root including one deliberately failing tactic.
- Interleaved active window of 24 roots. Exact per-session snapshots with root fallback on eviction. Three matched interleaved arms (root replay with LRU, 16-state snapshots with LRU, snapshots with recent-density session eviction), plus one block-scheduled control.
- 2/4/8 GiB hard cgroup memory caps and no swap. One CPU quota and 180 charged cgroup CPU seconds per scope, 480-second wall safety deadline. Measured CPU includes controller, Lean processes, startup, queries and double compilation. Cleanup can overrun; certificate credit stops at the budget. All 12 arms on the same runner with rotated order; one replicate is descriptive only.
- Preflight compiles 64 reference scripts and rejects a bad identifier. Scored certificates are separate, double compiled within the scope budget. Preflight uses predetermined proof tapes and measures execution/certification, not novel proof discovery.
- Startup reserve, memory high-water eviction, per-event memory.current/peak/anon/file and CPU, pressure events, query timing, process anonymous RSS proxy, actual attempted/restored commands, partial/unknown roots, raw mailbox and certificate logs. RssAnon is a session proxy, not a measured SavedState allocation. Exact proof bytes and root/script keys bind all credited proofs.
- Archive auditor compares raw mailbox files, exact goal observations across arms, session transitions, CPU-budget credit and certificate source/log hashes. Native failures remain in scope-exit and partial receipts.

## Executed software evidence

The deterministic fixture ran all 12 arm/cap combinations twice with identical summaries after receipt hardening, totaling 10,928 audited scheduler events per complete fixture. Four corrupted receipt mutations (certificate, work, missing reply, eviction victim) rejected. A short CPU budget fixture yields zero credited proofs; a startup failure fixture retains 31 successes and one unknown root. The mock backend uses declared 120 MiB/session and deterministic fake costs. Its results test mechanics, not actual memory/CPU performance.

Fixture example (only software simulation):
| Mode at 2 GiB | Opens | Pressure evictions | Attempted tactic commands |
|---|---:|---:|---:|
| Root replay, LRU | 187 | 155 | 2064 |
| Snapshots, LRU | 187 | 155 | 1540 |
| Snapshots, recent density | 150 | 118 | 1372 |
| Block snapshots | 32 | 0 | 896 |

The mock completes 32 roots in every arm; it does not establish that any policy succeeds or avoids OOM on real Lean. Native workflow is the measurement gate. No training, physical RAM benefit, per-state byte estimate or discovery improvement claimed.

## Reproduce

Fixture: `node experiments/cog13g_native_memory_pressure/run.mjs --fixture /tmp/cog13g-fixture.json` and `node experiments/cog13g_native_memory_pressure/test_budget.mjs`.

Native workflow prepares pinned Mathlib/Lean, verifies both scripts for every root, runs each scope, then invokes audit_archive.mjs on the native directory. Review aggregate.json together with scope-exit.tsv; if a scope fails, or unknowns occur, clean comparisons are withheld. A nonbinding cap is an inconclusive memory-pressure experiment even when proofs compile. If 8 GiB is unavailable or scope setup fails, the failure remains explicit. Repeated runs with a rotated order are necessary before a causal hardware conclusion. A future learned controller should be gated on a measured native baseline with enough pressure and actual overhead headroom.
