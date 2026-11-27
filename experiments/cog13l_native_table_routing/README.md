# COG-13L: Native table-backed certified routing

Protocol registered at fd129bca8d7e8c68b39461bb1c975911a77fa123.
Stacked on [COG-13K #45](https://github.com/MoralityLabAI/HybridTRMLDT/pull/45), whose hosted replay [45915616593](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/45915616593) passed. J's native experiment [45912260970](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/45912260970) also passed. This experiment leaves the J and K source blobs unchanged and explicitly adapts J's source emitter, routing callback and wall-clock limit. No checkpoint training or calibration.

## Frozen workload

24 new generated statements, 12 ordinary and 12 deeper, across six families (Nat inequalities, rational ring, Finset membership, mixed implications, propositional equivalences and modular arithmetic). Ordinary conjunction depths 3–4; deeper depths 7–9. Each statement's exact string is checked against the 56 C/H/J registry entries from pinned archives. Finset/logic variants may have similar structures; an unused distinguishing binder does not establish semantic independence.

All 24 reference scripts must compile in pinned Lean/Mathlib with permitted axioms before scoring. An invalid identifier must be rejected. The preflight scripts may exceed the scored 12-advancing-action depth limit, and are never supplied to a controller. Consequently a valid reference proof does not imply completion within the search budget. The harder workload is a generated development cohort, not a sealed or original mathematics test set.

## Native policies and budgets

| Arm | Recurring protocol | Installed bytes |
|---|---|---:|
| frequency | local fixed training-frequency action | 128 |
| local_dense | local dense action | 4424 |
| shared_dense | shared goal features, dense receiver | 4424 |
| decimal_progressive | int8 → binary16 → exact, Decimal interval receiver | 5960 |
| table_progressive | same packets, C table interval receiver | 38744 |

Installed representation accounting follows J/K; sizes are declared arrays, not measured process memory. The table receiver is compiled with no fast-math or FMA contraction. It uses the exact frozen 32,784-byte table from K, checks the table SHA, and executes K's endpoint and malformed-input tests. The adapter preserves J's sender, dense fallback and 16-tactic search grammar, with one native Lean process per root and 16 exact SavedState entries. Each of 120 episodes has up to 48 probes, 45 seconds wall time, 12 advancing actions, eight goals and two separate accepted full-source compilations before proof credit.

The workflow first exercises all five policies against J's synthetic fixture. Mutations of cache counts, proof certificates, packet bytes and interval receipts must be rejected. Then it verifies the full fresh native cohort. Each root runs all five arms in rotated order. Raw mailboxes, packet stages, compiled proof source/log hashes, interval bounds, exact action and script histories, CPU receipts, unknowns and failed audits are retained. The offline interval audit recomputes the packet/receiver trace using the same receiver and checks the bounds against the canonical dense scorer; it is not a separate formal floating-point implementation.

## Accounting and registered gates

Primary throughput is credited proofs per CPU-hour (waited Lean child CPU plus Python search CPU, with measured table setup charged once to the table arm). The common pinned archive load, compilation and reference preflight are separately recorded, outside the scored episodes. Host search CPU includes J's inline dense audit; extra offline replay CPU is reported separately. Each policy also reports sender/receiver CPU, native queries and actually executed/restored commands, wall time, setup/recurring bytes, per-family/per-split completion and paired root sets. No physical memory cap, real network I/O or matched CPU quota is claimed. Failed audits and unknowns close clean-comparison gates, with observed CPU retained.

The frozen criteria are in PROTOCOL.md. Require a clean 120-episode roster, identical nonempty dense completion sets, Decimal/table trajectory parity, table recurring traffic <=1.05x Decimal and <=0.5x shared dense, >=5x receiver speedup, and total table proof CPU <=1.05x Decimal including one-time table setup. Negative scientific results are preserved. The previously successful J and K readouts are source evidence, not results of L.

## Reproduce

Use the SHA-pinned C checkpoint and H/J cohort archives named in the workflow. Run the fixture:
```
python experiments/cog13l_native_table_routing/run.py C.zip --fixture --out /tmp/cog13l-fixture
```
For native execution compile J's StateCacheJ.lean using pinned Lean 4.35.0 and Mathlib commit 56a005855ea17e0b0958f73ec801ae140b0a1679, with Lake's compiled library path. Then run:
```
python experiments/cog13l_native_table_routing/run.py C.zip --old-h H.zip --old-j J.zip \
  --lean /path/to/lean --cwd /path/to/mathlib --out /tmp/cog13l-native
```
The native workflow uploads the fixture, preflight, table setup, per-episode transcripts and final result, including partial receipts after failure. A fresh output directory is required. Implementation committed; new fixture and native results are pending hosted execution.
