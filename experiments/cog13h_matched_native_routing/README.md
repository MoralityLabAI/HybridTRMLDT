# COG-13H: matched native compact proof routing

Frozen protocol: [ff756c2](https://github.com/MoralityLabAI/HybridTRMLDT/commit/ff756c2754b8a94c5b93afd79f5787c57775ea84). Extends the native COG-13C controller and COG-13D snapshot server. Physical pressure is evaluated separately in COG-13G #41.

## Executed so far

The exact COG-13C native archive (artifact 15136527625, SHA-256 in PROTOCOL.md) passed local preparation. Its 224 TRAIN-only probes from eight roots have 82 progress positives and 14 complete action menus. Preparation also collects 370 observed candidate labels across 121 distinct prior evaluation states, without using them to fit or select models.

Six matched fits on the actual archived training observations (regrouped in archived menu order) ran with the committed JS model. Each model has 553 trainable parameters and 4,480 SGD updates. Fit-only final cross-entropy:

| Seed | Flat | Tied three-pass |
|---|---:|---:|
| 1701 | .37147255 | .34037726 |
| 1709 | .38602182 | .34698757 |
| 1723 | .36714148 | .29735622 |

The stronger goal-aware symbolic ranker selects progress in 14/14 archived training states. Dense loop, dense flat and frequency each score 12/14 in all seeds. These are in-sample progress diagnostics, not certified theorem completion or evidence of recurrence-specific transfer.

The mock search fixture completes all seven arms in 20 synthetic probes and rejects forged cache and certificate receipts. Native search results are pending. No proof or timing credit from the mock.

## Native matrix

16 new generated Mathlib-backed roots, seven arms, three independent seeds (336 root episodes). Exact old/new statement inequality is checked before scoring. Native outcome and proof custody are checked from raw server mailboxes and distinct compiler processes. The search driver uses a fixed 32-probe/45-second per-root budget, common tactic grammar and exact 16-state cache (except root-replay control). Models share feature inputs; one-pass and tied-loop fit independently with the same update count, seed-specific shuffle and parameter count. Approximate multiply-add counts per candidate: flat dense 536, loop dense 828, rank-four loop 620. Feature/codec and other overhead are not included in those modeled counts. Host controller CPU, process CPU timing completeness, wall time and actual tactic commands are separately retained.

The BPD-inspired rank-four transport uses actual encoded/decoded values, including the four-bit arm. Setup JSON size describes a stored representation, not an executed setup wire protocol. Fresh evaluation is on public generated exercises, and repeated seeds on the same 16 roots do not create independent theorem cohorts. See PROTOCOL.md for preregistered gates; aggregate.mjs withholds all positive gates if unknown probes or incomplete cohorts occur.

Run fixture: `node experiments/cog13h_matched_native_routing/run.mjs --fixture /tmp/cog13h-fixture.json`.

For native reproduction: retrieve original ZIP from the successful COG-13C Actions run, use prepare.py to verify archive and produce input JSON, compile StateCacheH.lean against pinned Lean/Mathlib using Lake, and run the native driver with the new output dir, Lean path, Mathlib root, prepared input JSON and seed. The workflow runs all three seeds and archives results. Native compilation and fresh-root comparison remain the evidence gate. No new neural allocator training, actual BPD integration, or physical RAM conclusion is claimed.
