# Checkpoint: Sequencer Control Math and Verifiers v1

Date: `2026-07-13`

## Completed

- Frozen 667-episode matched replay across Sudoku, ARC-1, ARC-2, routing, and two storyworld objectives.
- Calibration-only sequence fitting with independent fold transports and RSITopology lineage/holonomy/orientation
  receipts.
- Registered global, lineage-only, fixed-typed, full-control, and local-reference sequencers.
- Family-macro endpoint, hierarchical context-cluster bootstrap CI, context-cluster sign-flip inference, Holm
  correction, family-balanced cluster `d_z`, exact McNemar audit, and epsilon sensitivity frontier.
- Raw episode receipts and protocol/calibration/evaluation hashes.
- Verifiers `0.1.14` v1 Taskset/Harness replay package, Prime eval config, smoke script, and adapter checks.

## Quantitative Finding

`control_math` reaches macro utility `0.900085` versus `0.893142` for global signed control: delta `+0.006943`,
95% clustered CI `[+0.004909,+0.009232]`, Holm-adjusted `p=0.000600`, family-balanced cluster `d_z=0.876`.
Accuracy is unchanged at `0.9667` and macro cost falls by `1.238`. The result is driven by conservative local
sectioning in 100% of contexts; orientation reversal is detected in 14.3%.

Against lineage-only control, the delta is `+0.000310` with clustered `p=0.502350`. This slice does not establish
incremental utility from holonomy/orientation checks beyond lineage localization.

## Boundary

This demonstrates sequencer-control effects over fixed deterministic skill implementations. It does not
demonstrate neural weight infusion, official ARC performance, or official INTELLECT-3 leaderboard performance.

## Validation Limitation

The v1 package contract tests pass. Exact `verifiers==0.1.14` runtime installation did not complete within a
bounded isolated attempt, and native Windows Prime remains blocked by the installed Unix-only `prime_tunnel`
dependency. Resume the exact smoke under Linux/WSL or a host with a current package cache.
