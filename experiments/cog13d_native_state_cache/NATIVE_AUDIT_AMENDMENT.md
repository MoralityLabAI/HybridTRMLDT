# COG-13D native receipt audit amendment

Original workflow [45852224081](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/45852224081), source commit 9337d3d15da5f3975f426c3cc41629992b9b9c4a, compiled the genuine Lean snapshot server and executed both hard-capped native jobs. Their final workflow status was FAILURE. The audit compared JSON objects by property insertion order; Lean's JSON emits cache entry keys in sorted order, so every nonempty cache failed that comparison. Also, the negative compiler sentinel checked only "error:" while Lean 4.35 prints "error(lean.unknownIdentifier):". The original raw artifacts and failure status remain intact.

## Independent retrospective replay

Both artifact archives were retrieved and independently replayed using Python dictionary equality, exact prefix matching, deterministic eviction, command conservation, complete native goals compared against uncached native replies, and source/log SHA-256 checks for certificates. The independent archive-only replay found zero mismatches. The companion archive_reaudit.py preserves the replay logic and validates exact archive hashes.

| Cap | Artifact ID | Archive SHA-256 |
|---|---|---|
| 2048 MiB | 15135165929 | dff129389f1877141795b2c9a65663e4bfe40df7c8c25241277db6a839a14a45 |
| 4096 MiB | 15133514634 | c610eba2575855087e4efc21cb92ae6f399a2fff091a7a549531d0f623c25bbd |

All 1,680 native receipts per tier pass; each tier has eight complete roots per arm, 16 proof scripts each compiled twice with matching source hashes, accepted axiom inventories, and an independently rejected unknown identifier sentinel. This is a retrospective audit of native execution, not a newly green CI run.

| Policy | Commands executed per tier | 2 GiB query ms | 4 GiB query ms | 2 GiB CPU s | 4 GiB CPU s |
|---|---:|---:|---:|---:|---:|
| Root replay | 1504 | 3370.15 | 2975.84 | 29.87 | 32.93 |
| LRU 1 | 1128 | 3019.12 | 2726.85 | 29.39 | 32.46 |
| LRU 4 | 364 | 2369.25 | 2214.71 | 28.24 | 31.60 |
| Value 4 | 704 | 2575.50 | 2378.14 | 28.74 | 31.54 |
| LRU 16 | 240 | 2237.45 | 2110.15 | 28.07 | 31.29 |

CPU includes startup and shutdown for the eight sessions, so command reduction and query wall-time savings are much larger than whole-process CPU savings. No root or arm dropped from the replay. Both cgroup memory limits and disabled swap were confirmed. Cgroup peak accounted bytes: 749457408 and 730247168, or 714.74 and 696.42 MiB, respectively; both tiers had zero max/oom/oom_kill events. GNU-time process RSS is much larger (about 7.3 GiB) due to mapped/shared pages, and is not equivalent to cgroup-accounted memory. Neither tier bound the working set. Different CI hosts and a fixed trace are insufficient evidence of a causal physical RAM advantage or learned theorem-search gain.

Audit corrections: order-insensitive object comparison and categorized Lean error recognition. They change receipt validation only. The native server and registered request tape remain unchanged. A rerun on the patched branch is required before declaring CI green.
