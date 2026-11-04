# COST: cca05_certified_trm_repair

Host: Windows 11, AMD Ryzen AI 9 365, Python 3.12.14, torch 2.11.0+cu128 (CPU only, 1 thread, Idle priority). No GPU was used. Receipts: `outputs/cost_live.json`, `outputs/environment.json` (`cost`) and the per-arm `timing_*` fields in `outputs/arms/{tag}.json`.

## Whole runs

| Step | Wall | CPU | Peak RSS |
|---|---:|---:|---:|
| Environment report (lemmas, kernels, certificates, cold-cost sample) | 8.1 s | 8.0 s | |
| Negative controls (TRM-free) | 13 s | | |
| Live run (per checkpoint), from verification and tape through all arms, trajectories and the pass analysis | 16.1 to 20.7 s | 15.7 to 19.8 s | 0.53 GB |
| Live run, six checkpoints | 105 s | 101 s | 0.53 GB |
| Replay from tapes, six checkpoints (no torch) | 55 s | | |
| Full test suite (repository plus experiment, including replay and parity) | 109 s | | |

Live runs were done three times:

* the first run;
* the amendment-001 rerun;
* a rerun after switching the receipts to gzip.

Each run took the time above. The decision-rule inputs were identical across all three.

## Unit costs

| Operation | Cost |
|---|---:|
| One `Verify(s, a)` at horizon 6, caches cleared (registered sample of 500) | mean 0.81 ms, max 5.6 ms |
| One certificate check (replay of at most 5 steps; 159 issued certificates in the sample) | mean 22 µs, max 74 µs |
| One TRM forward, single row, 4 passes (2,100 inputs, X211) | 0.37 ms |
| TRM forwards per checkpoint in the live run | 8,093 to 8,108 (4,200 full-mask on `D_full`, 3,893 to 3,908 refined-mask A4 queries) |

## Per-decision overhead (793 decisions with nonempty S)

| Arm | Verifier evaluations per executed decision | TRM queries (4 passes each) | Certificate checks |
|---|---:|---:|---:|
| A1 | 1.00 | 1 | 1 |
| A2id, A2tr | 1.18 to 1.23 | 1 | 1 |
| A3 | 1.13 to 1.19 | 1 | 1 |
| A4 | 1.06 to 1.09 | 1.06 to 1.09 | 1 |
| A5 | 5 | 1 | 1 |
| L0 | 2.32 | 0 | 1 |
| O | 5, plus `U` on S | 0 | 1 |

Every arm needs 5 evaluations to establish that a certified set is empty: 957 of the 1,750 decisions. Per viable trajectory:

| Arm | Verifier evaluations | TRM queries |
|---|---:|---:|
| A4 | 3.9 to 4.2 | 3.9 to 4.2 |
| A5 | 17.4 to 18.0 | 3.5 to 3.6 |
| L0 | 8.7 | 0 |
| O | 17.8 | 0 |

## Cost of certification relative to the TRM

At these unit costs, a certified A4 decision costs about 1.07 × (0.81 + 0.37) ≈ 1.3 ms, against 0.37 ms for the raw TRM and 4.1 ms for the oracle's five cold evaluations. Certification is cheap in absolute terms here, because the environment is small and the verifier's dynamic programme is memoised across decisions within a run. The CPU times per arm in `timing_single_step` include those caches: 0.1 to 0.3 s per 1,750 decisions, and 2.3 to 2.9 s for A4, whose refined-mask TRM forwards are computed on first use.
