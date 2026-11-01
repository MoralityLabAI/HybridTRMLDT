# Amendment A1 (2026-11-02): the "CPU" reference was not CPU-only

Written after the first run and before any rerun.

## What happened

The pre-registered run completed on 2026-11-02 (all passes exit 0; receipt `../receipts/logit_parity_20261102_asrun.json`). Passes 1 and 2 used `-ngl 0`, but this PrismML build defaults to `--op-offload` on. The pass-1 log shows the fused Gated Delta Net ops assigned to `CUDA0` ("layer 0 is assigned to device CPU but fused Gated Delta Net ... is assigned to device CUDA0"). Large-batch work therefore ran partly on the GPU. The reference is a mixed CPU+CUDA placement, not the CPU path that Q3b2 verified against the oracle.

## As-run results, kept as recorded

| Comparison | Top-1 agreement | Mean KLD | Max KLD | 99.9% KLD | RMS Δp | PPL ratio |
|---|---:|---:|---:|---:|---:|---:|
| PQ2_0 vs PTQ1_0, same mixed placement | 100.000% | 0.000000 | 0.000058 | 0.000050 | 0.001% | 1.000020 |
| PTQ1_0 full CUDA (`-ngl 99`) vs mixed reference | 99.510% ± 0.077 | 0.000186 | 0.103031 | 0.003491 | 0.459% | 1.000117 |

Reference perplexity: 7.4934 ± 0.2120 (wikitext-2 test, 32 × 512).

Both comparisons meet the primary ≥ 99.0% criterion. What they establish: the two packed formats are interchangeable under identical placement, and full CUDA agrees with the mixed placement. They do **not** establish CPU-versus-CUDA parity.

## Change

Passes 1 and 2 are rerun with `-ngl 0 --device none --no-op-offload` (new runner parameter `-CpuArgs`) in a fresh work directory. The runner reports "CPU-only" only if the log shows no CUDA device assignment. Pass 3 is unchanged and repeated against the new reference. Everything else in `PREREG.md` stands: inputs, criteria, and one attempt per pass.
