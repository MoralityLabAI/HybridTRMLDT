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

## A1 results (2026-11-02)

The rerun completed with every pass at exit 0. Receipt: `../receipts/logit_parity_20261102_a1.json`. Passes 1 and 2 log no CUDA device assignment, so the reference is CPU-only. Pass 1 took 5,472 s, pass 2 1,153 s and pass 3 50 s.

| Comparison | Top-1 agreement | Mean KLD | Max KLD | 99.9% KLD | RMS Δp | PPL ratio |
|---|---:|---:|---:|---:|---:|---:|
| PQ2_0 CPU vs PTQ1_0 CPU | 99.326% ± 0.091 | 0.000272 | 0.837389 | 0.005110 | 0.739% | 0.999930 |
| PTQ1_0 full CUDA (`-ngl 99`) vs PTQ1_0 CPU | 99.326% ± 0.091 | 0.000196 | 0.120659 | 0.005064 | 0.465% | 1.000105 |

CPU-only reference perplexity: 7.4935 ± 0.2119. The mixed reference gave 7.4934.

Both comparisons meet the primary ≥ 99.0% criterion. This establishes CPU-versus-CUDA logit parity for PTQ1_0 on this build.

Unlike the mixed run, PQ2_0 and PTQ1_0 are not bit-identical on the CPU path: top-1 drops from 100% to 99.326%, and one token reaches KLD 0.84. The CPU vec_dot kernels for the two formats therefore differ numerically, even though their dequantized weights are bit-identical (Q3b). Both comparisons give exactly the same top-1 figure (55 of 8,160 tokens disagree). That fits PTQ1_0's CPU kernel being the one that differs from the other two paths. This is unconfirmed, because the logs do not record per-token data.
