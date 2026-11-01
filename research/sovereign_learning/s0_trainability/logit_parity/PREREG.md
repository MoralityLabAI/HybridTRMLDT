# S0 step 2b: full-model logit parity, pre-registration (2026-11-01)

Written and committed before any logit pass ran.

## Question

Does the PrismML runtime give the same next-token distribution for Ternary-Bonsai-2-27B when (a) the backend changes from CPU to CUDA, and (b) the packed format changes from PTQ1_0 to PQ2_0, which holds the same trits and scales in a different encoding with different kernels?

This tests the runtime's self-consistency. It is **not** oracle parity: the branch has no independent full-model forward. `packed_linear.FrozenTernaryLinear` covers one projection, and Q3b2 showed that the fork's CPU kernels match it at that level. That is why the CPU PTQ1_0 run is the reference.

## Fixed inputs

- Runtime: `llama-perplexity.exe` from the PrismML Windows build at `adfffbe41` (`C:\Users\patri\tools\llama-prism-b10743\bin`). The pin `c96baf4` is not built.
- Reference model: `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, revision `94dd51cd`, sha256 `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`.
- Format comparison model: `Ternary-Bonsai-2-27B-PQ2_0.gguf`, same revision, sha256 expected `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1` (Hugging Face LFS). The download is checked against this hash before use.
- Text: wikitext-2-raw-v1 test split (HF parquet sha256 `5f1bea06…6d91`), rows joined unchanged; text sha256 `bbf94c53a05abe9ee670d3b6343608095822c85e26de37c70b24fc571964574a`.
- Context 512, 32 chunks (the README's "32 fixed prompts"), 8 CPU threads, default sampling-free evaluation. The KL statistics cover the second half of each chunk, as llama-perplexity scores it.

## Passes

1. `01_base_ptq1_cpu`: PTQ1_0, `-ngl 0`, writes the reference logits (`--kl-divergence-base`).
2. `02_pq2_cpu`: PQ2_0, `-ngl 0`, `--kl-divergence` against pass 1.
3. `03_ptq1_cuda`: PTQ1_0, `-ngl 99`, `--kl-divergence` against pass 1, under `vramorama run --vram 9G`.

The CPU passes wait for at least 10 GB of free physical RAM, so the shared machine does not thrash.

## Criteria

- Primary, per comparison: "Same top p" (top-1 agreement) **≥ 99.0%**, the S0 step-2 criterion.
- Reported for every comparison, with no pass/fail attached: mean, median, 99.9th-percentile and maximum KLD; RMS Δp; and the perplexity of each pass.
- Expectation, recorded so a miss is visible: PQ2_0 on CPU should be very close to bit-identical, since both formats feed the same trits and scales into Q8_0-activation dot products. CUDA should differ by kernel and accumulation order only.
- The S0 "max-logit discrepancy" bound is **not** evaluated here: llama-perplexity stores compressed log-probabilities, not raw logits. A raw-logit export is a separate step.

A failed pass is recorded as it ran, with its log. Nothing is rerun with changed settings without an amendment.
