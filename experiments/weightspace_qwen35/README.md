# Qwen3.5-0.8B weight-space repair: RTX 5080 reproduction

**Scope:** reproduce the **fixed, released** p50c intervention from Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab, and compare it with an identical-roster baseline and support/norm-matched random control. No Astra search, gradient training, or local GPU experiment has been conducted in preparing this harness.

## Exact source and provenance

- Authors' [article](https://rikiyat.github.io/blog/weight-space-programming/) (2026-11-02).
- [Experiment 8 logs and bit-exact diffs](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math), specifically `runs/p50c/` and `code/apply_edit.py`.
- [Reconstructed BF16 checkpoint](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming), which independently scored **51.0%** on MATH-500 in a 2,000-sample H200 run (authors' reported checkpoint **48.1%**).
- Base: `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`.
- Tensors: `model.language_model.layers.{15,19}.mlp.down_proj.weight`. Exactly 64 selected columns per matrix, 131,072 addressed entries; 130,188 differ in final BF16 representations.
- Authors' model **uses thinking on**, 32,768 generated tokens, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, seed 0, **4 samples per MATH-500 item**. Their evaluator grades the **last boxed final answer** with `math-verify==0.9.0`.

The tool verifies the original and edited tensor bytes against published SHA-256 values before it constructs controls. The edited Hugging Face checkpoint includes the vision architecture, while the authors evaluated an otherwise equivalent text-only export; the two edited language-model tensors have matching hashes.

## RTX 5080 use

Use **WSL2/Linux** with a recent NVIDIA CUDA driver; `vllm` on Windows itself is not the tested pathway. In the experiment folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Download model checkpoints and verify both patched tensors:
python repro.py fetch --base models/base --edited models/edited

# Fixed, test-independent roster. Keep identical file for every arm:
python repro.py roster --n 8 --seed 20261103 --out artifacts/roster-smoke.json
python repro.py run --roster artifacts/roster-smoke.json --model models/base --samples 1 --max-tokens 2048 --out artifacts/base-smoke.json
python repro.py run --roster artifacts/roster-smoke.json --model models/edited --samples 1 --max-tokens 2048 --out artifacts/edited-smoke.json
python repro.py compare --results artifacts/base-smoke.json artifacts/edited-smoke.json --out artifacts/smoke-comparison.json

# Ablation control: same tensor support and approximate per-tensor delta L2 in BF16:
python repro.py control --base models/base --edited models/edited --out models/random_control --seed 20261103

# Full protocol on a FROZEN 32-item roster, if smoke passes and budget allows:
python repro.py roster --n 32 --seed 20261103 --out artifacts/roster32.json
python repro.py run --roster artifacts/roster32.json --model models/base --samples 4 --max-tokens 32768 --out artifacts/base32.json
python repro.py run --roster artifacts/roster32.json --model models/edited --samples 4 --max-tokens 32768 --out artifacts/edited32.json
python repro.py run --roster artifacts/roster32.json --model models/random_control --samples 4 --max-tokens 32768 --out artifacts/random32.json
python repro.py compare --results artifacts/base32.json artifacts/edited32.json artifacts/random32.json --out artifacts/paired32.json
```

A 32-item, four-sample, 32K-token run is **expensive on a 5080**; each arm can request up to 4.2M output tokens. Start with the 8-item smoke and inspect GPU memory/time. For the exact MATH-500 sample protocol, set `--n 500 --samples 4 --max-tokens 32768`; this is a substantial multi-hour-or-longer consumer-GPU run and should be scheduled separately.

`run` stores per-sample predictions, correctness, truncation rate, generated token counts, and the model identifier. Use `--save-text` only if you want raw generations (large JSON). `compare` rejects different roster hashes or decoding settings and reports rescues, regressions, and problem-clustered bootstrap confidence intervals. The random baseline uses the **same nonzero BF16 edit support** and an approximately equal L2 magnitude after rounding; it does **not** preserve the original intervention's rank-one geometry.

## Interpretation gates

1. First confirm the published edited hashes and base hashes. A mismatch halts rather than silently evaluating different weights.
2. Check whether 32K cut rate falls and answer accuracy increases on held-out MATH-500 under identical prompts and sampling.
3. Confirm that accuracy recovery comes predominantly from formerly truncated responses. Track wrong-but-finished responses separately.
4. Compare edited vs random control using the same questions and per-problem samples; use confidence intervals clustered by problem.
5. Only then run VPD on layers 15/19 and test whether decomposition predicts the observed causal effect. This package intentionally does not label a random patch as a VPD intervention.

**Known limitations:** current score is an independent implementation of the authors' last-box + `math-verify` protocol, not their exact evaluation harness. The random control builder currently expects one `model.safetensors` per checkpoint. The `verify` command accepts sharded files. Presence-penalty semantics depend on the installed vLLM version. GitHub released logs omit the editor's intermediate activation dumps; direct reconstruction of its entire search trajectory is unavailable.

## CPU smoke tests

```bash
python -m unittest discover -s tests -v
```

This validates parsing, aggregation and paired results; inference, checkpoint hashing and random edits must be validated on a host with the dependencies and GPU installed.
