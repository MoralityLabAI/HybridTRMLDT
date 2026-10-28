# S0: Bonsai 2 PTQ1_0 trainability kill test

**Status: CPU reference only.** No whole-model GPU test, adapter training, model download, 27B trainability or GGUF adapter serving is claimed. Start here before investing in learned TRM/LDT routing.

## Working source

Target: **Bonsai 2 ternary 27B**, PTQ1_0 GGUF type 143, distinct from first-generation binary Bonsai 27B Q1_0. PrismML's format holds 128 ternary values in 28 bytes: 24 five-trit payload bytes (consumed in the fork's 16/8 staging order), 2 four-trit bytes and a trailing little-endian FP16 scale. PQ2_0 (type 142) uses 34 bytes/group for identical trits and scales. Both formats carry weights in a rotated basis.

The reference codec is pinned to [PrismML's llama.cpp source](https://github.com/PrismML-Eng/llama.cpp/tree/c96baf4abe24d11a50f326e40b6223b687449d63), specifically `ggml/src/ggml-quants.c` and `ggml-common.h`. See `source_pins.json`. The [model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) describes an input-axis Hadamard-1024 transform with fixed sign flips; real signs are GGUF metadata, not a universal all-ones vector.

## Implemented in this subfolder

- `ptq1_codec.py`: Python-standard-library PTQ1_0 and PQ2_0 block encoders/decoders and bounded already-located matrix payload decoder. The encoder serializes existing trits, not arbitrary FP16-to-ternary quantization. GGUF offset metadata and tensor selection remain external.
- `packed_linear.py`: an **unpacked** PyTorch reference with int8 trit buffers, g128 scales, differentiable signed Walsh-Hadamard input transform, and an unmerged LoRA applied in the original input basis. It demonstrates small synthetic forward and gradient contracts; packed on-device backward remains unimplemented.
- `parity_cli.py`: compares **complete** paired output vocabularies for at least 32 identically tokenized prompts, checks top-1 match and logit differences, and writes hash-bound local file comparisons. CLI requires `--expected-vocab-size` from the pinned checkpoint metadata. The default 0.03 logit tolerance is provisional and must be independently justified before a hardware claim.

Read-only local unit checks with existing Python dependencies:

```sh
python -m unittest discover -s research/sovereign_learning/s0_trainability/tests -v
```

PyTorch tests skip when the optional `neural` extra is unavailable. Test code launches no training or model inference and needs no external access.

## Measured S0 acceptance gate, still open

1. Pin the **full checkpoint file SHA**, fork runtime commit, GGUF metadata and actual sign vector. Verify PTQ1_0-vs-PQ2_0 weights against independently decoded real matrix rows. Confirm activation rotation, tensor offsets and tokenization.
2. Compare one actual fork projection against the PyTorch oracle, then full model logits on 32 fixed prompts, with and without an unmerged rank-16 GGUF adapter. Preregister a tolerance based on the fork's own quantization spread.
3. On the RTX 5080, measure 8K/32K inference peak VRAM, then one **single** 2048-token frozen-base adapter backward pass with a true packed-resident kernel. The presence of an autograd oracle in this directory does not establish such a kernel.
4. Report peak VRAM, train tokens/s, runtime configuration and source hashes. S0 go requires 99% top-1 agreement, measured max-logit discrepancy below the preregistered bound, <=14 GB peak training VRAM, >=300 train tokens/s and <=10 GB inference VRAM at 32K. All criteria remain **NOT_EVALUATED**.
5. If any step fails, stop and record it. No loops, no paid teacher calls or open-ended optimization. Subsequent S1 experiments would use BitAgent open-gate material for development and fresh sealed transfer examples, with an independent acceptor.

The earlier `Control-Harness/agent/signals/MYTHOS_RSI_2026-10-24.md` authority-cut and liveness obligations (P2/P4) remain open. This S0 branch neither certifies those invariants nor promotes any routing/model artifact.
