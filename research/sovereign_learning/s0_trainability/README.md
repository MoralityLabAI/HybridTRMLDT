# S0: Bonsai 2 PTQ1_0 trainability kill test

**Status: CPU reference only.** No whole-model GPU test, adapter training, model download, 27B trainability or GGUF adapter serving is claimed. Start here before investing in learned TRM/LDT routing.

## Working source

Target: **Bonsai 2 ternary 27B**, PTQ1_0 GGUF type 143, distinct from first-generation binary Bonsai 27B Q1_0. PrismML's format holds 128 ternary values in 28 bytes: 24 five-trit payload bytes (consumed in the fork's 16/8 staging order), 2 four-trit bytes and a trailing little-endian FP16 scale. PQ2_0 (type 142) uses 34 bytes/group for identical trits and scales. Both formats carry weights in a rotated basis.

The reference codec is pinned to [PrismML's llama.cpp source](https://github.com/PrismML-Eng/llama.cpp/tree/c96baf4abe24d11a50f326e40b6223b687449d63), specifically `ggml/src/ggml-quants.c` and `ggml-common.h`. See `source_pins.json`. The [model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) describes an input-axis Hadamard-1024 transform with fixed sign flips; real signs are GGUF metadata, not a universal all-ones vector.

## Implemented in this subfolder

- `ptq1_codec.py`: Python-standard-library PTQ1_0 and PQ2_0 block encoders/decoders and bounded already-located matrix payload decoder. The encoder serializes existing trits, not arbitrary FP16-to-ternary quantization. GGUF offset metadata and tensor selection remain external.
- `packed_linear.py`: an **unpacked** PyTorch reference with int8 trit buffers, g128 scales, differentiable signed Walsh-Hadamard input transform, and an unmerged LoRA applied in the original input basis. It demonstrates small synthetic forward and gradient contracts; packed on-device backward remains unimplemented.
- `gguf_reader.py`: standard-library, mmap-backed GGUF v2/v3 reader for metadata and tensor offsets. It does not dequantize.
- `real_row_parity.py`: compares real PTQ1_0 rows decoded by `ptq1_codec` with the fork's compiled `dequantize_row_ptq1_0` (`ggml_get_type_traits(143)->to_float`, loaded via ctypes), and audits the tensor layout and `prism.hadamard.*` metadata.
- `parity_cli.py`: compares **complete** paired output vocabularies for at least 32 identically tokenized prompts, checks top-1 match and logit differences, and writes hash-bound local file comparisons. CLI requires `--expected-vocab-size` from the pinned checkpoint metadata. The default 0.03 logit tolerance is provisional and must be independently justified before a hardware claim.

Read-only local unit checks with existing Python dependencies:

```sh
python -m unittest discover -s research/sovereign_learning/s0_trainability/tests -v
```

PyTorch tests skip when the optional `neural` extra is unavailable. Test code launches no training or model inference and needs no external access.

## Real-row parity, measured 2026-10-31

Receipt: `receipts/row_parity_20261031.json` (**PASS**).

- Checkpoint: `prism-ml/Ternary-Bonsai-2-27B-gguf` revision `94dd51cdbbec5309ed8b76a12f3268bc4fdceb93`, file `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, 5,946,648,928 bytes, sha256 `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` (matches the Hugging Face LFS hash).
- Reference: `ggml-base.dll` from a PrismML Windows build at commit `adfffbe41` (sha256 `8001b0cb…5d3c`). That commit is an ancestor of the pin `c96baf4`, and `ggml-quants.c`, `ggml-common.h` and the `ggml_type_traits` struct are identical between the two.
- Layout: 851 tensors (402 PTQ1_0, 353 F32, 96 BF16), GGUF v3, alignment 32. Payloads do not overlap, and the last one ends exactly at end of file.
- Rows: 0 mismatches, bit-exact. Every PTQ1_0 tensor was sampled (first, last and 2 seeded random rows: 11,600,896 values). `blk.0.attn_gate.weight` (31,457,280 values) and `blk.63.ffn_down.weight` (89,128,960 values) were compared in full. About 67% of the weights are non-zero.
- Rotation metadata: normalized Sylvester Walsh-Hadamard, block 1024, input last dimension, `sign_mode` `explicit`. Three sign vectors (widths 5120, 6144, 17408; 28,672 values, 14,504 of them −1, sha256 `3d1ea913…8bed`). Every name in the 401 `weight_names` exists, and each row length has a sign width. `token_embd.weight` is the one inverse-transform tensor; `gdn_v_grouped` is true.

This closes the weight-decoding part of gate step 1. The check compares this codec with the fork's C decoder on the real file, so a layout misreading shared by the two would now show up. Still open in step 1: the runtime order of sign and transform application, and tokenization. Steps 2 to 5 are not touched.

## Measured S0 acceptance gate, still open

1. Pin the **full checkpoint file SHA**, fork runtime commit, GGUF metadata and actual sign vector. Verify PTQ1_0-vs-PQ2_0 weights against independently decoded real matrix rows. Confirm activation rotation, tensor offsets and tokenization.
2. Compare one actual fork projection against the PyTorch oracle, then full model logits on 32 fixed prompts, with and without an unmerged rank-16 GGUF adapter. Preregister a tolerance based on the fork's own quantization spread.
3. On the RTX 5080, measure 8K/32K inference peak VRAM, then one **single** 2048-token frozen-base adapter backward pass with a true packed-resident kernel. The presence of an autograd oracle in this directory does not establish such a kernel.
4. Report peak VRAM, train tokens/s, runtime configuration and source hashes. S0 go requires 99% top-1 agreement, measured max-logit discrepancy below the preregistered bound, <=14 GB peak training VRAM, >=300 train tokens/s and <=10 GB inference VRAM at 32K. All criteria remain **NOT_EVALUATED**.
5. If any step fails, stop and record it. No loops, no paid teacher calls or open-ended optimization. Subsequent S1 experiments would use BitAgent open-gate material for development and fresh sealed transfer examples, with an independent acceptor.

The earlier `Control-Harness/agent/signals/MYTHOS_RSI_2026-10-24.md` authority-cut and liveness obligations (P2/P4) remain open. This S0 branch neither certifies those invariants nor promotes any routing/model artifact.
