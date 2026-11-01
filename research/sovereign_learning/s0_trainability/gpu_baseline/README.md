# Q3c GPU inference baseline

`run_gpu_baseline.ps1` runs `llama-bench` with full offload (`-ngl 99 -fa on -p 512 -n 128 -r 3`) at context depths of 8,192 and 32,768 tokens. While it runs, nvidia-smi samples VRAM every 500 ms. The run took place inside a vramorama lease (`--vram 11G --max-commit 80 --strict-fifo --guard-power 600`).

## Result (2026-11-02)

- Model: Ternary-Bonsai-2-27B-PTQ1_0.gguf (sha256 `53107f53…`, 5.94 GB).
- Build: llama-prism adfffbe41 (CUDA).
- GPU: RTX 5080 Laptop, 16 GB.
- Before the run: 0 MiB in use and no other GPU process.
- Raw files: `../receipts/gpu_baseline_20261102/`.

| Depth | Prompt 512 tok/s | Generate 128 tok/s | Peak VRAM (MiB) |
|---:|---:|---:|---:|
| 8,192 | 1,234 ± 33 | 40.5 ± 24.4 | 6,976 |
| 32,768 | 950 ± 54 | 47.7 ± 0.9 | 8,480 |

Peak VRAM at 32K is 8.3 GiB, which is under the S0 go criterion of 10 GB.

The 8K generation figure has a large spread: 8K was the first test, and its repetitions include the GPU clock ramp. The 32K run, at 47.7 ± 0.9, is the stable generation number.

The power column in `vram_samples.csv` contains one invalid nvidia-smi reading (628,233 W) and is not used.
