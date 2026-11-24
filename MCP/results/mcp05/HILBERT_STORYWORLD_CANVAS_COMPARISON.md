# Pocket Goblin MCP-05 — Matched Hilbert Storyworld + Canvas Art Benchmark

Head-to-head evaluation on the identical Pocket Goblin federation packet (`cfdcad17...`), seed (`523`), temperature (`0`), Goose `create_file` workspace contract, SWMD + SweepWeave topology validator, and headless Chrome 2-qubit Hilbert + Canvas verifier.

## 1. Cold Long-Context Planning (`16K` / `24K` / `32K`)

| Target | Model | Input Tokens | Out Tokens | Prefill tok/s | Decode tok/s | Wall (s) | Direct Plan Authored |
|---:|---|---:|---:|---:|---:|---:|:---:|
| 16,384 | Inky Mini (`IQ4_NL`, 2 loops) | 16,384 | 1,024 | 16.53 | 8.00 | 1119.20 | No (bounded repair) |
| 16,384 | Bonsai 2 27B (`Q4_0` unrotated) | 16,385 | 690 | 1100.68 | 31.87 | 36.55 | **Yes** |
| 24,576 | Inky Mini (`IQ4_NL`, 2 loops) | 24,577 | 1,024 | 54.08 | 17.34 | 513.58 | No (bounded repair) |
| 24,576 | Bonsai 2 27B (`Q4_0` unrotated) | 24,577 | 683 | 1003.84 | 29.15 | 47.97 | **Yes** |
| 32,768 | Inky Mini (`IQ4_NL`, 2 loops) | 32,768 | 1,024 | 47.72 | 17.59 | 745.16 | No (bounded repair) |
| 32,768 | Bonsai 2 27B (`Q4_0` unrotated) | 32,767 | 686 | 68.32 | 1.20 | 1051.72 | **Yes** |

## 2. Full 16K Hilbert Storyworld + 35-Scene Canvas Edition

| Metric | Inky Mini (`IQ4_NL`, 2 loops) | Bonsai 2 27B (`Q4_0` unrotated) |
|---|---:|---:|
| Storyworld Title | *The City That Lost Its Night Bell* | *The City That Lost Its Bell* |
| Encounters / Endings | 30 / 5 | 30 / 5 |
| Scene Words | 5,445 | 5,419 |
| Distinct Choice Labels | 23 | **85** |
| First-Attempt Stage Passes | 13 / 24 | **22 / 26** |
| Timed HTTP-200 Goose Turns | 89 | 36 |
| Preflight Token Parity | 89 / 89 | 36 / 36 |
| Total Prompt Tokens In | 234,586 | 97,194 |
| Uncached / KV-Cached Prompt In | 123,687 / 110,899 | 82,498 / 14,696 |
| Total Completion Tokens Out | 149,847 | 41,941 |
| Aggregate Prefill tok/s | 36.01 | **830.79** |
| Aggregate Decode tok/s | 9.48 | **8.20** |
| Cumulative Provider Wall (s) | 19351.5 | **5216.1** |
| Canvas Landmarks / Palettes | 35 / 33 | 30 / 19 |
| Visited Scenes / Distinct Canvas Hashes | 32 / 32 | **33 / 33** |
| Mean Quantized Colors | 87.41 | **124.76** |
| Edge Density / Mean Pairwise RGB L1 | 0.0194 / 15.91 | **0.0423 / 49.87** |
| Secret Endings Unlocked (Swap-Sensitive) | 2 / 2 (`verified`) | 2 / 2 (`verified`) |

## 3. 100% Pure-GPU (`0` CPU Offload) + Pocket Goblin Curated Context TPS Benchmark

Reloading `Ternary-Bonsai-2-27B-unrotated-Q4_0.gguf` (`14.48 GiB` weights) 100% in RTX 5080 VRAM (`-ngl 99`, zero `-ot` CPU tensor offload) and using Pocket Goblin (`MCP.mcp05.service` SetTRM shard routing + `WorkingSet` curation) to curate retrieval and Canvas stage budgets inside `4,608`–`7,168` tokens eliminates CPU FFN offload and WDDM PCIe paging:

### Pure-GPU Context Window & VRAM Sweep (`0` CPU Layers)

| Context Slot | VRAM Used / Free (`MiB`) | Prompt Tokens | Prefill `tok/s` | Decode `tok/s` |
|---:|---:|---:|---:|---:|
| 4,096 | 14,472 / 1,444 | 29 | 45.20 | **34.78** |
| 4,608 | 14,444 / 1,472 | 2,189 | 688.52 | **34.57** |
| 6,144 | 14,548 / 1,368 | 3,865 | 872.93 | **32.58** |
| 7,168 | 14,560 / 1,356 | 3,072 | 929.89 | **36.78** |
| 8,192 | 14,626 / 1,290 | 3,865 | 861.85 | **31.45** |
| 10,240 | 14,702 / 1,214 | 3,865 | 852.76 | **31.83** |
| 12,288 | 14,780 / 1,136 | 3,865 | 829.41 | **31.38** |

### Pure-GPU Pocket Goblin Curated Storyworld + Canvas Remainder Stages

- **Curated Federation Packet**: `4` chunks (`9,562` bytes) assembled in `0.570s` (`3,072` tokens vs `16,384`–`32,768` uncurated — **81.2%–90.6% input token reduction**)
- **Aggregate Throughput (`8` HTTP-200 Goose turns, `8/8` preflight parity)**: **`644.83 tok/s` prefill** (peak `929.89`), **`32.86 tok/s` decode** (range `28.53`–`36.78 tok/s`, **6.6× faster decode** than 10-layer CPU FFN offload at `5.01 tok/s` and **27.9× faster** than 32K WDDM paging at `1.19 tok/s`)
- **Headless Chrome Verification**: `33/33` distinct Canvas SHA-256 hashes, `187.33` mean quantized colors, both secret endings verified (`verified`)

| Phase | Accepted | Prompt In | Cached | Uncached | Out | Prefill `tok/s` | Decode `tok/s` | Wall (s) |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|
| `curated-plan-attempt-01` | Yes | 3,074 | 0 | 3,074 | 713 | 821.31 | **33.53** | 25.01 |
| `curated-math-draft-attempt-02` | Yes | 3,072 | 0 | 3,072 | 699 | 929.89 | **36.78** | 22.31 |
| `canvas-art-map-attempt-01` | Yes | 1,256 | 436 | 820 | 1,744 | 784.67 | **35.95** | 49.55 |
| `canvas-page-attempt-01` | Retry | 651 | 0 | 651 | 443 | 265.60 | **31.06** | 16.74 |
| `canvas-page-attempt-02` | Yes | 671 | 519 | 152 | 390 | 152.99 | **31.16** | 13.50 |
| `canvas-player-attempt-01` | Retry | 2,189 | 0 | 2,189 | 1,910 | 688.52 | **34.57** | 58.45 |
| `canvas-player-attempt-02` | Yes | 2,278 | 2,057 | 221 | 1,844 | 300.68 | **28.53** | 65.36 |
| `canvas-renderer-attempt-02` | Yes | 1,246 | 0 | 1,246 | 2,199 | 549.50 | **32.65** | 69.64 |
