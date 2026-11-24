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
