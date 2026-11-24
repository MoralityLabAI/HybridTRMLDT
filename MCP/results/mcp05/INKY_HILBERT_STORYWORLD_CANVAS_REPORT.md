# Inky Mini (IQ4_NL, 2 loops) — Goose Pocket Goblin Hilbert Storyworld + Canvas Art Benchmark

- **Model**: `inky-mini-varloop` (`TML-d24-varloop-research-IQ4_NL.gguf`, 2 loops, 10 CPU MoE layers, RTX 5080 Laptop GPU, 34,816-token slot)
- **Storyworld**: *The City That Lost Its Night Bell* — 30 encounters, 5,445 scene words, 90 options, 5 endings (3 public + 2 non-commuting Hilbert secret endings)
- **Browser & Canvas Verification**: 35 mapped scenes, 35 landmarks, 33 distinct palettes, **32/32** distinct Canvas SHA-256 hashes, **87.41** mean quantized colors (`124` max), `0.0194` edge density, `15.91` mean pairwise RGB L1, both secret endings unlocked in headless Chrome (`status: verified`)

## 1. Sized Long-Context Planning (`16K` / `24K` / `32K`)

| Target | Actual Input | Completion Out | Prefill tok/s | Decode tok/s | Wall (s) | Preflight Parity | Direct Plan Authored |
|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 16,384 | 16,384 | 1,024 | 16.53 | 8.00 | 1119.20 | Yes | No (bounded repair) |
| 24,576 | 24,577 | 1,024 | 54.08 | 17.34 | 513.58 | Yes | No (bounded repair) |
| 32,768 | 32,768 | 1,024 | 47.72 | 17.59 | 745.16 | Yes | No (bounded repair) |

## 2. 16K Conveyor + Canvas Turn Totals

- **Timed HTTP 200 Goose Turns**: `89` (`89/89` exact `/apply-template` + `/tokenize` preflight parity)
- **Prompt Tokens In**: `234,586` total (`123,687` uncached + `110,899` KV-cached)
- **Completion Tokens Out**: `149,847`
- **Aggregate Throughput**: `36.01 tok/s` prefill, `9.48 tok/s` decode (`19351.5s` cumulative provider wall)

## 3. Accepted Turn Per Conveyor & Canvas Stage

| Stage | Accepted Phase | Prompt In | Cached | Uncached | Out | Prefill tok/s | Decode tok/s | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `act-complete-1` | `act-complete-1-attempt-01` | 2,846 | 308 | 2,538 | 192 | 61.04 | 11.88 | 57.70 |
| `act-complete-2` | `act-complete-2-attempt-05` | 1,196 | 308 | 888 | 218 | 75.92 | 16.99 | 24.50 |
| `act-complete-3` | `act-complete-3-attempt-01` | 1,192 | 308 | 884 | 180 | 42.75 | 11.86 | 35.80 |
| `act-complete-4` | `act-complete-4-attempt-01` | 1,200 | 308 | 892 | 213 | 39.15 | 11.88 | 40.64 |
| `act-complete-5` | `act-complete-5-attempt-01` | 1,134 | 308 | 826 | 151 | 33.18 | 14.22 | 35.47 |
| `canvas-art-map` | `canvas-art-map-attempt-03` | 1,547 | 1,279 | 268 | 2,959 | 54.85 | 12.50 | 241.64 |
| `canvas-page` | `canvas-page-attempt-07` | 547 | 308 | 239 | 1,541 | 17.46 | 3.69 | 430.84 |
| `canvas-renderer` | `canvas-renderer-attempt-01` | 1,209 | 1,174 | 35 | 1,304 | 25.04 | 6.12 | 214.44 |
| `canvas-renderer-repair` | `canvas-renderer-repair-attempt-01` | 1,423 | 308 | 1,115 | 3,281 | 19.47 | 3.28 | 1058.80 |
| `characterize` | `characterize-attempt-04` | 1,186 | 308 | 878 | 333 | 76.00 | 17.68 | 30.38 |
| `encounter-01` | `encounter-01-attempt-01` | 1,611 | 308 | 1,303 | 1,524 | 80.33 | 17.22 | 104.66 |
| `encounter-04` | `encounter-04-attempt-05` | 2,849 | 308 | 2,541 | 1,189 | 58.28 | 14.61 | 124.94 |
| `encounter-07` | `encounter-07-attempt-06` | 2,937 | 2,904 | 33 | 1,370 | 42.06 | 12.07 | 114.25 |
| `encounter-10` | `encounter-10-attempt-01` | 3,001 | 308 | 2,693 | 2,017 | 62.13 | 12.32 | 207.02 |
| `encounter-16` | `encounter-16-attempt-01` | 3,382 | 308 | 3,074 | 1,870 | 37.03 | 17.66 | 188.88 |
| `encounter-19` | `encounter-19-attempt-02` | 3,401 | 3,356 | 45 | 1,844 | 35.20 | 11.69 | 158.94 |
| `encounter-22` | `encounter-22-attempt-01` | 3,342 | 308 | 3,034 | 1,939 | 41.10 | 11.76 | 238.61 |
| `encounter-25` | `encounter-25-attempt-05` | 2,243 | 308 | 1,935 | 1,715 | 37.92 | 11.31 | 202.55 |
| `encounter-28` | `encounter-28-attempt-01` | 2,148 | 308 | 1,840 | 1,930 | 35.34 | 11.15 | 225.06 |
| `endings` | `endings-attempt-01` | 2,421 | 308 | 2,113 | 1,012 | 32.59 | 10.17 | 164.27 |
| `late-stage-holistic` | `late-stage-holistic-attempt-05` | 1,433 | 308 | 1,125 | 324 | 41.69 | 10.44 | 57.94 |
| `math-draft` | `math-draft-attempt-13` | 1,687 | 308 | 1,379 | 998 | 79.02 | 17.61 | 74.08 |
| `plan-repair` | `plan-repair-attempt-02` | 620 | 587 | 33 | 1,015 | 53.34 | 18.57 | 55.25 |
| `recharacterize` | `recharacterize-attempt-01` | 1,380 | 308 | 1,072 | 334 | 48.81 | 12.79 | 48.00 |

## 4. Non-Commuting 2-Qubit Secret Endings

| Secret Ending | Draft Born Access | Final Born Access | Clue Access | Adjacent Gate-Swap Closes Gate |
|---|---:|---:|---:|:---:|
| `end_bell_as_question` | 20.536% | 20.536% | 44.006% | Yes |
| `end_bell_as_answer` | 32.833% | 32.833% | 70.363% | Yes |
