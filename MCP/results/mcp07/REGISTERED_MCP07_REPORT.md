# MCP-07 Registered Evaluation Report: Geometry-Adaptive Recurrent Memory Routing

- **Experiment ID**: `MCP-07-ADAPTIVE-GEOMETRY-V1`
- **Parent baseline**: `gym/mcp06-spatial-memory` (`3c03ccc6c4a0cac059c670011a55ea7ad9666de9`)
- **Frozen protocol commit (prior to held-out evaluation)**: `0620aff37dff5b8593f52686f4e7daa37748e651`
- **Frozen `manifest.json` SHA-256**: `4ac105df37ec9328e8b4088a1536396c5f5b3b7af62a7720a3a6c9be9c319b3b`
- **Execution environment**: CPU-only (`CUDA_VISIBLE_DEVICES=""`), 6 parallel 1-thread CPU workers, deterministic algorithms enabled (`torch.use_deterministic_algorithms(True)`)
- **Wall time**: `3,402.38 s` (`56.71 min`) | **Peak process RSS**: `676.62 MiB` (`709,488,640` bytes, includes Python + PyTorch + 128K multi-projection tables; strictly `< 2 GiB`)
- **Scale**: `3,317,760` paired root episodes (`3` seeds × `5` capacities `16K–128K` × `8` query families × `128` paired roots × `4` probe budgets × `3` corruption rates × `18` arms), `36` saved checkpoints (`24` core + `9` architecture/supervision ablations + `3` distribution-shift switching checkpoints).

> **Scope boundary**: Sections 1–6 evaluate the **synthetic multi-projection spatial retrieval proxy** (physical record slots, not tokens). Section 7 reports the **native Storyworld integration** (`MCP.mcp05.hilbert_storyworld`) on a strictly separate track and does **not** conflate synthetic retrieval gains with end-to-end story planning capability.

---

## 1. Registered Gate Verdicts (32K, 16 Probes, 0% Corruption; 3,072 Paired Roots)

| Gate | Specification | Observed Result | 95% Paired Bootstrap CI (`2,000` resamples) | Threshold | Status |
|---|---|---|---|---|---|
| **Primary Gate** | `recurrent_switch` (Arm E) vs strongest non-oracle fixed layout (`det_row`, Arm A) | **`+18.62 pp`** (`66.08%` vs `47.46%`) | `[+16.41, +20.80] pp` (`p > 0 = 1.0000`) | `>= +5.00 pp` | **PASS** |
| Primary (vs best learned fixed) | `recurrent_switch` (Arm E) vs strongest learned fixed layout (`fixed_col`, Arm C) | **`+21.26 pp`** (`66.08%` vs `44.82%`) | `[+19.24, +23.31] pp` (`p > 0 = 1.0000`) | `>= +5.00 pp` | **PASS** |
| Primary (adaptive arm) | `hier_adaptive` (Arm F) vs strongest non-oracle fixed layout (`det_row`) | **`+18.62 pp`** (`66.08%` vs `47.46%`) | `[+16.47, +20.87] pp` (`p > 0 = 1.0000`) | `>= +5.00 pp` | **PASS** |
| **Secondary Gate (Reads)** | Mean charged read reduction: `hier_adaptive` (Arm F) vs `recurrent_switch` (Arm E, shared weights) | **`-46.53%`** (`275.77` vs `515.72` reads) | `[+45.11%, +47.86%]` | `>= 15.0%` | **PASS** |
| **Secondary Gate (Retention)** | Verified completion loss: `recurrent_switch` minus `hier_adaptive` | **`0.00 pp`** (`0 / 3,072` premature stops) | `[0.00, 0.00] pp` | `<= 1.00 pp` | **PASS** |

### Additional Registered Paired Comparisons at 32K / 16 Probes / Clean
- `recurrent_switch` vs `router_single` (Arm D, single selection per episode): **`+17.12 pp`**, 95% CI `[+15.23, +19.04] pp` (`p > 0 = 1.0000`).
- `recurrent_switch` vs `det_router` (Arm B, geometry-aware deterministic router): **`+6.54 pp`**, 95% CI `[+4.26, +8.53] pp` (`p > 0 = 1.0000`). Notably, `hier_adaptive` Pareto-dominates `det_router` on **both** completion (`66.08%` vs `59.54%`) and mean charged reads (`275.8` vs `306.5`).
- `recurrent_switch` vs `rand_routing` (Arm G, matched-switch-rate randomized routing): **`+39.97 pp`**, 95% CI `[+37.96, +41.93] pp` (`p > 0 = 1.0000`).
- `oracle_proj` (Arm H, privileged upper bound) minus `recurrent_switch`: **`+19.56 pp`**, 95% CI `[+17.94, +21.16] pp` (`85.64%` vs `66.08%`), quantifying remaining non-oracle headroom.

### Per-Seed Breakdown (32K, 16 Probes, Clean; 1,024 Paired Roots per Seed)

| Seed | `recurrent_switch` | `hier_adaptive` | Best Fixed All (`det_row`) | Delta vs Best Fixed All | Best Learned Fixed (`fixed_col`) | Delta vs Best Learned Fixed | Secondary Read Reduction | Secondary Comp Loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `107` | `63.67%` | `63.67%` | `47.56%` | **`+16.11 pp`** | `42.19%` | **`+21.48 pp`** | **`-45.64%`** | `0.00 pp` |
| `109` | `67.87%` | `67.87%` | `46.97%` | **`+20.90 pp`** | `44.24%` | **`+23.63 pp`** | **`-47.33%`** | `0.00 pp` |
| `113` | `66.70%` | `66.70%` | `47.85%` | **`+18.85 pp`** | `48.05%` | **`+18.65 pp`** | **`-46.61%`** | `0.00 pp` |
| **Mean** | **`66.08%`** | **`66.08%`** | **`47.46%`** | **`+18.62 pp`** | **`44.82%`** | **`+21.26 pp`** | **`-46.53%`** | **`0.00 pp`** |

---

## 2. Full Arm Comparison & Resource Accounting (32K, 16 Probes, Clean)

All learned fixed-layout controllers (Arm C) use `56,024` active parameters (`>=` the `54,460` active parameters of routed Arms D, E, F, G), eliminating parameter-count advantage for switching. All index storage (`8.375` bytes/slot/projection), anchor lookups (`+1` read), projection switches (`+2` reads), and cross-projection physical slot overlaps are charged conservatively.

| Group | Arm | Active Params | Comp (%) | Probes | Charged Reads | Unique Phys Slots | Overlap Reads | Switches | Rec Updates | Premature Stops | Index Storage (32K) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | `det_row` *(best fixed all)* | 0 | **47.46%** | 11.71 | 375.8 | 374.6 | 0.1 | 0.00 | 0.0 | 0 | 268.0 KiB |
| A | `det_col` | 0 | 42.29% | 11.73 | 376.5 | 375.3 | 0.1 | 0.00 | 0.0 | 0 | 268.0 KiB |
| A | `det_hilbert` | 0 | 41.99% | 11.60 | 372.4 | 371.2 | 0.1 | 0.00 | 0.0 | 0 | 268.0 KiB |
| A | `det_morton` | 0 | 31.45% | 12.48 | 400.6 | 399.4 | 0.1 | 0.00 | 0.0 | 0 | 268.0 KiB |
| A | `det_rand_coloc` | 0 | 1.73% | 15.88 | 509.3 | 508.3 | 0.0 | 0.00 | 0.0 | 0 | 268.0 KiB |
| A | `det_rand_full` | 0 | 3.45% | 15.73 | 504.5 | 503.5 | 0.0 | 0.00 | 0.0 | 0 | 268.0 KiB |
| B | `det_router` | 0 | 59.54% | 9.50 | 306.5 | 299.1 | 4.8 | 0.73 | 0.0 | 0 | 1,072.0 KiB |
| C | `fixed_row` | 56,024 | 34.34% | 16.00 | 513.1 | 511.9 | 0.1 | 0.00 | 33.7 | 0 | 268.0 KiB |
| C | `fixed_col` *(best learned fixed)* | 56,024 | **44.82%** | 16.00 | 513.1 | 511.9 | 0.1 | 0.00 | 33.7 | 0 | 268.0 KiB |
| C | `fixed_hilbert` | 56,024 | 23.11% | 16.00 | 513.0 | 512.0 | 0.0 | 0.00 | 32.7 | 0 | 268.0 KiB |
| C | `fixed_morton` | 56,024 | 27.86% | 16.00 | 513.0 | 511.9 | 0.1 | 0.00 | 31.7 | 0 | 268.0 KiB |
| C | `fixed_rand_coloc` | 56,024 | 1.53% | 16.00 | 513.0 | 512.0 | 0.0 | 0.00 | 28.3 | 0 | 268.0 KiB |
| C | `fixed_rand_full` | 56,024 | 2.83% | 16.00 | 513.0 | 512.0 | 0.0 | 0.00 | 28.0 | 0 | 268.0 KiB |
| D | `router_single` | 54,460 | 48.96% | 16.00 | 513.1 | 511.9 | 0.1 | 0.00 | 33.0 | 0 | 1,072.0 KiB |
| E | **`recurrent_switch`** | **54,460** | **66.08%** | 16.00 | 515.7 | 486.7 | 25.3 | 1.31 | 34.0 | 0 | 1,072.0 KiB |
| F | **`hier_adaptive`** | **54,460** | **66.08%** | **8.53** | **275.8** | **264.4** | **8.6** | **0.83** | **18.9** | **0** | 1,072.0 KiB |
| G | `rand_routing` | 54,460 | 26.11% | 16.00 | 515.7 | 479.0 | 33.0 | 1.32 | 34.0 | 0 | 1,072.0 KiB |
| H | `oracle_proj` *(privileged upper bound)* | 54,460 | 85.64% | 16.00 | 518.1 | 482.1 | 29.9 | 2.47 | 33.7 | 0 | 1,072.0 KiB |

**Index overhead accounting**: Multi-projection arms (B, D, E, F, G, H) require `4×` the index storage (`1,072 KiB` vs `268 KiB` at 32K; `4,288 KiB` at 128K) and construct all 4 structured views (`~26 ms` at 32K, `~310 ms` at 128K). Cross-projection overlap reads average `25.3` slots/ep for fixed-budget `recurrent_switch` and `8.6` slots/ep for `hier_adaptive` (all charged in full without caching discounts).

---

## 3. Per-Family Geometry Separation (32K, 16 Probes, Clean; 384 Roots/Family)

| Query Family | `det_row` | `det_col` | `det_hilb` | `det_mort` | `det_router` | `fix_row` | `fix_col` | `fix_hilb` | `fix_mort` | `router_sgl` | **`rec_switch`** | **`hier_ad`** | `rand_rt` | `oracle` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `horizontal` | 81.8% | 41.1% | 55.5% | 57.0% | 81.8% | **94.8%** | 43.5% | 20.6% | 42.4% | 89.6% | **93.2%** | **93.2%** | 38.8% | 95.8% |
| `vertical` | 33.1% | 69.8% | 51.6% | 13.5% | 69.8% | 29.7% | **100.0%** | 26.3% | 25.8% | 93.8% | **100.0%** | **100.0%** | 45.3% | 100.0% |
| `diagonal` | 58.6% | 46.9% | 89.3% | 72.4% | 58.6% | 47.4% | 58.6% | 80.2% | 74.7% | 86.7% | **96.4%** | **96.4%** | 44.8% | 98.4% |
| `isotropic` | 94.5% | 95.6% | 85.2% | 82.6% | 84.9% | 43.5% | 77.1% | 36.2% | 28.1% | 42.7% | 68.8% | 68.8% | 47.7% | 88.3% |
| `sparse_long_range` | 12.5% | 0.0% | 0.0% | 1.3% | **36.5%** | 8.6% | 0.0% | 4.2% | 21.1% | 15.4% | 32.6% | 32.6% | 5.7% | 79.2% |
| `multihop` | 28.6% | 32.6% | 37.0% | 21.9% | 46.6% | 19.0% | 41.1% | 6.5% | 9.6% | 29.4% | **69.3%** | **69.3%** | 10.7% | 85.4% |
| `regime_shift` | 53.1% | 37.0% | 10.2% | 1.3% | **72.1%** | 11.2% | 20.3% | 5.2% | 4.7% | 22.4% | 40.1% | 40.1% | 13.5% | 64.8% |
| `corrupted_replica` | 17.4% | 15.4% | 7.3% | 1.6% | 26.0% | 20.6% | 18.0% | 5.7% | 16.4% | 11.7% | **28.4%** | **28.4%** | 2.3% | 73.2% |

Key per-family observations:
- Fixed layouts exhibit sharp geometry trade-offs: `fixed_row` wins `horizontal` (`94.8%`) but drops to `29.7%` on `vertical`; `fixed_col` achieves `100.0%` on `vertical` but drops to `43.5%` on `horizontal` and `0.0%` on `sparse_long_range`.
- `recurrent_switch` matches or beats every learned fixed layout on **all 8 families except `horizontal`** (`93.2%` vs `94.8%` `fixed_row`) and **`isotropic`** (`68.8%` vs `77.1%` `fixed_col`), while delivering massive gains on dynamic families: **`multihop` `+28.2 pp`** over best learned fixed (`69.3%` vs `41.1%`) and **`+39.9 pp`** over `router_single` (`29.4%`).
- Honest weakness: `det_router` outperforms `recurrent_switch` on `regime_shift` (`72.1%` vs `40.1%`) and `isotropic` (`84.9%` vs `68.8%`), while `recurrent_switch` decisively beats `det_router` on `vertical` (`+30.2 pp`), `diagonal` (`+37.8 pp`), `multihop` (`+22.7 pp`), and `horizontal` (`+11.4 pp`).

---

## 4. Capacity Transfer (16K → 128K), Probe Budgets, and Corruption Resilience

### 4.1 Capacity Extrapolation from 16K-Only Training (`b = 16, c = 0%`)

| Arm | 16K (train cap) | 24K | 32K (gate) | 64K | 128K (8× train) |
|---|---:|---:|---:|---:|---:|
| `det_row` | 47.59% | 46.48% | 47.46% | 40.82% | 40.98% |
| `det_col` | 48.54% | 31.09% | 42.29% | 42.55% | 47.82% |
| `det_hilbert` | 43.07% | 41.83% | 41.99% | 38.93% | 40.20% |
| `det_morton` | 33.30% | 32.19% | 31.45% | 29.95% | 30.44% |
| `det_router` | 64.03% | **67.90%** | 59.54% | 54.52% | 54.17% |
| `fixed_row` | 35.87% | 36.36% | 34.34% | 32.78% | 33.59% |
| `fixed_col` | 49.58% | 31.77% | 44.82% | 44.92% | 28.29% |
| `fixed_hilbert` | 27.41% | 23.86% | 23.11% | 21.81% | 20.67% |
| `fixed_morton` | 29.17% | 28.61% | 27.86% | 24.25% | 24.77% |
| `fixed_rand_coloc` | 2.86% | 1.46% | 1.53% | 0.62% | 0.36% |
| `fixed_rand_full` | 5.31% | 3.03% | 2.83% | 1.20% | 0.65% |
| `router_single` | 50.03% | 47.56% | 48.96% | 47.72% | 47.46% |
| **`recurrent_switch`** | **70.05%** | 65.49% | **66.08%** | **63.96%** | **61.62%** |
| **`hier_adaptive`** | **70.05%** | 65.49% | **66.08%** | **63.96%** | **61.62%** |
| `rand_routing` | 29.39% | 25.16% | 26.11% | 22.53% | 20.96% |
| `oracle_proj` *(upper bound)* | 83.17% | 85.81% | 85.64% | 88.90% | 89.26% |

- **8× capacity transfer**: Trained strictly at 16K, `recurrent_switch` / `hier_adaptive` retain **`61.62%` at 128K** (beating best fixed layout `det_col = 47.82%` by **`+13.80 pp`**, best learned fixed `fixed_row = 33.59%` by **`+28.03 pp`**, and `det_router = 54.17%` by **`+7.45 pp`**).
- **Non-power-of-two nuance at 24K**: At `24K` (`W=128, H=192`, `6` pages/column instead of a power of two `4` or `8`), `fixed_col` dips sharply (`49.58% → 31.77%`), and `det_router` (`67.90%`) slightly exceeds `recurrent_switch` (`65.49%`, `-2.41 pp`), whereas `recurrent_switch` still leads best fixed layout at 24K by `+19.01 pp`.

### 4.2 Probe Budget Scaling at 32K Clean (`Completion % (Mean Charged Reads)`)

| Arm | `B = 4` | `B = 8` | `B = 16` | `B = 32` |
|---|---:|---:|---:|---:|
| `det_row` | 17.0% (120) | 32.3% (221) | 47.5% (376) | 56.9% (622) |
| `det_router` | 35.1% (112) | 47.8% (188) | 59.5% (306) | 67.1% (494) |
| `fixed_col` | 22.8% (129) | 31.6% (257) | 44.8% (513) | 59.0% (1,025) |
| `router_single` | 35.5% (129) | 41.6% (257) | 49.0% (513) | 54.2% (1,025) |
| **`recurrent_switch`** | **42.7% (130)** | **54.9% (259)** | **66.1% (516)** | **73.6% (1,028)** |
| **`hier_adaptive`** | **42.7% (107)** | **54.9% (174)** | **66.1% (276)** | **73.6% (431)** |
| `rand_routing` | 11.2% (130) | 19.0% (259) | 26.1% (516) | 37.3% (1,028) |
| `oracle_proj` *(upper bound)* | 52.3% (131) | 72.0% (261) | 85.6% (518) | 93.0% (1,031) |

Adaptive halting scales cleanly with budget: `-17.7%` reads at `B=4`, `-32.8%` at `B=8`, `-46.5%` at `B=16`, and **`-58.1%`** at `B=32` (`431` vs `1,028` reads) with `0.00 pp` loss at every budget.

### 4.3 Corruption Sweep & Replica Co-Location Isolation (32K, 16 Probes)

| Arm | Replicas Co-Located? | `c = 0%` | `c = 5%` | `c = 15%` | `0% → 15%` Drop |
|---|---|---:|---:|---:|---:|
| `fixed_row` | Yes (`100%`) | 34.34% | 34.38% | 33.69% | `-0.65 pp` (`-1.9%` rel) |
| `fixed_col` | No (`0%`, adj cols) | 44.82% | 43.00% | 38.87% | **`-5.95 pp`** (`-13.3%` rel) |
| `fixed_hilbert` | Mostly (`~95%`) | 23.11% | 23.11% | 22.66% | `-0.45 pp` (`-1.9%` rel) |
| `fixed_morton` | Yes (`100%`) | 27.86% | 27.83% | 27.12% | `-0.74 pp` (`-2.7%` rel) |
| `fixed_rand_coloc` | Yes (`100%`) | 1.53% | 1.53% | 1.50% | `-0.03 pp` (`-2.0%` rel) |
| `fixed_rand_full` | No (`0.09%`) | 2.83% | 2.73% | 2.25% | `-0.58 pp` (`-20.5%` rel) |
| `router_single` | Mixed | 48.96% | 48.11% | 45.54% | `-3.42 pp` |
| **`recurrent_switch` / `hier_adaptive`** | Mixed (adaptive) | **66.08%** | **65.59%** | **63.41%** | `-2.67 pp` (`-4.0%` rel) |
| `det_router` | Mixed | 59.54% | 59.34% | 57.98% | `-1.56 pp` |

**Co-location isolation finding**: Directly addressing MCP-06's open question, separating replicas (`rand_full`) roughly doubles random-probe hit rate under clean memory (`2.83%` vs `1.53%`) because two random pages are hit twice as often as one co-located page, **but** under 15% physical corruption `rand_full` loses `20.5%` relative completion vs `2.0%` for `rand_coloc` (matching the theoretical $c^2 = 2.25\%$ dual-replica failure rate). The same mechanism explains why `fixed_col` drops `-5.95 pp` at `c=15%` while `fixed_row` drops only `-0.65 pp`.

---

## 5. Architecture, Supervision, and Distribution-Shift Ablations (32K, 16 Probes)

### 5.1 Supervision, Cross-Projection Communication, and Scaling (`<= 128K`)

| Variant | Active Params | Clean 32K | `c = 15%` 32K | Adaptive Reads | Paired Delta vs `action_set_54k` (95% CI) | Interpretation |
|---|---:|---:|---:|---:|---|---|
| `action_set_54k` *(baseline)* | 54,460 | 66.08% | 63.41% | 275.8 | — | Registered 54K baseline |
| `single_best_54k` | 54,460 | 66.21% | 63.31% | 273.8 | `+0.13 pp` `[-1.20, +1.46]` | **Null result**: no reliable accuracy gap over optimal-action-set loss |
| `no_comm_54k` | 54,188 | 65.95% | 63.22% | 274.8 | `-0.13 pp` `[-1.40, +1.17]` | **Null result from scratch** (cf. strong `-12.04 pp` inference lesion in Mechanistic Report) |
| `scaled_116k` *(<= 128K)* | 117,224 | 67.32% | 64.49% | 268.9 | `+1.24 pp` `[-0.20, +2.77]` | Modest gain across all 3 seeds; 54K is near capacity saturation |

### 5.2 Distribution-Shift Test (`80/20 → 20/80` Mixture; Trained at 16K, Evaluated at 32K, 3,072 Roots)

Models were trained at 16K on an `80%` majority (`horizontal, isotropic, sparse_long_range, multihop`) / `20%` minority (`vertical, diagonal, regime_shift, corrupted_replica`) mixture and evaluated frozen at 32K on the inverted `20% / 80%` mixture:

| Arm | 32K Shifted Completion | Mean Charged Reads |
|---|---:|---:|
| `fixed_row` | 33.33% | 513.0 |
| `fixed_col` *(best fixed under shift)* | 47.43% | 513.0 |
| `fixed_hilbert` | 25.85% | 513.0 |
| `fixed_morton` | 28.09% | 513.0 |
| `router_single` | 48.83% | 513.0 |
| `det_router` | 58.46% | 314.5 |
| **`recurrent_switch`** | **60.32%** | 516.0 |
| **`hier_adaptive`** | **60.32%** | **294.0** |

- Under inverted distribution shift, `recurrent_switch` drops `-5.76 pp` relative to balanced training (`66.08% → 60.32%`), yet still outperforms the strongest fixed layout under shift (`fixed_col`) by **`+12.89 pp`** (95% CI `[+10.87, +14.91]`) and `router_single` by **`+11.49 pp`** (95% CI `[+9.70, +13.22]`), while `hier_adaptive` beats `det_router` (`60.32%` vs `58.46%`) at lower read cost (`294.0` vs `314.5`).

---

## 6. Native Storyworld Integration (Strictly Separate from Synthetic Proxy)

- **Pinned Storyworld**: `MCP/results/mcp05/inky-hilbert-16k/storyworld.hilbert.json` (`SHA-256: 7685eee95e79698da3ca1a1d2a1817a25c69e294ad4adb6861a24049df426b5b`).
- Native verifier functions imported directly from `MCP.mcp05.hilbert_storyworld` (`START`, `transition`, `classify`, `probability`, `replay_path`, `counterfactual`).

### 6.1 Track A: MCP-05 Archived Secret-Ending Witness Replay (32K, 16 Probes/Decision, Clean; 6 Routes = 3 Seeds × 2 Endings)

All native archived witnesses and adjacent-swap controls passed exact verification (`swap_control_verified = True` on `100%` of runs). Each route chains 25 sequential decisions across mixed spatial geometries:

| Arm | Full 25-Step Routes Solved | Mean Route Step Coverage | Mean Charged Reads / Decision | Swap Control Verified |
|---|---:|---:|---:|---|
| `det_hilbert` | 0 / 6 | 5.3% | 477.4 | True |
| `det_router` | 0 / 6 | 16.7% | 271.8 | True |
| `fixed_row` | 0 / 6 | 8.7% | 513.0 | True |
| `fixed_hilbert` | 0 / 6 | 5.3% | 513.0 | True |
| `router_single` | 0 / 6 | 18.0% | 513.0 | True |
| **`recurrent_switch`** | **1 / 6** | **40.7%** | 514.0 | True |
| **`hier_adaptive`** | **1 / 6** | **40.7%** | **181.0** | True |
| `rand_routing` | 0 / 6 | 4.0% | 514.0 | True |

**Honest assessment**: `recurrent_switch` / `hier_adaptive` are the only arms to complete a full 25-step sequential witness chain (`1/6`, `40.7%` mean step coverage, `64.8%` lower per-decision reads for `hier_adaptive`), yet `1/6` full-chain completion demonstrates that compounding single-step retrieval errors over 25 sequential steps still prevent reliable witness replay without retry/verification loops.

### 6.2 Track B: Independent Witness-Withheld Route Discovery (32K, 6 Probes/Option; 6 Trials/Arm/Corruption)

`strip_archived_witnesses()` mechanically deletes `exact`, `balance.witnesses`, and `balance.counterfactuals` prior to execution (`witness_withheld_verified = True`, enforced by unit test). All 90 scene-option records `(operator, clue, advance)` are encoded into 32K physical memory across mixed geometries; a bounded 1-step native lookahead planner sees **only** retrieved records. Every discovered route is checked on all 5 criteria (legal transitions, required clues, terminal projectors, secret-ending eligibility, noncommuting adjacent-swap counterfactual).

| Condition | Arm | Retrieval Rate | Legal Trans | Req Clues | Projectors | Secret Elig | Counterfactual | **All 5 Verified** | Novel vs Witness | Mean Total Reads |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `c=0.00` | `blind_random` | 0.0% | 100% | 83.3% | 50.0% | 33.3% | 33.3% | 33.3% (2/6) | 100% | 0 |
| `c=0.00` | `det_hilbert` | 28.4% | 100% | 100% | 50.0% | 50.0% | 50.0% | 50.0% (3/6) | 100% | 27,612 |
| `c=0.00` | `det_router` | 40.7% | 100% | 100% | 66.7% | 66.7% | 66.7% | 66.7% (4/6) | 100% | 20,805 |
| `c=0.00` | `fixed_row` | 27.1% | 100% | 100% | 83.3% | 83.3% | 83.3% | 83.3% (5/6) | 100% | 32,553 |
| `c=0.00` | `fixed_col` | 27.2% | 100% | 66.7% | 66.7% | 66.7% | 66.7% | 66.7% (4/6) | 100% | 31,202 |
| `c=0.00` | `fixed_hilbert` | 10.4% | 100% | 16.7% | 33.3% | 0.0% | 0.0% | 0.0% (0/6) | 100% | 40,916 |
| `c=0.00` | `fixed_morton` | 12.5% | 100% | 66.7% | 50.0% | 33.3% | 33.3% | 33.3% (2/6) | 100% | 40,112 |
| `c=0.00` | `router_single` | 53.4% | 100% | 100% | 66.7% | 66.7% | 66.7% | 66.7% (4/6) | 100% | 24,640 |
| `c=0.00` | `recurrent_switch` | **73.2%** | 100% | 83.3% | 66.7% | 66.7% | 66.7% | 66.7% (4/6) | 100% | 21,042 |
| `c=0.00` | **`hier_adaptive`** | **73.2%** | 100% | 83.3% | 66.7% | 66.7% | 66.7% | 66.7% (4/6) | 100% | **11,296** |
| `c=0.00` | `rand_routing` | 18.2% | 100% | 50.0% | 50.0% | 33.3% | 33.3% | 33.3% (2/6) | 100% | 36,196 |
| `c=0.15` | `blind_random` | 0.0% | 100% | 83.3% | 50.0% | 33.3% | 33.3% | 33.3% (2/6) | 100% | 0 |
| `c=0.15` | `det_hilbert` | 27.2% | 100% | 100% | 100% | 100% | 100% | 100% (6/6) | 100% | 28,212 |
| `c=0.15` | `det_router` | 37.4% | 100% | 100% | 83.3% | 83.3% | 83.3% | 83.3% (5/6) | 100% | 22,401 |
| `c=0.15` | `fixed_row` | 25.3% | 100% | 100% | 83.3% | 83.3% | 83.3% | 83.3% (5/6) | 100% | 33,132 |
| `c=0.15` | `fixed_col` | 22.4% | 100% | 66.7% | 83.3% | 50.0% | 50.0% | 50.0% (3/6) | 100% | 33,678 |
| `c=0.15` | `fixed_hilbert` | 10.2% | 100% | 16.7% | 50.0% | 0.0% | 0.0% | 0.0% (0/6) | 100% | 41,045 |
| `c=0.15` | `fixed_morton` | 11.9% | 100% | 50.0% | 50.0% | 16.7% | 16.7% | 16.7% (1/6) | 100% | 40,691 |
| `c=0.15` | `router_single` | 46.2% | 100% | 66.7% | 83.3% | 66.7% | 66.7% | 66.7% (4/6) | 100% | 26,570 |
| `c=0.15` | `recurrent_switch` | **65.4%** | 100% | 83.3% | 66.7% | 66.7% | 66.7% | 66.7% (4/6) | 100% | 22,356 |
| `c=0.15` | **`hier_adaptive`** | **65.4%** | 100% | 83.3% | 66.7% | 66.7% | 66.7% | 66.7% (4/6) | 100% | **13,237** |
| `c=0.15` | `rand_routing` | 16.6% | 100% | 50.0% | 50.0% | 0.0% | 0.0% | 0.0% (0/6) | 100% | 36,682 |

**Critical separation of retrieval vs native planning**:
1. **Verified novel route discovery succeeds**: `hier_adaptive` independently discovers 100% novel (non-archived) routes passing all 5 native Storyworld verification checks for **both** `end_bell_as_question` (length 27–28, swap `[2, 3]`) and `end_bell_as_answer` (length 27–28, swap `[0, 1]` and `[2, 3]`), doubling `blind_random` (`66.7%` vs `33.3%`) while leading memory retrieval cleanly (`73.2%` clean / `65.4%` corrupted vs `10.2%–27.2%` for fixed layouts) and cutting planning read cost by **`-46.3%`** (`11,296` vs `21,042`).
2. **Honest negative/inconclusive planning comparison**: Despite `hier_adaptive` achieving nearly `3×` higher option retrieval than `fixed_row` (`73.2%` vs `27.1%`), `fixed_row` (`5/6 = 83.3%`) and `det_hilbert` at `c=15%` (`6/6 = 100%`) achieved equal or higher terminal secret eligibility over `n=6` trials. Because the underlying 30-scene storyworld has substantial baseline eligibility (`20.5%`/`32.8%`) and `fixed_row` retrieves option `j=0` reliably while falling back on unretrieved options, a 1-step greedy planner over `n=6` episodes is too noisy and myopic to turn higher memory retrieval monotonically into higher terminal ending accuracy. Improving end-to-end native planning requires deeper multi-step tree search over retrieved transitions and larger trial counts.

---

## 7. Protocol Deviations, Limitations, and Artifact Receipts

- **Deviations from frozen manifest**: **None**. Every checkpoint, seed, capacity (`16K–128K`), budget (`4, 8, 16, 32`), corruption (`0%, 5%, 15%`), ablation, native track, and mechanistic intervention ran verbatim from frozen commit `0620aff37dff5b8593f52686f4e7daa37748e651`.
- **Storage packaging note**: To keep the Git repository cleanly under 100 MB with zero >50 MB files, `summary.json` (`21.0 MiB` raw) is committed in the repo as lossless `summary.json.gz` (`~1.8 MiB`) alongside immediately readable `summary_metrics.json`, while the full uncompressed `summary.json` is included inside `MCP/results/mcp07/MCP07_evidence_packet.zip`.
- **Artifact SHA-256 hashes**:
  - `MCP/mcp07/manifest.json`: `4ac105df37ec9328e8b4088a1536396c5f5b3b7af62a7720a3a6c9be9c319b3b`
  - `MCP/results/mcp07/registered/roots.jsonl.gz`: `6e888f4fe7c76b2ce47248db0feac0dd5b9fce6f736db14dff242edf2c1937ad`
  - `MCP/results/mcp07/registered/failed_episodes.jsonl.gz`: `89da6ef0625399ce9d020ee0072f1245041c57763cd42ebe7f41307a02fb82d8`
  - `MCP/results/mcp07/MCP07_evidence_packet.zip` (`44.54 MiB`): `9b5c0f07f1db6abfc16352a0810f5b3522254743a7ba6215c2d3d165613cdbfd`
  - All 36 checkpoint SHA-256 hashes are recorded in `MCP/results/mcp07/registered/checkpoints_sha256.json`.
- See [`MECHANISTIC_REPORT.md`](MECHANISTIC_REPORT.md) for linear probes, router telemetry, switch-transition matrices, and all five paired causal interventions.
