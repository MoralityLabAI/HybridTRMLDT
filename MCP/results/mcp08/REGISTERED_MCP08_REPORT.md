# MCP-08 Registered Evaluation Report: Verifier-Guided Recursive Memory Search and Error Recovery

- **Experiment ID**: `MCP-08-VERIFIER-RECURSIVE-SEARCH-V1`
- **Branch**: `gym/mcp08-verifier-search`
- **Parent MCP-07 commit (preserved untouched)**: `e0bf57f760890856dbc87c26cefc5065a70f18de` (`gym/mcp07-adaptive-geometry`)
- **Frozen protocol commit (pre-sealed eval)**: `791a8d0a688a869e5dc2795f26da096cac14f15b`
- **Frozen `manifest.json` SHA-256**: `cc5d2028d0cf4fe664ed185467e5af75fb0a7a086489147488d84dd3065e746d`
- **Execution hardware & compute**: CPU-only (`CUDA_VISIBLE_DEVICES=""`), 6 single-thread worker processes (`OMP/MKL/OPENBLAS_NUM_THREADS=1`), total wall time **`2,757.9 s` (46.0 min)**, peak process RSS **`545.77 MiB`** (`< 2,048 MiB`), active controller size **`54,511` parameters** (within 51 parameters of MCP-07's `54,460`).
- **Receipt corpus**: `61,344` JSONL episode rows in `registered/episodes.jsonl.gz` (`3,744` Storyworld planning episodes + `57,600` 16K→256K retrieval rollouts) and **`23,118` failed episodes preserved in full** in `registered/failed_episodes.jsonl.gz`.
- **Protocol deviations**: None after freezing commit `791a8d0a688a869e5dc2795f26da096cac14f15b`. All MCP-07 files under `MCP/mcp07/` and `MCP/results/mcp07/` have zero diff.

---

## 1. Executive Gate Summary (Explicit PASS / FAIL)

| Preregistered Gate | Criterion (`manifest.json`) | Sealed Measurement | Status |
|---|---|---|---:|
| **Gate 1 — Long-Horizon Native Storyworld (`32K, c=10%`, 144 paired worlds, 5–50 decisions)** | `mcp08_full` beats `mcp07_orig` `>= +15.0 pp`, `greedy_planning` `>= +5.0 pp`, `uniform_retries` `>= +5.0 pp` | **`50.00%`** vs `mcp07_orig` `2.08%` (**`+47.92 pp`**, `[+40.28, +56.25]`), vs `greedy_planning` `1.39%` (**`+48.61 pp`**), vs `uniform_retries` `1.39%` (**`+48.61 pp`**); beats `beam_search` (`18.06%`) by **`+31.94 pp`** at *lower* reads, verifier calls, and recurrent updates; beats `bounded_symbolic` (`6.94%`) by **`+43.06 pp`** | **PASS** |
| **Gate 2 — 16K → 256K Extrapolation (`c=10%`, 1,152 paired roots across 8 families)** | `mcp08_full` beats `mcp07_orig` `>= +10.0 pp` AND `det_router` `>= +10.0 pp` | **`64.1%`** vs `det_router` `50.1%` (**`+13.98 pp`**, `[+10.50, +17.27]`, sub-criterion met), vs `uniform_retries` `57.1%` (**`+6.94 pp`**, `[+4.69, +9.20]`), vs `mcp07_orig` `59.9%` (**`+4.17 pp`**, `[+1.56, +6.51]`, statistically positive `p>0=1.0`, **below `+10.0 pp` threshold**) | **FAIL** |
| **Gate 3 — Calibration & Hierarchical Compute Allocation (`c=10%`)** | `AUROC >= 0.70`, `ECE <= 0.12` (8-family `256K`); read reduction vs fixed `uniform_deep` `>= 15%` at `<= 2.0 pp` Storyworld completion loss | `256K` calibration (`n=1,282`): **`AUROC = 0.7030`**, **`ECE = 0.0688`**, `Brier = 0.2283`; Storyworld read reduction vs `uniform_deep`: **`-55.73%`** (`[51.94%, 59.16%]`, `20,329.5` vs `45,919.5` reads) with **`+2.78 pp` higher completion** (`50.00%` vs `47.22%`); `256K` read reduction vs `uniform_deep`: **`-25.97%`** (`[24.22%, 27.85%]`) at `0.61 pp` loss | **PASS** |
| **Gate 4 — Cross-Domain Slow-State Transfer (`Storyworld ↔ Canvas ↔ Lean`, `32K, c=10%`)** | Zero-shot off-diagonal linear probe $R^2$ on slow state `s_t` for search progress `> 0.25` | Off-diagonal mean **`R² = +0.6228`** (`r = +0.8174`, all 6 pairs `+0.4615` to `+0.8056`; permutation null `R² = -0.2257`; fast state `h_t` off-diag `R² = +0.5355`) | **PASS** |
| **Compute Gate** | `48,000 <= active_params <= 60,000`, Peak RSS `< 2,048 MiB`, CPU-only | **`54,511` active params**, **`545.77 MiB` Peak RSS**, CPU-only | **PASS** |

> **Primary Research Answer**: Yes for long-horizon verified planning (5–50 sequential decisions, up to 256K memory) and cross-domain progress representation, **with explicit quantitative boundaries**: a `54,511`-parameter hierarchical recurrent controller improves verified Storyworld completion from `2.08%` (`mcp07_orig`) and `18.06%` (`beam_search`) to **`50.00%`** at `32K` (`55.56%` clean; **`39.58%`** at `256K`), cuts charged reads `55.7%` vs fixed deep compute, and learns a strongly cross-domain transferable slow progress state (`off-diag R² = 0.623`). However, on single-query `16K → 256K` 8-family synthetic retrieval (`Gate 2: FAIL`), `mcp08_full` beats `mcp07_orig` by only `+4.17 pp` (below the `+10 pp` threshold) while spending `2.67×` more reads on retries — and trained-from-scratch ablations reveal that **bounded backtracking** (`+48.61 pp`) drives completion, **transposition caching** drives efficiency (`-31.9%` reads), while persistent slow state improves recovery efficiency (`-12.4%` reads, `+10.1 pp` error recovery) rather than raw completion alone.

---

## 2. Track 1: Independent Long-Horizon Native Storyworld Benchmark (5–50 Decisions)

All worlds are independently synthesized over MCP-05's exact two-qubit Hilbert algebra (`MCP.mcp05.hilbert_storyworld`), mechanically stripped of archived witnesses (`witness_withheld_verified = true`), and evaluated strictly via all five native checks (legal transitions, required clues, terminal Pauli/basis projectors, secret-ending eligibility, and noncommuting adjacent-operation swap counterfactual).

### 2.1 Core Arms & Trained-From-Scratch Ablations at `32K, c=10%` (144 Paired Sealed Worlds)

All cost dimensions are reported **separately** (charged memory reads, verifier calls, recurrent updates, retrieval rate, error recovery, backtracks, post-failure switches, cache prunes, irreversible blocks, and memory storage footprint):

| Arm | Verified Completion | Charged Reads | Verifier Calls | Recurrent Updates (`F+S`) | Retrieval Rate | Error Recov. | Backtracks | Post-Fail Switches | Cache Pruned | Irrev. Blocks | Storage (`Index / Cache / Stack`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mcp07_orig` (frozen parent) | `2.08%` | `1,135.4` | `19.5` | `72.4` | `84.5%` | `25.0%` | `0.0` | `0.0` | `0.0` | `0.00` | `1,072 KiB / 0 / 0` |
| `det_router_greedy` | `2.08%` | `1,462.3` | `15.1` | `0.0` | `64.0%` | `2.1%` | `0.0` | `0.0` | `0.0` | `0.00` | `1,072 KiB / 0 / 0` |
| `uniform_retries` | `1.39%` | `1,743.9` | `19.6` | `114.5` | `89.5%` | `50.8%` | `0.0` | `0.0` | `0.0` | `0.00` | `1,072 KiB / 0 / 0` |
| `greedy_planning` | `1.39%` | `2,373.4` | `20.6` | `145.3` | `91.4%` | `65.9%` | `0.0` | `1.4` | `0.0` | `0.00` | `1,072 KiB / 0 / 0` |
| `beam_search` (width=3) | `18.06%` | `21,366.7` | `160.4` | `1,298.8` | `90.0%` | `56.6%` | `0.0` | `13.0` | `0.0` | `0.00` | `1,072 KiB / 0 / 0` |
| `bounded_symbolic` (DFS+cache) | `6.94%` | `6,826.1` | `49.4` | `0.0` | `49.4%` | `0.0%` | `7.3` | `0.0` | `6.3` | `1.20` | `1,072 / 64 / 1 KiB` |
| **`mcp08_full` (54.5K)** | **`50.00%`** | **`20,329.5`** | **`153.0`** | **`1,227.6`** | **`87.0%`** | **`54.1%`** | **`12.6`** | **`13.9`** | **`9.6`** | **`1.87`** | **`1,072 / 64 / 1 KiB`** |
| `mcp08_uniform_deep` (fixed T=2) | `47.22%` | `45,919.5` | `144.8` | `2,718.6` | `88.7%` | `59.5%` | `12.6` | `15.5` | `9.2` | `1.65` | `1,072 / 64 / 1 KiB` |
| *Scratch*: `no_persist_slow` | `48.61%` | `23,200.6` | `150.4` | `1,365.5` | `84.0%` | `44.0%` | `12.7` | `18.5` | `9.5` | `1.81` | `1,072 / 64 / 1 KiB` |
| *Scratch*: `no_unc_alloc` (fixed T=1) | `49.31%` | `13,704.3` | `147.3` | `876.7` | `82.3%` | `33.0%` | `12.2` | `17.5` | `9.2` | `1.96` | `1,072 / 64 / 1 KiB` |
| *Scratch*: `no_backtrack` | `1.39%` | `2,387.0` | `20.6` | `145.3` | `91.4%` | `65.9%` | `0.0` | `1.4` | `1.1` | `0.00` | `1,072 / 64 / 0 KiB` |
| *Scratch*: `no_cache` | `52.08%` | `29,872.9` | `198.9` | `1,819.6` | `88.9%` | `55.6%` | `14.8` | `20.4` | `0.0` | `1.97` | `1,072 / 0 / 1 KiB` |
| *Scratch*: `no_postfail_switch` | `50.69%` | `19,257.3` | `158.1` | `1,165.2` | `87.5%` | `51.9%` | `12.6` | `0.0` | `9.4` | `2.06` | `1,072 / 64 / 1 KiB` |

> **Clean (`c=0.00`) 32K Summary (144 paired worlds)**: `mcp08_full` achieves **`55.56%`** (`19,242.1` reads, `163.3` verifier calls, `1,151.9` RU) vs `beam_search` `19.44%` (`21,594.6` reads), `bounded_symbolic` `6.25%` (`6,969.9` reads), `mcp07_orig` `2.08%`, `det_router_greedy` `1.39%`, `uniform_retries` `1.39%`, and `greedy_planning` `1.39%`.

### 2.2 Paired Bootstrap 95% CIs (`32K, c=10%`, 2,000 Resamples, `n=144`)

| Comparison (`mcp08_full` minus Comparator) | Point Delta | 95% Bootstrap CI | `P(delta > 0)` |
|---|---:|---:|---:|
| vs `mcp07_orig` | **`+47.92 pp`** | `[+40.28, +56.25]` | `1.0000` |
| vs `det_router_greedy` | **`+47.92 pp`** | `[+40.28, +55.56]` | `1.0000` |
| vs `uniform_retries` | **`+48.61 pp`** | `[+40.28, +56.25]` | `1.0000` |
| vs `greedy_planning` | **`+48.61 pp`** | `[+40.28, +56.27]` | `1.0000` |
| vs `beam_search` | **`+31.94 pp`** | `[+24.29, +40.28]` | `1.0000` |
| vs `bounded_symbolic` | **`+43.06 pp`** | `[+34.72, +51.39]` | `1.0000` |
| vs `scratch_no_backtrack` | **`+48.61 pp`** | `[+40.28, +56.25]` | `1.0000` |
| vs `scratch_no_persist_slow` | `+1.39 pp` | `[-5.56, +8.33]` | `0.6140` |
| vs `scratch_no_unc_alloc` | `+0.69 pp` | `[-6.25, +7.64]` | `0.5410` |
| vs `scratch_no_postfail_switch` | `-0.69 pp` | `[-7.64, +6.25]` | `0.3775` |
| vs `scratch_no_cache` | `-2.08 pp` | `[-6.94, +2.78]` | `0.1555` |

### 2.3 Horizon Scaling Across 5–50 Sequential Decisions (`32K, c=10%`) & Per-Seed Consistency

| Horizon Bucket (36 worlds/bucket) | `mcp07_orig` | `det_router` | `uniform_ret` | `greedy_plan` | `beam_search` | `bounded_sym` | **`mcp08_full`** | `mcp08_full` Reads / Ver / BT / Irrev |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `short_5_10` (5–10 steps) | `5.6%` | `8.3%` | `5.6%` | `5.6%` | `41.7%` | `25.0%` | **`83.3%`** | `7,328 / 67.1 / 5.7 / 0.19` |
| `medium_12_24` (12–24 steps) | `2.8%` | `0.0%` | `0.0%` | `0.0%` | `22.2%` | `2.8%` | **`55.6%`** | `16,331 / 129.4 / 11.6 / 1.28` |
| `long_25_38` (25–38 steps) | `0.0%` | `0.0%` | `0.0%` | `0.0%` | `2.8%` | `0.0%` | **`27.8%`** | `23,494 / 166.4 / 12.8 / 3.33` |
| `ultra_40_50` (40–50 steps) | `0.0%` | `0.0%` | `0.0%` | `0.0%` | `5.6%` | `0.0%` | **`33.3%`** | `34,164 / 249.0 / 20.3 / 2.67` |

- **Per-seed `mcp08_full` completion (`32K, c=10%`)**: seed 107 = **`52.1%`**, seed 109 = **`52.1%`**, seed 113 = **`45.8%`** (vs `beam_search`: `20.8% / 16.7% / 16.7%`; `bounded_symbolic`: `8.3% / 8.3% / 4.2%`; `mcp07_orig`: `2.1% / 2.1% / 2.1%`).
- **Failure-mode taxonomy (`72` failed `mcp08_full` worlds out of `144` at `32K, c=10%`)**:
  - **Irreversible-transition lockouts (`irreversible_blocks_hit > 0`)**: **`46 / 72` (`63.9%`)** — confirming irreversible act gates and trap doors are genuine, un-bypassable search hazards.
  - **Checkpoint stack depth (`K=8`) or retrieval exhaustion**: `20 / 72` (`27.8%`).
  - **Backtrack cap exhausted**: `5 / 72` (`6.9%`).
  - **Episode read budget exhausted**: `1 / 72` (`1.4%`).

### 2.4 Native Storyworld Extrapolation on `256K` (`262,144` Slots, `c=10%`, 48 Paired Worlds)

| Arm (`256K` Memory, Trained Exclusively at `16K`) | Verified Completion | Charged Reads | Verifier Calls | Recurrent Updates | Retrieval Rate | Backtracks | Index Storage |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mcp07_orig` | `2.08%` | `1,156.0` | `19.7` | `73.6` | `84.6%` | `0.0` | `8,576 KiB` |
| `det_router_greedy` | `2.08%` | `1,664.0` | `13.0` | `0.0` | `50.1%` | `0.0` | `8,576 KiB` |
| `uniform_retries` | `2.08%` | `2,259.6` | `22.1` | `148.4` | `87.0%` | `0.0` | `8,576 KiB` |
| `greedy_planning` | `0.00%` | `3,103.2` | `21.0` | `187.8` | `85.5%` | `0.0` | `8,576 KiB` |
| `bounded_symbolic` | `6.25%` | `5,963.5` | `31.1` | `0.0` | `37.0%` | `5.3` | `8,576 KiB` |
| **`mcp08_full`** | **`39.58%`** | **`21,101.4`** | **`122.6`** | **`1,273.8`** | **`77.8%`** | **`10.5`** | **`8,576 KiB`** |

Even at **16× memory capacity extrapolation (`16K → 256K`)** under `10%` physical corruption, `mcp08_full` retains **`39.58%`** end-to-end 5–50 step Storyworld completion — **`+33.33 pp`** above `bounded_symbolic` (`6.25%`) and **`+37.50 pp`** above `mcp07_orig` (`2.08%`).

---

## 3. Track 2: `16K → 256K` Synthetic Retrieval Extrapolation (`Gate 2: FAIL`)

Controllers trained exclusively at `16K` (`16,384` slots) are frozen and evaluated across `{16K, 32K, 64K, 128K, 256K}` on all 8 MCP-07 query families (`3` seeds × `48` roots/family × `8` families = `1,152` paired roots per `(capacity, corruption)` cell).

### 3.1 Capacity Scaling Curve (Completion % and Mean Charged Reads)

| Capacity $N$ | Corr. $c$ | `mcp07_orig` (`B=16`) | `det_router` (`B=16`) | `uniform_retries` | `mcp08_uniform_deep` | **`mcp08_full`** | `mcp08_full` vs `mcp07_orig` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `16,384` (train cap) | `0.00` | `70.4% (265)` | `61.7% (306)` | `63.6% (395)` | `84.5% (876)` | **`84.5% (554)`** | `+14.1 pp` |
| `16,384` (train cap) | `0.10` | `68.8% (272)` | `61.0% (310)` | `62.3% (405)` | `83.4% (890)` | **`83.4% (572)`** | `+14.6 pp` |
| `32,768` (`2×`) | `0.00` | `67.3% (275)` | `60.1% (305)` | `61.1% (401)` | `76.3% (934)` | **`76.6% (622)`** | `+9.3 pp` |
| `32,768` (`2×`) | `0.10` | `66.4% (279)` | `59.7% (307)` | `60.2% (409)` | `75.6% (943)` | **`75.7% (638)`** | `+9.3 pp` |
| `65,536` (`4×`) | `0.00` | `63.8% (279)` | `54.9% (315)` | `58.9% (407)` | `70.1% (1,003)` | **`70.1% (702)`** | `+6.3 pp` |
| `65,536` (`4×`) | `0.10` | `62.4% (286)` | `54.1% (318)` | `57.0% (420)` | `68.0% (1,022)` | **`68.1% (730)`** | `+5.7 pp` |
| `131,072` (`8×`) | `0.00` | `62.1% (284)` | `55.6% (317)` | `58.6% (412)` | `66.7% (1,026)` | **`66.2% (744)`** | `+4.1 pp` |
| `131,072` (`8×`) | `0.10` | `59.8% (294)` | `54.4% (321)` | `56.3% (427)` | `64.8% (1,045)` | **`64.3% (773)`** | `+4.5 pp` |
| **`262,144` (`16×`)** | `0.00` | `62.4% (281)` | `51.4% (328)` | `59.8% (405)` | `66.5% (1,033)` | **`66.1% (747)`** | `+3.7 pp` |
| **`262,144` (`16×`)** | **`0.10`** | **`59.9% (293)`** | **`50.1% (333)`** | **`57.1% (424)`** | **`64.7% (1,056)`** | **`64.1% (782)`** | **`+4.17 pp` (FAIL `<10 pp`)** |

### 3.2 Per-Family Diagnostic at `256K, c=10%` (144 Paired Roots/Family) — Why Gate 2 Failed

| Query Family (`256K, c=10%`) | `mcp07_orig` | `det_router` | `uniform_retries` | `mcp08_uniform_deep` | **`mcp08_full`** | Key Finding |
|---|---:|---:|---:|---:|---:|---|
| `horizontal` | `99.3% (70)` | `64.6% (221)` | `99.3% (74)` | `99.3% (525)` | **`99.3% (87)`** | Ceiling saturated; `-83.4%` reads vs `uniform_deep` |
| `vertical` | `90.3% (128)` | `61.1% (241)` | `88.9% (163)` | `93.1% (614)` | **`93.1% (228)`** | `+2.8 pp` vs `mcp07_orig`, `+32.0 pp` vs `det_router` |
| `diagonal` | `97.2% (143)` | `54.9% (295)` | `94.4% (225)` | `92.4% (861)` | `92.4% (545)` | `-4.8 pp` multi-task capacity trade-off vs `mcp07_orig` |
| `isotropic` | `64.6% (295)` | `77.8% (220)` | `50.7% (473)` | `62.5% (1,056)` | `62.5% (853)` | `-2.1 pp` vs `mcp07_orig`; `det_router` leads (`77.8%`) |
| `multihop` | `63.9% (312)` | `31.9% (396)` | `77.1% (379)` | `81.2% (1,358)` | **`76.4% (745)`** | **`+12.5 pp`** vs `mcp07_orig`, **`+44.5 pp`** vs `det_router` |
| `corrupted_replica` | `28.5% (447)` | `19.4% (463)` | `16.7% (689)` | `34.7% (1,307)` | **`34.7% (1,221)`** | **`+6.2 pp`** vs `mcp07_orig`, **`+18.0 pp`** vs `uniform_retries` |
| `regime_shift` | `18.8% (464)` | `60.4% (369)` | `12.5% (708)` | `29.9% (1,365)` | **`29.9% (1,281)`** | **`+11.1 pp`** vs `mcp07_orig`, **`+17.4 pp`** vs `uniform_retries` |
| `sparse_long_range` | `16.7% (486)` | `30.6% (458)` | `17.4% (684)` | `24.3% (1,359)` | **`24.3% (1,292)`** | **`+7.6 pp`** vs `mcp07_orig`, `+6.9 pp` vs `uniform_retries` |

**Diagnosis of Gate 2 FAIL**: Post-failure projection switching delivers large, genuine recovery gains on all four hard families (`+6.2` to `+12.5 pp` over `mcp07_orig` and **`+17.4` to `+18.0 pp` over blind `uniform_retries`** on `regime_shift` and `corrupted_replica`). However, at `256K` (`512×512`), `mcp07_orig` is already near ceiling on `horizontal`/`vertical`/`diagonal` (`90–99%`), and `mcp08_full` loses `-4.8 pp` on `diagonal` and `-2.1 pp` on `isotropic` while spending `2.67×` more average reads (`782` vs `293`) on failed-query retries. Consequently, the 8-family pooled advantage narrows from **`+14.6 pp` at `16K`** to **`+4.17 pp` at `256K`** (`95% CI [+1.56, +6.51]`), passing vs `det_router` (`+13.98 pp`) but **failing** the preregistered `+10.0 pp` threshold vs `mcp07_orig`.

---

## 4. Gate 3: Calibration, Compute Efficiency, and Preserved Negative / Null Findings

1. **Calibration under `16×` capacity shift (`256K, c=10%`, `n=1,282`)**: **`AUROC = 0.7030`**, **`ECE = 0.0688`**, `Brier = 0.2283` (PASS).
2. **Storyworld `32K, c=10%` calibration (`n=10,071`, honest secondary finding)**: `AUROC = 0.5516`, `ECE = 0.1347`, `Brier = 0.2266`. Because Storyworld stores options exclusively across the 4 primary structured geometries (`row/col/morton/hilbert`), pre-probe (`t=0`) failures are dominated by unobservable uniform `c=10%` physical slot corruption rather than observable geometric hardness, bounding `t=0` AUROC near `0.55`.
3. **Hierarchical compute allocation vs fixed `uniform_deep` (`tier=2`, 16 probes)**:
   - On Storyworld `32K, c=10%`: **`-55.73%` charged reads** (`95% CI [51.94%, 59.16%]`, `20,329.5` vs `45,919.5`) while **gaining `+2.78 pp` completion** (`50.00%` vs `47.22%`, because `uniform_deep` exhausts episode read budgets earlier on long worlds).
   - On `256K, c=10%` retrieval: **`-25.97%` charged reads** (`[24.22%, 27.85%]`, `782` vs `1,056`) at only `0.61 pp` completion loss.
4. **Honest negative/null ablation findings (preserved in full)**:
   - **`scratch_no_unc_alloc` (fixed `tier=1` = 8 probes *with* adaptive stopping)**: Achieves `49.31%` Storyworld completion (`-0.69 pp`, statistically indistinguishable from `mcp08_full`) at **fewer charged reads (`13,704.3` vs `20,329.5`)**, though with substantially lower error-recovery rate (`33.0%` vs `54.1%`). Thus, while hierarchical tiering + adaptive stopping crushes fixed deep compute (`-55.7%` reads), a fixed medium tier with adaptive stopping is a surprisingly strong efficiency baseline on 4-geometry Storyworlds.
   - **`scratch_no_cache` & `lesion_no_cache`**: Removing the transposition cache does **not** reduce Storyworld completion (`52.08%` scratch, `50.0%` lesion; CIs cross `0`), **but increases charged reads by `+46.9%` (`+9,543` reads/episode) and verifier calls by `+30.0%`**. The cache is a pure search-efficiency mechanism, not a completion bottleneck under the `2200*L+6000` budget.
   - **`scratch_no_postfail_switch` on Storyworld**: Null on 4-geometry Storyworld completion (`50.69%`, `-0.69 pp` CI crosses `0`), contrasting sharply with its **`+6.94 pp`** pooled (`+17.4` to `+18.0 pp` on `regime_shift`/`corrupted_replica`) benefit on 8-family `256K` retrieval.
   - **`lesion_corrupt_uncertainty`**: Exact `0.00 pp` / `0.0` reads null on Storyworld because inverting calibrated `u_prob` in `[0.25, 0.75]` still exceeds the `0.20` selective-retry threshold.

---

## 5. Companion Artifacts & Reproducibility

- Mechanistic probes (`3×3` slow vs fast transfer), 6-pair cross-domain activation patching, and trained-from-scratch vs inference-time lesion comparison: see `MCP/results/mcp08/MECHANISTIC_REPORT.md`.
- Raw frozen JSON receipts & checkpoints: `MCP/results/mcp08/registered/` (`summary.json`, `storyworld_native.json`, `extrapolation_256k.json`, `mechanistic.json`, `checkpoints_sha256.json`, `episodes.jsonl.gz`, `failed_episodes.jsonl.gz`, and 18 trained `.pt` checkpoints).
- Portable reviewer bundle: `MCP/results/mcp08/MCP08_evidence_packet.zip` (+ `.sha256`).
