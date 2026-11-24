# MCP-08 Formal Specification: Verifier-Guided Recursive Memory Search and Error Recovery

- **Parent results commit**: `e0bf57f760890856dbc87c26cefc5065a70f18de` (`gym/mcp07-adaptive-geometry`)
- **Preservation invariant**: Zero modifications to `MCP/mcp07/` or `MCP/results/mcp07/`. All MCP-07 checkpoints, receipts, manifests, and negative findings remain intact and unmodified; original MCP-07 checkpoints are loaded directly as the `mcp07_orig` baseline.
- **Primary research question**: *Can a compact (~54K-parameter) hierarchical recurrent controller learn to recover from its own search errors and reliably execute long verified plans (5–50 sequential decisions) over external structured memory up to 256K slots?*

---

## 1. Extended External Memory (16K → 256K) & Transposition Cache

### 1.1 Physical Memory Lattice
Physical slot capacities $N \in \{16384, 32768, 65536, 131072, 262144\}$ with page size $P = 32$:

| Capacity $N$ | Width $W$ | Height $H$ | Pages $M = N/32$ | 4-Projection Index Storage |
|---:|---:|---:|---:|---:|
| 16,384 (train only) | 128 | 128 | 512 | 536.0 KiB |
| 32,768 | 128 | 256 | 1,024 | 1,072.0 KiB |
| 65,536 | 256 | 256 | 2,048 | 2,144.0 KiB |
| 131,072 | 256 | 512 | 4,096 | 4,288.0 KiB |
| **262,144 (256K track)** | **512** | **512** | **8,192** | **8,576.0 KiB** |

Structured address projections $\Pi_{\text{struct}} = \{\text{row}, \text{col}, \text{hilbert}, \text{morton}\}$ share identical physical keys, values, and corruption draws ($u_s \sim \text{Uniform}(0,1)$, slot valid iff $u_s \ge c$).

### 1.2 Bounded Transposition Cache (`TranspositionCache`)
A bounded LRU cache ($K_{\text{cache}} = 1024$ entries, $64$ bytes/entry $= 64\text{ KiB}$ explicitly charged) keyed by canonical domain state signature $\text{sig}(x)$ memoizes verified refuted dead ends and explored branch masks:
- Every lookup charges $+1$ cache-lookup read-equivalent in `charged_memory_reads`.
- Cache hits on refuted states prune redundant subtrees (`cache_pruned_states`), preventing backtracking cycles and re-expansion of dead branches.

### 1.3 Separate Cost Accounting
Every arm reports all resource dimensions **separately**:
1. **Charged memory reads**:
   $$C_{\text{reads}} = 32 \cdot N_{\text{probes}} + N_{\text{anchor\_lookups}} + 2 \cdot N_{\text{switches}} + N_{\text{cache\_lookups}}.$$
2. **Unique physical slots & overlap reads**: exact set union $|S_{\text{vis}}|$ and overlap $32 N_{\text{probes}} - |S_{\text{vis}}|$.
3. **Verifier calls**: exact count $N_{\text{verify}}$ of native retrieval/transition/dead-end/terminal verifier invocations.
4. **Recurrent updates**: $N_{\text{fast}} + N_{\text{slow}}$ (reported both summed and separately).
5. **Search recovery telemetry**: `retries_triggered`, `post_failure_switches`, `backtracks_executed`, `irreversible_blocks_hit`, `cache_hits`, `cache_pruned_states`.
6. **Storage footprint**: `index_storage_bytes` + `transposition_cache_bytes` + `checkpoint_stack_bytes`.

---

## 2. MCP-08 Hierarchical Recursive Controller (~54K Parameters)

Extending MCP-07's two-timescale coupled GRU (`d_h = 49, d_s = 49, d_p = 16`, **54,511 active parameters**, matched within `< 0.1%` of MCP-07's `54,460`), `MCP08RecursiveController` consumes a `42`-dim strictly non-oracle observation vector combining MCP-07's `34`-dim spatial/progress observation with `8` verifier-search channels:
`[plan_progress_frac, goal_deficit_frac, verify_fail_cur, retry_count_norm, backtrack_count_norm, irreversible_risk, cache_hit_cur, compute_tier_norm]`.

Seven coupled mechanisms:
1. **Persistent slow-state search progress**: Slow state $s_t \in \mathbb{R}^{49}$ directly integrates `[h_t, search_ctx]` and persists across sequential decisions, retries, and backtracks (trained end-to-end across 4-step coherent search chains) rather than resetting between lookups.
2. **Calibrated uncertainty head**: $\hat{u}_t = \sigma(w_u^\top [h_t, s_t, \text{ctx}_t] + b_u) \in [0, 1]$, trained with Brier + BCE proper scoring loss against observed retrieval/branch failure outcomes; evaluated by held-out **ECE** and **AUROC**.
3. **Explicit hierarchical compute allocation**: `compute_head` maps `[s_persist, scale_cues, event_flags]` to 3 probe-budget tiers `{tier_0: 4 probes, tier_1: 8 probes, tier_2: 16 probes}`, allocating minimal probes on confident steps and deeper budgets on uncertain or post-failure steps while relying causally on `s_persist` for accumulated search progress.
4. **Selective retrieval retries**: Triggered selectively when retrieval fails verification or uncertainty exceeds learned threshold $\tau_u$, bounded by per-decision and episode read caps.
5. **Adaptive projection switching after verification failure**: On retrieval/verification failure, `verify_fail_cur = 1` triggers an immediate slow-state update and masks/penalizes the failed initial projection on retry so the router shifts to a complementary addressing geometry.
6. **Bounded backtracking with verified state restoration**: A learned `meta_action_head` (`[proceed, retry, backtrack, halt]`) and bounded reversible checkpoint stack ($K_{\text{bt}} \le 8$) restore verified prior states when the verifier signals a dead end or unreachable goal. **Irreversible transitions** place a hard restoration barrier (`lock_depth`): states prior to a committed irreversible transition cannot be restored.
7. **Transposition cache**: Consulted before expanding any candidate state to avoid revisiting verified dead ends.

---

## 3. Benchmarks

### 3.1 Independent Long-Horizon Native Storyworld Benchmark (5–50 Sequential Decisions)
Built directly on MCP-05's exact two-qubit Hilbert algebra (`MCP.mcp05.hilbert_storyworld`):
- Independent procedural worlds across four horizon buckets: **`5–10`**, **`12–24`**, **`25–38`**, **`40–50`** sequential decisions (plus MCP-05's pinned `storyworld.hilbert.json` compatibility check).
- Features: branching DAGs, distractor trap scenes, **irreversible transitions** (both mandatory act gates and irreversible trap doors), tight phase-sensitive Pauli projector intervals (`X`/`Y` + computational basis), required clues, and exact native `counterfactual()` verification of noncommuting adjacent operations.
- Every world is algebraically verified at generation to contain a valid secret-ending route; all witnesses are **strictly stripped** (`strip_world_witnesses()`) prior to evaluation.
- All scene-option transition records reside in external structured memory (`32K` main sweep; `256K` extrapolation sweep) across mixed spatial geometries under `0%` and `10%` physical corruption.

### 3.2 256K Memory Extrapolation Track (Trained Exclusively at 16K)
Frozen 16K-trained checkpoints evaluated at `16K`, `32K`, `64K`, `128K`, and **`256K` (`262,144` slots)** on:
- Multi-geometry error-recovery retrieval (including selective retries + post-failure projection switching under `0%` and `10%` corruption), and
- Long-horizon native Storyworld planning backed by `256K` external physical memory.

### 3.3 Cross-Domain Mechanistic Transfer Suite (Storyworld, Canvas, Lean-Style)
Three distinct native verifier domains sharing the exact same external-memory + controller interface:
1. **Storyworld**: Native two-qubit branching narrative planning (`5–50` steps).
2. **Canvas**: 2D visual scene-graph layout planning (`8–20` steps) with bounding-box collision checks, contrast/balance invariants, distractor clipping assets, and irreversible raster-flatten locks.
3. **Lean-Style Proof Search**: Symbolic subgoal-stack tactic kernel (`6–20` steps) with typed goals/hypotheses, distractor dead-end generalizations, and irreversible context-clearing commits.

---

## 4. Baselines, Trained-From-Scratch Ablations & Inference-Time Lesions

### 4.1 Core Experimental Comparisons (Matched Read / Verifier Budgets Where Applicable)
1. `mcp07_orig`: Original frozen MCP-07 checkpoints (`MCP/results/mcp07/registered/checkpoints/{107,109,113}-recurrent_switch.pt`), greedy planning, no retries/backtracking/cache.
2. `det_router_greedy`: Deterministic geometry-aware router + greedy planning.
3. `uniform_retries`: Fixed uniform retries on every decision (matched read budget, no uncertainty selectivity, no backtracking).
4. `greedy_planning`: MCP-08 adaptive retrieval + selective retries, strictly greedy action selection (no backtracking or transposition cache).
5. `beam_search`: External-memory-backed beam search (`beam_width = 3`, matched charged read & verifier budget).
6. `bounded_symbolic`: Bounded verifier-guided DFS with transposition table & backtracking over deterministic geometry retrieval (matched read & verifier budget; `0` recurrent updates).
7. `mcp08_full`: Full ~54K verifier-guided recursive search controller with all 7 extensions.

### 4.2 Trained-From-Scratch Ablations vs Inference-Time Lesions
- **Trained from scratch (16K, 3 seeds)**: `scratch_no_persist_slow`, `scratch_no_unc_alloc`, `scratch_no_backtrack`, `scratch_no_cache`, `scratch_no_postfail_switch`.
- **Inference-time lesions (on frozen `mcp08_full`)**: `lesion_reset_slow`, `lesion_shuffle_slow`, `lesion_uniform_compute`, `lesion_no_postfail_switch`, `lesion_no_backtrack`, `lesion_no_cache`, `lesion_corrupt_uncertainty`.

---

## 5. Frozen Preregistered Gates (`manifest.json`)

- **Gate 1 (Long-Horizon Native Storyworld, 32K, c=10%, 3 seeds × 48 sealed worlds = 144 paired worlds)**:
  - `mcp08_full` beats `mcp07_orig` by `>= +15.0 pp`, beats `greedy_planning` by `>= +5.0 pp`, and beats `uniform_retries` by `>= +5.0 pp` (at matched or lower mean charged memory reads).
- **Gate 2 (256K Extrapolation from 16K Training, c=10%)**:
  - `mcp08_full` beats `mcp07_orig` and `det_router` by `>= +10.0 pp` on 256K error-recovery retrieval.
- **Gate 3 (Uncertainty Calibration & Hierarchical Compute Allocation)**:
  - Held-out 8-family `256K` (`c=10%`) calibration achieves `AUROC >= 0.70` and `ECE <= 0.12` (with Storyworld 4-geometry `AUROC`/`ECE` also reported transparently), and hierarchical compute allocation reduces charged memory reads by `>= 15%` vs uniform-deep fixed tier at `<= 2.0 pp` Storyworld completion loss.
- **Gate 4 (Cross-Domain Slow-State Transfer — Storyworld ↔ Canvas ↔ Lean)**:
  - Mean zero-shot off-diagonal linear probe $R^2$ for search progress across `{Storyworld, Canvas, Lean}` exceeds `0.25`, with directional activation-patching steering reported honestly.
- **Compute Gate**: `48,000 <= active_params <= 60,000`, peak process RSS `< 2,048 MiB`, CPU-only execution.
