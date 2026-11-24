# MCP-07 Formal Mathematical Specification: Geometry-Adaptive Recurrent Memory Routing

Parent baseline: `gym/mcp06-spatial-memory` (`3c03ccc6c4a0cac059c670011a55ea7ad9666de9`).

## 1. Shared Physical Memory & Multi-Projection Addressing

Let capacity $N \in \{16384, 24576, 32768, 65536, 131072\}$ denote physical record slots (**not tokens**), arranged on a canonical 2D lattice of width $W$ and height $H = N / W$:

| Capacity $N$ | Width $W$ | Height $H$ | Pages $M = N / P$ ($P=32$) |
|---:|---:|---:|---:|
| 16,384 | 128 | 128 | 512 |
| 24,576 | 128 | 192 | 768 |
| 32,768 | 128 | 256 | 1,024 |
| 65,536 | 256 | 256 | 2,048 |
| 131,072 | 256 | 512 | 4,096 |

Every physical slot $s \in \{0, \dots, N-1\}$ has canonical lattice coordinates $x(s) = s \bmod W$, $y(s) = \lfloor s / W \rfloor$.

### 1.1 Logical Records and Replicas
There are $L = N / 2$ logical records $r \in \{0, \dots, L-1\}$, each storing:
- A unique permuted integer key $k_r \in \{0, \dots, L-1\}$.
- A payload value $v_r \in [0, 2^{30})$.
- Two adjacent physical replicas: $s_0(r) = 2r$ and $s_1(r) = 2r + 1$ (same row $y$, consecutive columns $2k, 2k+1$ since $W$ is even).
- Per-slot corruption scores $u_s \sim \text{Uniform}(0, 1)$ drawn once per world seed. Under global corruption rate $c \in \{0, 0.05, 0.15\}$, slot $s$ is **valid** iff $u_s \ge c$ and **stale/corrupted** iff $u_s < c$.

All projections share the **exact same** physical arrays $(k, v, u)$.

### 1.2 Address Projections
Each projection $\pi \in \Pi = \{\text{row}, \text{col}, \text{hilbert}, \text{morton}, \text{rand\_coloc}, \text{rand\_full}\}$ is a bijection $\sigma_\pi : \{0, \dots, N-1\} \to \{0, \dots, N-1\}$ mapping projection address $a$ to physical slot $s = \sigma_\pi(a)$, packed into $M = N/32$ contiguous pages:
$$\text{cells}_\pi[p] = \{\sigma_\pi(32p + i) : 0 \le i < 32\}, \quad \text{inv}_\pi[s] = \lfloor \sigma_\pi^{-1}(s) / 32 \rfloor.$$
Each page stores its physical centroid $\bar{c}_\pi(p) = (\frac{1}{32}\sum_{s \in \text{cells}_\pi[p]} x(s), \frac{1}{32}\sum_{s \in \text{cells}_\pi[p]} y(s))$ and deterministic cursor slot $s^*_\pi(p) = \arg\min_{s \in \text{cells}_\pi[p]} (|x(s)-\bar{c}_x| + |y(s)-\bar{c}_y|)$.

The six projections are:
1. **`row`**: $\sigma_{\text{row}}(yW + x) = yW + x$. Consecutive $x$ values pack into single pages; moving $\pm 1$ page shifts $x$ by $\pm 32$ independent of $N$. Replicas $(2r, 2r+1)$ are always co-located.
2. **`col`**: $\sigma_{\text{col}}(xH + y) = yW + x$. Consecutive $y$ values pack into single pages; moving $\pm 1$ page shifts $y$ by $\pm 32$ independent of $N$. Replicas $(2r, 2r+1)$ lie in adjacent columns ($H/32$ pages apart).
3. **`hilbert`**: Clipped 2D Hilbert curve over side $S = 2^{\lceil \log_2 \max(W, H) \rceil}$, retaining $(x, y)$ with $x < W, y < H$. Preserves continuous 2D neighborhood locality across quadrant seams.
4. **`morton`**: Clipped 2D Z-order curve sorting slots by bit-interleaved key $Z(x, y) = \text{spread}(x) \mid (\text{spread}(y) \ll 1)$. Preserves anti-diagonal quadrant transitions ($Q_1 \leftrightarrow Q_2$) and monotone diagonal ordering, and strictly co-locates $(2r, 2r+1)$ since $x_0$ is the LSB.
5. **`rand_coloc`**: Random permutation of logical records $r \in \{0, \dots, L-1\}$, preserving adjacent physical replica co-location $(2r, 2r+1)$ on the same page while destroying 2D spatial locality.
6. **`rand_full`**: Uniform random permutation of all $N$ physical slots, destroying both spatial locality and replica co-location.

The four structured routing projections are $\Pi_{\text{struct}} = \{\text{row}, \text{col}, \text{hilbert}, \text{morton}\}$.

### 1.3 Charged Cost Model & Shared Physical ID Accounting
Every arm is charged transparently for storage, preprocessing, and runtime reads:
1. **Index Storage (`index_storage_bytes`)**: Each projection $\pi$ stores `cells` ($4N$ bytes), `inv` ($4N$ bytes), `centers` ($2 \times 4M = 0.25N$ bytes), and `center_slot` ($4M = 0.125N$ bytes) $\Rightarrow 8.375 N$ bytes per active projection. An arm with $K$ projections ($K=1$ for fixed arms, $K=4$ for routed arms) is charged $8.375 K N$ bytes.
2. **Preprocessing Cost**: Measured construction wall time and analytical op count ($\mathcal{O}(N)$ for `row`, `col`, `rand_coloc`, `rand_full`; $\mathcal{O}(N \log S)$ for `hilbert`, `morton`), summed over the $K$ projections used by each arm.
3. **Runtime Charged Reads**:
   - Each page probe charges $P = 32$ physical record reads (never discounted for cross-projection overlap).
   - Initial anchor index translation (`inv[π][anchor]`) charges $+1$ lookup read (and $+1$ on multihop re-anchoring).
   - Each mid-episode projection switch ($\pi_{t+1} \neq \pi_t$) translates cursor slot $s^*_{\pi_t}(p_t)$ via `inv[π_{t+1}]` and fetches page metadata, charging $+2$ read-equivalents.
   - Total episode charged reads:
     $$C_{\text{reads}} = 32 \cdot N_{\text{probes}} + N_{\text{anchor\_lookups}} + 2 \cdot N_{\text{switches}}.$$
4. **Shared Physical Slot Tracking**: Every episode tracks the exact set of physical slot IDs $S_{\text{vis}} = \bigcup_{t} \text{cells}_{\pi_t}[p_t]$ and logical record IDs $R_{\text{vis}} = \{\lfloor s / 2 \rfloor : s \in S_{\text{vis}}\}$, reporting:
   - `raw_slot_reads` $= 32 N_{\text{probes}}$
   - `unique_physical_slots` $= |S_{\text{vis}}|$
   - `overlap_slot_reads` $= 32 N_{\text{probes}} - |S_{\text{vis}}|$
   - `unique_logical_records` $= |R_{\text{vis}}|$.

## 2. Query Families & Observable Interface

A query never exposes oracle target coordinates $(t_x, t_y)$, target record ID $r^*$, or target pages $\text{inv}_\pi[2r^* : 2r^*+2]$ at inference. Only observable query geometry and retrieval outcomes are provided:
- Anchor coordinate $(a_x, a_y)$ and dyadic phases $(a_x \bmod 16, a_y \bmod 16, a_x \bmod 32, a_y \bmod 32)$.
- Coarse directional signs $h_x = \text{sign}(t_x - a_x), h_y = \text{sign}(t_y - a_y) \in \{-1, 0, +1\}$.
- Coarse 3-level per-axis scale indicators $b_x, b_y \in \{0, 0.5, 1.0\}$ (`0`: exact axis $|d|=0$; `0.5`: near $1 \le |d| \le 18$; `1.0`: far $|d| \ge 19$).
- Probe outcomes: exact-key valid hit indicator, stale-replica indicator, cursor page centroid $(\bar{c}_x, \bar{c}_y)$, remaining budget, hop index, and legal stride mask.

### 2.1 Eight Distinct Query Families
1. **`horizontal`**: $d_y = 0$, $|d_x| \in [8, 64]$. Strongly favors `row`.
2. **`vertical`**: $d_x = 0$, $|d_y| \in [8, 64]$. Strongly favors `col`.
3. **`diagonal`**: Anti-diagonal Z-corridor across $16\times 16$ or $32\times 32$ dyadic tiles ($Q_1 \leftrightarrow Q_2$, $h_x h_y = -1$). Strongly favors `morton`.
4. **`isotropic`**: Uniform 2D cluster $|d_x| \in [3, 9], |d_y| \in [3, 9]$. Strongly favors `hilbert` (and stays capacity-invariant from 16K to 128K).
5. **`sparse_long_range`**: 2D long-range block jump ($|d_x| \in [59, 69]$ and $|d_y| \in [59, 69]$, centered on $\pm 64$ with $\pm 5$ jitter). Favors multi-projection switching (`row` $\pm 2 \leftrightarrow$ `col` $\pm 2 \to$ `hilbert`) with hierarchical strides.
6. **`multihop`**: Two-hop pointer chain inside a single probe budget. Valid retrieval of Hop 1 key $k_1$ reveals Hop 2 anchor, key $k_2$, and an orthogonal/distinct Hop 2 geometry. Favors recurrent projection switching over both fixed layouts and single-selection routers.
7. **`regime_shift`**: Single-key two-regime L-shaped displacement: one axis far ($|d_{\text{major}}| \in [59, 69]$) and the orthogonal axis local ($|d_{\text{minor}}| \in [4, 10]$). Optimal geometry transitions mid-episode from 1D axis traversal (`row`/`col`) to 2D/orthogonal capture (`hilbert`/`col`/`row`).
8. **`corrupted_replica`**: Primary replica at a nearby location is forced stale/corrupted; probing it returns `stale=True` and a coarse forwarding direction/scale toward a healthy mirror displaced along an orthogonal geometry ($|d_{\text{orth}}| \in [59, 69]$ where in-bounds). Tests stale-replica detection, recovery, and mid-episode projection switching.

### 2.2 Distribution-Shift Protocol
In addition to the balanced-mixture primary evaluation, a dedicated distribution-shift benchmark trains controllers on a skewed 16K mixture (80% weight on `{horizontal, isotropic, sparse_long_range, multihop}`, 20% on `{vertical, diagonal, regime_shift, corrupted_replica}`) and evaluates frozen checkpoints at 32K under the inverted 20%/80% mixture.

## 3. Coupled Recurrent Metacontroller

The controller operates at two timescales over shared observation $x_t$ plus projection-specific state memory $P_t \in \mathbb{R}^{4 \times d_p}$:
- **Fast state** $h_t = \text{GRU}_{\text{fast}}([x_t, s_{t-1}, c_{t-1}], h_{t-1}) \in \mathbb{R}^{d_h}$ (updated every step).
- **Slow state** $s_t \in \mathbb{R}^{d_s}$, updated on slow clockwork + event trigger ($t \bmod 2 = 0$ or hop advance or stale detection): $s_t = \text{GRU}_{\text{slow}}(h_t, s_{t-1})$.
- **Projection-specific states & cross-projection communication**: Active slot $P_t[\pi_t]$ updates via gated residual projection from $h_t$; cross-projection message $c_t = \frac{1}{3}\sum_{j \neq \pi_t} W_{\text{comm}} P_t[j]$ (ablated by setting $c_t = 0$).
- **Projection router**: $\ell^{\text{proj}}_t = f_{\text{router}}(s_t, h_t, c_t, x_t^{\text{geom}}) \in \mathbb{R}^4$, executed at $t=0$ (initial selection) and after every probe (recurrent switching).
- **Factored resolution & stride head**: Given selected $\pi_{t+1}$ and its legal mask $m \in \{0, 1\}^{13}$ over strides $\mathcal{D} = (-32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32)$ (with $d=0$ legal only upon projection switch to an unvisited page):
  - Resolution logits $r \in \mathbb{R}^2$ (`fine`: $|d| \le 4$; `coarse`: $|d| \ge 8$).
  - Combined stride logits $\ell_d = \ell^{\text{raw}}_d + r_{\text{res}(d)}$, masked by $m$.
- **Adaptive stop head**: $\ell^{\text{stop}}_t = w_{\text{stop}}^\top [h_t, s_t] + b_{\text{stop}}$; Arm F halts when $\ell^{\text{stop}}_t \ge 0$.

### 3.1 Parameter Scales
- **Primary 54K scale**: $d_h = 56, d_s = 56, d_p = 16$ ($\approx 54\text{K}$ trainable parameters, closely matching MCP-06's 54,345). All learned arms (C, D, E, F, G) share identical architecture and parameter counts.
- **Scaled variant ($\le 128\text{K}$ cap)**: $d_h = 80, d_s = 80, d_p = 24$ ($\approx 108\text{K} \le 128\text{K}$ parameters).

### 3.2 Optimal-Action-Set Supervision
From verified shortest-path traces generated exclusively at 16K, let $A^*(t)$ denote the set of legal actions achieving the minimum verified distance-to-goal at step $t$. Optimal-action-set loss minimizes:
$$\mathcal{L}_{\text{set}}(\ell, A^*) = -\left(\operatorname{LogSumExp}_{a \in A^*}(\ell_a) - \operatorname{LogSumExp}_{a \in A_{\text{legal}}}(\ell_a)\right),$$
which reduces to standard cross-entropy when $|A^*| = 1$ and avoids arbitrary tie-breaking penalties when multiple projections or strides are equally optimal. We ablate `action_set` against `single_best` cross-entropy.

## 4. Registered Evaluation & Statistical Gates

- **Training**: Exclusively at 16K, seeds `{107, 109, 113}`, frozen before evaluation.
- **Evaluation**: Capacities `{16K, 24K, 32K, 64K, 128K}` (128K included as registered extrapolation), 128 paired roots per seed, capacity, and query family ($128 \times 8 = 1024$ roots per seed/capacity; $3072$ paired roots per capacity), budgets `{4, 8, 16, 32}`, corruptions `{0%, 5%, 15%}`.
- **Primary Gate (32K, 16 probes, 0% corruption)**:
  $$\Delta_{\text{primary}} = \text{Comp}(\text{Arm E: recurrent\_switch}) - \max_{a \in \text{Fixed non-oracle (A, C)}} \text{Comp}(a) \ge +0.05 \text{ (+5 pp)}.$$
- **Secondary Gate (32K, 16 probes, 0% corruption)**:
  $$\frac{\overline{C}_{\text{reads}}(\text{Arm E}) - \overline{C}_{\text{reads}}(\text{Arm F})}{\overline{C}_{\text{reads}}(\text{Arm E})} \ge 0.15 \quad \text{and} \quad \text{Comp}(\text{Arm E}) - \text{Comp}(\text{Arm F}) \le 0.01.$$
- Paired bootstrap 95% confidence intervals (2,000 resamples), per-seed deltas, and per-family outcomes are reported unconditionally.

## 5. Native Storyworld Integration (Separate Track)

1. **Archived Witness Replay**: Replays MCP-05 `storyworld.hilbert.json` secret witnesses and adjacent-swap counterfactuals through native `transition`, `classify`, and `replay_path`, using bounded 32K multi-projection memory.
2. **Independent Route Discovery**: Strips `exact` and `balance` archived witnesses from the world object before execution. Encodes scene-option records `(operator, clue, advance)` into the 32K physical memory lattice. Each controller arm retrieves scene-option records under bounded probe budget and plans transitions using only retrieved records, native `transition`/`probability`/`classify`, and the target secret rule. Verifies legal transitions, required clues, terminal projector intervals, secret-ending eligibility, and noncommuting adjacent-operation counterfactuals.

## 6. Mechanistic Interpretation Protocol

At 32K (16 probes, 0% corruption), extract router probabilities, slow/fast trajectories, and switch events. Fit held-out linear probes on $s_t$ and $h_t$ for query geometry, search progress, and uncertainty. Execute five paired causal interventions:
1. Freeze projection router at $\pi_0$.
2. Shuffle slow states $s_t$ across episodes.
3. Remove cross-projection communication ($c_t = 0$).
4. Force incorrect projection ($\arg\min$ router probability).
5. Patch slow states $s_t$ between matched vs mismatched query families and measure steering & recovery.
