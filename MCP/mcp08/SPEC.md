# MCP-08 Formal Specification: Context Economics and Autonomous World Construction

**Target Paper:** *The Cost of Context II: Hardware-Aware Recurrent Memory for Autonomous Local Agents*  
**Parent Baselines:** `MCP-05` (`gym/mcp05-controller-evidence`), `MCP-06` (`gym/mcp06-spatial-memory`), `MCP-07` (`gym/mcp07-adaptive-geometry`).

MCP-06 and MCP-07 established spatial memory, adaptive multi-projection geometry, and recurrent routing in controlled synthetic gyms. **MCP-08** tests whether those capabilities translate into verified end-to-end work from actual local language models (**Bonsai 2 27B** and **Inky Mini**) operating under a strict physical compute and memory budget.

---

## 1. Four New Evaluation Environments (32 Held-Out Roots)

Each environment supplies **8 fresh held-out evaluation roots** (32 total, namespace `810000+`) plus separate **development fixtures** (namespace `800000+`) for harness debugging. Every root strictly partitions:
- `federation_corpus`: searchable records exposed to retrievers and models (containing deliberate stale/contradictory/distractor records), and
- `private_verifier_witness`: ground-truth answers, mathematical witnesses, unseen test maps, and invariant checks held **strictly outside** the searchable federation.

| # | Environment | Domain | Adversarial Corpus Hazards | Independent Verifier Criteria |
|---|---|---|---|---|
| 1 | **The Flooded City** | Narrative memory | Contradictory & obsolete lore, superseded faction treaties, flood-level changes | State consistency, character relationships, inventory & flood invariants, causal continuity, valid endings |
| 2 | **Quantum Labyrinth** | Mathematical planning | Noncommuting operators, resource budgets, hidden endings, witness withheld from model & retriever | Native complex-amplitude Hilbert replay ($\|\psi\|^2=1$), noncommuting order check, route efficiency, valid terminal state |
| 3 | **Arena Architect** | Executable game design | Broken JS combat mechanics, faulty enemy AI/pathfinding, missing Canvas draw invariants | Headless execution on previously unseen maps & adversarial player actions: functionality, playability, invariants, rendering |
| 4 | **The Broken Archive** | Agentic research | Stale specifications, renamed functions, conflicting commits, plausible irrelevant documents | Exact multi-hop answers, valid canonical source citations, full dependency chain discovery, stale-claim recovery |

---

## 2. Experimental Treatment Arms (224 Scored Episodes)

Every model faces identical roots, federation snapshots, tool interfaces, verifiers, and completion budgets.

| Arm | Context Policy | Evidence Budget | Roots per Model | Episodes (2 Models) |
|---|---|---:|---:|---:|
| **A** | BM25 retrieval | 3K tokens | 32 (all) | 64 |
| **B** | Hybrid lexical + embedding retrieval | 3K tokens | 32 (all) | 64 |
| **C** | Learned SetTRM + adaptive working set (Pocket Goblin) | 3K tokens | 32 (all) | 64 |
| **D** | Wide-context baseline | 16K tokens | 8 (2/env preset) | 16 |
| **E** | Oracle evidence selection | 3K tokens | 8 (2/env preset) | 16 |
| **Total** | | | **112 / model** | **224** |

### Cost & Interface Accounting
- Charge the recurrent controller's **runtime wall-clock**, **index storage bytes**, and **preprocessing cost** into total episode accounting.
- Audit and record which MCP-07 capabilities are available through the live runtime interface vs. requiring an explicit integration adapter.

---

## 3. Hardware-Aware Inference Experiment (Pareto Frontier)

Independently vary actual prompt length and allocated context capacity on the RTX 5080 Laptop GPU (16 GB VRAM), anchored against earlier MCP-05 observations (e.g., 7,168-token slot, 14.22 GiB VRAM, 36.78 t/s, 1,355 MiB headroom):

- **Actual input length:** `{1K, 3K, 6K, 12K, 16K, 24K, 32K}` tokens
- **Allocated context capacity:** `4K` through `40K` (feasible combinations)
- **KV cache precision:** `{F16, Q8_0, Q4_0}` where supported
- **Execution regime:** `{pure_gpu, explicit_cpu_offload, paging}`
- **Diagnostic generation:** fixed `256`-token output
- **Protocol:** (1) Hold a matched `3K` prompt fixed across context allocations to isolate memory-allocation cost; (2) Sweep actual prompt length at fixed allocation; (3) For Inky Mini, add a recurrence sweep over `{1, 2, 4}` loops where supported.
- **Output:** Context–Recurrence–Accuracy–Latency Pareto frontier per model.

---

## 4. Full Storyworld Generation (8 Complete Builds)

Two fresh playable worlds authored from scratch by both models under conventional hybrid retrieval (Arm B) and Pocket Goblin (Arm C) — yielding **2 worlds × 2 models × 2 policies = 8 builds** (with all failed stages and retries preserved):

1. **The Clockmaker's Republic**: Political RPG with 8 districts, competing factions, persistent economic resources, multiple endings, and decision-reactive NPCs. Native verification checks faction transitions, resource conservation, unreachable states, and decision-altered later encounters.
2. **The Garden of Unmeasured Things**: Quantum-mechanical storyworld where measurements alter available branches and hidden endings depend on transformation order and accumulated knowledge. Native verification checks state normalization, projector probabilities, reachability, and order-sensitive noncommuting counterfactuals.

Deterministic verifiers govern functional correctness; blinded external review (Fable or Opus) scores narrative coherence, creativity, character development, and visual composition.

---

## 5. Primary Metric & Preregistered Success Gates

### Primary Performance Metric
Verified task completions per GPU-hour, measured alongside raw success rate:

$$\eta = \frac{3600 \sum_{i} \mathbf{1}[\text{verified}_i]}{\sum_{i} T_i}$$

where $T_i$ is full end-to-end wall time in seconds (retrieval + controller + inference + tools + verification + repair attempts, including failures).

### Secondary Telemetry
Task success (independent verifier), time to success, evidence recall, retrieval precision, context compression vs. Arm D, repair burden (extra turns/tokens), peak GPU & system RAM, energy efficiency (verified tasks/kWh if measurable), narrative quality (blinded review), controller overhead (runtime + index bytes).

### Preregistered Success Gates
1. **Learned routing**: $\ge +5\text{ pp}$ verified completion over the strongest conventional equal-budget arm ($\max(\text{A}, \text{B})$).
2. **Context economy**: $\ge 25\%$ lower end-to-end wall time at comparable success rates.
3. **Memory residency**: `0` CPU tensor offload and `0` detected GPU paging in pure-GPU treatments.
4. **Autonomous construction**: Complete fresh browser-verifiable worlds with zero inherited solution artifacts.
5. **Transfer**: Positive efficiency gains ($\Delta \eta > 0$) on both Bonsai 2 27B and Inky Mini.

Paired bootstrap 95% confidence intervals (2,000 resamples) are computed by root. The 32-root suite is an engineering validation before expanding to fresh roots for strong statistical claims.

---

## 6. Staged Run Order

| Stage | Workload |
|---|---|
| **MCP-08A** | CPU fixture construction, strict witness isolation, and deterministic verifier tests |
| **MCP-08B** | Eight separate pilot roots, both models (Bonsai first; calibrate Inky compute budget) |
| **MCP-08C** | Bonsai pure-GPU hardware-aware memory/context/KV frontier sweep |
| **MCP-08D** | Registered 224-episode comparison across Arms A–E |
| **MCP-08E** | Eight complete storyworld builds (*Clockmaker's Republic* & *Garden of Unmeasured Things*) |
| **MCP-08F** | Blinded Fable/Opus review and paper preparation |
