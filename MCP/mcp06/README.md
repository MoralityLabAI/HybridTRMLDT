# MCP-06: Learned spatial memory and adaptive recurrence

Parent evidence: `1307e7f9a38eb1afd63fb585b0cc80e297ddf1d9`.
MCP-05's Hilbert space describes two-qubit storyworld state. MCP-06 adds a **separate** Hilbert-curve layout for memory addressing. This is a CPU retrieval proxy with a native fixed-witness replay bridge, not a model that authors stories, renders Canvas, calls an MCP server, or independently discovers secret endings.

## Running

From the repository root, with Python 3.11+ and CPU PyTorch:

```sh
python -m pip install -r MCP/mcp06/requirements.txt
python -m unittest MCP.mcp06.test_gym -v
python -m MCP.mcp06.gym --smoke --threads 2 --out MCP/results/mcp06/smoke
python -m MCP.mcp06.gym --threads 4 --out MCP/results/mcp06/registered
```

Use a fresh output directory for each run. On a standalone checkout without MCP-05, native replay is skipped and explicitly reported. `--native-world` can point to the archived storyworld JSON in a complete checkout; the native verifier comes from `MCP.mcp05.hilbert_storyworld`. On Windows, optional `psutil` enables peak working-set reporting. No GPU, provider credits or external MCP endpoint needed.

## Formal environment

A capacity N denotes physical record slots, **not tokens**. The rectangular memory lattice has width 128 for N=16,384, 24,576, 32,768; width 256 at N=65,536. Logical records have two adjacent physical replicas, a randomly permuted key and a payload. Each arm uses identical spatial records, queries and corruption draws. Address layout is a bijection of slots, packed into pages of 32. The 24K and 32K Hilbert layouts clip the next enclosing power-of-two square; the clipped curve need not preserve adjacency across gaps.

A query contains an anchor location/address, requested key and a coarse quadrant hint. The requested record lies within a clipped radius-12 square of the anchor. The learned observation includes normalized page/anchor position, page-center displacement, hint signs, hit/stale indicators, remaining budget and legal-action mask. The key is used only for exact match inside a paid page probe. The target location and correct payload are privileged to the training teacher and verifier; neither is a learned input. The page-centroid metadata and replica-validity flag are supplied by the environment and are assumed trustworthy. Stale copies are rejected, rather than simulated as undetectably wrong content. Corruption is independently assigned per physical replica; both copies can become unavailable.

Controller (54,345 trainable parameters): two coupled GRU cells (64 state units each), a fast update each visit and a slow update every fourth visit, with eight movement logits and a stop logit. This is a TRM-style two-timescale proxy and an LDT-inspired stride hierarchy; it does not implement the full TRM/LDT stack. Training uses teacher-forced cross-entropy on legal movements toward the closest replica, plus weighted binary cross-entropy for stopping on a valid match. The fixed and adaptive hierarchical arms share the very same trained checkpoint; no explicit compute penalty or policy-gradient training is included in v1. The adaptive arm learns a stop classifier and is scored on actual probes. Fixed arms store the first valid payload and continue probing until the cap or until legal moves run out. Deterministic scan stops on the first valid match.

| Arm | Layout | Movement / termination |
|---|---|---|
| scan | Hilbert | alternating outward page scan; verified hit stops |
| row | row-major | learned local ±1,2,3,4; fixed cap |
| random | random permutation | same learned local movements and inputs |
| hilbert | Hilbert | same learned local movements and inputs |
| hier | Hilbert | learned ±1,2,8,32 page strides; fixed cap |
| adaptive | Hilbert | identical hier checkpoint, learned stop |

No repeated pages in an episode. Every probe costs 32 record reads. A legal-action dead end terminates. Fixed budget means a matched maximum, with realized costs reported separately. The random control changes both geometry and replica co-location; the row-major control helps distinguish locality from Hilbert-specific ordering.

## Registered proxy matrix (v1)

Training: one independently generated 16K layout per model seed, 1,024 roots, 8 epochs, AdamW lr .002, batch size approximately 32, teacher horizon 16; seeds 107,109,113. Test layouts and root draws use a separate deterministic namespace. These are reproducible held-out fixtures, not secret test data. All models frozen during evaluation. Capacities 16K/24K/32K/64K, budgets 4/8/16/32 probes, corruption 0/5/15%, 128 paired roots per seed per capacity. Smoke uses seed 107, a distinct evaluation namespace, 64 training roots, two epochs, eight test roots, two capacities and two budgets. Do not count smoke as registered evidence.

Primary comparison: mean of the three paired per-seed differences, Hilbert-local vs random-local at 32K, 16 probes and no corruption. Point gate >= 0.05 (percentage-point fraction). This is an engineering threshold, not a statistical significance claim. Report individual seed differences and all arms, including row and deterministic scan, regardless of the result. Inspect 24K/64K transfer and corruption recovery separately. Native replay is an additional descriptive test, never the primary proxy gate. Any tuning after the registered run requires a new experiment ID and fresh held-out namespace.

## Native witness bridge

Reads the pinned MCP-05 `storyworld.hilbert.json`. Before testing retrieval, the bridge replays each archived exact secret witness through native `transition` and `classify`. It also verifies the archived adjacent-swap control with native `replay_path`, when supplied. For each witness decision, the synthetic memory stores its encoded option ID as the requested record's payload. A retrieved ID must match the actual scene and action; the native transition updates the two-qubit state and clues. A route passes only if every retrieval succeeds, the terminal scene is reached and the requested secret ending is eligible. This evaluates memory-assisted replay of known decisions, not story planning or browser verification. The 16-probe cap applies per decision. Native world hash is recorded. The future native task should select actions without supplying the known witness as a query schedule.

## Receipts

`roots.jsonl`: paired per-root successes, values, unique visited pages, probe counts, charged record reads, matched valid replicas, stale/recovery flags, premature stops, and mean page-center Manhattan displacement. `summary.json`: per-cell totals and timings, matched seed deltas, checkpoint hashes, native witness outcomes, peak process RSS (where available), overall wall time and gate status. Process RSS includes Python/PyTorch overhead and is not incremental memory. Four learned checkpoints per seed; adaptive reuses hier. No sealed claims from smoke or unavailable native data.
