# COG-13B: cache-aware proof scheduling and exact checkpoint replay

Stacked on [COG-13 #34](https://github.com/MoralityLabAI/HybridTRMLDT/pull/34), whose [workflow 45849589142](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/45849589142) passed. COG-13B separates scheduler effects from restoration cost and includes operation-count sensitivity.

## Design

The standard-library Node benchmark implements a new deterministic synthetic Horn generator inspired by COG-02. The public seed is 2026112603; 64 roots each for train, calibration, ordinary, and deeper (depth 6–10). It forbids duplicate exact unpermuted or presented graphs within this generated cohort; graph-isomorphism equivalence across cohorts is not audited. These are fresh generated worlds, not the original COG-02 snapshot, and not Cogentic paper traces. No sealed evaluation.

Ten arms across eight disjoint eight-root portfolios per evaluation split. Budget prefixes are 48, 96, 144 selected verifier calls per portfolio. The two-hop local router and deterministic tie-breaking are shared except for the explicitly privileged full symbolic ancestry arm. A ridge predictor is fit once on 64 training roots, with each root weighted equally, 645 training checks, and nine public features. It predicts remaining local-router calls and prioritizes a root using residual estimate plus restoration cost. Quantum is ceil(estimate) clipped to 1–8, and the estimate is refreshed after every nonterminal fresh step. This is an LDT precursor, not a trained neural LDT. No calibration tuning.

The one-resident-slot tape policies evict LRU states and replay all prior actions on resume, charging each selected verifier call. Packed policies instead move an exact host-owned bitset of verified propositions to a cold backing map. On restoration the host reads it without re-verifying past facts. The independent auditor constructs each packed bit vector from its own checked state, verifies every saved/restored bitset and tape prefix, and rejects false fact, false checkpoint, and false cost sentinels. This trust model assumes the same trusted checker controls the backing store. A remote/untrusted checkpoint would require authenticated state and proof replay or a sound verified certificate. The action tape remains available even when packed checkpoints are used.

## Results

240 runs, 26,876 independently replayed selected verifier calls. All three negative sentinels rejected. Deeper split: 64 roots, including 19 with at least 40 public rules (a descriptive size cut used in the benchmark, not a confirmatory subgroup).

| Arm | 48 calls | 96 calls | 144 calls | Replays at 144 | Larger roots solved | Peak live + backing packed payload bytes |
|---|---:|---:|---:|---:|---:|---:|
| RR1, tape, 1 slot | 0 | 0 | 0 | 800 | 0/19 | 7 |
| RR4, tape, 1 slot | 1 | 1 | 3 | 512 | 0/19 | 7 |
| RR8, tape, 1 slot | 1 | 7 | 12 | 346 | 0/19 | 7 |
| RR1, 8 slots | 0 | 6 | 27 | 0 | 1/19 | 41 |
| RR1, packed, 1 slot | 0 | 6 | 27 | 0 | 1/19 | 41 |
| RR4, packed, 1 slot | 1 | 6 | 27 | 0 | 1/19 | 41 |
| Static-size block, 1 slot | 18 | 34 | 48 | 0 | 4/19 | 7 |
| Adaptive quantum, tape, 1 slot | 18 | 35 | 49 | 0 | 13/19 | 7 |
| Adaptive quantum, packed, 1 slot | 18 | 35 | 49 | 0 | 13/19 | 7 |
| Full ancestry, static-size block, 1 slot | 26 | 49 | 64 | 0 | 19/19 | 7 |

Packed RR1 exactly matches all-resident RR1 completion curves on all three partitions. It stores the same logical amount of packed state in live + backing storage; it is not a total-memory reduction relative to eight packed resident states. The deepest RR1 tape run wastes 800/1152 checks on restoration. Block scheduling removes that cost with a single resident ledger and completes 48/64. The adaptive predictor completes 49/64 and 13/19 larger roots vs static 4/19; both have zero replay on this cohort, so the adaptive packed and tape arms are identical. The larger-root readout is exploratory; the learned arm starts 57/64 roots and static starts 56/64. Full symbolic ancestry completes all 64 using 988 selected verifier calls.

Peak packed-state bytes exclude static graphs, maps, object overhead, estimates, run logs, and tapes. The tape payload is separately accounted at one byte/action plus two-byte length per root (indices bounded below 128). Actual JS Set/Array/Map sizes are larger; the report uses logical byte-equivalent payloads, not a process-memory comparison. See results.json for peak payload including tapes, checkpoint copy bytes, evictions, ordinary/calibration results, and all counters.

## Cost sensitivity

We count selected verifier calls, rule visits (including backward graph expansion and candidate screening), membership probes, ranking candidates, features, dot-product elements, and checkpoint/tape byte reads and writes. This is a declared synthetic operation model, not a CPU instruction count. Counters are monotone; the independent checker recomputes proof, tape and checkpoint counts and verifies weighted-prefix accounting from receipts, but does not independently reconstruct all graph/feature/scheduler operations or policy decisions. Ridge solve arithmetic, set construction, allocations, sorting and bit-pack CPU loops are not individually counted. The Node CLI emits whole-process elapsed time and RSS separately, not per-arm hardware measurements.

For weight W on each verifier call and unit weight for the other counters, read the last event with total cost <= W*144 per portfolio. At the largest weighted budget, completions out of 64:

| Policy | W=10 | W=100 | W=1000 |
|---|---:|---:|---:|
| RR1 tape | 0 | 0 | 0 |
| RR1 packed | 0 | 1 | 15 |
| Static-size block | 0 | 25 | 44 |
| Adaptive quantum | 0 | 18 | 41 |
| Full symbolic ancestry | 0 | 28 | 59 |

The learned arm's call-count advantage disappears under every tested nonzero screening-cost model at the largest budget. It visits 84,680 rules and performs 177,969 membership probes on the deeper split; static visits 42,633 and performs 86,515 probes. In this public Horn setting readiness is cheap and as strong as local checking. An expensive native verifier may shift the operating point; measure it before drawing neural/RAM conclusions.

## Reproduce and export

Node 20+, no external npm packages:

```bash
node experiments/cog13b_checkpoint_quantum/benchmark.mjs /tmp/cog13b-run.json /tmp/cog13b-exports
```

Use new output paths. The optional export bundle contains 645 root-weighted training-only residual-cost rows (nine public features, target remaining verifier calls), the full 256-task registry, a 192-episode index for deeper static/adaptive/full-ancestry arms (including all 31 failures), 161 constructive Lean proof terms, a wrong-term sentinel, and SHA-256 manifest. The export construction and summary equality were executed in the JS sandbox. Native compilation of those terms is a separate CI job using Lean 4.35.0. No new native compiler result is claimed before CI completes. These certificates concern generated propositional tasks only.

## COG-13C gate

Next use actual Mathlib goals with proof-state snapshots, host-verified proof progress, and measured per-call costs. Replace exact symbolic readiness with actual candidate tactics. Test a trained LDT allocator against static and cheap cost estimators, including cost-aware fairness, on fresh root-clustered portfolios. Keep certified ledger facts exact; use BPD low-rank channels for untrusted routing features or proposal scores, not as authority to admit a fact. Compare packed exact checkpoints, authenticated snapshots, and recomputation under equal total bytes (including graph, tape and checkpoint metadata) and measured RAM/CPU. DataGoblin can ingest the train-only residual labels after reviewing the new source manifest; the current PR exports a file and does not modify DataGoblin or train a neural controller.
