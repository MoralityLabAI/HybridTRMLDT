# RLM Architecture Neighborhood v0.1 Results

## Result

The transport-qualified run completed all 16 registered cells with no cell errors. Direct full-context, RLM depth 1, and RLM depth 2 each solved three of four tasks; deterministic four-way map/reduce solved two. The preregistered Pareto frontier contains direct full-context and RLM depth 2.

RLM depth 2 matched direct full-context's aggregate accuracy while using 26,394 rather than 79,649 provider tokens, a 66.86% reduction. This efficiency came with 13 rather than 4 provider calls and 24.9708 rather than 4.9202 seconds of summed cell execution time. The result is therefore a token/latency tradeoff, not a scalar win.

Neither RLM arm made a recursive subcall. Both used the restricted local REPL to inspect and compute over the external context, but maximum observed depth remained zero. The run therefore demonstrates a useful iterative external-context neighborhood, not a recursion-depth benefit.

## Aggregate Endpoints

| Architecture | Solved | Accuracy | Calls | Input tokens | Output tokens | Total tokens | Cell time (s) | Observed depth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct full-context | 3/4 | 0.75 | 4 | 79,630 | 19 | 79,649 | 4.9202 | 0 |
| Map/reduce 4 | 2/4 | 0.50 | 20 | 80,867 | 76 | 80,943 | 16.1605 | 0 |
| RLM depth 1 | 3/4 | 0.75 | 17 | 32,721 | 2,270 | 34,991 | 46.5834 | 0 |
| RLM depth 2 | 3/4 | 0.75 | 13 | 24,738 | 1,656 | 26,394 | 24.9708 | 0 |

The capability gate used one additional call and 22 tokens. Its usage is sealed separately and excluded from the table.

## Task Split

| Task | Direct | Map/reduce 4 | RLM depth 1 | RLM depth 2 |
|---|---|---|---|---|
| Exact needle retrieval | PASS | FAIL | PASS | PASS |
| Three-hop join | PASS | PASS | PASS | FAIL |
| Global sum | FAIL | FAIL | PASS | PASS |
| Latest revision | PASS | PASS | FAIL | PASS |

The task split matters more than the tied aggregate score:

- Only the RLM arms solved the exact global sum. Their REPL traces parsed all `category=amber` records and computed `7049` locally. Direct returned `6921`; map/reduce returned `9266`.
- Map/reduce lost the exact needle even though one partition contained it. The fixed summaries are a lossy interface for sparse evidence.
- RLM depth 1 parsed `AUDIT` filler records instead of the target `REV` records on latest revision, producing an unrelated payload.
- RLM depth 2 searched unrelated log identifiers after finding the alias on the three-hop join and exhausted its control loop without a release code.

These are complementary failures rather than a shared capability ordering. No architecture solved every task.

## Realized Neighborhood

The frozen feature space specified four configurations:

1. Direct full-context inference.
2. Fixed four-way partition and reduction.
3. External context plus restricted REPL with recursion permitted to depth 1.
4. External context plus restricted REPL with recursion permitted to depth 2.

The observed control flow collapsed the last two into the same topology: iterative root-model calls plus local code execution, with zero model subcalls and zero recursive subcalls. Their registered root iteration limits also differ (`4` for depth 1 and `3` for depth 2). Consequently, the depth-1/depth-2 difference in this single run is jointly attributable to stochastic model behavior and control-budget differences; it is not evidence that recursion depth 2 is better.

The local REPL nevertheless changed the computational regime. It let the model transform a long prompt into compact program outputs, reducing repeated provider input. RLM depth 2 used 33.14% of direct's total tokens, but required 5.08 times its summed cell latency. This identifies a concrete architecture neighborhood axis: provider-token conservation through local context computation versus low-call direct latency.

## Instrument And Resource Audit

- Configured model: `gpt-4.1-mini` through the official RLM `OpenAIClient` chat-completions path.
- Capability gate: exact response passed; 16 input and 6 output tokens, excluded from architecture metrics.
- Official RLM source: commit `72d6940142ddfb84ee6be573dc999a37e633e671`, package `rlms==0.1.3`.
- Task suite: four deterministic prompts, SHA-256 `8b5607ca597c68831dbd8f739a289cb944ac8facf34a30e742088d2301236528`.
- Run attempt 1: stopped before process launch because an unrelated local GPU workload was active; no provider calls.
- Run attempt 2: completed in 109.406 seconds; peak process RAM 4.172 MB, measured peak I/O 0 MB/s, peak VRAM 0 MB, cleanup passed.
- Integrity: 16 records, 16 trajectories, and all registered result, event, manifest, capability, and trajectory hashes re-verified.
- Regression suite: 355 tests passed after the run.

## Claim Boundary

This is one stochastic observation per architecture-task cell with no provider seed. The pilot can establish the observed task split, resource use, control-flow realization, and Pareto relation for these receipts. It cannot estimate variance, separate prompts from architecture, establish recursion value, generalize to other context lengths or models, compare trained neural architectures, or support a general RLM-superiority claim.

## Next Discriminating Experiment

The next study should force and verify recursion rather than only permit it. Use tasks whose contexts are split into independently generated shards and whose exact solution requires at least one model subcall, register a minimum-subcall manipulation check, and compare:

1. REPL-only RLM with recursive calls disabled.
2. RLM with depth 1 and the same root iteration budget.
3. RLM with depth 2 and the same root iteration budget.

Replicate each cell across at least three provider runs. This isolates recursive delegation from the external-context and iteration-budget effects measured here.
