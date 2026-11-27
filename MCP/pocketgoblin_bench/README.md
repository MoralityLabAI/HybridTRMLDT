# PocketGoblin Bench

Measures how well a model (an LLM agent) trains a Pocket Goblin router mesh for a given MCP.

Pocket Goblin answers questions from a compact context packet that a learned router builds over an MCP
surface. [MCP-02](../mcp02/README.md) generates synthetic multi-server MCP catalogs from seeds, builds
train and dev datasets, fits router models and evaluates them against BM25 and oracle controls. This
benchmark puts a trainer behind a tool surface and scores what it submits.

A **mesh** is the router stack that decides which catalog branch, tool or resource to read next: encoder
features (`tokens` or exact-`match`), a core (`dense`, `transformer`, `looped`, `trm`, `trm_ldt`, `hrm`),
training hyperparameters, and the filters and budgets (`prefilter_k`, `max_tool_calls`).

## Episode contract

An episode is one MCP plus a compute budget. The MCP is a family of MCP-02 worlds (catalog size, arm,
index layout) and `mcp_seed` selects its train, dev and test worlds. The model under test sees only the
trainer tools, served as an MCP stdio server (newline JSON-RPC 2.0, protocol 2025-06-18):

| Tool | Effect | Charged |
| --- | --- | --- |
| `describe_mcp` | Catalog roles, record kinds, concepts, the routed tool surface, split sizes, scoring rules | free |
| `sample_train_queries(n, offset)` | Train requests with answer and gold evidence records | free |
| `list_mesh_families` | Families, hyperparameter ranges and defaults, charge rules | free |
| `estimate_cost(candidate)` | Upper bound of a fit in units | free |
| `fit(candidate)` | Trains the mesh on the train worlds, returns `candidate_id` | training GFLOPs + 0.5 per step |
| `evaluate(candidate_id)` | Dev score against BM25 and oracle; repeats are free | inference GFLOPs + 2 per dev episode |
| `submit(candidate_id)` | Scores once on the hidden test worlds; ends the episode | free |
| `status` | Budget, fits, evaluations | free |

Test worlds are generated inside `submit` and never appear in any tool result or receipt. Train, dev and
test worlds use disjoint entity and server name pools and disjoint request templates.

**Budget.** One unit is about 25 ms of single-process CPU. Charges depend only on the candidate and the
seeds, so a run is reproducible. A `fit` whose upper bound exceeds the remaining budget is refused before
any training (`budget_exceeded`); `max_fits` and `max_evals` are also enforced. Wall-clock CPU seconds are
recorded in the receipt but do not enter the score.

**Candidate.**

```json
{"family": "trm",
 "core": {"d": 24, "heads": 2, "net_layers": 1, "inner": 1, "cycles": 2},
 "encoder": "match",
 "train": {"worlds": 96, "epochs": 2, "lr": 0.0015, "batch_states": 32, "seed": 7},
 "filters": {"prefilter_k": 32, "max_tool_calls": 40}}
```

Missing keys take defaults; unknown keys and out-of-range values are rejected with the offending path.

## Scoring

The hidden test worlds are run with the submitted mesh through MCP-02's `run_episode` on the episode's
surface. Per episode:

* `packet_recall`: gold-evidence recall of the final packet, counted as 0 when the packet exceeds the
  handoff budget (2048 bytes);
* `packet_tokens`: handoff bytes / 4;
* `tool_calls`; `accepted` (MCP-02's strict success) is also reported.

Metrics are means over test worlds. The score is the position of the mesh's `packet_recall` between the
`lexical_bm25` control (0) and the `oracle_upper_bound` (1), clipped to [0, 1]:
`score = clip((mesh - bm25) / (oracle - bm25))`. The unclipped `normalized_packet_recall` is reported too.
`deterministic_rules` and `sham_random` are reported as extra calibration points. The receipt carries the
score with the budget spent. Scoring is deterministic given the config, `mcp_seed` and candidate. A run that
never submits scores 0 (`no_submission`).

## Commands

Run from the repository root with torch installed. Use at most 8 threads.

```bash
# smoke episode: both reference trainers, about a minute on CPU
python -m MCP.pocketgoblin_bench reference --config MCP/pocketgoblin_bench/configs/smoke.json --out RUN
# over the stdio server instead of in process
python -m MCP.pocketgoblin_bench reference --config MCP/pocketgoblin_bench/configs/smoke.json --transport stdio --out RUN
# standard episodes 0..2
python -m MCP.pocketgoblin_bench reference --config MCP/pocketgoblin_bench/configs/standard.json --mcp-seeds 3 --out RUN
# controls and oracle only (standard library)
python -m MCP.pocketgoblin_bench controls --config MCP/pocketgoblin_bench/configs/smoke.json
# serve one episode to any MCP client
python -m MCP.pocketgoblin_bench.server --config MCP/pocketgoblin_bench/configs/standard.json --mcp-seed 0 --receipt RUN/episode.json
# LLM runner: loopback OpenAI-compatible endpoint, native tool calls, one receipt line per episode
python -m MCP.pocketgoblin_bench llm --config MCP/pocketgoblin_bench/configs/smoke.json \
    --base-url http://127.0.0.1:8091/v1 --model NAME --mcp-seeds 3 --out RUN/receipts.jsonl
# tests
python -m pytest -q MCP/test_pocketgoblin_bench.py
```

Configs: `smoke.json` (catalog 96, 24 train / 6 dev / 12 test worlds, 400 units) and `standard.json`
(catalog 128, 96 / 16 / 32 worlds, 3000 units). Both use the `layered_ldt` surface with layout `PSCB`, so
every family applies.

## Reference trainers

* `default`: fits one fixed mesh, a width-24 TRM with two cycles and the match encoder, trained on all train
  worlds with default hyperparameters, and submits it.
* `random`: draws meshes uniformly over families and ranges (log-uniform learning rate, quartile world
  counts), skips draws whose cost bound exceeds the remaining budget, fits and dev-evaluates until the budget
  or the draw limit is reached, and submits the best dev mesh. Seeded.

## Baselines

Measured on CPU (6 threads), `standard.json`, `mcp_seed` 0 to 2, 32 test worlds per episode. Controls are
the same trainer-independent calibration points on every episode; the mean over three episodes is shown.

| Policy | packet_recall | accepted | packet_tokens | tool_calls | score (per episode) |
| --- | --- | --- | --- | --- | --- |
| `sham_random` control | 0.18 | 0.06 | 55 | 39.6 | |
| `lexical_bm25` control (score 0) | 0.76 | 0.61 | 170 | 31.3 | 0 |
| `deterministic_rules` control | 0.87 | 0.81 | 195 | 27.2 | |
| `default` trainer, 1 fit, 1000 to 1060 of 3000 units | 0.90 | 0.85 | 202 | 25.8 | 0.75, 0.59, 0.34 (mean 0.56) |
| `random` search, 7 to 9 fits, 2995 of 3000 units | 0.63 | 0.43 | 142 | 19.2 | 0.0, 0.0, 0.0 (unclipped -0.44, -0.85, -0.45) |
| `oracle_upper_bound` control (score 1) | 1.00 | 1.00 | 227 | 16.8 | 1 |

The fixed default mesh (TRM, match encoder, all train worlds) beats BM25 on every episode but does not reach
the deterministic-rules control. Random search spends the whole budget and lands below BM25 on every episode; its
winning meshes were a looped, an HRM and a transformer core. The default mesh scores above random search and
above BM25 at a third of the budget.

`smoke.json` (one episode, stdio transport, 53 s wall for both trainers): `default` packet_recall 0.79
(score 0, unclipped -0.11), `random` 0.25 (unclipped -3.0), against BM25 0.81, deterministic rules 0.88,
oracle 1.00 on 12 test worlds. The smoke config checks the pipeline; its 12 test worlds are too few to rank
trainers.

Receipts for the three standard episodes hold the per-fit dev scores and the chosen candidate.

## Files

| File | Role |
| --- | --- |
| `space.py` | Families, ranges, validation, sampling |
| `episode.py` | Episode state, tools, budget, scoring, receipt |
| `server.py` | MCP stdio server around one episode |
| `reference.py` | Scripted trainers, in-process and stdio clients |
| `llm_runner.py` | Loopback OpenAI-compatible runner |
| `__main__.py` | `reference`, `controls`, `llm` commands |
