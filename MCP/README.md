# MCP gym

`MCP-01` supplies an executable gym for comparing how local models discover tools and retrieve evidence through MCP. The initial environment uses synthetic records with an exact judge. It includes a real stdio server for a pinned MCP subset, a deterministic reference controller, and an adapter for a locally served chat model.

Data Goblin's longer-term goal is to become a local retrieval specialist that spends local inference on finding evidence and gives a frontier agent a compact, useful packet. [ARCHITECTURE.md](ARCHITECTURE.md) describes that path and the interfaces for other Transformer families. [PROTOCOL.md](PROTOCOL.md) separates the current smoke from a later frozen model comparison.

## MCP-02

[MCP-02](mcp02/README.md) adds multi-server compositional tasks, layered indices with certified candidate refinement, and interchangeable learned routers (dense, Transformer, looped, TRM, TRM+LDT, HRM) with deterministic, sham and oracle controls. MCP-01 below remains the reference baseline; its code and receipts are unchanged.

## MCP-03 and Goose

[MCP-03](mcp03/README.md) applies the same routers to code-repository MCP surfaces on SWE-bench (training on django and sympy, testing on ten held-out SWE-bench Lite repositories) and measures token-budgeted context packets. [goblin-gdk](goose_gdk_app/README.md) is a Goose Development Kit app that injects the packet into a goose-agent state machine; `mcp03/goblin_server.py` exposes it as an MCP extension. Results: [`results/mcp02`](results/mcp02/README.md), [`results/mcp03`](results/mcp03/README.md); paper: [`papers/mcp_data_goblin`](../papers/mcp_data_goblin).

[`hubrun`](hubrun/README.md) runs Prime Environments Hub agentic tasksets and the MoralityLab storyworld environments against local Ternary Bonsai 2 27B, through a relay with a raw arm and a goblin-packet arm. Results: [`results/hubrun`](results/hubrun/README.md).

## Run the gym

Run from the repository root with Python 3.10 or newer. The gym's runtime uses the standard library and the repository's existing lattice kernel.

```bash
python -m MCP.run --controller reference --transport stdio --cases 24 --out MCP/runs/smoke
```

This runs paired episodes in all four discovery arms. The client initializes a child MCP server, discovers its tools, and executes calls over stdio. An independent in-process replay checks each returned tool result. The final judge requires the correct answer and evidence that the controller read. The reference controller uses public task requirements and tool observations.

For a catalog-width and index-depth sweep:

```bash
python -m MCP.run --controller reference --catalog-sizes 8 32 128 --depths 1 3 6 --cases 10 --out MCP/runs/shape-sweep
```

The catalog width changes the number of records and the eager leaf-tool count. Index depth changes the tree used by the layered arms. The task and required evidence remain paired across widths; larger catalogs add distractors. The `flat` and `search` arms also run at each depth as controls.

The task generator covers exact lookup, arithmetic across branches, provenance decoys, and missing evidence. Split names `train`, `dev`, and `test` create disjoint synthetic source groups. They reuse the task templates, so this smoke establishes plumbing and accounting rather than semantic generalization.

## Use a local model

Start a model server on the same machine with a chat-completions endpoint. Replace `local-model` with its served model ID:

```bash
python -m MCP.run --controller local --model local-model --base-url http://127.0.0.1:8080/v1 --model-family transformer --max-output-tokens 4096 --max-steps 64 --out MCP/runs/local-pilot
```

The controller receives the public task and currently advertised tool schemas. It returns one JSON action per turn:

```json
{"tool":"catalog_search","arguments":{"query":"entity from the task"}}
```

The terminal action has this shape:

```json
{"tool":"finish","arguments":{"answer":42,"evidence_refs":["reference returned by a read"],"abstain":false}}
```

Those values illustrate the response format. Each episode supplies its own records and answer. The adapter uses JSON action mode; native model function-calling integration is a later adapter. A malformed action receives a format-error observation and consumes a model turn. Endpoint or transport failures remain failed episodes with their receipts.

Local exploration has separate `--max-output-tokens`, `--max-steps`, and `--max-runtime-seconds` bounds. Set these to suit the model and machine. `--handoff-budget-bytes` controls the compact evidence packet, with a default of 4096 bytes. A larger local exploration budget can coexist with a small handoff budget.

`--max-context-bytes` bounds the serialized message history; `--context-window-tokens` records a known server context limit and checks it against reported usage. Exact token preflight requires the server's tokenizer, which this adapter leaves to the serving backend. The gym records local token use even when the experiment permits extensive local deliberation. A later downstream comparison can reward the extra local work when it improves the evidence packet or reduces paid model use.

Model identity fields are explicit:

| Option | Recorded meaning |
| --- | --- |
| `--model` | Requested server model ID |
| `--model-family` | Declared architecture label, such as Transformer, TRM, or hybrid |
| `--parameter-count` | Declared parameter count |
| `--checkpoint-sha256` | Declared checkpoint digest |
| `--quantization` | Declared precision or quantization label |

Supply the actual checkpoint details when available. Missing fields remain unknown. The adapter also records the server's returned model identity. A name alone does not establish architecture or checkpoint provenance. Endpoints are restricted to loopback and HTTP redirects are rejected.

## Discovery arms

| Arm | Model-visible interface | Implementation |
| --- | --- | --- |
| `flat` | All leaf read-tool schemas | Eager catalog |
| `search` | `catalog_search`, `evidence_read` | Exact metadata substring retrieval with pagination |
| `layered` | `index_children`, `evidence_read` | Traversal through entity ranges |
| `layered_ldt` | Same layered tools | Host-checked metadata exclusions through `certify_and_apply_verified` |

The LDT arm checks revision, scope, epoch, and units. Unknown registry metadata keeps a candidate available. Each check operates on the children of the requested node; persistent lattice state and learned proposal networks belong to later experiments. Counters include metadata scans and checker calls.

The discovery arm and controller architecture are independent variables. The local adapter can compare different served Transformer models using the same environment. Future native adapters can expose TRM state and recurrent depth directly, with checkpoint and reset contracts preserved.

## Results and accounting

Each run directory contains:

- `summary.json`, with paired arm results, declared model details, configuration, and source hashes.
- `episodes.jsonl`, preserving tool actions and results, model receipts, failures, and the final evidence packets.
- `source_manifest.json`, with exact SHA-256 hashes of the gym and imported kernel files.

The report separates answer/evidence correctness from packet-budget adherence. Joint acceptance requires both. Failed episodes contribute to total model use and time. Token counts come from the local server when supplied; absent usage remains `null`, with partial known totals recorded separately. Exact UTF-8 byte counters describe their serialization boundary and remain separate from tokens.

Reference runs make zero model calls and carry no neural token measurement. `reference_context_utf8_bytes` shows the simulated context exposure under the reference transcript; `model_input_utf8_bytes` counts message payloads for attempted local model requests. `model_observation_utf8_bytes` measures the single result representation exposed to the controller, while complete MCP results remain in the receipts. Handoff bytes measure a possible downstream packet. Frontier token savings require a downstream experiment.

Generated runs are ignored under `MCP/runs/`. A small reviewed smoke receipt can be preserved under `MCP/results/`. Keep raw records with the result; record an external artifact URI and checksum if a later run exceeds the repository's artifact budget.

## MCP transport

The fixture server is callable independently:

```bash
python -m MCP.serve --architecture layered_ldt --split test --seed 70919 --case-index 0 --n 24 --catalog-size 32 --depth 3
```

It implements `initialize`, `notifications/initialized`, `tools/list`, and `tools/call` for MCP `2025-06-18`. The initialization result includes the public task as a gym extension. Protocol traffic alone goes to stdout. This is a bounded experimental server; its supported methods are detailed in [PROTOCOL.md](PROTOCOL.md).

## Verification

```bash
python -m pytest -q MCP tests/test_cca_verified_membrane.py tests/test_hybrid_membrane.py
```

Tests exercise the exact evidence contract, LDT pruning, paired fixture construction, actual subprocess stdio, a local HTTP test server, and failure-inclusive accounting. The dedicated [MCP gym workflow](../.github/workflows/mcp-gym.yml) runs these checks and a deterministic stdio pilot. Live local-model runs use the machine hosting that model.
