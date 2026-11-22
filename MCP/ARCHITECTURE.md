# MCP architecture

The MCP gym studies how a local controller finds evidence and returns a bounded packet to a larger agent. The first release supplies a synthetic environment, deterministic controllers, a local chat-model adapter, and an MCP stdio interface. Neural routing quality requires a later run with a recorded checkpoint.

## State and authority

The design follows the repository's [hybrid contract](../docs/hybrid_contract.md) and [architecture taxonomy](../docs/looped_transformer_trm_ldt_taxonomy.md). A future recurrent controller may maintain private state `h_t`. The environment owns the catalog epoch and evidence receipts. Public candidate state `a_t` provides a checked boundary for proposed refinements.

The current `layered_ldt` arm builds a `CandidateState` over each index query's children and passes its proposed filter to `certify_and_apply_verified` from [the reference kernel](../research_gym/core/hybrid.py). Its environment-owned checker establishes that every excluded record violates registered revision, scope, epoch, or unit constraints. It permits an unchanged page through `require_progress=False`; a removal must still shrink the candidate set. Counters record both verification calls and metadata checks, including scans of descendants.

Persistent lattice state across calls remains a later extension. That extension must bind state and receipts to the catalog epoch, and require strict progress for each durable refinement. A changed epoch starts a new state.

Ranking determines what to inspect next. Pruning requires a check. A low retrieval score alone supplies no certificate that an unread branch is irrelevant. The synthetic environment's exact judge evaluates a submitted answer separately from these checks; the controller cannot inspect its private expected answer.

The initial checked state is symbolic. A future learned LDT can propose refinements through this boundary, with its weights, projection semantics, and verification cost recorded as additional experimental variables.

## Information paths

| Path | Data carried | Consumer |
| --- | --- | --- |
| Public task | The requested result and explicit constraints | Controller |
| Catalog discovery | Tool descriptions, available schemas, and exposed index metadata | Controller |
| Evidence retrieval | Requested records and environment-issued receipts | Controller and evidence store |
| Final packet | Selected facts with the evidence required to verify them | Larger agent or exact judge |
| Evaluation record | Expected answer, correctness, failures, accounting, and provenance | Offline evaluator |

The local controller may spend several calls exploring before it constructs a packet. Its reported input and output tokens have a separate ledger from the packet's byte budget. Raw records can remain in the environment's store and be referenced through scoped handles. Any facts needed by the next agent's decision must survive in the packet or in a resource that agent can explicitly fetch.

Data Goblin can use a generous local generation budget. The study should measure the resulting quality against wall time and hardware use, while separately measuring the packet given to a paid model. More local tokens can be a useful trade when they improve evidence coverage or save downstream work. The current runtime provides bounded episodes for reproducibility; later budget sweeps can vary those limits explicitly.

## Discovery arms

| Arm | Discovery behavior | Purpose |
| --- | --- | --- |
| `flat` | Exposes the full synthetic leaf-tool catalog | Measures eager schema exposure |
| `search` | Retrieves candidates through a search interface | Supplies the ordinary retrieval baseline |
| `layered` | Traverses the catalog's entity-range index | Measures the cost of hierarchical discovery |
| `layered_ldt` | Uses the layered interface with checked metadata filtering | Measures the added behavior and cost of the membrane |

All arms use the same task and underlying records for a paired episode. Tasks specify explicit entity keys and arithmetic operations. Their interfaces control when evidence becomes visible. Results must distinguish different exposure costs from differences in controller quality. The deterministic policies validate the available paths and supply traces for later experiments.

## Controller adapters

The present local-model adapter calls a loopback endpoint compatible with chat completions. It requests one JSON action containing `tool` and `arguments`; the parser enforces this shape. The adapter records the requested model, the server's returned identity, and any supplied checkpoint metadata. A model name alone supplies no evidence about its size or architecture.

The same task/observation/action boundary can support several future controllers:

| Family | Variables to record |
| --- | --- |
| Search or linear reranker | Encoder, index, features, fitted parameters, candidate count |
| Feed-forward Transformer | Parameter count, layers, width, attention scheme, tokenizer |
| Shared-weight looped Transformer | Shared stack, number of passes, state reset, halting policy |
| TRM | Answer/latent state layout, update schedule, recurrent depth, termination rule |
| HRM | Fast/slow state sizes and their update clocks |
| Learned LDT or hybrid | Proposal network, explicit lattice, projection/checker contract, recurrence |

A native tensor adapter can emit tool IDs or candidate scores directly. Hidden-state shapes, cache ownership, and reset IDs belong in its contract. Episode resets must clear private state unless cross-episode memory is itself the registered treatment.

Architecture and discovery arm are independent axes. Parameter count, recurrent passes, retrieved bytes, and local generated tokens measure different resources. The later comparison must record each, together with latency and peak memory. The gym can accept another Transformer family without changing the exact judge.

## GDK integration

A future Rust host can run the router before inference and store the chosen catalog binding in its session. GDK's `ToolProvider` supplies dynamic tool definitions and dispatch. `InferenceRequestPreparer` controls the system prompt, tool list, and additional messages. Tool result projection and application persistence keep bulky evidence outside the conversation sent to the provider.

Tool names and handlers must stay stable from advertisement until all calls produced by that inference finish. The host pins the catalog version for this interval. Execution checks remain in the host-owned dispatch path, with the current evidence and authorization state.

The initial gym's stdio server makes the tasks callable by an MCP client. A native GDK adapter, Goose integration, and MCP Apps UI require their own integration runs. They are later consumers of the same environment.

For an App UI, the provider-visible result should stay compact. Large tables can be fetched through app-only tools. Host behavior determines which result fields reach its model, so inspect the serialized provider request before claiming a token saving from UI metadata.

## Data Goblin milestone

The intended specialization is local retrieval that turns a research request into a small, useful evidence packet. Training examples can record the public task, exposed metadata, chosen actions, retrieved evidence, checked refinements, and a successful packet. Failed routes remain in the corpus with their failure labels.

The first training target is tool and index selection. Later targets can learn when another read improves the packet, how to allocate recurrent depth, and when to widen a search. A final integration milestone asks Data Goblin to serve a frontier agent with the same packet budget and checks whether the frontier agent completes more work per paid token.

That milestone requires real downstream agent runs. Synthetic packet size and correctness alone establish the behavior of this gym.

## Sources

- [GDK operations](https://goose-docs.ai/docs/gdk/concepts/operations/).
- [GDK ToolProvider](https://docs.rs/goose-agent/latest/goose_agent/tool/trait.ToolProvider.html).
- [GDK inference source](https://docs.rs/goose-agent/latest/src/goose_agent/inference.rs.html).
- [Goose Code Mode](https://goose-docs.ai/docs/guides/managing-tools/code-mode/), the later integration baseline for on-demand discovery and local chaining.
- [MCP Apps patterns](https://apps.extensions.modelcontextprotocol.io/api/documents/patterns.html), including app-only tools and chunked retrieval.
