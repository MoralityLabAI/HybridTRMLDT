# GDK lean context: architecture and rollout plan

## Core idea

Treat an agent session as two objects with different lifetimes:

- **Durable execution history:** tool requests and results, approval decisions, verifier receipts, source snapshots, inference usage, and user messages. The host persists this append-only or reconstructible history.
- **Ephemeral inference view:** a generated packet for one call, containing phase, task, selected evidence handles, optional verified summaries, and a catalog of currently offered tools. The view can be rebuilt after restart and should never become the sole copy of a tool result.

The learned router chooses retrieval actions and which evidence to place in the inference view. The host retains authority over execution and fact verification.

## Phase lifecycle

`INSPECT -> ACT -> VERIFY -> FINAL`. An unsuccessful verification goes back to `INSPECT` for a new evidence route. The transition guard in `lean_context.py` requires a tool-selection event before ACT, a receipt before VERIFY and trusted verifier references before FINAL. These flags are **caller-supplied assertions**; the host must authenticate them against its durable receipt store.

- `INSPECT`: read a small task sketch, current catalog epoch, latest verified evidence and tool discovery results. Decide whether to search, expand an index node, or read a symbol.
- `ACT`: execute the selected action through Goose's normal approval and tool-dispatch path. Keep complete tool output in the durable store.
- `VERIFY`: check source digest, catalog epoch, relevance, required coverage and terminal claim against host-owned evidence. Reject stale or unsupported conclusions.
- `FINAL`: emit the verified answer or an explicit abstention with evidence handles.

Operations can contribute prompt content through `prompt_parts` or `moim_parts`. The request preparer has the wider responsibility: select which conversation messages and tool definitions reach the model. A prompt contribution alone cannot enforce a model context bound.

## Integration boundaries

| Boundary | Proposed implementation | Current status |
| --- | --- | --- |
| MCP retrieval | Existing MCP-03 `goblin_server.py` and `MCP/mcp03/agent.py` | Executable baseline |
| Packet assembly | `MCP/GDK/lean_context.py` | Deterministic scaffold |
| Durable evidence store | Host-owned handles for full results, revisions and SHA-256 | Design; not implemented here |
| oi-kit operation | `LeanContextOperation` contributes phase and packet metadata | Planned; avoid premature alpha.14/15 dependency mix |
| Request preparer | `LeanRequestPreparer` selects bounded history, tool schemas and packet; preflights complete request | Planned |
| Approval and dispatch | Existing Goose host path, including MCP and Code Mode | Must remain authoritative |
| Learned selector | SetTRM/LDT utility head for `expand` and `read_symbol` | Later MCP-04 integration |
| Evaluator | Existing MCP-01/02/03 exact judges and frozen SWE-bench split | Reuse; new Goose controls needed |

Do not insert a second tool loop around Goose. oi-kit reloads state after applied effects. The state machine must include approval and tool execution ahead of inference, and must yield on an authorization or budget gate. A model-proposed final result is not a verifier receipt.

## Typed working set (target)

The future durable record should carry these fields:

```json
{
  "version": "gdk.working_set.v0",
  "session_id": "session-opaque-id",
  "catalog_epoch": "repo-tree-or-worktree-revision",
  "phase": "INSPECT",
  "active_evidence_refs": ["sha256:.../symbol:..."],
  "verified_refs": [],
  "pending_tool_call_ids": [],
  "tool_catalog_snapshot_id": "tools-revision",
  "packet_digest": "sha256-of-last-inference-view"
}
```

For every evidence handle, store separately the raw output or resource URI, acquisition method, source digest, repository revision, provenance, time, verifier verdict and scope. Rebuild packets from immutable evidence; reject cross-epoch reuse unless the source is revalidated.

## Token accounting

Let `W` be the model window, `O` the output reservation, `S` the safety reservation, and `F` the exact number of tokens from all other request components. The packet maximum is:

```text
B_packet = min(packet_cap, W - O - S - F)
```

The Python builder counts a canonical JSON packet with an injected counter. The Goosе/oi-kit adapter must count the final request as actually tokenized by that provider, including role framing, system strings, tool-schema encodings and selected conversation messages. The provider's observed usage should be recorded separately. An estimate is suitable for exploratory routing only; a hard bound requires a proven tokenizer/accounting interface or a conservative provider-enforced preflight.

Do not truncate UTF-8, JSON or cited evidence mid-record. The builder skips an oversized record and keeps trying smaller ones. It never replaces a missing citation with a generated one. If the fixed task and phase already exceed budget, fail before inference.

## Trust model and failure tests

1. **Authorization:** allowlisted tool IDs in prompt are advisory. Every call still goes through host approval and scope checks. Dynamic catalogs are pinned from request construction until resulting tool calls finish.
2. **Source binding:** a score chooses reading order, never proves a fact. Verifier decisions bind to immutable source digests and catalog epoch; unknown metadata stays eligible until examined.
3. **Failure completeness:** rejected calls, empty model outputs, invalid JSON, tool timeouts, cancellations, compaction errors and budget exhaustion produce receipts. Replayed effects need idempotency keys where actions can have side effects.
4. **Context exposure:** exclude private judge targets, raw passwords, authorization secrets and unselected tool results from inference views. The gym's hidden labels remain outside every controller input.
5. **Comparability:** model, tokenizer, decoder settings, task set, catalog version, GPU, warmup, context cap and tool permissions stay matched across arms.

## Evaluation design

Start with `MCP/mcp03` and the current `goblin-gdk` smoke. Add a scripted provider that stores **the complete request** so a test can assert what Goose transmitted. Compare paired runs on the same frozen SWE-bench instance set and repository revision. Then run a small live-model pilot before training a recurrent selector.

Measure joint answer/evidence success, file recall, gold-symbol recall, number of reads, tool schema exposure, request tokens, local router tokens, verifier calls, latency p50/p95, compaction recovery and peak working-set memory. Count failed attempts and restarts. A smaller packet is an optimization only when outcome requirements continue to pass.

Reference arms: vanilla Goose, Code Mode, existing `goblin-gdk` packet injection, deterministic lean context, learned SetTRM/LDT lean context, and Hermes Lite under the same input/output accounting. All model-routing variants use the same strict permission membrane.

## Implementation order

- Wire real tokenizer-aware request preflight into the adapter; add a safe rollback to vanilla behavior when the opt-in flag is disabled.
- Persist the evidence index and packet digest. Use read-only epoch invalidation to handle edits and resumed sessions.
- Add separately trainable `expand` and `read_symbol` heads, using success-per-token and symbol recall at fixed budgets as primary decisions.
- Only then connect DataGoblin's recurrent repair/hill-climbing controller and longer training rollouts.

Existing Goose GDK pins `goose-agent 0.1.0-alpha.14`. Official oi-kit examples now use `oi-kit-agent 0.1.0-alpha.15`. Treat migration as a separately verified dependency change, not a find-and-replace.
