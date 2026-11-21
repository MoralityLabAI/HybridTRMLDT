# GDK: Goose Lean Context Engine

This folder stages the Hermes Lite-style **lean context** work for Goose and the Open Intelligence Kit (oi-kit). The immediate goal is to control the **model-visible working set**, while keeping the authoritative session, retrieved artifacts, MCP results, and tool authorization outside that small prompt.

This is **MCP-04 preparation**, built on top of [MCP-01](../README.md), [MCP-02](../mcp02/README.md), and [MCP-03](../mcp03/README.md).

## Existing working baseline

- [`../goose_gdk_app`](../goose_gdk_app/README.md) is an existing Rust `goblin-gdk` binary using `goose-agent` and `goose-providers` `0.1.0-alpha.14`. Its `CompactContextOperation` injects an MCP-03 DataGoblin packet into the system prompt; `--no-packet` supplies an ablation.
- [`../mcp03/goblin_server.py`](../mcp03/goblin_server.py) is the stdio MCP endpoint for `compact_context` and `read_symbol`. Its packet router and receipts stay the starting comparison.
- Goose's newer **oi-kit** documentation uses `oi-kit-agent` and examples targeting `0.1.0-alpha.15`. Treat its request-preparer API as a migration target; preserve the existing alpha.14 app and its CI until the API transition has been tested.

This folder introduces a **stdlib-only, deterministic packet/budget contract**. It does **not** yet change Goose's provider messages or claim that a 12K model window is enforced in the Rust binary.

## Intended request path

```text
Goose/oi-kit durable session, approvals and tool effects
       |
       +--> MCP-03 / MCP-04 router: search, expand, read_symbol
       |                         (future SetTRM + LDT utility heads)
       |
       +--> GDK working-set builder (this folder)
       |       phase + catalog epoch + scoped evidence handles
       |       checked token budget + compact model-facing packet
       |
       +--> request preparer: prune provider-visible history,
       |       pin allowed schemas, preflight the FULL provider request
       |
       +--> inference (Inky Mini, Bonsai, ALEPH-1B or other model)
       |
       +--> Goose-hosted tool authorization -> MCP execution -> receipts
       |
       +--> verifier -> persistent evidence -> next lean packet
```

The model sees references to the small selection of evidence it needs. The durable store retains complete tool outputs, original source hashes, call IDs and their authorization outcomes. A model-visible tool hint is never an execution permit.

## Token budget

Default planning envelope: **12,288 tokens total**. Reserve 2,048 for output and 2,048 for safety, leaving **8,192 maximum input tokens**. The input allowance must include system instructions, task, message history, all advertised tool schemas, message framing and the compact packet.

`assemble_packet` only enforces the **packet's share** of that allowance. Its caller must supply `external_input_tokens`, covering other prompt components; the future oi-kit request preparer must independently preflight the exact final serialized model request with the chosen provider tokenizer. Missing tokenizer metadata must be reported as an estimate or treated as a hard-budget failure.

## Developer quick start

From the repository root (Python 3.10+; standard library only):

```bash
python -m unittest MCP.GDK.test_lean_context -v
python -m MCP.GDK.lean_context
```

The smoke prints an example compact packet and a receipt using an **explicitly labeled byte estimator**. The unit tests use a deterministic synthetic counter to verify packing invariants; neither result claims provider-token accounting.

## Files

- [`DESIGN.md`](DESIGN.md): lifecycle, integration points, trust and failure boundaries, phased work.
- [`CONTRACT.md`](CONTRACT.md): wire packet and state-transition contract.
- [`lean_context.py`](lean_context.py): pure budgeted packet builder and phase transition guard.
- [`test_lean_context.py`](test_lean_context.py): stdlib regression tests.
- [`config.example.json`](config.example.json): an illustrative opt-in policy for the future Goose adapter.

## Proposed milestones

1. **GDK-00 (this change)**: freeze packet and phase contracts; baseline tested builder; no changes to Goose runtime.
2. **GDK-01**: implement opt-in oi-kit request preparer. Construct provider-visible conversation from an immutable snapshot, account for *all* tool schemas/history, and preflight the actual request. Keep alpha.14 production baseline intact.
3. **GDK-02**: persist typed working-set state and source-bound evidence handles across compaction, cancellation, and restart. Invalidate on catalog epoch or worktree changes. Preserve complete tool transcripts outside prompt view.
4. **GDK-03**: connect MCP-04 `expand` / `read_symbol` heads and SetTRM-LDT learned routing behind the same deterministic packet builder.
5. **GDK-04**: compare default Goose, Goose Code Mode, existing goblin-gdk, deterministic Goose Lean, learned Goose Lean, and the Hermes Lite reference at matching task/checkpoint/tokenizer budgets.

Promotion gates: verified outcome parity, no unauthorized tool execution, no stale evidence or hidden-judge leakage, exact provider budget compliance, and paired efficiency gains on held-out tasks. Keep failure and abstention counts in the denominator.

## Sources

- [Open Intelligence Kit](https://goose-docs.ai/docs/oi-kit/)
- [oi-kit operations](https://goose-docs.ai/docs/oi-kit/concepts/operations/)
- [oi-kit state machine](https://goose-docs.ai/docs/oi-kit/concepts/state-machine/)
- [oi-kit Hello World](https://goose-docs.ai/docs/oi-kit/hello-world/)
