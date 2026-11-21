# goblin-gdk: a Goose GDK app with compact MCP context

`goblin-gdk` is a Rust app built on the Goose Development Kit (GDK), from [`goose-agent`](https://crates.io/crates/goose-agent) and [`goose-providers`](https://crates.io/crates/goose-providers), pinned at `0.1.0-alpha.14`. It assembles a `goose-agent` state machine with two steps:

1. `CompactContextOperation`, a GDK `Operation`, contributes one prompt part named `retrieved_code_context`. Its content is the compact packet returned by the Data Goblin MCP server ([`MCP/mcp03/goblin_server.py`](../mcp03/goblin_server.py)). The server runs a small local router over the repository's code MCP surface (search, layered index, symbol reads, certified exclusions) and returns only a token-budgeted packet.
2. `InferenceRunner` sends the conversation and the collected prompt parts to an OpenAI-compatible endpoint through `goose-providers`' `OpenAiCompatibleProvider`.

The goblin server is spoken to over stdio JSON-RPC (MCP `2025-06-18`). `--no-packet` runs the same machine without the operation's contribution, so a pair of runs isolates what the packet adds to provider input tokens.

## Build and run

```bash
cargo build --release
python -m MCP.mcp02.hf_server --model-dir SNAPSHOT --served-name qwen2.5-0.5b-instruct --port 8093 \
    --device cpu --dtype float32 &   # or any loopback OpenAI-compatible server
./target/release/goblin-gdk --repo /path/to/python/repo --task-file issue.txt \
    --base-url http://127.0.0.1:8093/v1/ --model qwen2.5-0.5b-instruct --budget 2048 \
    --python python3 --goblin-cwd /path/to/HybridTRMLDT
./target/release/goblin-gdk --task-file issue.txt --no-packet --base-url http://127.0.0.1:8093/v1/ --model qwen2.5-0.5b-instruct
```

`MCP/mcp02/hf_server.py` serves a local Transformers checkpoint on loopback with exact token usage. The app prints one JSON receipt:

- the answer;
- `provider_input_tokens` and `provider_output_tokens`;
- the goblin packet accounting: packet tokens, the tokens the router read locally, tool calls, router name and FLOPs;
- retrieval and inference seconds.

Set `MCP03_ROUTER_CHECKPOINT` in the goblin server's environment to swap the BM25 control for a trained looped, TRM, TRM+LDT or HRM router.

## GDK-01 optional lean provider view

The existing `goblin-gdk` command now accepts `--lean-context --lean-request-byte-limit 32768 --lean-history-turns 0`. See [MCP/GDK/GDK01.md](../GDK/GDK01.md).

The feature uses a request preparer for system/packet text and a provider wrapper for conversation selection. It keeps the current user turn together, excludes prior turns from the provider request by default, and checks the **internal serialized input byte** limit before calling the OpenAI-compatible endpoint. `--history-file` accepts a JSON array of simple user/assistant text turns for reproducible comparison. The authoritative session history remains unchanged. This byte counter does not establish exact provider tokens or the model's context-window limit.

`--lean-context` is disabled by default. `--no-packet` still isolates the DataGoblin packet contribution. The scripted endpoint smoke exercises the full provider path, the new Rust unit suite tests pruning and rejection, and the JSON receipt contains `lean_request` with the observed boundary counters.

## GDK-02 exact token gate and durable MCP packet capture

See [GDK02.md](../GDK/GDK02.md) for the three supported profiles and sidecar launch instructions. Pass `--lean-profile inkling-2-mini`, `--lean-profile bonsai-8b` or `--lean-profile bonsai-27b` when the `--base-url` points to the local exact token-gate proxy; the app probes `/gdk/health` for an exact profile/model binding and requires a bounded `max_tokens`. `--working-set-root DIR --session-id ID` saves the full MCP-03 compact packet and metadata under a revision-bound content-addressed handle. The original Goose session stays authoritative for conversation and tool execution.
