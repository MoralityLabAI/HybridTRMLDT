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
