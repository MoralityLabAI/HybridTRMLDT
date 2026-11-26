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

## Multi-turn repair sessions

Pass `--max-agent-iterations N` to activate the Pocket Goblin session. Set
`--context-window 16384`, `--memory-mode raw|packet|typed`, and optionally
`--controller-checkpoint PATH`, `--recurrence-depth N`, `--controller-reset`,
`--session-file PATH`, and `--verifier PATH`. The verifier manifest is created by
the experiment operator and controls the editable source paths and native
verification command. Use `--json-actions` with the Inkling variable-loop
checkpoint for typed parsing of textual or marker-encoded action proposals.

See the [MCP-05 gym and experiment guide](../mcp05/README.md) for the pinned
runtime, training and matched evaluation commands.

## Titan: Pocket Goblin as a TitanAgent squad member

[TitanAgent](https://wchung.tw/blog/20260511-openab-en) is Blake Hung's agent squad. Each member is an Agent Client Protocol ([ACP](https://agentclientprotocol.com)) command bridged into Discord by [OpenAB](https://github.com/openabdev/openab). Tasks are routed to members by role, each member works in its own `.agents/<name>/` area, and every operation is recorded in a shared audit log. `goblin-gdk --acp` makes Pocket Goblin one of those commands. By default it fits the read-only scout role: it answers code questions over the shared source tree from a token-budgeted Data Goblin packet, using a local model. With the repair-session flags it is an edit-capable member instead (see [Repair sessions over ACP](#repair-sessions-over-acp)).

```bash
./target/release/goblin-gdk --acp --agent-name goblin --repo /workspace/src \
    --base-url http://127.0.0.1:8093/v1/ --model qwen2.5-0.5b-instruct --budget 2048 \
    --python python3 --goblin-cwd /path/to/HybridTRMLDT \
    --history-turns 6 --audit-log /workspace/shared/audit-log/goblin.jsonl
```

[`titan/openab.toml`](titan/openab.toml) is the matching OpenAB config. Run it with `openab run -c titan/openab.toml`. Pocket Goblin speaks ACP protocol version 1 as newline-delimited JSON-RPC 2.0 on stdio, the frames OpenAB's `AcpConnection` sends:

| ACP frame | Pocket Goblin |
|---|---|
| `initialize` | `protocolVersion` 1, `agentInfo.name` from `--agent-name`, `loadSession: false`, no auth methods |
| `session/new` `{cwd, mcpServers}` | a session over `--repo`, or over `cwd` if no repo is pinned (it must be an absolute directory). Client MCP servers are not attached; context comes from the goblin server. |
| `session/prompt` | one GDK turn over the session's conversation. The packet is a `tool_call` of kind `search`, then a `tool_call_update` with the packet accounting as `rawOutput`. The answer is one `agent_message_chunk`. The result holds `stopReason: end_turn`, provider `usage`, and the turn receipt under `_meta.pocketGoblin`. |
| `session/cancel` | stops retrieval or inference and answers the prompt with `stopReason: cancelled`. The cancelled turn is not kept in the history. |
| `session/close`, `session/end` | drops the session |
| anything else | JSON-RPC error -32601 |

- **History:** a session keeps its last `--history-turns` completed turns (default 6). Each prompt gets a fresh packet for its own text.
- **Errors:** a failure carries a stable `error.data.category`: `invalid_params`, `session_not_found`, `retrieval_error`, `provider_error`, `repair_error` or `audit_log_error`.
- **Audit log:** `--audit-log` appends one JSON line per retrieval (`op: retrieve`, `path`, packet tokens, router) and per inference (`op: infer`, `model`, tokens). Each line carries `ts`, `agent`, `session` and `status` (`ok`, `failed` or `cancelled`). If the log cannot be written, the turn fails.
- **Process model:** one goblin server per agent process, shared by its sessions. The server is restarted after a failed or cancelled retrieval. Stdout carries protocol frames only.

### Repair sessions over ACP

Add the repair-session flags to `--acp` and each `session/prompt` runs one multi-turn repair session over the session repository: the MCP-05 workspace service, the exact-preflight relay and the edit/verify loop described above. The prompt text is the task.

```bash
./target/release/goblin-gdk --acp --agent-name fixer --repo /workspace/src \
    --base-url http://127.0.0.1:8093/v1/ --model qwen2.5-0.5b-instruct --budget 2048 \
    --python python3 --goblin-cwd /path/to/HybridTRMLDT \
    --max-agent-iterations 8 --max-tool-calls 24 --memory-mode typed --verifier /workspace/verifier.json \
    --audit-log /workspace/shared/audit-log/fixer.jsonl
```

| Repair step | ACP |
|---|---|
| each tool call | a `tool_call` and, when it ends, a `tool_call_update` (`completed` or `failed`) with the tool result as `rawOutput`. Kinds: `context_query` and `context_expand` are `search`; `context_read`, `context_state` and `inspect_edit` are `read`; `apply_edit` and `create_file` are `edit`; `run_verifier` is `execute`. Calls made by a controller checkpoint or by the host's automatic verify are reported the same way. |
| `apply_edit`, `create_file` | the `tool_call` starts `pending` with the file in `locations` and a `diff`. Before the workspace is touched, Pocket Goblin sends `session/request_permission` with `allow_once` and `reject_once` options. There is no `allow_always`, so every write asks. A rejected or cancelled answer becomes a tool error that the model sees, and the file stays unchanged. OpenAB as published answers every permission request itself: it picks the most permissive allow option and logs the tool call's title. A rejection needs a client that answers `reject_once` or `cancelled`. |
| end of the session | one `agent_message_chunk` with the answer, then `stopReason: end_turn`, provider `usage`, and the session receipt (the same JSON the command line prints) under `_meta.pocketGoblin`. |
| `session/cancel` | stops the session, including a tool call or permission request in flight, closes that tool call as `failed`, and answers `stopReason: cancelled`. |

- **Audit log:** an `op: write` line per write with the edited `path`, `tool`, `origin` and `status` (`ok` with the before and after digests, `rejected`, `failed`, or `cancelled`); an `op: verify` line per verifier run with `passed` and `exit_code`; and one `op: repair` line per prompt with the model, `verified`, tokens and call counts.
- **History:** prompts do not share history. The session's own conversation is multi-turn, up to `--max-agent-iterations` inference rounds.
- **Clean worktree:** the MCP-05 service needs a clean git worktree at the repository root, so it can attribute the patch exactly. After a repair has edited the tree, the next repair prompt fails with `repair_error` until the patch is committed or discarded.
- **Errors:** `repair_error` covers a session that could not start or whose provider or relay failed. `--session-file` is refused with `--acp`, because every prompt would overwrite the same checkpoint.

[`ci_acp_smoke.py`](ci_acp_smoke.py) checks the whole contract end to end. It sends OpenAB's frames to the binary and uses a scripted OpenAI-compatible endpoint. It runs four turns on this repository, cancelling one of them during inference. Then a second agent with the repair flags runs a repair turn on the MCP-05 gym fixture. The client rejects the first edit and approves the second, as OpenAB would, and the verifier passes. A follow-up prompt on the edited worktree is refused.

```bash
python MCP/goose_gdk_app/ci_acp_smoke.py target/release/goblin-gdk
```
