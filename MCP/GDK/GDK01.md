# GDK-01: provider-boundary compact view (experimental)

Status: **opt-in Rust runtime integration**, built on the existing `MCP/goose_gdk_app` at pinned `goose-agent 0.1.0-alpha.14`. This change does not mutate or compact the persisted conversation.

## Why a provider wrapper

`InferenceRequestPreparer::prepare` controls the system prompt, tool definitions and extra messages. The GDK `InferenceRunner` independently forwards the current agent-visible conversation to the provider. Therefore a preparer alone cannot prune historical messages. `LeanRequestPreparer` contributes a short system policy plus the existing DataGoblin packet; `LeanProvider` intercepts the actual `Provider::stream` arguments to choose a compact view.

The wrapper receives the system prompt, advertised tools and all final projected messages. It keeps **the entire newest user task and its subsequent assistant/tool messages**, and can include zero or more complete preceding user turns. Prior turns are omitted from the *provider input* while the original conversation remains intact inside the state machine.

## Run

From `MCP/goose_gdk_app`:

```bash
cargo test --locked
cargo build --release --locked

# Existing behavior, unchanged
./target/release/goblin-gdk --task "Locate the parser" --repo /repo \
  --goblin-cwd /path/to/HybridTRMLDT --model local-model

# Opt-in lean view and byte gate
./target/release/goblin-gdk --task "Locate the parser" --repo /repo \
  --goblin-cwd /path/to/HybridTRMLDT --model local-model \
  --lean-context --lean-request-byte-limit 32768 --lean-history-turns 0

# Simulate persisted earlier turns, keeping the newest task mandatory
./target/release/goblin-gdk --task "Locate the parser" --no-packet \
  --history-file conversation_fixture.json \
  --lean-context --lean-request-byte-limit 32768
```

The history fixture is a JSON list of `{"role":"user"|"assistant","text":"..."}` entries. It is used to exercise the runtime path and is distinct from the future durable session store. `--no-packet` remains the ablation against the MCP-03 retrieval controller.

## Request-boundary semantics

1. Collect the newest user kickoff and every later message, including tool responses and current-turn assistant calls.
2. Serialize the candidate provider inputs as deterministic internal Goose JSON, including system text, all advertised schemas and the model name.
3. Fail before sending to the provider if the mandatory newest turn exceeds `--lean-request-byte-limit`. For `--lean-history-turns N`, include earlier complete turns nearest first while under the same bound.
4. Pass the selected messages to the **existing Goose provider**, preserving the state-machine effects and provider implementation.
5. Emit a `lean_request` receipt: `original_messages`, `visible_messages`, `omitted_messages`, `serialized_input_bytes`, `max_serialized_input_bytes`, `blocked`.

The guard measures **internal Goose JSON bytes**. It is deterministic, includes whole schema objects, and provides a useful explicit input-size budget. It does **not** establish the final OpenAI-compatible wire size, provider chat-template token count, or 12K-token compliance. For those, the next stage needs an exact tokenizer/chat-template renderer (or provider endpoint preflight) tied to the actual backend, with output reservation and per-model safety headroom. The received token-usage fields remain separately available.

### Security boundaries

- Tool authorization remains in Goose's operation and dispatch path. No new tool executor is introduced.
- The packet and history are source content that must be treated as untrusted data.
- A failure to fit the mandatory current turn ends the request, with an error receipt, before provider I/O.
- This stage keeps in-memory history intact; restoring it across process restarts remains a host persistence milestone.
- Signed reasoning and tool-request pairing across old turns require provider-specific tests before enabling nonzero `--lean-history-turns` for production use.
- A restored provider session might retain earlier context server-side. The current loopback OpenAI-compatible provider has no such server-held history contract. Stateful provider adoption needs explicit session-resume controls.

## Verification

`python MCP/goose_gdk_app/ci_smoke.py PATH_TO_BINARY` pairs original request injection with a separate history-pruning test. The scripted local HTTP endpoint checks that the baseline sends old messages, lean mode excludes them, the current task remains, the durable session has equal message count, and an undersized limit produces **zero** HTTP calls. Its mock token usage is a labeled byte-based fixture.

`cargo test --locked` exercises Rust view selection and mandatory-current-turn checks. See the [GDK-00 packet contract](CONTRACT.md) for the later verified evidence and true tokenizer-aware request model.

## Next implementation

GDK-02 should add a provider-specific exact preflight adapter, durable evidence-handles and worktree-epoch invalidation. Then evaluate learned MCP-04 selectors against the same frozen SWE-bench fixtures using the existing Goose Code Mode baseline.
