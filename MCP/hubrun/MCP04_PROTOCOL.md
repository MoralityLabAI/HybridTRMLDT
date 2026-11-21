# MCP-04: exact preflight and typed working-set memory (CPU implementation)

**Status:** experimental `mcp04` relay arm under PR #18, separate from
historical `raw` and `goblin` arms. Protocol tests are synthetic; no
pinned Bonsai 27B replay or native task score is yet asserted.

## Interface

The relay accepts `/mcp04/<tag>/v1/chat/completions`. It forwards `raw`
requests whenever the *actual applied template* fits within the reserved
window. On overflow it produces a typed working-set packet from completed tool
turns and remeasures the full request after packing.

The upstream must support these exact response contracts. Current llama.cpp
server documentation advertises `messages` for `/apply-template` and a
`prompt` string in its response. When a tool schema is present, this arm
issues a no-tools control to verify that the applied template includes the
schema's contribution. A server that silently ignores `tools` fails the
preflight rather than giving a false token-safety receipt:

https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md



- `POST /apply-template` with `{messages, tools, add_generation_prompt: true}`
  returns `{"prompt": "<entire applied template>"}`.
- `POST /tokenize` with `{content: <entire applied template>, add_special: false}`
  returns `{"tokens": [integer token IDs]}`.

The model's actual chat-template implementation must use the **same** tool
schema serialization and beginning/end-of-message tokens as the preflight
endpoint. Otherwise the raw token count can diverge. On a provider reply,
the relay records `preflight_vs_provider_delta`, and a nonzero discrepancy is
a provider-specific contract failure to audit before deployment. If either
endpoint is missing, malformed or unsupported, the new arm returns HTTP 422
and does not send the chat-completion call.

The default context gate is `32768 - 4096 - 1024 = 27648` prompt tokens.
Use `--context-window`, `--completion-reserve`, `--safety-margin` to
match the running server. Requests receive a single extra preflight-and-compact
attempt when the provider reports a context error despite the preflight fit.
Raw and historical v1 Goblin remain unchanged.

## Typed working set

The packet preserves the first system/user messages, later user constraints,
the chronological list of completed edit tool calls with available line
anchors and digests, result handles, observed-empty queries, and complete
recent assistant/tool pairs. Previously read material receives content-addressed
handles (`mcp04:<sha256>`). A local retrieval endpoint can return it:

`GET /mcp04/<tag>/v1/working-set/mcp04:<sha256>`

The handle store is scoped to the relay task tag and resides in process
memory. Requests using another tag receive a 404 for that handle.
Restart durability, authorization of tags and cross-machine handle retrieval
remain future engineering work. Keep the relay on loopback until stronger
authentication is introduced. "Observed empty" is a tool output fact; it does not
constitute an LDT certificate for ruling out entire subtrees. Compaction
refuses unpaired tool calls or missing essential edit anchors/constraints
at the imposed packet budget. No approximate count grants admission to
model inference.

## Regression and replay

Run stdlib contracts on a CPU checkout:

```bash
python -m pytest -q MCP/test_mcp04.py MCP/test_hubrun.py
```

A native local replay needs a pinned llama-server fork confirmed to expose
the endpoints above. Compare raw, v1-early, v1-late and MCP-04 with the same
model, sampling, task allocations and tool schemas. Archive both preflight
counts and provider `usage.prompt_tokens`. The gates are:

- Zero context overflow on the five event-recon tier-1 tasks.
- Match or exceed raw `5/8` doc-edit solutions without increasing total
  provider tokens, while preserving edit-site correctness and counting
  repeat reads.
- Preserve tool-call/result pairings and complete all required finalization
  actions.
- Distinguish actual task solves from report submissions and include failed
  assigned tasks in total cost.

Native pinned-server verification and the above replay are outstanding.
MCP-04 remains a draft experiment pending those results.

Related: https://github.com/MoralityLabAI/HybridTRMLDT/issues/15
