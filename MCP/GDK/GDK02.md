# GDK-02: exact local-model preflight and durable MCP working set

GDK-02 adds one **backend-counted prompt budget** for three model profiles: **Inkling 2 Mini**, **Bonsai 8B**, and **Bonsai 27B**. It preserves GDK-01's old-history pruning and the existing MCP-03 code router.

The exact gate is a **local loopback sidecar**. Goose sends the full OpenAI-compatible chat request to the sidecar. Before forwarding that same request to inference, the sidecar calls the *same serving backend's chat preprocessing endpoint*. The recorded prompt token count includes the template, tool schemas, special tokens and other supported request fields, as rendered by that backend. A missing counter, model mismatch, budget overflow or other unsupported request fails before model inference.

| Profile | Weight/model reference | Recommended backend | Preflight API |
| --- | --- | --- | --- |
| `inkling-2-mini` | [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) | vLLM with chat render enabled | `POST /v1/chat/completions/render` -> `token_ids` |
| `bonsai-8b` | [prism-ml/Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf) | Current llama.cpp GGUF server | `POST /v1/chat/completions/input_tokens` -> `input_tokens` |
| `bonsai-27b` | [prism-ml/Bonsai-27B-gguf](https://huggingface.co/prism-ml/Bonsai-27B-gguf) | Current llama.cpp GGUF server | Same backend chat counter |

The Bonsai entries select the **1-bit** GGUF family. Ternary Bonsai and Bonsai 2 checkpoints need separate provenance records; they are not silently substituted. GGUF embeds the model tokenizer and template. Inkling 2 Mini's original mixed-precision checkpoint is around 20GB; a 16GB GPU alone may need CPU offload or a different backend/quantization. The model paths in this document describe inputs, not weights downloaded or performance verified by this change.

## Budget

The default envelope uses **12,288 context tokens**, **2,048 output reserved**, **2,048 safety reserved**, leaving **8,192 maximum prompt tokens** when the backend context is at least 12,288. If the configured backend context is smaller, the gate reduces the input allowance. A chat request must explicitly specify positive `max_tokens` (or `max_completion_tokens`) no larger than 2,048.

The proxy enforces **text-only**, explicitly `stream:false` chat completions and one generation. Multimodal inputs, server-side truncation and streaming fail closed until tested. The gate requires that the operator provide the actual backend context size; it cannot independently attest the live server's `--ctx-size` or `--max-model-len` setting. A rendered count is exact for the backend's reported preprocessing response, while the serving version, model identity and chat template still require a matching configuration.

## Launch a Bonsai 8B or 27B GGUF backend

Use a recent llama.cpp build that implements `/v1/chat/completions/input_tokens`. Example:

```bash
# Model and alias must match the selected profile; repeat with
# Bonsai-27B.gguf and --alias bonsai-27b for the 27B profile.
llama-server --model /models/Bonsai-8B.gguf \
  --alias bonsai-8b --ctx-size 16384 --host 127.0.0.1 --port 8080

python -m MCP.GDK.token_gate serve \
  --profile bonsai-8b --served-model-id bonsai-8b \
  --backend-url http://127.0.0.1:8080 \
  --backend-context-tokens 16384 --port 8094 \
  --receipt-file MCP/runs/gdk02/bonsai-8b-receipts.jsonl
```

For Bonsai 27B use `--profile bonsai-27b --served-model-id bonsai-27b` and the corresponding loaded GGUF model, on a separate serving port if running simultaneously.

## Launch Inkling 2 Mini

Inkling 2 Mini can be served with vLLM if the model's runtime dependencies are supported. The render endpoint is only available on the relevant vLLM versions with scale-out enabled or on a dedicated renderer. Check that it is present before running a model campaign.

```bash
vllm serve thinkingmachines/Inkling-2-Mini --enable-scale-out \
  --served-model-name inkling-2-mini --max-model-len 16384 \
  --host 127.0.0.1 --port 8081

python -m MCP.GDK.token_gate serve \
  --profile inkling-2-mini --served-model-id inkling-2-mini \
  --backend-url http://127.0.0.1:8081 \
  --backend-context-tokens 16384 --port 8095 \
  --receipt-file MCP/runs/gdk02/inkling-receipts.jsonl
```

An Inkling GGUF served through recent llama.cpp can instead select `--counter llama_cpp_input_tokens` on its gate. The counter and inference backend must still be the same server process or share an identical tokenizer/template configuration.

## Run Goose through the gate

Build `MCP/goose_gdk_app` and point it at the **sidecar**, with the same served-model ID:

```bash
cd MCP/goose_gdk_app
cargo build --release --locked
./target/release/goblin-gdk \
  --repo /path/to/code --task "Find the symbol handling retries" \
  --goblin-cwd /path/to/HybridTRMLDT \
  --base-url http://127.0.0.1:8094/v1/ \
  --model bonsai-8b --lean-profile bonsai-8b \
  --max-tokens 2048 --budget 2048 \
  --working-set-root /path/to/private/gdk_state \
  --session-id coding-session-001
```

`--lean-profile` implies `--lean-context`, sets an explicit output cap if absent and verifies the sidecar's `/gdk/health` model/profile binding **before MCP retrieval or inference**. A mismatch stops the run. Use profile `bonsai-27b` and its sidecar URL for 27B; `inkling-2-mini` for Inkling. GDK-01's `--lean-request-byte-limit` remains an additional coarse limit, whose count differs from the backend's token count.

To inspect the saved packet from a restarted process:

```bash
python -m MCP.GDK.working_set load \
  --root /path/to/private/gdk_state \
  --session-id coding-session-001 \
  --catalog-epoch WORKTREE_REVISION_FROM_GOOSE_RECEIPT
```

The capture store writes original MCP-03 **compact packets** into SHA-256 content-addressed blobs, records the router receipt, and maintains an atomic session index. Changed worktree epochs invalidate the active list while preserving old blobs for audit. Stored blobs are private and may contain source code. Files use best-effort 0600/0700 permissions and should be placed in an access-controlled directory.

GDK-02 does not yet persist the complete Goose action/event journal, certify fact claims or automatically restore cross-episode model history. Packet rehydration is a guarded operator interface; further runtime integration can select handles from it once the verifier accepts the revision.

## Verification and baselines

`python -m unittest MCP.GDK.test_token_gate MCP.GDK.test_working_set -v` tests the three profiles, exact backend-route selection, rejection, untouched wire forwarding, model mismatch, worktree change, restart, tampering and private receipts. `ci_gdk02_smoke.py` runs the real Rust adapter against three **scripted loopback** backends, using fake backend token counts explicitly marked as fixtures. Those tests establish integration behavior, not local-model correctness or actual backend parity.

Before accepting efficiency claims run the same frozen MCP-03 / SWE-bench tasks with actual local GGUF/vLLM instances, and compare against Goose default, Code Mode, GDK-01 and the preflighted GDK-02 profile. Keep actual provider usage, token-gate receipt and verifier results separate, counting failures.

Sources: [llama.cpp chat token-counting endpoint](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md); [vLLM chat-render API](https://docs.vllm.ai/en/latest/serving/online_serving/renderer/); [Inkling model card](https://thinkingmachines.ai/model-card/inkling-2-mini/); [Bonsai models](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/Bonsai1_README.md).
