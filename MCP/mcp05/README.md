# Pocket Goblin

Pocket Goblin extends the existing Goose GDK executable with a re-entrant repair session. Goose applies typed effects after authorizing model or controller proposals. A workspace-bound Data Goblin MCP service executes lookups, resolves source handles, applies exact replacements, and runs the configured verifier. The same live working-set engine builds training and inference slates.

The original one-shot CLI remains available. Pass the new iteration flag to activate session mode. The application's turn checkpoint records the original task, paired tool calls and receipts, admitted evidence, budgets, edit state, and unresolved verifier failures. The checkpoint is an audit artifact for the invocation. The trained controller's recurrent fast state and candidate embedding cache live in CPU memory for that session.

## Build

    cargo +1.99.0 build --locked --manifest-path MCP/goose_gdk_app/Cargo.toml
    python -m unittest -v MCP.mcp05.test_mcp05 MCP.mcp05.test_federation

Set GOBLIN_GDK_BINARY to the built binary to include the deterministic provider replay in the unittest run. The two replays exercise native tool calls and Inkling's marker-encoded actions and verify a real test fixture after the same edit. The CPU workflow pins the Rust toolchain, the cargo lockfile, and the CPU PyTorch version.

## Generate and train

Run these from the HybridTRMLDT root with PyTorch CPU installed.

    python -m MCP.mcp05.gym --out work/mcp05-cases --seed 523 --train 18 --test 6
    python -m MCP.mcp05.train --stage work/mcp05-cases --out work/mcp05-controller --threads 4 --depth 3 --epochs 6

Every platformer repository starts from a frozen buggy commit. Generation executes the regression on the buggy tree and on the reference patch. Held-out roots use renamed symbols, a different source layout, unseen defect families, and multi-file repairs. The operator's sealed manifest holds gold patches and localization targets. Evaluation gives Goose a fresh copy of only the assigned buggy tree, the public task and the verifier command.

All fitted heads consume the same verifier-grounded replay, including deliberately failing edits and their native results. The shared and dual models reuse the original RU02 looped encoder and utility heads on matched MCP-05 slates. The recurrent variant runs permutation-equivariant set attention and shared latent refinement, with optional resets across tool calls. A seeded target permutation controls for the relevance of utility labels. Training checks peak process RSS against 3 GiB and limits PyTorch to four threads by default.

## Run matched evaluations

### Thinking Machines Lab variable-loop GGUF

Start the existing `TML-d24-varloop-research-IQ4_NL.gguf` with llama.cpp b12602 and Vulkan. The RTX 5080 tests use model revision `618c48f1e3c54094333e3e652a92b1d99c52e34d`, a fixed 16,384-token slot, one loop, and greedy decoding. The separate three-loop compatibility probe is excluded from the matched comparison.

    llama-server --model PATH_TO_EXISTING_GGUF --host 127.0.0.1 --port 8097 --alias inky-mini-varloop --ctx-size 16384 --parallel 1 --threads 4 --threads-batch 4 -b 256 -ub 64 -ngl 99 --n-cpu-moe 17 --fit off --no-mmproj --no-context-shift --cache-ram 0 --reasoning off --reasoning-budget 0 --no-reasoning-preserve --seed 523 --temp 0 --override-kv inkling2_loop.loop.count=int:1

The shared-GPU profile keeps the first 17 layers' expert weights in system RAM and measured about 7.8 GiB on the GPU with the full slot allocated. When system RAM is the tighter constraint, `--n-cpu-moe 16` moves one expert layer back to the GPU. That configuration generated successfully at 16K and used 8.3 GiB of dedicated GPU memory in a shared sample. The earlier 11-layer CPU profile used about 11 GiB. The Windows pilot set `GGML_VK_VISIBLE_DEVICES=1` after verifying the device inventory. Use measured host and device memory when choosing a profile.

    python -m MCP.mcp05.evaluate --binary MCP/goose_gdk_app/target/debug/goblin-gdk --stage work/mcp05-cases --checkpoints work/mcp05-controller --out work/mcp05-inky --json-actions --context 16384 --completion 512 --arms A B C D E shuffled

For the external cases, replace `--stage` with `--cases PATH_TO_EXTERNAL_CASES_JSON` and use `--arms D A`. Use a separate output directory when comparing `--deduplicate-reads` so each result points to its generation settings.

### Earlier native CUDA comparison

For native CUDA, start the LookAtMyQuant provider from its `runtime_cuda` directory. The tested provider commit is `89439a2d63d1bdef0a19fba0ce5c2eeb5bd9488a`. Use the existing 5.4 GB model pack and DSpark drafter with a CUDA-enabled Python environment.

    python -m inkling_cuda serve PATH_TO_CACHED_INKY_5_4 --port 8101 --context 16384 --no-thinking --drafter PATH_TO_CACHED_DRAFTER --draft-block-size 2

From the HybridTRMLDT root, run the same cases across the raw, packet, learned, recurrent, reset, and shuffled arms. Set `--binary` to the executable built above. The model ID is the cached pack's directory name. Every arm uses the native tool protocol and the same inference limits.

    python -m MCP.mcp05.evaluate --binary PATH_TO_GOBLIN_GDK --stage work/mcp05-cases --checkpoints work/mcp05-controller --out work/mcp05-inky-cuda --base-url http://127.0.0.1:8101/v1/ --model inky-5.4 --context 16384 --completion 512 --iterations 10 --calls 18 --packet-budget 1024 --request-timeout 480 --episode-timeout 1200 --arms A B C D E shuffled

### Benchmark arms and grading

The A arm uses raw history and the existing no-packet control. B uses BM25 and the current packet history. C uses the shared utility model. D keeps the SetTRM latent state, E resets it at every step, and the shuffled arm uses permuted utilities. All arms share the same solver, assignments, iteration limit, context allowance and tool-call cap. B spends some of those calls on its initial packet.

Use MCP.mcp05.benchmarks to stage a native GPTStoryworld building exercise, SWE-bench Lite cases from the local dataset and bare repos, and evidence-grounded research planning cases. The Storyworld case invokes the repository's own validator and checks reachability of an added outcome. SWE-bench Lite uses targeted host regressions for interactive correction. Export the resulting final.patch as a prediction for the official SWE-bench harness when reporting benchmark resolution. Research planning scores are rubric checks on the written design.

For the two Flask assignments on Windows, run `grade_swe_wsl.ps1` after the A and D prediction files are complete. Set `-Campaign` to the evaluation directory and `-DatasetDirectory` to the directory containing the pinned `swe-bench-lite-test.parquet` and DuckDB under `reader`. The launcher projects the selected rows, checks for the corresponding local container images, and runs the WSL evaluator with one worker. For a single-case pilot, set `-InstanceIds` and `-Arms` to the pilot assignments and choose a separate `-OutputName`. Use `-ProbeOnly` to check the dataset and container availability before predictions arrive. [Official SWE-bench harness](https://github.com/SWE-bench/SWE-bench/tree/main/swebench/harness)

## Cross-repository Data Goblin lookup

Build an immutable index from committed source blobs. The input JSON contains a `repositories`
array. Each object has an `id`, the local checkout `path`, an optional display `label`, and an
optional pinned `commit` (defaults to that checkout's HEAD). The builder never indexes uncommitted
work, symlinks, Git LFS placeholders, or paths matching its protected-data exclusions.

    python -m MCP.mcp05.federation --spec repositories.json --out work/repo-indexes
    python -m MCP.mcp05.index_router --indexes work/repo-indexes --out work/index-router

The first command creates one on-disk FTS index per repository plus a compact vocabulary catalog.
The second fits a three-pass CPU SetTRM with two threads and a 3 GiB process RSS cap. It reports
lexical, recurrent, and reset recall on disjoint source-path probes. This is a routing proxy.
Use a live rollout to test whether the retrieved evidence actually helps a solver.

For a normal Goose multi-turn session, set `MCP05_FEDERATION` to the built index directory and,
optionally, `MCP05_INDEX_ROUTER` to its `router.pt`. Both can also be supplied directly to
`python -m MCP.mcp05.service --repo WORKSPACE` as `--federation` and `--index-router`.
The `--repo` workspace still controls edits and verification. Indexed sources are read-only.

Use `context.query` for a cross-repository search. The receipt records queried, deferred,
and independently excluded indexes. An exclusion is certified for the query's exact OR-term
postings at the catalog revision. Semantic relevance remains open. Use `@index-id terms` to
search a deferred repository, `@workspace terms` to find editable sources, and
`context.expand(node_id="indexes")` to inspect the catalog. Pass returned URIs to `context.read`.

For a long-context research pass, `python -m MCP.mcp05.evidence_packet` builds a packet
from selected repository lanes and measures it with the loopback model tokenizer. Pass
the resulting text with `--evidence-file` in a multi-turn Goose session and keep
`--task-file` short. The relay measures the complete rendered prompt against the
actual context and output reserve. Set `MCP05_WIRE_LOG` to a fresh local file when the
operator wants the exact forwarded messages and the provider's response. Keep these
transcripts with the local run records.

Use `--answer-only` for a session whose repository evidence has already been assembled. The app requests a final answer without advertising or dispatching tools. For cold context comparisons, `MCP05_FRESH_PROMPT=1` asks the llama.cpp relay to disable prefix reuse. Verify `cache_n` and `prompt_n` in the response timing record.

## Execution and accounting

The session sends the actual Goose wire request through a loopback preflight relay. It applies the upstream chat template, tokenizes with special tokens recognized, and checks for tool-schema serialization when schemas are present. It forwards raw messages if they fit. Packet and typed modes may condense closed history and remeasure it. Each provider completion records preflight parity. A mismatch prevents further completions in that session.

Use `--deduplicate-reads` with typed memory to keep the full text of the newest copy of an identical source read. Older copies retain their tool IDs, source handles, and other metadata. The relay rechecks the provider token count and only uses the shorter request if it fits. The evaluator accepts the same flag and applies it to typed arms. The raw baseline keeps its original wire history.

The controller packet retains the newest complete source receipts that fit its budget. When a newly selected source fits, earlier receipts can leave the packet while remaining in the session audit trail. A proposal to reuse a handle already in the packet avoids another source read.

For a JSON object, the working set indexes multiline array and object members at the top two levels. Use `context.query` to find a section or expand the JSON file's candidate node to list its sections. A URI such as `world/diplomacy_min.json::/rules/outcomes` reads the original lines with the normal digest-bound source handle. Reading the file URI still supports line-based pagination.

GDK controls which tool names, action budgets and workspace edits are authorized. A replacement requires a current digest-bound source handle and an exact unique match within the read span and file. Host-certified LDT exclusions carry the current revision and are reset on source changes. A native repair receipt is tied to the final patch digest. The harness derives the final patch from the worktree and independently reruns the verifier, including after an episode timeout. The patch must stay within the host-approved paths and retain its digest through verification. Verifier manifests are operator-supplied. The Python verifier guard limits routine Python filesystem mutations and blocks subprocess and socket creation. Use an OS sandbox when executing untrusted third-party code or grading general SWE-bench tasks.

In read-only sessions, an exact repeated search after a source read triggers one final answer turn using the collected evidence. Retrying the same failed expansion also triggers this handoff once the matching source has been read. Three successive successful solver reads of the same source handle and revision trigger the same handoff. Goose suppresses further lookup dispatch for that final turn.

Per-episode receipts record provider prompt and completion tokens, retrieval postings, controller time, repeated reads, invalidated source handles, verifier calls, patch digests and wall time. On a timeout the evaluator recovers completed source reads and verifier calls from the session checkpoint. The summary indicates how many episodes supplied final runtime metrics. Token cost per verified repair includes tokens from failed episodes. A repair result and a localization or rubric result have separate denominators.

Use `--goblin-cwd` to evaluate Goose with a specific checkout of the MCP service. This lets the host-side evaluator change while a matched run retains its chosen source lookup behavior. The alternate checkout path is recorded in the evaluation plan.

For a shared GPU, pass the same local path as `--yield-file` to the guarded CUDA provider and `--gpu-yield-file` to the evaluator. When the scheduler creates that file, the provider stops at a generation boundary and the evaluator archives any interrupted episode outside the scored campaign.

Start a campaign with `--provider-prefix-id` set to the fixed model, provider revision, and generation settings to enable resumption. On a restart, the evaluator recreates the worktree and reruns its tools. The relay retokenizes each request and reuses a recorded completion only when both the complete outbound request and rendered prompt match at the same turn. It requests a fresh completion when they differ. Each arm has its own provider response journal. Replayed turns retain their original token counts and inference times, with replay separately identified in the receipts.

If the device needs to release CUDA allocations between turns, the recycling build of the provider accepts `--recycle-completions`. Goose still uses one stable loopback URL and 16K context. Each request reloads the model, so compare those request times separately from a resident model. The provider log reports load, inference, and release time for each worker.
