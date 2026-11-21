# Agent handoff: MCP-04 Bonsai 27B native compaction replay

Continue in `MoralityLabAI/HybridTRMLDT` from **draft PR #18**,
`research/mcp04-stateful-preflight-20261121`. Preserve PR #14 as the
parent feature branch and do not merge either draft merely because the
CPU tests pass. Adjacent PR #17 has the CPU-only retrieval experiments.
This handoff is for the **local 27B inference campaign**, with native
doc-edit/event-recon task grading and a fixed model.

## Mission and safety gate

Test whether MCP-04's typed, stateful working set prevents context
overflows and repeated reads while retaining usable edit state. The
historical raw baseline resolved **5/8 doc-edit** tasks. Its v1 lexical
compaction arm resolved **1/8** while using more total provider prompt
tokens. Historical event-recon tier-1 arms submitted **0/5** reports;
raw and late packet arms overflowed **5/5**, while early compaction
avoided overflow but also submitted zero reports. Keep resolution and
report submission separate.

Use the existing native grader and original task allocation. Keep all
GPU runs on the user's local system after confirming the active RTX 5080
is free. Use the pinned `Ternary-Bonsai-2-27B-PTQ1_0.gguf` and its
server, with model hash, fork commit, prompt template, context per slot
and quantization recorded. The prior server was PrismML llama-server
b10743 (`adfffbe41`), `-c 65536 -np 2` (two 32,768 slots),
DeepSeek reasoning format, and a 4,096-token completion cap. Confirm
the actual active binary and model before replay; preserve the device's
VRAM lease policy and keep total concurrent episodes within its slots.

## CPU and zero-inference preflight

1. Fetch PR #18 into a separate working tree so existing jobs are
   undisturbed. Run
   `python -m pytest -q MCP/test_mcp04.py MCP/test_hubrun.py`,
   `bash -n MCP/hubrun/scripts/mcp04_replay.sh` and
   `python -m MCP.hubrun.preflight_probe --help`.
2. On WSL, while the pinned server is active on the Windows host,
   run from this branch:

   ```bash
   mkdir -p /root/hub-overnight/logs
   python3 -m MCP.hubrun.preflight_probe \
     --upstream http://127.0.0.1:18127 \
     --via-curl /mnt/c/Windows/System32/curl.exe \
     --context-window 32768 --completion-reserve 4096 --safety-margin 1024 \
     --out /root/hub-overnight/logs/mcp04_preflight.json
   ```

   The probe performs **zero inference calls**. Require
   `status=compatible_sample`, `tool_schema_delta > 0`, and
   a prompt within the reserved 27,648-token context. The server's
   `/apply-template` and `/tokenize` must count the *actual* chat
   prompt with the same tool schemas and special tokens. A missing
   endpoint or silently ignored tool schema is an admission failure.
   Investigate the pinned fork and adapt the exact template preflight
   against its real chat-completion prompt builder. Record source
   hashes, failing receipts and a CPU regression. Never replace the
   exact gate with `CHARS_PER_TOKEN` or a guessed token budget.
3. After the first **single** real model completion, compare
   `preflight.prompt_tokens` to `usage.prompt_tokens` in the relay
   record. Any mismatch must be diagnosed before the full campaign;
   examine BOS/special token behavior, reasoning/chat format, tool
   schemas and the fork's actual chat formatting.

## Paired campaign

The prepared runner is `MCP/hubrun/scripts/mcp04_replay.sh`.
It uses new relay ports **18132** and **18133**, keeping the historical
18130/18131 proxies available. Run a one-task smoke per arm, confirm
the provider token-count parity, then perform the matched allocation:

```bash
bash MCP/hubrun/scripts/mcp04_replay.sh smoke
bash MCP/hubrun/scripts/mcp04_replay.sh full
```

The script checks for the pinned WSL verifiers venvs and task arguments:
`/root/hub-overnight/venvs/doc-edit`,
`/root/hub-overnight/venvs/event-recon`, and
`/mnt/c/Users/patri/Documents/Codex/hub-overnight/args/docedit.json`.
Adapt those paths only after locating the original pinned installations
and preserving the verifiers versions. It runs:

- 8 native `dp-learning-rl/doc-edit` tasks per arm with the original
  generated-task seed/args and `vf-eval` grader.
- 5 public tier-1 `briahnloo/event-recon` tasks per arm, using the
  original subprocess tool runtime and up to 40 turns.
- `raw`, `goblin` v1 early packet, v1 late packet, and the new
  `mcp04` typed packet. Match task IDs, seed, sampling
  temperature 0.6, model, context, tool schema, per-call cap,
  and native grader. Tags and per-arm logs are isolated. Report any
  sampled task-ID mismatch instead of presenting an unpaired run
  as paired.

**Important interface limit:** MCP-04 archives full tool results with
task-scoped content handles and exposes
`GET /mcp04/<tag>/v1/working-set/<handle>`. The native agent may
have no tool capable of resolving these URLs. Record when handles
were referenced and whether they were actually retrieved. Do not claim
benefits from handle retrieval solely because the store contains them.
The name `observed-empty` in the packet describes a prior tool result,
not a certificate that a code subtree is irrelevant.

## Receipts and acceptance

For **every assigned task and arm**, save native grader outcome,
report-submission status, peak prompt size, provider usage (prompt and
completion), exact preflight and any count discrepancy, number of
context errors, repeat reads, stale line anchors, tool-call/response
pairing violations, final artifact/patch state, total turns, latency,
and GPU/RAM observations. Include failures in the cost denominator:

```
total_model_tokens_per_resolved_task =
    sum_all_assigned(prompt_tokens + completion_tokens) / resolved_count
```

Return null/undefined when resolved_count is zero. Preserve the
raw-history branch whenever it fits exact preflight. Attribute
new request errors and repairs to the correct arm and retain the
historical v1 compaction code for comparison.

The historical MCP-04 acceptance target is zero context overflows
among five event-recon tier-1 tasks and at least **5/8 doc-edit**
solutions without increasing total provider tokens relative to a
matched raw run. Track `tool_submit_report` separately from
event-recon reward. Exact provider/preflight parity, evidence
integrity, and replay of matched native task IDs are additional
hard gates. A small smoke is diagnostic only.

Push a reproducible report to PR #18 under `MCP/results/mcp04/`:
`manifest.json`, `protocol_probe.json`,
`provider_counts.jsonl`, `paired_metrics.json`,
`resource_measurements.json` and a redacted methodology note.
Keep private storyworld prompts, API keys, personal data and
large raw local paths out of the public commit. Record
task-generation seeds, environment/venv pins and source SHA values.
If a gate fails, preserve its negative receipts and leave PR #18
draft with a clear next change. On completion stop only the server
and proxy processes this campaign launched, release the RTX 5080
lease and verify zero remaining campaign GPU processes.
