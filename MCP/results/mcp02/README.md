# MCP-02 results

Design and protocol: [`MCP/mcp02/PROTOCOL.md`](../../mcp02/PROTOCOL.md). Configs: [`MCP/mcp02/configs`](../../mcp02/configs). Comparative write-up: [`papers/mcp_data_goblin`](../../../papers/mcp_data_goblin).

| Run | What ran | Episodes |
| --- | --- | ---: |
| [`pilot_gpu/`](pilot_gpu) | Width-128 routers with the exact-match encoder, trained on the RTX 5080. Main grid at 256 and 1024 resources, plus test-time depth, tool-budget and layout sweeps | 8,064 |
| [`pilot_cpu/`](pilot_cpu) | Width-48 routers with the tokens-only encoder trained and run on CPU, plus a match-encoder ablation pair. Same grid and sweeps | see `summary.json` |
| [`llm_pilot/`](llm_pilot) | Qwen2.5-0.5B-Instruct as an end-to-end JSON-action controller through `MCP.local_model.LocalModel`. Stopped after 6 episodes to stay within its VRAM lease | 6 |

Each directory holds:
- `episodes.jsonl.gz`: one lean record per episode;
- `tapes.jsonl.gz`: decision tapes, or raw LLM responses for the LLM run;
- `summary.json`: per-cell success with Wilson intervals, tool calls, bytes, router FLOPs and latency, certified exclusions, failure-inclusive cost per accepted task under four price scenarios, and paired bootstrap comparisons;
- `model_receipts.json`: parameters, recurrent depth, FLOPs per candidate, checkpoint SHA-256 and training receipt;
- `memory/`: inference memory probes;
- `dataset_meta.json`: training and dev group IDs and template groups;
- host, source and run configuration manifests.

Checkpoints stay outside Git; their hashes are in `model_receipts.json`.

## Findings

Every cell has 32 held-out test worlds unless noted.

- **Controls.** Search and eager listing are solved by BM25 and the deterministic rules at both catalog sizes. Layered discovery is not: at 1024 resources the controls accept 1–6 of 32, while the oracle accepts 26–31.
- **Learned routers (match encoder).**
  - On layered discovery at 256 resources they accept 22–32 of 32 with 15–30 tool calls; BM25 and the rules accept 7–22 with 25–37.
  - At 1024 resources, four times the training size, the attention-based routers accept 14–23 on the kind-first index and 4–14 elsewhere.
- **Encoder.** The tokens-only Transformer accepts 16 and 8 of 32 where its match-encoder twin accepts 28 and 25. Shuffled-label training accepts nothing on layered tasks.
- **Architecture.**
  - Transformer (1–4 layers), looped (4 or 12 passes), TRM, TRM+LDT and HRM all reach dev ranking accuracy 0.90–0.92.
  - The looped, TRM and HRM cores match the four-layer Transformer with 0.21–0.41M non-embedding parameters instead of 0.80M.
  - Training with 12 passes or applications adds nothing over 4.
- **Test-time depth.** The looped block improves with more passes than it was trained with: 25, 27, 29 and 30 of 32 at 1, 2, 4 and 8 passes on layered/SCB. TRM, TRM+LDT and HRM are flat across test depths.
- **Tool budget.** Raising it from 24 to 64 lifts learned routers from 26–28 to 30–31 of 32, and they still use only 19–22 calls. The rules reach 19 and BM25 15.
- **Index depth.** At 1024 resources, flat paging and one- or two-facet layouts fail even for the oracle. Three-facet layouts let it finish in 14–19 calls with about half the tool-result bytes.
- **Certification.** No certified exclusion removed a required record in any run.
- **LLM controller.** Qwen2.5-0.5B accepted 0 of 6 search-arm episodes. It used 39.8 calls and 78.5K tokens per episode, while the looped router solves the same cell 32 of 32 times in 5.5 calls.

Replay without torch:

```bash
python -m MCP.mcp02.run replay --run MCP/results/mcp02/pilot_gpu --every 9
```
