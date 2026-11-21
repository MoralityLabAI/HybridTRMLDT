# MCP-03 results: compact packets on SWE-bench Lite

Configuration: [`MCP/mcp03/configs/swebench_lite.json`](../../mcp03/configs/swebench_lite.json). Design and metrics: [`MCP/mcp03/README.md`](../../mcp03/README.md).

| Directory | Contents |
| --- | --- |
| [`swebench_lite/`](swebench_lite) | Held-out evaluation, 109 SWE-bench Lite tasks from 10 repositories with no training data. Files: `episodes.jsonl.gz`, `tapes.jsonl.gz`, `baselines.jsonl.gz` (BM25 whole-file and path-map baselines, gold context sizes), `summary.json`, `model_receipts.json`, `memory/`, `dataset_meta.json` and host/source manifests. |
| [`swebench_e2e/`](swebench_e2e) | End-to-end SWE-bench Lite resolution on 26 tasks (requests, flask, pytest). Qwen3-4B-Instruct-2507 writes search/replace edits through goblin-gdk, and the official harness (swebench 5.0.2, Docker under WSL) scores them. Three contexts: issue only, BM25 whole files (13K tokens), goblin packet (4K). Files: predictions, receipts with answers, patches and token usage, harness reports, and the gold-patch check (21/26 resolvable on this host). Results: issue only 0 patches, 0 resolved; BM25 14 patches, 0 resolved at 13,430 input tokens; goblin 12 patches, 1 resolved (pytest-7373) at 4,039 tokens. Scripts are in `MCP/mcp03/scripts` and use this host's paths. |
| [`gdk_demo/`](gdk_demo) | The goblin-gdk Goose GDK app with Qwen2.5-0.5B-Instruct on CPU, with and without packets, plus the server identity receipts. |

`summary.json` reports per arm and router:
- packet tokens under the pinned Qwen2.5-0.5B-Instruct tokenizer;
- gold-file recall, all-gold-files hits and gold-symbol recall, at budgets of 512, 1,024, 2,048 and 4,096 tokens. Keys `cNNN` are the composed packets (BM25 path map plus router snippets);
- tokens the router read locally;
- router FLOPs and latency;
- certified exclusions.

Replay without torch:

```bash
python -m MCP.mcp03.run replay --run MCP/results/mcp03/swebench_lite --every 9
```

Replay rebuilds each snapshot from the bare clones named in the config's `work_dir`, so it needs the SWE-bench data and repositories. It compares the tool-result digests, reads and tool calls of every replayed episode. Packet token counts are also compared when the same tokenizer is configured.
