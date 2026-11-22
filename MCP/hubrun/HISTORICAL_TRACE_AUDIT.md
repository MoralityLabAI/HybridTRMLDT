# MCP-05T: historical public replay audit (2026-11-22)

This report freezes the **pre-MCP-05** Bonsai 27B historical raw/Goblin/late comparison. It is not evidence of a native MCP-04 or MCP-05 model replay. The source is the committed public archives in [MCP/results/hubrun/traces](../results/hubrun/traces), originally run on November 21, 2026. Private storyworld traces are outside this audit.

## Source identity and contracts

The CPU extractor reads all five relevant compressed archives, hashes their decompressed JSONL contents, and requires a match to the published `MCP/results/hubrun/manifest.json` source SHA-256 entries. Case IDs were matched separately for the two doc-edit arms and for all three event-recon arms. All five archives had five matched event-recon task IDs; doc-edit had eight matched IDs per arm. The code emits aggregate behavior and hashed request signatures without copying message bodies, file paths or tool arguments into receipts.

Run:

```bash
python -m MCP.hubrun.trace_schema --root MCP/results/hubrun/traces --limit 3 --out MCP/runs/historical_schema.json
python -m MCP.hubrun.trace_metrics --root MCP/results/hubrun/traces --out MCP/runs/historical_metrics.json
python -m pytest -q MCP/test_mcp05_trace_metrics.py MCP/test_mcp05_trace_schema.py
```

The schema census showed that `er-t1-*.traces.jsonl.gz` exposes tool calls at `traces[].nodes[].message.tool_calls[]`, under the actual `workspace_read_file` name. Doc-edit archives instead provide first-class environment counters in `results.jsonl`, including `read_page_calls` and `stale_sites`. These two measurement boundaries remain distinct.

## Event-recon tier 1: five matched historical cases

| Historical arm | File read calls | Exact repeated read requests | Solved tasks |
| --- | ---: | ---: | ---: |
| Raw history | 14 | 0 | 0/5 |
| Late Goblin packet | 24 | 3 | 0/5 |
| Early Goblin packet | 166 | 53 | 0/5 |

An **exact repeated request** is an additional call within one episode with the same normalized tool name and complete parsed argument dictionary, hashed before aggregation. All recorded `workspace_read_file` arguments were parsed. Repeated requests are observable; the metric alone does not establish whether any given repeat was avoidable. The original run summary records up to 47 file reads in one Goblin episode, which is compatible with the new full-archive total.

## Doc-edit: eight matched historical cases

| Environment-reported measure | Raw | Early Goblin | Goblin minus raw |
| --- | ---: | ---: | ---: |
| Reward >= 0.999 (episodes) | 5/8 | 1/8 | -4 |
| Page reads per episode | 11.375 | 25.000 | +13.625 |
| Document searches per episode | 9.375 | 19.000 | +9.625 |
| Stale edit sites per episode | 1.125 | 2.875 | +1.750 |
| Edit calls per episode | 2.625 | 1.125 | -1.500 |
| Submit calls per episode | 0.875 | 0.250 | -0.625 |
| Site recall | 0.7344 | 0.4688 | -0.2656 |

These measurements come from the environment's recorded counters and grader. They do not constitute a fresh independent site-level code-diff verification.

## Acceptance boundary for future native replay

Run [mcp05_replay.sh](scripts/mcp05_replay.sh) on the pinned Bonsai host, with task-ID and exact-token parity checked by [paired_replay.py](paired_replay.py). Compare **raw versus MCP-04 versus MCP-05** on the same seed, tool schema, episode allocations and model checkpoint. Repeat the offline trace audit on the new native outputs. Promotion requires solved-task retention, edit/finalization safety, and fewer repeated reads, with unsuccessful and rejected episodes included.

The historical rows above are regression reference data, not trained-agent gains, native MCP-05 results, or a claim that the new memory policy is superior.
