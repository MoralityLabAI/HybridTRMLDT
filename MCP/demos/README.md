# Pocket Goblin Field Tests

Two original playable browser projects use procedural Canvas art with no image assets or frontend dependencies. Lantern Below is a compact action RPG about recovering the embers that open a tide gate. The Hour Between Tides is an illustrated branching story where the chart, bell, and ferryman’s seal unlock different routes. Both projects include public briefs for a new playable variant.

From the repository root, serve `MCP/demos` and open the local URL printed by Python.

```sh
python -m http.server 8765 --bind 127.0.0.1 --directory MCP/demos
node --test MCP/demos/smoke.test.mjs
node MCP/demos/playthrough.mjs
node MCP/demos/browser-smoke.mjs /path/to/chrome /path/to/captures
```

## Matched development pilot

Inky authors an alternate RPG encounter and an optional story scene with an ending. Both arms receive the same task, model, settings, completion allowance and project brief. The assisted arm asks the live MCP workspace for a few relevant sections and uses its recurrent SetTRM federation route to read and clip two repository chunks. The wide arm pages through the whole project evidence slice and fetches a broad cross-repository reference packet. Inky generates one candidate in answer-only mode and the same contract verifier grades both arms. The campaign alternates assisted, wide, assisted, wide. The RPG goes assisted first and the story goes wide first.

This measures two retrieval policies and their prompt budgets on two development tasks. A matched lexical short-packet ablation would be needed to isolate the ranker's contribution. The wide arm is a bounded token-intensive control built from the shared index, not a dump of the whole federation.

```sh
python -m MCP.mcp05.creative_bench --indexes "$INDEXES" --router "$ROUTER" --gdk "$GDK" --out "$CAMPAIGN"
```

Each archived attempt includes the exact Inky exchange, retrieval MCP and Goose MCP wire logs, chunk digests and route metadata, verifier verdict, and candidate where parseable. The local training index labels the attempts and retains both successes and failures. Its development rows carry `training_ready=false` pending admission. The game imports validated data only.

[Read the pilot results](RESULTS.md).

Open the approved patrol and passage from the local portal’s benchmark variants. To validate and normalize the local training archive, supply the frozen Q4 runtime receipt used during the campaign.

```sh
python -m MCP.mcp05.creative_trace "$CAMPAIGN" --runtime "$RUNTIME_RECEIPT"
```
