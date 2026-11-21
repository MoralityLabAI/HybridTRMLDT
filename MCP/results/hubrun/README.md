# Hub runs: Bonsai 27B on Prime Hub tasksets and storyworlds (2026-11-21)

One night on the RTX 5080 Laptop GPU (80 W limit), running Ternary Bonsai 2 27B PTQ1_0 on:
- held-out lanes of the two MoralityLab storyworld environments;
- seven sets from five agentic Prime Environments Hub tasksets.

Agentic sets ran in a raw-history arm and a goblin-packet arm. Setup, pins and relay fixes are in [`MCP/hubrun/README.md`](../../hubrun/README.md); paper section: `papers/mcp_data_goblin`, "Hub tasksets and storyworlds with a local 27B model".

## Files

| File | Contents |
| --- | --- |
| `summary.json`, `summary.md` | Per run: score, solved with Wilson 95% interval, errored episodes, provider tokens, compaction counts and context overflows. Per pair: paired task outcomes and token ratios. |
| `traces/` | Gzipped verifiers outputs of the public-environment runs (`results.jsonl` from `vf-eval`, `traces.jsonl` from the v1 `eval` CLI). |
| `calls.jsonl.gz` | One line per model call from both relays: tag, arm, status, exact prompt and completion tokens, cached tokens, latency, finish reason and compaction receipt. No message text. |
| `manifest.json` | Model server, verifiers version and `pip freeze` hash of each environment's venv, trace hashes, and the list of private runs that publish aggregates only. |
| `server.json` | Model file, revision and hash, server arguments, relay settings, run times. |
| `queue.log` | Start and end of every job. |

The storyworld environments (`sw-*`, `jbm-*`) are private hub environments, so their prompts and completions stay out of this repository.

## Storyworlds

Held-out splits only: `eval` for storyworld-constitution, `development` for jinn-beast-metta. No `candidate_train`, `confirmatory` or village-replay rows were run.

| Lane | Tasks | Mean reward | Format valid | Legal action | Best action | Critical action |
| --- | --- | --- | --- | --- | --- | --- |
| storyworld-constitution, eval | 15 | 0.00 | 0.00 | – | 0.00 | – |
| jinn-beast-metta cross-frame, dev | 80 | 0.80 | 0.96 | 0.95 | 0.95 | 0.00 |
| jinn-beast-metta constructs, dev | 4 | 0.88 | 1.00 | 1.00 | 1.00 | 0.00 |
| jinn-beast-metta moral reasoner, dev | 16 | 0.87 | 1.00 | 1.00 | 0.94 | 0.00 |
| jinn-beast-metta control mesh v2 (tools), dev | 48 | 0.93 | 1.00 | 1.00 | 1.00 | 0.00 |

- **Jinn-Beast lanes: cleared.** No lane produced a critical action. On cross-frame decisions the weak components are the uncertainty label (matches 15% of tasks) and the review flag (80%).
- **storyworld-constitution: not cleared.**
  - The prompt ends with `Output format: <option_id>`, and the model answered `<opt_page_0000_1>`, angle brackets included, on every task. The environment's strict parser accepts only the bare id, so every answer scores zero.
  - With the brackets stripped, 3 of 15 choices match the historical frontier option, which is near chance over three options. The model would not clear this lane even if the format clash were fixed.
- **A version pin matters for this environment.** It must run on verifiers 0.1.10. On 0.1.11 and later, rubrics receive pydantic messages, and its `isinstance(message, dict)` check turns every answer into an overlong invalid string.

## Agentic tasksets

| Environment | Arm | Episodes | Mean score | Solved | Errored | Prompt tok./ep. | Peak prompt | Compacted calls | Context overflows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| event-recon tier 1 | raw | 5 | 0.00 | 0 | 5 | 159,908 | 31,066 | 0 | 5 |
| event-recon tier 1 | goblin | 5 | 0.00 | 0 | 0 | 294,898 | 32,248 | 185 | 0 |
| event-recon tier 1 | late | 5 | 0.00 | 0 | 5 | 210,853 | 32,119 | 39 | 5 |
| doc-edit | raw | 8 | 0.70 | 5 | 0 | 48,088 | 9,468 | 0 | 0 |
| doc-edit | goblin | 8 | 0.33 | 1 | 0 | 60,187 | 6,650 | 87 | 0 |
| feishu-office hard | raw | 7 | 1.00 | 7 | 0 | 24,820 | 6,612 | 0 | 0 |
| feishu-office hard | goblin | 7 | 1.00 | 7 | 0 | 21,992 | 5,402 | 0 | 0 |
| feishu-office expert | raw | 7 | 1.00 | 7 | 0 | 22,045 | 5,659 | 0 | 0 |
| feishu-office expert | goblin | 7 | 0.86 | 6 | 0 | 19,537 | 5,240 | 0 | 0 |
| mcp-sqlite (exact match) | raw | 57 | 0.39 | 22 | 0 | 14,199 | 8,916 | 0 | 0 |
| durable-notebook (hardened) | raw | 8 | 0.95 | 6 | 0 | 22,044 | 1,751 | 0 | 0 |

What the runs show:

- **What Bonsai 27B can do locally:**
  - every feishu-office hard-tier task, in both arms;
  - 6 of 8 durable-notebook episodes (mean 0.95);
  - 22 of 57 mcp-sqlite questions by exact match (Wilson 95% interval 0.27 to 0.52).
- **event-recon tier 1: nothing in any arm.**
  - **raw**: every episode overflowed the 32,768-token slot within about 13 turns, and the environment ended it with an error.
  - **goblin**: the packet kept prompts near 4K tokens and no episode overflowed. But every episode reached the 40-turn cap without filing a report. The agent kept re-reading files the packet had dropped, up to 47 `read_file` calls in one episode, and used 1.84 times the raw arm's prompt tokens.
- **doc-edit: early compaction hurts.**
  - Raw prompts peak at 9.5K, well inside the context. Compacting from 6K cut success from 5/8 to 1/8; raw was better on 6 tasks and goblin on none.
  - The goblin arm took more turns (15.25 against 9.75), read twice as many pages, left more stale edit sites (2.9 against 1.1) and submitted in a quarter of episodes.
- **Late compaction without an exact count still overflows.** The late arm compacts only past 18,000 estimated tokens, with a 6,000-token packet and four kept turns. It overflowed in all 5 event-recon episodes:
  - the 3.2 characters-per-token estimate undercounts number-heavy tool output;
  - four kept turns of large file reads can exceed the slot by themselves.
- **feishu-office and mcp-sqlite never crossed the threshold.** Their packet arm sent the same requests as raw. The 7/7 against 6/7 on the feishu expert tier is sampling variation.

The goblin packet, as built for MCP-03 retrieval (BM25 plus exact-match line scoring with an action ledger), is not a good history compactor for a 27B agent. Both of its fixes for a long history failed here: compacting early starves the agent of detail, and compacting late without an exact count still overflows. The next version needs two things:
- the server's exact token count (llama-server `/tokenize`), instead of a character estimate;
- packet slots that keep the state the agent has acted on (files read, rows queried, sites edited), not only task-relevant lines.

## Not run

- event-recon tier 2 (3 tasks per arm): dropped once tier 1 scored zero in both arms, to save GPU time.
- Prime-sandbox environments, such as the vet-clinic benchmark and the Lean, SWE and terminal tasksets: they need paid Prime sandboxes.
- mcp-sqlite's API judge (`openai/gpt-5.4-nano`): no key on this host, so mcp-sqlite is scored by its `exact_match` metric.
