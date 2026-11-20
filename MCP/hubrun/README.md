# Hub runs: Bonsai 27B on Prime Environments Hub tasksets, raw versus goblin packet

This directory runs agentic environments from the Prime Intellect Environments Hub and the MoralityLab storyworld environments against a local Ternary Bonsai 2 27B. A relay between the environment and the model offers two arms:

- **raw**: messages are forwarded unchanged;
- **goblin**: messages pass through `packet.compact_messages` first.

The goblin packet is the MCP-03 packet scorer applied to a growing agent history. Once the history estimate passes 6,000 tokens:

- the task and the last two assistant turns (with their tool results) stay verbatim;
- the middle becomes one packet: an action ledger of every earlier tool call, plus the earlier lines that score highest against the task and the recent turns. Scoring is BM25 (`MCP.mcp02.textutil.BM25`) plus an exact-match channel over the task's salient terms (`MCP.mcp03.agent.salient_terms`) and identifier-like tokens;
- the packet is cut to 1,800 estimated tokens;
- any single tool result over 6,000 estimated tokens in the kept tail is cut to its most relevant lines.

The learned MCP-03 routers are not used here. They score code-repository candidates, and nothing in these environments has that shape.

## Model and server

| Item | Value |
| --- | --- |
| Model | `prism-ml/Ternary-Bonsai-2-27B-gguf` rev `94dd51cd`, `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, sha256 `53107f53…3ee3` (verified after download) |
| Runtime | PrismML llama-server b10743 (`adfffbe41`), CUDA, all layers on an RTX 5080 Laptop GPU (16 GB) |
| Context | `-c 65536 -np 2`: two slots of 32,768 tokens |
| Reasoning | `--reasoning-format deepseek --reasoning-budget 1536`; the relay strips `reasoning_content` from replies, so environments see the final answer only |
| Sampling | temperature 0.6; completion cap 4,096 tokens per call (relay) |
| Lease | `vramorama run --vram 11G --guard-power 300 --restart 5`; the server held 10.3 GB |

WSL in NAT mode cannot open Windows loopback ports. The relay therefore runs in WSL and reaches the Windows server through the Windows `curl.exe` (`proxy.py --via-curl`).

## Environments

Every environment runs in its own uv venv in WSL. The verifiers version is pinned per environment (see the table). The pins matter for two reasons:

- **Message types.** verifiers 0.1.11 and later pass pydantic message objects to rubrics. `storyworld-constitution` checks `isinstance(message, dict)`, so on those versions it scores every answer as an overlong, invalid string. 0.1.10 is the newest version with dict messages that also meets the environment's declared floor (`>=0.1.10`).
- **Endpoints.** verifiers 0.4.x replaced `vf-eval` and moved endpoint configuration. Environments that accept 0.3.1 are pinned to 0.3.1 so that one CLI shape covers them.

| Environment (hub version) | verifiers | Mode | Split used | Notes |
| --- | --- | --- | --- | --- |
| moralitylab/storyworld-constitution 0.1.1 | 0.1.10 | single turn | `eval` (15) | private; aggregates only |
| moralitylab/jinn-beast-metta 0.1.15 | 0.2.1 | single turn and tool lane | `development` only | private; aggregates only; `candidate_train`, `confirmatory` and the village replay gate are not run |
| briahnloo/event-recon 1.0.0 | 0.3.1 | v1 MCP tool loop, up to 40 turns | public split, tiers 1 and 2 | agent and tools on the local subprocess runtime |
| emerge/feishu-office-v1 0.4.1 | 0.3.1 | tool loop | hard and expert tiers, shuffled with seed 0 | |
| dp-learning-rl/doc-edit 0.1.3 | 0.3.1 | tool loop | generated, seed 1234 | |
| mikqu/mcp-sqlite 0.1.0 | 0.3.1 | v1 MCP tool loop, up to 20 turns | all 57 | the paid API judge is removed; the score is the taskset's `exact_match` metric |
| aravind-k/durable-notebook 0.1.4 | 0.3.1 | tool loop with environment-side compaction | held-out episodes, hardened grader | |

Each environment ran in two smoke runs first (one or two tasks). Prompts in mcp-sqlite (peak 1.4K tokens) and durable-notebook (1.6K) stay below the packet threshold, so the goblin arm would send identical requests. Those two run in the raw arm only. Each A/B pair runs its raw and goblin arms at the same time, one server slot each, on the same tasks.

## Relay fixes found in smoke runs

- **Null fields.** verifiers sends `content: null` on assistant tool-call turns, and `null` for unset fields. llama-server rejects both with HTTP 500. `proxy.normalize_messages` sets content to `""` and drops other null fields, for both arms.
- **Hidden reasoning.** Reasoning is stripped from replies, as with a hosted model that hides its reasoning.

## Run

The scripts in `scripts/` use this host's paths (`/root/hub-overnight` in WSL and `C:\Users\patri\Documents\Codex\hub-overnight` on Windows):

```bash
bash scripts/setup_wsl.sh <env>...            # copy pulled envs, one venv each
bash scripts/pin.sh <verifiers-version> <env>...
bash scripts/proxy_run.sh                     # relay on 127.0.0.1:18130
bash scripts/queue.sh                         # resumable overnight queue
python3 -m MCP.hubrun.summarize --runs /root/hub-overnight/runs \
  --calls /root/hub-overnight/logs/calls.jsonl --out MCP/results/hubrun
```

Results and the write-up are in [`MCP/results/hubrun`](../results/hubrun).
