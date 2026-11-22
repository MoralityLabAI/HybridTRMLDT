# MCP-02: learned routing over compositional MCP catalogs

MCP-02 adds the following to the [MCP-01 gym](../README.md), which stays unchanged as the reference baseline:

- multi-server synthetic worlds;
- layered indices over servers, collections, capabilities, provenance, name buckets and entities;
- candidate refinement certified through `certify_and_apply_verified`;
- seven interchangeable router families plus deterministic, sham and oracle controls, all behind one observation and action boundary;
- failure-inclusive economics.

The design, splits, controls and history are in [PROTOCOL.md](PROTOCOL.md). Measured results are in [`MCP/results/mcp02`](../results/mcp02/README.md).

## Layout

| File | Role |
| --- | --- |
| [`world.py`](world.py) | World and task generator; hidden `oracle`; grouped splits |
| [`env.py`](env.py) | MCP tool surface per arm; index layouts; certified `candidate_refine`; epoch roll; private judge |
| [`agent.py`](agent.py) | Shared plan tracker, frontier, observation, routing loop; deterministic, BM25, sham and oracle policies; tape format |
| [`features.py`](features.py) | Shared hashed-token encoder for learned routers (standard library) |
| [`models.py`](models.py) | Dense, Transformer, looped, TRM, TRM+LDT and HRM routers; parameter, FLOP and depth receipts (torch) |
| [`train.py`](train.py) | Imitation datasets from `train` worlds and listwise training (torch) |
| [`run.py`](run.py) | Evaluation grid and sweeps, summary, torch-free replay, memory probe |
| [`economics.py`](economics.py) | Cost per accepted task, Wilson intervals, paired bootstrap differences |
| [`llm.py`](llm.py), [`hf_server.py`](hf_server.py) | LLM controller through MCP-01's `LocalModel` adapter; loopback Transformers server with measured identity |
| [`serve.py`](serve.py) | One MCP-02 world over MCP-01's stdio server |

## Run

Run these from the repository root. The controls, the tests and replay use the standard library only.

```bash
python -m pytest -q MCP/test_mcp02.py
python -m MCP.mcp02.run evaluate --config MCP/mcp02/configs/smoke_cpu.json --controls-only --out MCP/runs/mcp02-smoke
python -m MCP.mcp02.run summarize --run MCP/runs/mcp02-smoke
python -m MCP.mcp02.run replay --run MCP/runs/mcp02-smoke
```

Learned routers need torch:

```bash
python -m MCP.mcp02.train dataset --config MCP/mcp02/configs/pilot_cpu.json --split train --out WORK/train.json.gz
python -m MCP.mcp02.train dataset --config MCP/mcp02/configs/pilot_cpu.json --split dev --out WORK/dev.json.gz
python -m MCP.mcp02.train fit --config MCP/mcp02/configs/pilot_cpu.json --model trm_c2i1_d48 \
    --train WORK/train.json.gz --dev WORK/dev.json.gz --out WORK/ckpt/trm_c2i1_d48.pt
python -m MCP.mcp02.run evaluate --config MCP/mcp02/configs/pilot_cpu.json --checkpoints WORK/ckpt --out WORK/run
python -m MCP.mcp02.run memprobe --checkpoint WORK/ckpt/trm_c2i1_d48.pt
```

For the GPU pilot, pass `--device cuda --cuda-memory-fraction F` to `fit` and `evaluate`. On a shared card, run the whole sequence under a VRAM lease. Checkpoints stay outside Git. Their SHA-256 values, training receipts and measured FLOPs are recorded in each run's `model_receipts.json`.

To use a local LLM as an end-to-end controller, start any loopback OpenAI-compatible server, or the bundled one:

```bash
python -m MCP.mcp02.hf_server --model-dir PATH/TO/SNAPSHOT --served-name qwen2.5-0.5b-instruct --port 8093
python -m MCP.mcp02.llm --config MCP/mcp02/configs/llm_pilot.json --out WORK/llm --model qwen2.5-0.5b-instruct \
    --base-url http://127.0.0.1:8093/v1
```

## Records

A run directory holds:

- `episodes.jsonl`: one lean record per episode, with outcome, judge reasons, tool and verification counters, compute totals, a digest of every tool result, and the call history;
- `tapes.jsonl`: the decision tape, holding the observation digest, chosen index and cost per decision, plus exclusion proposals and raw LLM responses;
- `model_receipts.json`, `host.json`, `source_manifest.json`, `run_config.json`, and `summary.json` from `summarize`.

`replay` re-runs every episode, or every Nth with `--every`, from its tape without torch or a model server, and compares outcome, tool calls, handoff bytes, result digests, environment counters and FLOP totals.

## Data Goblin: future training objective

Data Goblin is the intended local retrieval specialist of about one billion parameters. It should spend inexpensive local inference on retrieving, filtering, cross-referencing and checking evidence, then hand a frontier model a compact, verifiable packet. MCP-02 is the gym and evidence-collection stage for that objective. It trains no Data Goblin.

The specialisation, once the gym's evidence selects an architecture:

1. **Inputs.** The public task, MCP tool schemas or a frontier of candidate expansions, read results, and the explicit candidate state of a layered index.
2. **Outputs.** One MCP action per step: an expansion, a certified-exclusion proposal or a final packet. Packets cite evidence by environment-issued refs and fit a strict byte budget.
3. **Supervision.** First, imitation of oracle-routed trajectories from grouped `train` worlds, keeping failed routes with their failure labels. Then policy optimisation against the exact judge, with reward equal to negative total cost per accepted task under the failure-inclusive accounting in [`economics.py`](economics.py). The downstream context budget stays a hard constraint.
4. **Deliberation.** Local tool calls, recurrent depth and generated tokens are experimental variables. All of their compute is counted, and none of it reaches the frontier model except through the packet.
5. **Certification.** Hard exclusions only through `certify_and_apply_verified`. Learned relevance stays soft, and unknown metadata keeps a candidate.
6. **Promotion gate.** Data Goblin must beat the registered BM25, deterministic and best small-router controls on held-out `test` worlds. Then it must pass a downstream run in which a frontier agent receives only the packet and its resource handles and completes more tasks per paid token.
