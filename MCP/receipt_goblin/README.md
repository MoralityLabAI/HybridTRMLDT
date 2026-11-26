# Receipt Goblin: check a write-up's numbers against its run receipts

Receipt Goblin checks the numeric claims in a Markdown write-up (a results README, a summary table)
against the JSON receipts the numbers were taken from. It targets one failure in particular: a number
copied from the wrong cell, arm or run. A number search cannot catch that error, because the wrong
number does exist in the receipts.

The receipts are too large for a small model's context (`MCP/results/mcp02/pilot_gpu/summary.json`
is 867 KB). As in the Data Goblin packet server (`MCP/mcp03/goblin_server.py`), a local server builds
a small packet for each claim, about 1.3K tokens by default, and the model only has to link the claim
to one packet leaf. Code then compares the numbers.

## Pieces

| File | Role |
| --- | --- |
| `numbers.py` | Numbers as written: `13,430`, `0.86`, `78.5K`, `15%`. A claim matches a receipt value at the claim's written precision. |
| `claims.py` | One claim per number in a Markdown file: the sentence, or the table row prefixed by the paragraph before the table. Range endpoints (`0.90–0.92`), years, identifiers and code spans are skipped. |
| `receipts.py` | Flattens every `*.json` receipt under a root into numeric leaves. A list element is labelled by its identity fields (`runs/[arm=goblin policy=looped]/accepted`), not its position. BM25 over leaf paths. |
| `packet.py` | Packet for one claim: the top 40 lexical leaves, the leaves whose value reads as the claimed number, and up to 8 path neighbours of each of the first 4 number matches (the same metric in adjacent cells). `blind` withholds the values. `certify` compares the claim with a linked leaf. |
| `server.py` | MCP stdio server: `extract_claims`, `claim_packet`, `certify`. |
| `bench.py`, `score.py` | Benchmark construction, rule baselines, scoring. |

The GDK app is `receipt-goblin` in [`MCP/goose_gdk_app`](../goose_gdk_app). For each claim, a
`goose-agent` state machine runs three steps in priority order:

1. `CertifyOperation` applies after the model answers. It parses `{"leaf": "L07"}` or `NONE` and,
   once, asks for a repair when the answer names no packet leaf. It then calls `certify`.
2. `ReceiptPacketOperation` contributes the instructions and the packet as prompt parts.
3. `InferenceRunner` calls an OpenAI-compatible endpoint through `goose-providers`.

```bash
cd MCP/goose_gdk_app && cargo build --release
./target/release/receipt-goblin --doc ../results/mcp03/README.md --receipts ../results/mcp03 \
    --server-cwd ../.. --base-url http://127.0.0.1:18090/v1/ --model inkling-2-mini --out receipts.jsonl
./target/release/receipt-goblin --bench ../receipt_goblin/bench.jsonl --server-cwd ../.. --arm packet --out run.jsonl
python -m MCP.receipt_goblin.score run.jsonl
```

`--blind` withholds packet values. `--k-lexical`, `--k-neighbours` and `--budget` shape the packet.
Re-running with the same `--out` and `--arm` skips the claims already written.

Inside a Goose session the server works as an extension:
`goose session --with-extension "python -m MCP.receipt_goblin.server"`, run from the repository root.

## Benchmark

`bench_config.json` pairs five result write-ups in this repository with their receipt directories
(MCP-01, MCP-02, MCP-03, and the hub run's README and `summary.md`). Building the benchmark:

1. `bench.py candidates` sampled up to 22 claims per document among numbers that match 1 to 50
   receipt leaves. It listed the top 6 matching leaves for each, ranked by lexical score.
2. I (Claude Opus 5.5, in Claude Code) reviewed all 87 sampled claims. 58 received gold leaves, and
   equivalent copies of one value are all gold. The other 29 have no single receipt leaf:
   configuration lists, approximations such as "13K", counts held outside the receipts, and
   coincidental matches. `gold.json` records every decision.
3. `bench.py build` writes `bench.jsonl` with three variants per gold claim:
   - `clean`: the claim as written; the label is `supported`.
   - `off_receipt`: a nearby value that no receipt leaf reads as; the label is `contradicted`. Only 23
     claims admit one, because small counts match some leaf almost everywhere.
   - `decoy`: the value of a sibling of a gold leaf, meaning the same metric in an adjacent cell or an
     adjacent metric of the same cell; the label is `contradicted`. 53 claims have one.

That gives 134 items. Gold leaves could only be picked from the top 6 lexical candidates, so the
benchmark leans toward claims that lexical search can locate. The neighbour expansion in the packet
draws on the same sibling relation that defines the decoys.

### Rule baselines (`MCP/results/receipt_goblin/baselines`)

A claim *passes* when the verdict is `supported`; flagging it any other way counts as `contradicted`.

| Arm | clean passed | off_receipt flagged | decoy flagged |
| --- | ---: | ---: | ---: |
| number lookup over all receipts | 58/58 | 23/23 | 0/53 |
| number lookup inside the packet | 58/58 | 23/23 | 0/53 |
| BM25 top-1 leaf, then certify | 1/58 | 23/23 | 53/53 |

BM25 top-1 links a gold leaf for 0 of 134 items. It flags everything because it links the wrong
leaf. The default packet holds a gold leaf for 58/58 clean, 51/53 decoy and 15/23 off-receipt
items, at a mean of 1,235, 1,297 and 900 packet tokens. Packet tokens use the labelled estimate
ceil(UTF-8 bytes / 4) unless `MCP03_TOKENIZER` names a tokenizer.

### Inky at 16K (`MCP/results/receipt_goblin/inky_16k`)

Inkling-2-Mini on the squid5080 engine (nvfp4, MTP on, RTX 5080 Laptop), run through `receipt-goblin`
with a 16,384-token context limit, 2,048 output tokens per call, temperature 0 and at most one repair.
`scripts/inky_16k.sh` ran all 134 items in each arm on 2026-11-26. The server sized its own context
to 51,200 tokens (an explicit `--ctx 16384` exits when another process takes VRAM during the load). The
largest provider input count for any item, summed over its calls, was 8,735 tokens, so every call fit
in 16,384 with its output budget.

| Arm | clean passed | decoy contradicted on a linked leaf | decoy flagged, any verdict | off_receipt contradicted on a linked leaf | off_receipt flagged |
| --- | ---: | ---: | ---: | ---: | ---: |
| `inky_packet`: values shown, 40 lexical + 8 neighbours | 58/58 | 42/53 | 52/53 | 13/23 | 23/23 |
| `inky_blind`: values withheld, same packet | 52/58 | 47/53 | 52/53 | 14/23 | 23/23 |
| `inky_packet_nonb`: values shown, no neighbours, 1,500 tokens | 49/58 | 25/53 | 51/53 | 12/23 | 23/23 |

The other flags are `unsupported` (Inky answered `NONE`) and `unparsed`. Every `unparsed` row (8, 7
and 5 per arm) is two empty answers: the reasoning used the whole 2,048-token budget on the first
call and again on the repair.

- Linking: with values shown, Inky linked a gold leaf for 57/58 clean, 42/53 decoy and 13/23
  off-receipt items. Each decoy it contradicted was contradicted on its gold leaf. The packet held
  a gold leaf for 51 of the 53 decoys, so 42 of those 51 were linked.
- Off-receipt items: the packet held a gold leaf for 15 of 23, and Inky linked 13 of those 15. It
  answered `NONE` for 9 items.
- Passed decoy: one decoy passed in both shown-value arms. "BM25 ⟦26⟧ patches" was linked to
  `submitted_instances = 26`; the write-up's original 14 is `completed_instances`.
- Blind mode: withholding values cost 6 clean items and added 5 decoys contradicted on a linked
  leaf. With values shown, the
  decoy failures are mostly reasoning that ran out of budget (7 unparsed, against 4 when blind).
- Neighbours: without them the packet holds a gold leaf for 30/53 decoys instead of 51/53, and decoys
  contradicted on a linked leaf drop from 42 to 25.
- Speed: the shown-value arm took 12 minutes for 134 items, 5.6 s of inference per item on average.
  The mean provider input was 1.8K to 2.7K tokens per variant.

### Through the goose CLI (`MCP/results/receipt_goblin/goose_cli_16k`)

`scripts/goose_demo.py` runs claims through goose 1.53.0 itself: `goose run --no-profile` with the
Receipt Goblin server as its only extension, the same Inky server, and `GOOSE_CONTEXT_LIMIT=16384`.
Goose is told to call `claim_packet`, choose a leaf and call `certify`. The script reads the tool
calls and the last certify result from goose's JSON transcript. It ran on 12 items, 4 per variant,
sampled with seed 0.

| | clean | decoy | off_receipt |
| --- | ---: | ---: | ---: |
| runs that called `claim_packet` then `certify` | 4/4 | 4/4 | 4/4 |
| right verdict (passed / flagged) | 4/4 | 4/4 | 4/4 |
| gold leaf linked | 3/4 | 3/4 | 3/4 |

Two of the three runs without a gold leaf flagged with `NONE`. The third, a clean "109", linked the
top-level `swebench_lite/summary.json :: instances = 109`; the gold annotation lists only the per-arm
copies. A run took 5.7 to 8.9 s. Goose's token count for a whole run, its system prompt and tool
schemas included, was 6.1K to 12.2K.

Both runs need the squid5080 server; `scripts/with_squid.sh WORK -- COMMAND` queues it for the card,
runs the command, and stops it afterwards. `inky_16k.sh` uses it when no server answers.
