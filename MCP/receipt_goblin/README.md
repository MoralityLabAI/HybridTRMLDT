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
`--request-params JSON` is merged into the provider request. Re-running with the same `--out` and
`--arm` skips the claims already written.

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
