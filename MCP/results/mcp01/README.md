# MCP-01 recorded smoke

Published source: [`1794c31f`](https://github.com/MoralityLabAI/HybridTRMLDT/commit/1794c31f4e5f72a2458422e360227611b44c8390).

All **360 reference episodes passed** across catalog sizes 8, 32, and 128; index depths 1, 3, and 6; and four discovery arms. Each configuration has ten paired tasks. The stdio result of every tool call matched its independent in-process replay.

The suite passed **83 tests and 55 subtests**. The local chat adapter was exercised with a loopback HTTP test server. This receipt contains zero neural model calls; model token measurements remain unknown.

## One shape from the sweep

The following counts cover ten tasks at catalog size 128 and index depth 3. Context bytes are simulated reference transcript exposure, including repeated history. They carry no tokenizer or model-performance claim.

| Arm | Accepted | Tool calls | Initial catalog bytes | Reference context bytes | Metadata checks |
| --- | ---: | ---: | ---: | ---: | ---: |
| flat | 10/10 | 16 | 818050 | 2565604 | 0 |
| layered | 10/10 | 84 | 10560 | 962013 | 0 |
| layered_ldt | 10/10 | 81 | 10560 | 973124 | 1490 |
| search | 10/10 | 32 | 9780 | 182426 | 0 |

Ordinary search provides an inexpensive control on the explicit-key fixture. Index depth changes the number of local operations. LDT filtering excludes incompatible metadata, with its checker work counted. Later semantic retrieval and trained-router studies use the frozen-comparison procedure in [PROTOCOL.md](../../PROTOCOL.md).

## Files

- [summary.json](summary.json) records every configuration, source hashes and failure-inclusive metrics.
- [episodes.jsonl.gz](episodes.jsonl.gz) preserves every original episode.
- [source_manifest.json](source_manifest.json) binds the harness and imported kernel bytes.
- [publication.json](publication.json) verifies equality between the executed local source tree and its published Git commit.
- [tests.txt](tests.txt) records the test outcome.

The source commit was published through the connected repository service. Its Git tree exactly matches the tested local commit recorded by the runner; the publication receipt binds both commit IDs. Gzip decompression reproduces the original JSONL byte-for-byte.

Reproduce the sweep from the repository root:

```bash
python -m MCP.run --controller reference --transport stdio --cases 10 --catalog-sizes 8 32 128 --depths 1 3 6 --out MCP/runs/replay
```
