# MCP-06 registered CPU proxy receipt (v1)

Source: `ef30ef3af5928928324fd9cc63c913d3769ff731`; CI bridge workflow: `fe52ab7d420f35a1273860ce58fb2c1671e99b0a`.

Standalone local CPU run, two threads. 179.14 seconds; peak process RSS 341,508,096 bytes (325.69 MiB, includes Python and PyTorch). 54,345 parameters per learned controller. 12 trained checkpoints, 864 evaluation cells, 110,592 root receipts; unique root keys, probe budgets, record-read counts, aggregate solved counts and checkpoint hashes independently checked. Native Storyworld replay was unavailable in the standalone checkout and was not run.

## Registered primary point gate

At 32K records, 16 probes, clean memory, Hilbert-local vs randomized-local:
- Seed 107: +10.9375 pp
- Seed 109: +18.75 pp
- Seed 113: +14.84375 pp
- Mean +14.84375 pp; preregistered threshold +5 pp; PASS_POINT_ONLY.

## Clean transfer at 16-probe cap

Each cell is solved / 384 paired roots across three seeds.

| arm | 16K | 24K | 32K | 64K |
|---|---:|---:|---:|---:|
| scan | 237 | 228 | 240 | 226 |
| row | 330 | 340 | 328 | 216 |
| random | 32 | 12 | 12 | 7 |
| hilbert | 70 | 51 | 69 | 59 |
| hier | 112 | 79 | 108 | 102 |
| adaptive | 111 | 78 | 108 | 102 |

32K mean probes: scan 10.27; row, random, hilbert, hier 16.00; adaptive 12.49. Adaptive and hier share weights; adaptive maintained the 108 solved cases in aggregate while reducing mean probes by approximately 21.9%. At 64K, both solved 102/384.

At 32K and 15% physical replica corruption: scan 234, row 315, random 11, hilbert 67, hier 105, adaptive 105. These are detected stale replicas under the specified fixture.

## Interpretation and follow-up

The registered gate is an engineering point estimate rather than a significance claim. Hilbert-local exceeds the randomized layout, but deterministic scan and learned row-major routing are substantially stronger. Randomization also breaks replica co-location. Hierarchical strides improve Hilbert routing over local moves and learned stopping saves probes on the hierarchical arm. These are spatial retrieval results, not full TRM/LDT or independent native Storyworld planning results. A follow-up should isolate the effects of replica co-location, compare geometry-aware deterministic routing, and test learned policies against matched realized read budgets before another architecture claim.

Full receipts and checkpoints were packaged locally; hashes:
- roots.jsonl SHA-256: `cf13730dec8c89fc7e6fb51022ae36eb46278c17604b9004db93832cfca8443f`
- summary.json SHA-256: `0005a484e7e4de9aebd1757f942dc504d2323587c6c8f899ba3b58a5774fb07c`
- MCP06_registered_CPU_receipts.zip SHA-256: `b646b0aa1c5781556e2f035001d79acea1425958c256640ef637cab27af253b9`
