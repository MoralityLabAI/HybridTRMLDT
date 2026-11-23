# Pocket Goblin evidence budget pilot

The pilot uses the same Aleph preparation task and the same IQ4_NL Inkling checkpoint across arms. The source snapshots, task, authoritative execution plan, operator advice, decoding settings and output cap are pinned. The 100K baseline uses the original weighted packet of indexed evidence. The smaller arms use the federation's learned SetTRM shard order, exact vocabulary exclusions checked through the lattice membrane, and BM25 within the selected shards.

| Arm | Evidence policy | App mode | Server slot | CPU MoE layers |
| --- | --- | --- | ---: | ---: |
| Wide baseline | Weighted seven-repo evidence | Repository tools | 100,096 | 18 |
| Routed 16K | Top-two routed shards per query | Repository tools | 100,096 | 18 |
| Routed 50K | Same ranked candidate stream as routed 16K | Answer only | 100,096 | 18 |
| Routed 50K, higher GPU placement | Byte-identical routed 50K request | Answer only | 100,096 | 10 |

The first two arms measure the practical change from wide to routed evidence at the smaller budget. Both generated control markers through the output cap. The separate 50K pair uses answer-only mode, which removes the repository-tool instructions after evidence assembly. The placement comparison holds the request and server slot fixed. A matched lexical route can isolate the learned ranker's contribution in a follow-up.

## Results

| Arm | Input tokens | Prefill wall time | Prefill tokens/s | Decode tokens/s | Task score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wide baseline | 97,982 | 3,008.06 s | 32.57 | 10.61 | 0/12 |
| Routed 16K | 14,579 | 535.66 s | 27.22 | 12.83 | 0/12 |
| Routed 50K | 48,417 | 1,109.75 s | 43.63 | 13.20 | 4/12 |
| Routed 50K, higher GPU placement | 48,417 | 676.27 s | 71.59 | 17.93 | 4/12 |

All four requests were cold and reached the 1,024-token output cap. The 16K routed prompt used 85.1% fewer input tokens and reduced prefill from 50:08 to 8:56. Neither tool-mode run returned a usable answer. The CPU navigator took 9.63 seconds to select and retrieve candidates across the local federation. Tokenization and packing took 0.36 seconds for the smaller packet and 1.57 seconds for the 50K packet after routing. The vocabulary verifier did not exclude any shard on these broad queries.

The matched 50K pair reduced prefill from 18:30 to 11:16 and GDK request time from 19:48 to 12:14. The server moved eight layers of expert weights to the RTX. GPU model buffers grew from 6,489 to 10,863 MiB, while host model buffers fell from 10,311 to 5,937 MiB. Observed device use increased from 9,566 to 13,964 MiB. The higher-residency run had 1,952 MiB of device memory free, and the one-time startup took 27.29 seconds. KV and compute buffer sizes stayed fixed at 1,664 and 1,396 MiB. The matched request SHA-256 was `33b4e496e5471f2181ca5e95ffb82a542e4461a6a8c3c9bdd40d968c909b04b7`.

Both answer-only drafts hit the output cap before completing the required evidence review. Both described a conditional GO for the engineering pilot where its currently unresolved gates call for a HOLD. The higher-GPU draft gave a more precise restart check but omitted the three grounded source citations that the original-placement draft provided. Equal rubric totals conceal those differences. This comparison measures deployment latency on one task. It cannot establish that wider or routed evidence preserves answer quality, or that the learned ranker improves it, because the prompt budgets differ and the two pairs use different app modes.

## Measurement

- Record one fresh request per arm with exact input-token preflight and the same 1,024-token generation cap. Keep prefix reuse off for the benchmark arms or record the provider's actual cached and newly evaluated token counts. Preserve exact request and response bytes, source manifests, finish reasons, and model/runtime versions.
- Time packet assembly, local tokenization, prefill, decoding and end-to-end wall clock separately. Report actual input length, new token count, prefill and decode throughput, and observed process RAM and GPU memory. Keep cold model startup separate from per-turn time.
- Score each answer using the criteria below before attributing utility to a shorter input. Each item is binary. A contradiction is a miss. If the answer terminates before a criterion, count the miss and record the finish reason.
- This is one development task and one run per arm. The comparison is suitable for a latency and feasibility case study. Generalization or a learned-router quality claim needs independent tasks and budget-matched controls.

The prompt assembler reads the seven pinned indexes and runs the existing SetTRM ranker over the available shard candidates. For each query, it takes the first two ranked repositories with eligible snippets under the predeclared path filters, then interleaves the resulting BM25 lists. It drops duplicate section IDs, limits repeated sections from the same file, and packs two nested budgets against the real llama.cpp tokenizer. Both budgets start with identical execution-plan and advisor bytes.

```sh
python -m MCP.mcp05.routed_context --indexes "$INDEXES" --spec "$CAMPAIGN/evidence-lanes.json" \
  --lead "$CAMPAIGN/routed-lead.md" --router "$ROUTER" \
  --out "$CAMPAIGN/routed-settrm-v1" --budgets 14300 48300 --port 18097
```

For the 50K requests, use the `48300/evidence.md` packet, the common `long-task.md`, `--answer-only`, `--context-window 100096`, `--max-agent-iterations 1`, and `--max-tokens 1024`. Keep the original IQ4_NL file, llama.cpp `b12602`, `Vulkan1`, one recurrent loop, and the same batch and sampler settings. Set `MCP05_FRESH_PROMPT=1` for each cold request. Check the request SHA-256 and provider-reported cache usage before comparing placement.

## Frozen answer criteria

1. Gives a HOLD for the real pilot while allowing bounded local preparation.
2. Distinguishes the Friday 10M processed non-padding gate from the later 10B starting target.
3. Separately counts accepted unique, processed non-padding, loss-bearing and padding positions.
4. Preserves original rights, deduplication, transitive family custody and sealed-root separation before packing.
5. Treats the tokenizer as an unresolved freeze and binds production tokenization and packing to immutable identifiers.
6. Describes exact sample-order restart and includes model, optimizer, scheduler, RNG and sampler or packing state.
7. Uses direct feedback and the two-visit initial point, with an independently checked tied-pass comparison.
8. Keeps the source-conflict first- or latest-report rule as a negative control and limits the inference from beating it.
9. Gives local deliverables with owners and a separate measured TPU allocation, step, memory and throughput gate.
10. Defers base-freeze trace collection and Silico or adapter work to separately governed evidence.
11. Cites at least three identifiable evidence paths. Confirm each path is in the packet or central pinned plan and supports its claim.
12. Avoids presenting proposed runs, unreviewed source or local reference behavior as achieved TPU or full-scale results.
