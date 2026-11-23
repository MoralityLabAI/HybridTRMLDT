# Pocket Goblin evidence budget pilot

The pilot uses the same Aleph preparation task and the same IQ4_NL Inkling checkpoint across arms. The source snapshots, task, authoritative execution plan, operator advice, decoding settings and output cap are pinned. The 100K baseline uses the original weighted packet of indexed evidence. The smaller arms use the federation's learned SetTRM shard order, exact vocabulary exclusions checked through the lattice membrane, and BM25 within the selected shards.

| Arm | Evidence policy | Server slot | CPU MoE layers |
| --- | --- | ---: | ---: |
| Wide baseline | Weighted seven-repo evidence | 100,096 | 18 |
| Routed 16K | Top-two routed shards per query | 100,096 | 18 |
| Routed 50K | Same ranked candidate stream as routed 16K | 100,096 | 18 |
| Routed 50K, placement change | Byte-identical routed 50K prompt | 100,096 | Measured on launch |

The first three arms measure the practical change from wide evidence to routed evidence at two budgets. The final arm isolates expert placement by holding the input and server slot fixed. A matched lexical route can isolate the learned ranker's contribution in a follow-up.

## Measurement

- Record one fresh request per arm with exact input-token preflight and the same 1,024-token generation cap. Keep prefix reuse off for the benchmark arms or record the provider's actual cached and newly evaluated token counts. Preserve exact request and response bytes, source manifests, finish reasons, and model/runtime versions.
- Time packet assembly, local tokenization, prefill, decoding and end-to-end wall clock separately. Report actual input length, new token count, prefill and decode throughput, and observed process RAM and GPU memory. Keep cold model startup separate from per-turn time.
- Score each answer using the criteria below before attributing utility to a shorter input. Each item is binary. A contradiction is a miss. If the answer terminates before a criterion, count the miss and record the finish reason.
- This is one development task and one run per arm. The comparison is suitable for a latency and feasibility case study. Generalization or a learned-router quality claim needs independent tasks and budget-matched controls.

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
