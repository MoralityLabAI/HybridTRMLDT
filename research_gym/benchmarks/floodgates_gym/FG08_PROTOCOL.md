# FG-08: batched CPU serving, verified cache compaction and RAM pressure

Prospective public development protocol. FG-01–06 verified greedy parity on
small synthetic targets. The previous FG-07 campaign failed at import due
to an unclosed action tuple; a separate correction is committed. FG-08
does not rely on FG-07 results or its controller.

## Workload and comparisons

Two independent fresh final-only four-loop targets, seeds 137 and 163,
trained using the existing GSF-10 fitter on 256 synthetic prefix roots.
Fit first-loop and final-loop ordinary rank-4 one-pass draft adapters on
96 distinct adapter roots; freeze the target and both adapters.
Use new generator offset 40000 for all panels; verify prefix-content
disjointness. 32 requests each for dev, shift and trace_train, all eight-token
prefixes generating twelve new tokens. The shift is the same period-three
stress test as the earlier gym.

Policies:
- serial: full-depth ordinary autoregressive step, batched across requests;
- first4: four masked draft slots, conditioned on first-loop verified KV;
- deep4: same using final-loop verified KV.

Every speculative action uses one draft pass and the unchanged full-depth
verifier. Near the output limit, shorten the draft to remaining allowance.
Exact greedy token IDs must equal independently cold-checked serial references.
Physically truncate provisional KV after rejection; finished requests release
their committed cache. No adapter writes into the verifier cache.

The scheduler admits deterministic waves of requests. Within a wave, active
requests are bucketed by committed-prefix length; a bucket uses one actual
batched tensor forward for each draft/target call. No padded slots, oracle
future decisions, continuous arrivals or multi-threaded service claims.
Requests that advance unevenly may split into smaller buckets. Admit the next
wave only after the current one completes. Baselines use identical workload,
wave admission and the same cache representation and verifier.

## Profiles

All policies see the same five profiles: batch ceilings 1, 4, 16, 32 with a
1 MiB committed-cache reservation budget, plus ceiling 32 under 128 KiB.
Reserve the worst-case committed per-request cache
  2 * loops * width * (prefix + generated - 1) * sizeof(float32)
throughout its wave. This is 19,456 bytes per request. The pressure profile
therefore admits at most six concurrent requests. The budget is a declared
persistent committed-KV reservation cap, not an RSS, allocator or temporary
tensor cap. Meter actual live committed cache tensor bytes, stacked inputs,
provisional target KV and retained cache bytes separately. Verify that live
committed KV never exceeds the reserved bound. Release caches at completion.
Do not claim memory-bandwidth savings from logical byte counts.

## Timing and metrics

Each seed/policy/profile runs in its own fresh subprocess with one torch
thread, fixed CPU numeric profile, loaded frozen weights and one untimed
warmup wave. Three measured repeats per panel; each repeat starts fresh
requests/caches. Actual elapsed wall and process CPU time includes prefill,
draft/verify, scheduler, rollback and host receipt accounting; excludes
checkpoint loading, independent serial reference generation, output
serialization and audit. Report median tokens/s from three repeats, median
batch latency, request TTFT and completion p50/p95 from individual repeats
with request admission time as the latency origin. Also record queue-to-
completion latency from panel start. Timing is diagnostic, single-worker CPU
toy fixture, with no claim of significance or production GPU throughput.

Primary descriptive comparison: first4 and deep4 versus serial at matched
seed, panel and profile. Report exact generated IDs, requests completed,
actual batch sizes and target/draft calls, actual token-block work, prefill
work, committed-cache peaks, target provisional peaks, logical stack-copy
bytes and fresh-process peak RSS (Linux ru_maxrss, includes weights, warmup,
reference work and all panels). Do not attribute process RSS exclusively to
KV storage. Work totals are conserved independently from measured time.
The complete outputs and scheduling ledger must be identical across repeats
for a cell. No timed correctness oracle calls.

## Receipts and limits

Default: two target fits, four adapter fits, 30 subprocess cells and 960
trace records over 32 distinct trace roots. Same model hashes across serving
profiles; each root's views remain grouped. Save checkpoints, source hashes,
profile bindings, per-panel stable ledgers and individual repeat timings.
Audit reloads weights, regenerates reference paths and independently reruns
the trace workloads, checks every exact emitted token, batching schedule,
cache limit and work ledger, and reconciles paired summaries. Timing samples
are checked for finite positive values and count, not reproduced bitwise.
Every rollout remains training_ready=false. No sampling, real network
server, actual Ouro/Inky/Aleph inference, GPU allocation or corpus admission.
