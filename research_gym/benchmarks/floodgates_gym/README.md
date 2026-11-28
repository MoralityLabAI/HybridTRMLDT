# GSF-10 / FG-01: Floodgates-style self-speculative decoding

Public CPU development fixture inspired by Suleymanzade et al.,
[Looped Models are Fast Multi-Token Predictors](https://arxiv.org/abs/2611.33896v1),
25 November 2026 (CC BY 4.0). Upstream
[code and adapters](https://github.com/MisakiTaro0414/floodgates).
Our companion research note is in
[LoopedTransformers/papers/floodgates](https://github.com/MoralityLabAI/LoopedTransformers/tree/research/floodgates-20261128/papers/floodgates).

## Research hypothesis and protocol

A four-pass tied causal attention/MLP target is trained on synthetic
token sequences. The teacher observes a random eight-token prefix and
learns to repeat its final one or two tokens, with the period determined
by the first token. Shift uses a period of three (an intentionally
out-of-contract task). Ground-truth continuation quality and exact
greedy equivalence to the trained target are separate metrics. A bad
target can be decoded perfectly losslessly.

Train two targets, final-only versus uniform all-depth next-token
supervision, with paired initialization, batch draws, unique parameters,
updates and block-token work. Freeze each target. Train a learned mask
vector and draft-only LoRA on its Q/V/output projections, using greedy
target-generated streams on independent adapter-training roots. The
training cache slices are strictly before the last committed anchor.
Test a matched rank-4 first/deep/null prefix comparison (null retains
the same attention shape with zero prefix KV) and a rank-8 deep arm.
The null control retains positional access to prefix length, and is
not equivalent to removing the prefix positions from softmax.

Each draft runs the shared block once, on an anchor token followed by
1, 2 or 4 mask slots. The native full-depth verifier reads its own
per-loop prefix caches; it never reads adapter-modified cache entries.
The draft uses a read-only first-loop or final-loop cache (or null
control), causal slot attention and a frozen readout. This is a small
Floodgates-inspired variant, not an exact upstream reproduction.

A single batched causal verification processes the anchor and draft.
Accept the matching prefix and append the verifier's next token. On
rejection, truncate provisional KV to the committed prefix excluding the
new last token. Keep that last token as the next anchor. Physically
clone the retained slices to avoid retaining rejected suffix storage.
Selftests force every mismatch position, full acceptance and a capped
generation boundary, and compare rollback to freshly computed cache.
Every ordinary cached greedy token is checked against cold forward
evaluation. Require exact generated IDs on every speculative rollout.
Use tolerances only for numerical cache/logit diagnostics, never token
identity. Adapter fitting and decoding must leave target weight hashes
unchanged.

Default CPU campaign: width 32, 4 loops, vocab 11, 240 base updates and
180 adapter updates, batch 16, 256 base-training roots, 96 distinct
adapter-training roots, 24 each CAL/dev/shift/trace-train. One seed:
two targets, eight adapters, 192 planned trace episodes. CAL selects
draft length from 1/2/4 by lowest verification calls, then lowest total
block-token work, then smaller length. This selection optimizes a
structural surrogate and does not guarantee latency speedup.
No final-panel selection. Repeated arms are paired views of a root.

## Receipts

Report emitted tokens per verification, target calls, accepted prefix
lengths, draft and verification block-token visits, discarded cache
positions, provisional target KV tensor bytes, training work and true
continuation accuracy. Prefill is recorded separately; wall time is a
diagnostic and excludes prefill, audit cold forwards and training.
The serial denominator counts one ordinary target call per generated
token. Token-block visits measure a structural cost; they do not
include the quadratic attention term, cache copies, adapter work,
head cost, memory bandwidth or allocator overhead. Do not call a
reduction in serialized calls an end-to-end speedup.

Save frozen target and adapter state_dicts, training receipts, CAL
selection and trace-train-only records. Every record contains proposed
drafts, verified accepted prefix, corrections, output parity, exact
synthetic target and work. The replay audit loads saved checkpoints,
regenerates split roots, recomputes CAL selection and replays every
trace with cold serial parity. All records are training_ready=false.
The public deterministic corpus is unsealed; no corpus admission or
native language-model performance claim.

```sh
python -m research_gym.benchmarks.floodgates_gym.gym --selftest
python -m research_gym.benchmarks.floodgates_gym.gym --smoke --out /tmp/fg-smoke
python -m research_gym.benchmarks.floodgates_gym.gym --audit /tmp/fg-smoke
python -m research_gym.benchmarks.floodgates_gym.gym --out /tmp/fg-dev
python -m research_gym.benchmarks.floodgates_gym.gym --audit /tmp/fg-dev
```

CPU, one PyTorch thread. Use a machine-specific RAM/wall cap externally.
No model download or accelerator allocation. CPU CI pins Python and
Torch and uses the portable replay environment from GSF-01–09.

## Successors

1. Add independent seeds and a stronger positive-control synthetic task.
   Add marginal and conditional per-position acceptance, and a competent
   released Ouro target with upstream draft adapter as a reference.
2. Test elastic drafting, rank and mask-block size at measured work,
   cache transport interventions and depth-supervision differences.
   Sampling requires separate distribution-preservation tests.
3. Add BPD-guided low-rank subspaces with rank-matched random/shuffled
   controls, using training-only predictive probes.
4. Test native Aleph: preserve prelude/coda, map per-physical-layer
   final-loop K/V, verify rollback and output parity of the frozen
   native checkpoint. The 10M pilot is only a plumbing reference.
5. Register the bounded researcher-facing choice of draft cache, rank,
   block length and refinement budget in GoblinBench and stage traces
   under its existing held-lineage protocol.

The current GSF-10 branch deliberately leaves the verified GB-SF-09
source pin and its 14,562-record staged artifact unchanged.
