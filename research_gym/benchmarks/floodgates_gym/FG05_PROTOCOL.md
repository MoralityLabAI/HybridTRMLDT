# FG-05: output-gradient subspace initialization for masked-block drafting

Prospective CPU development screen, following the one-root GSF-10 pilot
([run 46096326247](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/46096326247)).
The first-loop cache beat final-loop cache in that fixture. Final-only target
supervision gave better ordinary development continuation accuracy. This
experiment fixes final-only supervision and crosses both cache depths;
it uses fresh synthetic prefix roots and two independently trained targets.
It remains exploratory, not a replication of upstream Floodgates or BPD.

## Question

Can the output-gradient geometry of a frozen shared decoder identify
directions that make low-rank masked-token adaptation more efficient?

Inspired by the uncentered label-aligned gradient covariance in
[BPD NC-03](https://github.com/MoralityLabAI/BPD/blob/376e85bfc7aca49f02cc1167208603f414183b8d/mechinterp/bpd/experiments/nc03/PROTOCOL.md).
NC-03 studies a native BPD factor-read intervention and physical operator
equivariance. This gym instead estimates output-space gradients at the Q,
V and output projections of the toy shared block. It does not implement
the BPD decomposition or claim a recovered causal circuit.

For frozen target, public cache mode c, future slot j and projection p,
take the derivative of the teacher-token log probability with respect to
a zero additive perturbation of the projection output. Accumulate the
uncentered second moment of the resulting output-gradient vectors
over discovery prefixes, future slots and projection positions in float64.
Use the leading rank-four eigenspace U_p. Per-slot losses are differentiated
separately; the batch sum is valid because batch elements do not interact.
Discovery data are target-generated continuations from independent
adapter-training roots reserved for discovery. No CAL/dev/shift outcome
participates in choosing directions.

## Matched initialization arms

For each cache depth, train the existing rank-four LoRA Q/V/O adapter:

- ordinary: original random A / zero B;
- guided: zero A / orthonormal B = U_p;
- Haar: zero A / seeded QR orthonormal B;
- shuffled: zero A / coordinate-permuted U_p (same singular values).

All arms start with zero physical adapter update B A, the same learned mask
initialization, equal trainable parameter count, rank, data and update
budget. The primary matched comparison is guided versus Haar and shuffled:
these share the zero-A/nonzero-B convention. Ordinary LoRA is the practical
baseline, with a different factor initialization. Both factors remain
trainable in every arm; no hard subspace restriction is imposed. Report
initial physical zero parity, eigenvalues and boundary eigengap, and
post-fit overlap between the final B-column space and discovered projector.
An unresolved boundary eigengap limits geometric interpretation.

## Fixed campaign

Two seeds 137 and 163. New prefix roots at index offset 10000, disjoint
from the original GSF-10 default panels. 256 base fit, 96 adapter roots
(first 32 discovery, remaining 64 fit), 24 each CAL/dev/shift/trace.
Target width 32, four loops, 240 updates, batch 16; adapters rank 4,
180 updates, batch 16. Each discovery root supplies two cuts, fixed by
seeded PRNG, with four teacher continuation slots per cut. Teacher stream
and full prefix-cache construction costs, gradient probe counts and
adapter fit work recorded separately. Baselines do not get extra updates
in exchange for avoided discovery work, so this is not matched total
end-to-end compute.

Fixed draft length four, shortened at generation boundary, to isolate
initialization. No CAL selection in FG-05. Record CAL as a diagnostic.
Compare mean emitted tokens per verifier call, actual target-call and
token-block work, accepted-prefix counts by horizon (only eligible drafts
in denominators), target truth accuracy, paired root-level deltas, and
zero greedy mismatches. Keep every arm including negative results.
A two-seed result does not justify statistical superiority or native
model speedup. No sampling, elastic drafting or concurrent serving.

Planned 2 frozen target fits, 16 adapter fits, 384 trace episodes,
24 independent prefix roots (repeated across seed, cache, arm).
Artifacts preserve discovery matrices and directions, weights and hashes.
Audit regenerates discovery from the frozen target and source-only split,
checks model/adapter identity, all trace roots and exact greedy replay.
Numerical diagnostics may use tolerance; decoded IDs and acceptance
sequences must remain exact. All records remain training_ready=false.
