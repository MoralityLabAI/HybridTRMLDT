# MOLT-HYBRID-01: prefix-memory TRM with an LDT membrane

Status: design and development-only fixture. Source: https://arxiv.org/abs/2611.34923.
Companion: DataGoblin and LoopedTransformers branch `research/molt-goblin-01-20261126`.

## State and boundary

Use the existing HybridProposal / HybridDecision and MembranePolicy contract.
Let a_t be the public typed LDT candidate state, h_{i,r} the private recurrent workspace
for token/event i at pass r, and M_{<i,r} a versioned memory bank of earlier events.
TRM proposes q and a halt/branch request. Jinn can rank admissible alternatives. Beast or
an environment-owned verifier rechecks action identity, evidence digest and epoch; unknown,
stale or unverified evidence fails closed for hard state changes.

For one frozen public-state epoch e:

    h[i,r+1] = F_theta(x[i], a[e], M[<i,r], gate[i, <i, r])
    K[i,r+1], V[i,r+1] = P_theta(h[i,r+1])
    gate[i,j,r] = G_theta(x[i], a[e], K[j,r], V[j,r], public_receipt[j]) for j < i
    q = ProposalHead(h[i,r+1], a[e])
    decision = independently_verify(q, current_epoch=e)
    a[e+1] = meet(a[e], certified_refinement) only on acceptance

G may depend on stable prefix state and frozen parameters. It must not use same-position
previous-pass state, future state, global softmax across forbidden edges, changing pass
index, or dynamic a[e+1] within the same frozen epoch if finite stabilization is claimed.
Normalize over allowed prefix entries and a null slot. If a hard refinement changes a[e],
start a new epoch, invalidate affected bank receipts and revalidate. Keep certified lattice
updates separate from cached latent suggestions. Recurrent TRM with self feedback is a
separate arm without the finite stabilization claim.

## Proof obligation and compaction

Given a fixed deterministic map at each position and a strict-prefix DAG, the longest
dependency path plus one bounds stabilization from arbitrary initial states. A full
strictly triangular Jacobian is nilpotent at a fixed point, and products of T possibly
different strictly lower triangular T-block Jacobians also vanish. Large finite products
before that bound are possible. This gives a useful dependency-local cache invalidation
rule. A proof says nothing about competence, conditioning, stochastic training or
the cost of materializing memory. A reduced representation must preserve exact
membrane-relevant observable decisions before it can replace public certified state.

## Stage 0 and 1

A small dependency-free fixture (prefix_memory.py) checks parallel versus topological
fixed-point computation, strict-prefix structural validation, and exact changed-prefix
descendants. This is a scalar symbolic witness, not a transformer.
Build a neural adapter against LoopedTransformers only after the development smoke.

Compare: typed LDT alone, TRM residual, prefix-KV TRM, prefix-KV+LDT soft ranking,
prefix-KV+certified membrane, BPD-selected prefix components with a matched sham,
and exact symbolic oracle. A no-memory control and matched random eviction control
are mandatory. Keep public lattice semantics identical; count actual executed
actions, not only predicted permissions. Separate fixed-length and length-shift tasks.

Task surfaces:
- delayed constraint chains and graph reachability with an exact solver;
- LBF proof candidate retention, with native Lean checking on a later adapter;
- MCP index/query routing with immutable tool/receipt identities and stale-response tests;
- Jinn preference arrays inside Beast admissibility, using only approved public synthetic traces;
- Hadamard partial sequence constraints, where admissibility checks are exact and
  promising trajectories receive additional compute.

Endpoints: verified closure, verifier calls, utility per wall time, exact lattice replay,
invalid action acceptance, false holds, cache invalidation errors, loop-depth curves,
peak RAM and GPU memory, token/visit cost, and root-level paired uncertainty. An
unauthorized hard update is a structural failure. A reject-all policy fails utility.
Use EvalMath mechanistic power calibration with actual independent trained roots and
matched intervention doses before any causal claim. Keep test labels and future seeds
with the steward.

## DataGoblin connection

Export public task IDs, candidate arrays, predicted cost intervals, graph and receipt
digests, verified outcomes, and refusal/failure status. Private h/K/V or sensitive
tool contents are excluded by default. Train an array proposer/sorter on train-only
records; operator approval is required before executing a resource-consuming job.
Stage 2 is a future native research_gym adapter and version-pinned Verifiers environment.

## Resource hypotheses

Sweep 2, 4, 8, 16 GiB allowed *total* RAM with fixed CPU threads (the bank budget is the
remaining headroom), then repeat on the 3050/5080 under separate caps. Track hit rate,
bytes moved, serialization and invalidation, stale reads, and verified successes per hour.
Caching calibration receipts and frozen features is safe only for the matching version.
Training-dependent K/V may be reused inside one autograd graph; a cross-update cache
must be recomputed or explicitly treated as an approximation with a fresh-cache control.

No production promotion, training, upstream implementation reproduction or closed-loop
performance result is established by this proposal.
