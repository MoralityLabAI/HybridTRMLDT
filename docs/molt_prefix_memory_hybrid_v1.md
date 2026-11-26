# MOLT-HYBRID-01: prefix memory with a certified LDT membrane

Status: development design with a scalar reference fixture. The neural bridge and
epoch/cache adapter described here remain proposed work. The linked paper is
https://arxiv.org/abs/2611.34923; this repository contains an independent
source-inspired design. Reproduction of the upstream implementation is outside
the current fixture's scope.

Companions on `research/molt-goblin-01-20261126`:

- DataGoblin: `docs/molt_goblin_gym_v1.md` and the task generator/scorer.
- LoopedTransformers: `experiments/molt_goblin_v1/`, containing a neural prefix fixture.
- BPD: `mechinterp/bpd/experiments/molt_prefix_memory/PROTOCOL.md`.
- Compactification-Theory: `research/molt_prefix_memory_v1/README.md`.

## The question this hybrid should answer

Test whether retaining useful earlier computations helps a small proposer finish
verified tasks at a fixed compute budget. The explicit lattice keeps the current
admissible choices inspectable. Learned memory supplies candidate refinements and
helps choose which admissible branch deserves the next verifier call.

The first useful result would be an improvement in verified completion or verifier
cost on fresh development worlds, accompanied by exact replay of the public state.
A high refusal rate cannot substitute for task completion. Any accepted hard update
without an independent certificate fails the structural contract.

## State and authority

Use the existing `HybridProposal`, `HybridDecision`, and `CandidateState` types.
The strict entry point is `certify_and_apply_verified` in
[`research_gym/core/hybrid.py`](../research_gym/core/hybrid.py). It requires an
environment-owned verifier, rejects missing or inconclusive checks, and requires
strict lattice progress by default. The historical `certify_and_apply` path allows
a claimed-provenance fallback when its optional checker returns `None`; reserve
that behavior for replaying experiments that registered it.

| State | Owner and lifetime | Permitted role |
| --- | --- | --- |
| Public candidate state `a[e]` | Environment; immutable during epoch `e` | Defines admissible actions and the state against which proposals are checked |
| Workspace `h[i,r]` | Proposer; event `i`, reasoning pass `r` | Produces suggestions, confidence, and branch/halt requests |
| Prefix memory `M[j,r]` | Versioned model cache; `j < i` for a read by `i` | Supplies earlier features and candidate evidence references |
| Verification receipt | Independent checker; bound to a specific action and state | Supports the exact property checked, within its recorded scope |

Jinn can rank choices within the candidate set authorized by Beast. A preference
score has no authority to eliminate a candidate. The same restriction applies to
a latent probe or a proposer-authored `env_sound_dead` claim.

### Frame boundaries

Monotone refinements share one environment-owned problem/frame version and its
valid premises. A source update that retracts earlier evidence, changes the world
interpretation, or changes the candidate universe requires a new frame. The environment
initializes that frame's lattice from the assumptions that remain valid and links
it to the earlier frame through provenance and root lineage. Monotonicity is checked
within a frame; states from different frames have separate candidate universes.
The current `CandidateState` contract rejects mismatched domain keys.

Cache invalidation removes stale computations. An earlier certified elimination
also needs semantic repair when its premise expires, so clearing the cache cannot
restore the eliminated candidate. Opening a new frame performs that reset and
requires fresh checks before any old receipt supports a hard update. Preserve
related frame versions under the same independent root for statistical accounting.

## One frozen epoch

Let `x[i]` encode an event and its immutable inputs. Freeze the model parameters,
public state, graph, and read policy for the duration of epoch `e`:

```text
h[i,r+1] = F_theta(x[i], a[e], M[<i,r], gate[i,<i,r])
K[i,r+1], V[i,r+1] = P_theta(h[i,r+1])
gate[i,j,r] = G_theta(x[i], a[e], K[j,r], V[j,r], receipt[j]), j < i
q = ProposalHead(h[i,r+1], a[e])
decision = certify_and_apply_verified(
    a[e], q, provenance_verifier=epoch_checker, verifier_context=epoch_context
)
```

The checker context binds the proposal to the current action identity, frame and
problem version, public-state digest, epoch, input evidence, and checker version.
It checks the proposed semantic transition through environment mechanics. A digest establishes which bytes were
checked; a matching digest alone supplies no evidence that their claim is correct.

Run the epoch as follows:

1. Snapshot the public state and validate the strict-prefix dependency graph.
   Build an immutable epoch record and open the resource meter.
2. Fill missing eligible entries, then run the registered number of reasoning
   passes. The mask includes a null slot so an event with an empty prefix has a
   defined read. Normalization includes only permitted entries and that slot.
3. Emit a typed proposal with its evidence references. Verify it against the
   snapshot, then check monotonicity and the existing bottom/abstention rules.
4. If the proposal is rejected, retain the reason and spent budget. Another pass
   may continue under the same frozen inputs. A changed read policy or refinement
   under the same valid premises starts a new epoch. A retracted premise or changed
   candidate universe opens a new frame before reasoning resumes.
5. On an accepted strict refinement, commit `a[e+1]`, close the old epoch, and
   invalidate affected memory. Revalidate queued proposals against the new state
   before any further hard update.

Each synchronous pass reads an immutable bank for pass `r` and writes a separate
bank for pass `r+1`, followed by an atomic swap. Charge both live banks to peak RAM.
In-place topological evaluation is a separate schedule with its own visit and
latency accounting.

The first adapter should serialize commits. Concurrent proposers can submit
against one snapshot, but each later commit needs a fresh check after the state
changes. The default strict entry point rejects no-ops. If a later registered arm
allows one, its acceptance is recorded separately and contributes no progress.

## Prefix safety and cache semantics

The finite-stabilization claim requires a fixed deterministic local map. Each
event may read its own immutable input and earlier events' changing workspaces.
It cannot read its own previous-pass workspace, any future workspace, a changing
pass counter, or the next public state inside the same epoch. An ordinary residual
TRM arm is retained as a separate comparison with its own recurrence semantics.

A gate may change numerically as its allowed prefix inputs change. This still
defines a fixed map when the gate function and tie rule are frozen. A pass-dependent
temperature, changing eviction policy, or history-dependent lookup can change the
map itself. Record those variants as separate algorithms.

For the first exact-cache adapter, freeze which entries are eligible during an
epoch. An evicted value can be recomputed through the same map and charged to the
budget. Substituting a null value or a lossy reconstruction changes the computation
and belongs in an explicit approximation arm. Apply LRU or learned eviction choices
between epochs unless a later proof covers their within-epoch behavior.

Proposed record fields, to be implemented at the adapter boundary:

| Record | Required fields |
| --- | --- |
| `EpochSpec` | `root_id`, `task_id`, `split`, `frame_id`, `problem_version`, `parent_frame_id`, `premise_digest`, `epoch_id`, `public_state_digest`, `input_digest`, `graph_digest`, `checkpoint_digest`, `gate_digest`, `gate_version`, `read_policy_digest`, `verifier_digest`, `checker_version` |
| `MemoryEntry` | `epoch_key`, `event_id`, `pass_index`, `parent_ids`, `dependency_digest`, `source_input_digest`, `payload_digest`, `tensor_shape`, `dtype`, `bytes`, `created_at_update`, `evidence_refs`, `validity` |
| `ProposalEnvelope` | `epoch_key`, `action_id`, `proposed_state`, `evidence_refs`, `requested_mode`, `checkpoint_digest`, `spent_budget` |
| `EpochReceipt` | `epoch_key`, `before_digest`, `after_digest`, `proposal_digest`, `checker_result`, `executed_action`, `rejection_reason`, `resource_totals` |

`epoch_key` binds all fields that determine the computation, rather than just an
integer counter. Tensor payloads stay private by default. An evidence reference
points to an environment-owned receipt with a property and scope that the current
checker understands.

These are proposed adapter-envelope fields; existing core and fixture JSON keys
stay unchanged. The mathematical index `e` maps to within-frame `epoch_id`, and
`theta` is bound by `checkpoint_digest`. Digests bind content or a canonical
configuration. `problem_version`, `gate_version`, and `checker_version` remain
human-readable version labels; they accompany their content bindings.

### Invalidation rules

| Change | Required invalidation |
| --- | --- |
| Input or local map at event `k` changes | `k` and every descendant in the declared dependency graph |
| Edge set changes | Changed local maps and their descendants in the union of old and new graphs |
| Shared model weights or gate function changes | Every affected activation entry; the default is the whole bank |
| Checker or external evidence changes | All dependent receipts and proposals; open a new frame if accepted deductions lose their premises |
| Public lattice state changes | Every event whose local map reads a changed field, then all of its descendants |

The formula above gives every event access to `a[e]`, so its conservative default
is whole-bank invalidation after a public-state change. Local reuse requires a
declared per-event projection of `a[e]` and a checked dependency index. Copying an
old entry into a new epoch is allowed only after establishing that its entire
dependency signature is unchanged.

For example, with parents `[[], [0], [], [1], [2]]`, editing event `0` affects
`{0, 1, 3}`. Events `2` and `4` remain reusable under unchanged parameters and
unchanged public inputs. The existing `invalidate` fixture checks this case.

## Training and supervision boundaries

Keep the incoming TRM/HRM adapter's registered detach and supervision schedule.
Treat the following indices explicitly in logs:

| Index | Meaning | What changes at its boundary |
| --- | --- | --- |
| `i` | Event within one task | Prefix eligibility |
| `r` | Reasoning pass | Private workspace values |
| `e` | Public-state epoch | Snapshot, proposals, and cache validity |
| `u` | Optimizer update | Parameter version and all dependent activations |

Within one autograd graph, recurrent K/V reads can preserve the registered gradient
paths. At a scheduled detached carry boundary, detach exactly the state specified
by the original adapter and record that event. Extending the gradient horizon or
adding supervision at every cache write creates a new training arm.

An optimizer step changes the meaning of cached activations. Recompute them before
claiming a fresh-cache result. Cross-update reuse belongs in a named stale-cache
arm paired with a fresh-cache control; it has different gradient and inference
semantics. Immutable task inputs and verifier receipts can persist when their own
version and scope still match.

Keep training tokens, optimizer updates, and active token-block visits visible as
separate costs. Compare data-matched arms first. A compute-matched comparison needs
its own schedule because adding reasoning passes changes work per example.

## Task sequence and comparison plan

Start with delayed constraint chains. The environment reveals public facts over
several events; the model proposes candidate eliminations, and an exact solver
checks each elimination. Include an irrelevant-event distractor and a later source
change that opens a new frame and invalidates a previously useful cached path.
Score the whole episode so a locally plausible step that destroys eventual closure receives the appropriate
failure outcome.

| Surface | Main question | Checker and next dependency |
| --- | --- | --- |
| Constraint chains and reachability | Does memory reduce repeated search while retaining valid candidates? | Exact finite solver; first native adapter |
| LBF proof candidates | Does retaining a useful branch reduce proof-search calls? | Native Lean replay; requires a later LBF adapter |
| MCP index/query routing | Can the model reuse a valid receipt and reject stale replies? | Immutable tool/action identity plus environment-owned receipt verification |
| Jinn preference arrays | Can ranking improve utility inside the Beast candidate set? | Public synthetic traces and exact admissibility checks |
| Hadamard partial sequences | Does retained search state help allocate a fixed visit budget? | Exact partial-constraint checks; utility defined before the run |

Run a compact architecture comparison before adding BPD selection or a large RAM
sweep:

| Arm | Purpose |
| --- | --- |
| Typed LDT with a fixed proposer | Measures the value available from the explicit state alone |
| Residual TRM with the same membrane | Tests conventional recurrent workspace |
| Prefix-KV proposer with the same membrane | Isolates the prefix recurrence and memory change |
| Single-pass proposer with the same membrane | Establishes the cost and utility of one reasoning pass |
| Exact symbolic solver | Supplies attainable task answers and a separate cost reference |

All learned arms receive the same public information and checker interface.
Match parameter budgets or report their difference explicitly. Once a prefix arm
shows usable competence, add no-memory and matched random-eviction controls, then
compare soft ranking with certified hard refinement. BPD-selected components enter
only after reconstruction parity and the registered intervention controls pass.

Report verified closure per root, actual verifier calls, exact public-state replay,
invalid hard updates, false holds, loop-depth curves, wall time, and peak memory.
Count attempted and executed actions separately. Report in-distribution and length
shift performance separately so a pooled mean cannot hide either behavior.

Use EvalMath mechanistic power calibration before a causal claim. Pair task-world
comparisons by independent root and report trained-model seed variation separately.
Repeated passes, intervention doses, and episodes from the same world do not create
additional independent worlds. Development results determine the next preregistered
design; the steward retains later test labels and future seeds.

## RAM experiment at a fixed CPU budget

Use proposed total process-tree caps of 2, 4, 8, and 16 GiB, with the same registered
CPU quota, thread count, task stream, model, and software versions in every cell.
The model, activations, runtime, checker, and reporting process all consume that
cap. Compute the bank allowance from measured remaining headroom. A configuration
that cannot fit is recorded as infeasible at that cap.
Each cell must fit its registered host envelope. Larger caps use the existing
operator path; this proposal leaves DataGoblin's 2,048 MiB mock limit in force.

For each feasible cell compare fresh recomputation with version-correct reuse.
Keep a matched random-eviction condition to separate capacity from a learned
retention policy. Record bytes resident and moved, lookup/serialization time,
recomputed events, invalidation work, hit rate, and verified successes per hour.
Include the RAM used by copies and pinned buffers. A thread setting accompanies an
enforced resource wrapper; it does not by itself establish a CPU quota or RAM cap.

Start with frozen inference. Test training-dependent reuse after the cache receipt
and invalidation checks pass. The 3050/5080 comparison remains a later experiment
with separate device memory caps and host/device transfer accounting.

## DataGoblin handoff

DataGoblin receives public task IDs, graph and input digests, candidate arrays,
predicted cost intervals, and outcome receipts. Its train-only export can teach a
proposer to select a promising admissible array and explain an abstention. Preserve
aborted runs and verifier failures so its training distribution includes costly
mistakes.

The operator-owned executor applies the registered resource limits to a proposed
job. The scorer joins predictions to steward-held labels. Private h/K/V values,
sensitive tool contents, and held-out answers stay outside the ordinary export.
Any later activation dataset needs an explicit scope and versioned receipt.

## Implementation checkpoints

| Checkpoint | Deliverable | Acceptance evidence |
| --- | --- | --- |
| Existing scalar fixture | `prefix_memory.py` and its small-graph tests | Synchronous/topological equality, invalid-edge rejection, exact descendant sets |
| Epoch adapter | Typed envelopes plus strict membrane integration | Missing checker, stale epoch, altered evidence, and lattice-widening proposals fail before execution; accepted transitions replay exactly |
| Neural bridge | Prefix-KV proposer behind the same interface | Prefix mask and neural fixed-point checks; registered gradient/detach behavior; whole-episode receipts |
| Cache adapter | Version checks, exact recompute path, dependency index | Cached and fresh runs produce the same certified trajectory; injected stale entries are rejected |
| Development comparison | Matched architecture runs on fresh development roots | Competence above the registered trivial baseline, cost accounting, independent-root uncertainty, and recorded failed cells |
| Later extensions | BPD intervention, native LBF, version-pinned Verifiers environment | Each extension passes its own source, parity, and preflight requirements before broader claims |

The existing dependency-free smoke command is:

```bash
python -m unittest research_gym.benchmarks.molt_hybrid.test_prefix_memory -v
```

## Mathematical scope

For a fixed deterministic map on a finite strict-prefix DAG, define
`d[i] = 1 + max(d[j] for j in parents[i])`, with the empty maximum equal to zero.
Synchronous updates stabilize from arbitrary initial values by `D = max(d)`.
Topological evaluation computes that same fixed point directly. Under smoothness,
products of `D` Jacobians with the same DAG support vanish even when evaluated at
different states; finite products before that bound can still amplify perturbations.

These are this fixture's indexing conventions: the same map applies from the first
update, and `D` counts updates from an arbitrary `r=0` state. The paper's Appendix
B.1 treats the initial pass separately and states its position bound as `t+2` for
zero-based position `t`. The paper also allows same-position keys/values from the
current loop's depth computation while blocking same-position previous-loop
feedback. Our strict-prefix block supplies a simpler independent comparison. Its
bound and training schedule must be reported under their own definitions.

The separate Compactification-Theory note develops the proof and its limitations.
The scalar fixture is a finite implementation witness. Competence, numerical
conditioning, stochastic training behavior, and the cost of materializing memory
require their own evidence. Exact compaction must preserve the public decisions
and their verification semantics through a commuting representation map. A small
reconstruction error alone does not meet that obligation.

This document establishes a proposed build sequence. Neural training results,
native adapter performance, production promotion, and formal proof completion
remain open work.
