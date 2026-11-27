# GSF-07: geometry-aware MCP retrieval

Public CPU development campaign using the native MCP-02 environment, composer,
frontier, exact judge and certified lattice refinement. The source fixtures are
`MCP/mcp02/world.py`, `env.py` and `agent.py`. No new MCP service or live LLM
endpoint is invoked. Geometry here means a graph of public candidate metadata,
not activation-space BPD or RSITopology geometry.

## Skill and data

The agent sees an open evidence need, already read evidence, remaining tool
budget, and a frontier of public descriptors. It ranks one candidate to expand.
The native runtime owns the actual tool calls, provenance checks, epoch rolls,
revalidation, checked exclusions, and final answer/evidence packet. The private
oracle supplies frontier-usefulness labels during training and the oracle
control; it is not a feature available to learned or other deployable policies.

Train on 32 native train-world groups, with search and PSCB layered-LDT layouts.
Collect oracle and deterministic-rule trajectories, retaining counts of
zero-positive observations (excluded from listwise fitting). Each learner receives
the same training observations, initialization, sampled observation indices and
update count. The listwise objective rewards mass on candidates with a private
oracle-usefulness label; these labels indicate reachable required evidence, not
a unique optimal action or a proof of downstream success.

The evaluation uses 16 groups each in native dev (64 documents) and native test
(128 documents, new template/name pools). A separate 16-group train-name-pool
partition supplies rollout roots. All roots and source hashes are bound in the
report. The two native layouts, seed and arm variants of each world stay grouped.
The shift combines catalog width with native name/template shift; it does not
isolate any one factor. Public deterministic panels are not sealed confirmation.

## Router tournament

Every neural arm consumes the native nine structural features and eight public
lexical, deterministic, provenance and budget features. The graph is a
row-normalized public lexical-overlap/shared-origin affinity with self-loops.
A shared cell updates each candidate using its encoding, latent state and graph
message. The same graph and feature construction is performed in all learned arms.

| Arm | Visits | Graph |
| --- | --- | --- |
| direct | 1 | Identity |
| local | 4 | Identity |
| geometry | 4 | Public candidate affinity |
| shuffled | 4 | Permutation conjugate of the same affinity |
| deterministic_rules | 0 | Native heuristic |
| lexical_bm25 | 0 | Native lexical policy |
| sham_random | 0 | Native random control |
| oracle_upper_bound | 0 | Private frontier-usefulness reference |

The shuffled arm preserves graph size and spectrum but breaks candidate
alignment (trivial graphs can remain unchanged). Local/geometry/shuffled share
weight count and dense message computation, and use matched training updates.
Direct is lower-work. The oracle is an upper-bound-style reference from MCP-02,
not guaranteed to find a globally cheapest successful route.

## Resource and evidence contract

Default: 160 updates, four sampled observations per update, width 24, two model
seeds, 32 calls per episode, 2,048-byte handoff cap, native prefilter k=16.
Evaluate fixed checkpoints also at 16 calls on dev/shift. The handoff limit is
measured in UTF-8 bytes, not tokenizer tokens. The composer is shared across
arms, so this measures routing, not learned answer generation or packet
compression. Native epoch-roll behavior gives a stale-evidence recovery task;
the learned router prioritizes candidates but the composer decides when
revalidation is needed.

Report joint acceptance (correct answer, evidence, packet budget and no
environment failure), evidence recall, failure reasons, family metrics, tool
calls per accepted task, observation/result/handoff bytes, graph preparation
and message pairs, candidate-cell visits and dense multiply/add FLOP counts
(excluding nonlinearities, graph preparation and scalar logic). Do not equate
these structural work measures with end-to-end latency or paid-agent token
savings. A native infrastructure exception aborts the campaign visibly; ordinary
failed/budget-exhausted episodes remain in the denominator.

## Rollouts and replay

Default: eight neural fits, sixteen seed-policy cells, two layouts, 512 planned
trace-train episodes. Each contains the public frontier observations and chosen
indices, separate evaluator oracle-usefulness labels, native decision and tool
tape, answer/evidence receipt, byte/work ledger, root/checkpoint bindings, and
an accepted-packet or retrieval-repair target kind. The private-oracle control
is explicitly marked. Preserve failures; a useful-candidate label is not
equivalent to a successful packet. Keep all records training_ready=false pending
DataGoblin admission. No dev/shift transcripts enter the rollout file.

The audit verifies source hashes, root inventory, weight artifact bytes and
state-dict identity, then regenerates and reruns every trace root through the
native policy and exact judge; timing fields are excluded from exact replay.
It compares trace summaries and verifies coverage. This is public development
replay, not independent private benchmark custody.

```sh
python -m research_gym.benchmarks.goblin_skill_factory.mcp_geometry --selftest
python -m research_gym.benchmarks.goblin_skill_factory.mcp_geometry --smoke --out /tmp/mcp-smoke
python -m research_gym.benchmarks.goblin_skill_factory.mcp_geometry --audit /tmp/mcp-smoke
python -m research_gym.benchmarks.goblin_skill_factory.mcp_geometry --out /tmp/mcp-dev
python -m research_gym.benchmarks.goblin_skill_factory.mcp_geometry --audit /tmp/mcp-dev
```

Next: GSF-08 certified search and receipt reuse, then a researcher-facing
adapter with bounded fit and evaluation actions across all eight gyms.
