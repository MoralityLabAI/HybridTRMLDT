# GSF-05: loop-clock and schedule discovery

This CPU development gym tests the LSA register/write and schedule-word axes.
It is inspired by the repository's [LSA-02 typed calculus](https://github.com/MoralityLabAI/LoopedTransformers/blob/c42044c5ea0927382dd910a173ef9142a8d49d02/docs/lsa_typed_architecture_calculus_v0.md) and the fast/slow registers in Conductor-HRM. It is a graph-local recurrent MLP, not the full HRM-Text, MoLT or Aleph implementation.

## Task contract

An eight-node world contains two independently sampled permutations A and B, a
start node, and an explicit traversal program. Chain tasks use one operator;
composition tasks use two operators in either order (one-hop composition
examples degenerate to chain queries). The learner sees graph topology,
start indicator, both repetition counts and order. The exact solver traverses
the specified permutations; a separately implemented full-permutation
composition checks the target. Labels and intermediate exact path remain
outside the neural input.

Fit on lengths 1–4; CAL and development also use 1–4. Shift uses lengths 5–8.
Each root has a fresh graph pair. Public deterministic roots are development
only, and a graph-pair overlap across splits within a seed fails the run.

## Architecture arms

All arms have exactly the same initialized encoder, two separately learned
physical blocks A/B, and readout. Each block reads the fixed input embedding,
a writable local register, the other register and the two incoming graph
messages. The flat arm aliases the registers; each coupled arm updates only
one register at a visit. Readout uses both registers. All fit for eight ticks
with identical parameter count, batch order, update count, full gradient
horizon and final plus auxiliary loss.

| Arm | Word | Writes |
| --- | --- | --- |
| flat_ab | ABABABAB | One shared register |
| coupled_ab | ABABABAB | A fast / B slow register |
| coupled_grouped | AABBAABB | A fast / B slow register |
| hier_fast | AAABAAAB | A fast / B slow register |
| hier_reverse | BAAABAAA | A fast / B slow register |

The first two isolate register topology at fixed word. The second and third
hold operator visit counts fixed and change their order. The last two compare
fast/slow phase placement at matched A6/B2. Across the two frequency groups,
the distribution of applied parameters changes, even though total ticks and
unique parameter counts match. This is not an isolated test of frequency
independent of learned specialization. The physical block names are distinct
from the symbolic graph operators: each neural block sees both graph edges.

## Adaptive computation and runtime swaps

For each fitted model, choose a confidence threshold from the fixed grid on
CAL, minimizing actual ticks with no more than a one-point accuracy reduction
from its eight-tick result. The threshold is then frozen. Actual early exit
terminates the execution loop; no full unroll is hidden in inference. Four-tick
prefixes are a lower-compute diagnostic. Schedule swaps on the identical
checkpoint change AB to grouped or fast to reverse and vice versa, preserving
A/B counts within each pair. No selection uses dev or shift. Confidence
thresholding is a basic adaptive-depth policy, not learned per-token routing
or a compute-matched randomized exit comparator.

Report correct answers, accuracy by family, executed ticks, node visits,
graph-message gathers, exact solver hops and CPU time including grading.
These are different cost units; do not interpret equal tick counts as proven
equal wall time, and do not call the exact symbolic solver a neural arm.
The chance reference is 1/8 under uniform guessing.

## Run and custody

Default: width 24, 300 updates, batch 32, 512 train and 64 each CAL/dev/shift/
trace-train per seed, two seeds. Ten planned fits, 1,280 fixed/adaptive
rollout records over 128 dedicated roots. Trace records include the chosen
word, actual visit sequence, predictions, confidence, verified final answer,
exact path (post-action grading only), and work vector.

```sh
python -m research_gym.benchmarks.goblin_skill_factory.loop_clocks --selftest
python -m research_gym.benchmarks.goblin_skill_factory.loop_clocks --smoke --out /tmp/clocks-smoke
python -m research_gym.benchmarks.goblin_skill_factory.loop_clocks --audit /tmp/clocks-smoke
python -m research_gym.benchmarks.goblin_skill_factory.loop_clocks --out /tmp/clocks-dev
python -m research_gym.benchmarks.goblin_skill_factory.loop_clocks --audit /tmp/clocks-dev
```

CPU only, one Torch thread; enforce external RAM/wall limits on local runs.
Audit reloads every checkpoint, verifies schedule identity, regenerates
CAL threshold choice and every exported public world, then replays actual
visits and exact grading. Training records remain training_ready=false;
fit/CAL/dev/shift are excluded from the export. Public seeds are not sealed.
A failed competence or adaptive-depth comparison remains valid research
evidence; do not tune the reported panel.

## Researcher-facing successor

Expose architecture words, register topology, objective sites and compute
caps to GoblinBench as bounded experiment choices. GSF-06 can ask the
researcher to use CAL evidence to diagnose failure, choose a schedule swap,
continue from identical weights with a matched repair budget, and evaluate
its intervention on fresh roots. Distill should capture testable hypotheses,
work accounting and reasons for preferring symbolic traversal when warranted.
