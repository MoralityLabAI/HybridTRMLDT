# FG-07: learned budget allocation for exact greedy self-speculation

Prospective public CPU development protocol. FG-05 and FG-06 passed
[run 46099110333](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/46099110333).
The guided adapter advantage over Haar was small; extra elastic passes
failed to pay for their work. Keep ordinary adapters and allow elastic
refinement as an optional action.

## Frozen computation and action set

Use two new GSF-10 final-only synthetic targets (seeds 137, 163) and one
rank-four ordinary adapter per first/deep cache per target. Freeze both
target and adapter weights. Every policy receives the same accepted prefix,
full target cache and generation limit. The controller sees only the
committed token IDs, positional progress and exact repetition indicators;
it does not get draft confidence or any future target ID for free.

Thirteen legal actions:
- ordinary serial target step (no drafting);
- first/deep cache x single pass x nominal block lengths 1, 2, 4;
- first/deep cache x two ordinary draft passes x lengths 2, 4;
- first/deep cache x four cyclic RLT-style draft passes x length 4.

Actual draft length is min(nominal length, remaining token allowance).
All speculative actions use FG-06's exact draft schedule, followed by
GSF-10's frozen full-depth causal verification and physical rollback.
Serial steps run one full target call on the current anchor. All paths
must reproduce ordinary cold-checked greedy decoding exactly. The
controller cannot change target weights, alter verification or commit
an unverified token.

## Offline targets and tariff

A deterministic reference target generates the ordinary twelve-token
continuation. At every possible committed prefix, the evaluator builds
its target cache and exhausts all thirteen actions. Each produces a
positive number of verified emitted tokens and a measured work vector.
The directed acyclic graph advances only along that target continuation.
For this declared structural tariff:

  cost = 4 * serialized_shared_block_calls + total_block_token_visits

the exact hindsight dynamic program is

  V(T)=0;
  Q(t,a)=cost(t,a)+V(t+emitted(t,a));
  V(t)=min_a Q(t,a).

Tie break by canonical action order. Each label is a full vector of
counterfactual cost-to-go values, not a clairvoyant action in the model
observation. DP tables are constructed only by the offline evaluator;
held tables score oracle regret and supply evaluator-only repair labels
on the trace partition. Teacher computation, prefix builds and exhaustive
trials are separately metered; none is represented as online savings.

The tariff makes a serialized block call worth four token-block visits.
This is an explicitly chosen cost proxy, not measured latency, FLOPs or
an A100/5080 cost model. The oracle is optimal only for this action set,
fixed target path and tariff, with per-prefix deterministic transition
semantics. Controller inference and memory overhead are counted separately.

## Controller comparison

Train two equal-size width-32 MLPs with identical initialization, batches,
160 optimizer steps and batch 32 on controller-fit roots: one imitates
the DP's canonical optimum via cross-entropy, one predicts all action
cost-to-go regrets normalized by 100 via mean squared error. At deployment
the latter chooses minimum predicted regret. The network receives only
committed-prefix features (last four token one-hots, first-token one-hot,
four repetition flags and normalized generated/remaining counts).
No trainable state in the frozen decoder.

Compare:
1. ordinary serial;
2. fixed first-cache single-pass length four;
3. best fixed action selected using summed CAL tariff across the full
   declared fixed-action menu;
4. public repetition heuristic (serial for one remaining token, first
   single four on repeated suffix, otherwise first single two);
5. learned imitation;
6. learned cost regression;
7. private hindsight DP (analysis bound only, clearly labeled).

CAL selects the fixed baseline only. Both learned policies use their
terminal trained weights with no CAL hyperparameter selection.
Primary endpoint: summed structural tariff per emitted token on dev and
shift, relative to CAL-selected fixed. Report paired root wins/ties/losses
and mean regret relative to the hindsight bound, serialized calls, token-block
visits, logical prefix reads, draft/verify work, controller calls and dense
MAC counts, actions chosen and exact output parity. No promotion or
claim of statistical superiority from two target seeds.

## Splits and resource bounds

Fresh generator offset 30000, disjoint from FG-01/05/06 default roots.
256 target-training prefixes, 96 adapter-training prefixes, 64 dedicated
controller-fit prefixes (separate adapter_train namespace index offset
31000), 24 each CAL/dev/shift/trace. Validate prefix content disjointness.
Target/adapter fit budgets as GSF-10 (240/180 updates, batch 16).
One CPU thread. Two target fits, four adapter fits, four controller fits,
14 policy cells, 336 planned trace episodes on 24 independent trace roots.

Audit saves and checks all checkpoint hashes, replays controller-fit and
CAL tables, refits the two controllers per target, checks held trace DP
labels and every actual decision with frozen weights, and recomputes trace
summaries. Public online observations and executed actions remain separate
from future output, DP targets and oracle-only results. All traces are
training_ready=false; no sealed model ranking or corpus admission.
