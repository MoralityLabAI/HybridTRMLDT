# GSF-08: certified anytime search and bounded verdict memory

Public CPU development using native CCA-05 Storyworld semantics, CCA-09
anytime search, CCA-10 optimistic count envelopes and the GSF-03 repair
learner. This is a new trained action orderer with an exact host cache,
not a Lean proof cache or cross-context certificate transport.

## Task

Each root is one nonterminal Storyworld state. Six requests visit four
possible contexts (two scenarios crossed with two horizons). Development
and trace roots use horizons 3/6; shifted roots use horizons 1/2. The
request generator publishes weights for the next context, with a hot
context and repeat preference. Shift changes the hot context over time.
The realized future requests are withheld from each decision.

Every query has at most three verdict requests. The CCA-09 search verifies
all action upper bounds, evaluates candidates in the proposed order, and
maintains a certified incumbent and regret gap. Moral optimization uses
the CCA-10 count-envelope upper bound; secret ending uses the global
bound 1. A zero gap certifies optimality. No available incumbent is not
proof of emptiness unless the native search certifies it. The evaluator
separately checks the selected action and regret against the exact oracle.

## Neural ordering and memory

Train direct, refresh, RLT and matched damping policies with the existing
GSF-03 utility and masked-action objective. All learned arms have the same
initialization, examples, steps and width. Direct executes one visit;
the other arms execute four visits. A fixed lexical action order is the
non-neural comparator. The same frozen ordering policy is crossed with:

| Memory | Rule |
| --- | --- |
| cold | No verdict store |
| lru | At most six verified verdicts, evict least recently used |
| forecast | At most six verified verdicts, evict lowest next-context forecast weight, then oldest |

The forecast is a symbolic baseline, not a learned eviction policy.
All modes must produce identical anytime decisions for each fixed order,
root and query sequence. Cached answers cannot expand the search budget.

The trusted in-process store admits a verdict only after checking it.
Keys bind environment digest, scenario, state, horizon, and action.
The seal checks content consistency, not external authentication.
An exact positive verdict on a hit still replays its witness. A negative
verdict on a hit is authorized only by the trusted exact-context store.
Exact utility and negative admissibility are independently checked at
admission; they are not recomputed on every hit. This relies on immutable,
deterministic semantics and a host-owned store. Context changes prevent
reuse unless the previous exact context is requested again. Stale, foreign
action, forged seal and forged upper-bound tests must fail closed.

Report verdict requests, cold issues, cache hits, cache binding checks,
positive witness checks/steps, negative exact checks, exact utility checks,
upper-bound calls/envelope terms, evictions, peak entries and neural visits.
Native CCA semantics already use memoization; these are counts of API-level
work and structural terms, not measured cold wall-time savings. Upper-bound
construction/checks and final grading are separate from the verdict cache.
The native witness checker and utility oracle share the deployed environment;
this is not an independently formalized game engine.

## Campaign and custody

Default width 24, 100 updates, batch 32, two model seeds, 256 fitting
state roots, 64 roots in each evaluation/trace partition, exact GSF-03
state partition and root identifiers. Eight neural fits and 30 policy/cache
cells. Each cell sees the same 64 roots in each panel; planned 1,920
trace trajectories / 11,520 query records over 64 independent trace state
roots. Seeds and cache/order variants are paired views, not independent
examples. GSF-03 and GSF-08 share root custody and must be grouped on intake.

The trace contains current observation and forecast, order, scores,
bounds, checked verdict/witness events, incumbent/gap, realized query
sequence, exact evaluator grade and work vector. Successful trajectory
labels require every query to have a certified optimal or empty result,
not merely zero hindsight regret. Unresolved searches are kept as repair
targets. The audit regenerates worlds, checks source/partition/checkpoint
hashes, replays every exported trace and validates cache parity. Every
record remains training_ready=false. Public roots and source are accessible;
no sealed evaluation or production deployment is claimed.

```sh
python -m research_gym.benchmarks.goblin_skill_factory.certificate_search --selftest
python -m research_gym.benchmarks.goblin_skill_factory.certificate_search --smoke --out /tmp/cert-smoke
python -m research_gym.benchmarks.goblin_skill_factory.certificate_search --audit /tmp/cert-smoke
python -m research_gym.benchmarks.goblin_skill_factory.certificate_search --out /tmp/cert-dev
python -m research_gym.benchmarks.goblin_skill_factory.certificate_search --audit /tmp/cert-dev
```

Successor: learn an eviction or retrieval policy with matched capacity,
compare cross-context witness transport separately from exact-utility
reuse, and expose these experiments as bounded GoblinBench actions.
