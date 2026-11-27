# GSF-02: delayed-query MoLT-style receipt retention

Status: executable CPU development experiment, synthetic scalar bank; no neural K/V cache.
Uses the existing `molt_hybrid.epoch.ScalarBank` for exact incremental evaluation and
checks each update against independently computed dependency invalidation and cold query evaluation.

## Task

A six-node strict-prefix dependency DAG over Z/17Z is observed with the current inputs and
a public distribution over four next events: no-op, input edit, edge edit and new context.
Each event has a conditional delayed-query distribution. Choose exactly two entries before the
realized event and query are supplied. The host retains only those two usable receipts;
the full scalar bank belongs to the evaluator. Receipt authorization checks root, node,
context and affected-node membership before reuse. Missing or stale entries fall back to
an exact query-local cold traversal. The policy never controls authorization.

The utility is saved query-local node evaluations; receipt lookups, initial bank cost,
refresh and independent host verification are reported separately. This is a structural
cost proxy, not a wall-time or end-to-end efficiency claim. Only direct query hits save
work: partial ancestor reuse is outside this experiment. Context switches conservatively
invalidate every receipt. Local edits invalidate the descendant closure in the union
of the old and new graphs. Even coincidentally unchanged values do not authorize stale reuse.

For each candidate entry i, compute the exact expected saving as the sum over event e
of P(e) P(query=i | e) * cold_cost(e,i) * 1{i unaffected by e}. Choose the two largest
values with deterministic index tie-breaking. This exact additive solution is a ceiling
for the stated utility. Query-frequency ranking and a uniform fixed-seed random policy
are controls. The realized event and query are pseudorandom functions of the root in
this public fixture; their indices are withheld from learned input features. It is not
a hidden test against a source-aware agent.

## Neural comparisons

Five actual PyTorch arms with a shared encoder, recurrent cell and readout:
direct one-visit, residual carry, input refresh, RLT, and matched constant damping.
All recurrent arms execute four visits, have equal parameter count, identical initial
weights, data order, AdamW steps and per-visit supervised expected-value targets.
The direct arm is a lower-work comparator. The model predicts six expected savings;
choose the top two. No inference-time access to the exact scorer by neural policies.
No claim that this reproduces MoLT attention, TRM or HRM.

Default: 2 seeds, 256 fitting worlds and 64 each CAL/dev/shift/trace-train per seed.
Shift increases graph density and edit/context frequency. 10 neural fits and 6 rule
cells; 1,024 candidate trace records on 128 dedicated trace roots. CAL is diagnostic
only in this fixed-policy stage. No policy selection on evaluation partitions.

Run:
```sh
python -m research_gym.benchmarks.goblin_skill_factory.molt_memory --selftest
python -m research_gym.benchmarks.goblin_skill_factory.molt_memory --smoke --out /tmp/molt-smoke
python -m research_gym.benchmarks.goblin_skill_factory.molt_memory --audit /tmp/molt-smoke
python -m research_gym.benchmarks.goblin_skill_factory.molt_memory --out /tmp/molt-dev
python -m research_gym.benchmarks.goblin_skill_factory.molt_memory --audit /tmp/molt-dev
```

Each trace includes public forecast, per-visit masks/scores, post-action event/query,
exact receipt, expected regret and independently derived correction. The audit regenerates
the public packet and outcome, verifies artifact and receipt hashes, loads the fitted weights
on CPU and replays every action. The native bank is checked against independent closure and
cold recursion on every episode; selftest enumerates every mask, event and query for 60 worlds.
All records remain `training_ready=false`; train-only roots are distinct from fit/CAL/dev/shift.
Use expected rather than hindsight-optimal mask for correction. Group alternate arms and
outcomes by root during intake. The deterministic public split is development-only, never
a sealed benchmark.

## Successors

Introduce a genuine sequential multi-query eviction policy and query-conditioned
partial ancestor reuse before attempting neural KV adaptation. Then expose a bounded fit
interface to GoblinBench: compare exact vs learned policies, choose loop budgets and explain
regret/cost on CAL. Keep held evaluation separate, preserve negative results, and use
Distill forms to supervise research decisions rather than merely copying exact answers.
