# GSF-06: bounded research and checkpoint meta-repair

Status: public CPU development. This is a RIVET-inspired propose/test/repair
protocol over the existing GSF-01 skill-module learner, not a reproduction
of a pinned RIVET implementation. The initial researchers are scripted
policies. The interactive JSONL interface supports later external LLM
researchers; no model endpoint is called by this campaign.

## Research task

The host trains a real small refresh-TRM checkpoint on one of three skill
contracts (repair, receipt retention, persona ranking). Baseline training
uses a registered uniform, abstention-skewed or veto-underexposed sampling
regime. These are explicit exposure interventions, not corrupt labels.
All baseline checkpoints are included; no outcome-based selection of
underperformers. The public observation reports the skill semantics,
baseline regime and fit receipt, failure count on fit roots, calibration
counts, per-pass correctness, and raw unsafe proposal count.

The researcher has four turns and at most three experiments. It must run a
matched ordinary continuation first. The remaining two slots can test:

| Intervention | Controlled change |
| --- | --- |
| control | Ordinary refresh-TRM continuation |
| rlt | Scheduled interpolation at recurrent updates |
| damp | Constant mean-matched interpolation with terminal replacement |
| failure_replay | Replace half of each batch with baseline-failure examples |
| final_only | Supervise the final visit rather than all visits |
| residual | Remove repeated input injection after the first visit |

Each continuation starts from the exact same checkpoint, resets AdamW
and executes the same optimizer-update count, batch size and four cell
visits per example. Ordinary arms have identical batch draws. Failure
replay uses baseline errors on the fitting partition only. Its additional
screening work is charged as a shared baseline diagnostic. Cell visits
are a structural work unit; objective computation, sampling and wall
time need not be identical. No continuation is initialized from the
result of another experiment. This isolates interventions at matched
checkpoint and work without requiring expensive accelerator allocations.

After each experiment the host returns actual fit and CAL receipts plus
remaining budgets. The researcher can submit the baseline or any
experiment artifact. The claim is relative to the matched continuation
on CAL: improved, equivalent, or regressed. Raw unsafe count must not
increase; maximize mean balanced raw quality across normal and stress
panels; prefer unchanged weights on exact ties, then the control. Within
each panel balance viable-action accuracy and correct abstention when
there is no viable action (average the represented strata). Calibration
and final stress panels use separate seeds. The final score is the selected
artifact's independently graded balanced quality if the CAL claim and
selection are correct, there were no invalid actions, and the final unsafe
count does not exceed the control. Otherwise the score is zero. Retain
quality delta, guardrail failures, work and negative results separately.
The exact finite solver always remains the deployment reference.

## Protocol

```json
{"action":"run_experiment","intervention":"control","hypothesis":"ordinary_continuation"}
{"action":"run_experiment","intervention":"rlt","hypothesis":"scheduled_interpolation"}
{"action":"run_experiment","intervention":"damp","hypothesis":"constant_interpolation"}
{"action":"submit","artifact_id":"a2","claim":"improved"}
```

The last action is an illustrative claim, not a predetermined correct
answer. Use the returned CAL receipts. Hypothesis labels are structured
experiment intentions, not verified explanatory prose. No arbitrary code,
paths, models, hyperparameters, or final-label requests are accepted. Invalid
actions consume turns; duplicate experiments are rejected. An exhausted
or abandoned episode with no submission scores zero.

Interactive local harness:

```sh
python -m research_gym.benchmarks.goblin_skill_factory.research_repair \
  --session repair abstain_skew 0 --private-receipt /tmp/gsf06-local.json
```

Send one JSON action per line, read one response. The observation never
includes final evaluation metrics. The evaluator receipt and the selected
state_dict (if submitted) are written separately; keep their paths outside
the proposer's filesystem. The process is a local reference, not a security
sandbox. Public source and seeds are accessible; a sealed evaluator must
hold its roots, exact labels and result storage separately. The JSONL runner
bounds input-line size, but use an external timeout and RAM limit and a
trusted transport for production. A malformed or timed-out action in a real
agent adapter must be recorded as a failure, not omitted.

## Reference campaign and post-training export

Five policies compare control-only (lower cost), fixed RLT/damping pair,
fixed failure-replay/final-only pair, seeded random matched budget, and
a deterministic diagnostic policy that inspects the public baseline.
Four matched-budget policies spend three experiments; control-only spends
one. All see the same root-level checkpoint and data. The diagnostic policy
is a heuristic, not a trained researcher.

Default: two roots per family/regime/split, 36 independent research roots,
180 reference episodes, 90 trace-train episodes (234 continuation fits on
the trace partition), 192 fitting examples, 64 per normal or stress CAL/final
panel, 24 baseline updates, 32 updates per experiment, batch 32, width 24.
Two meta splits have disjoint roots. The evaluator verifies independent
labels and rejects duplicate world content across inner roles within a
root. Alternate policies for one research root must remain grouped.
All trace-train records remain training_ready=false. Their observations,
experiments and final grading are explicitly separated; development metrics
are stored in dev_results.jsonl, excluded from research_rollouts.jsonl.
The audit deterministically refits each exported root and all attempted
experiments, checks receipt identity and reproduces aggregate trace metrics.
Development aggregates are custody-checked but not refitted by this audit.
No final test cases or their item-level answers are exported.

```sh
python -m research_gym.benchmarks.goblin_skill_factory.research_repair --selftest
python -m research_gym.benchmarks.goblin_skill_factory.research_repair --smoke --out /tmp/research-smoke
python -m research_gym.benchmarks.goblin_skill_factory.research_repair --audit /tmp/research-smoke
python -m research_gym.benchmarks.goblin_skill_factory.research_repair --out /tmp/research-dev
python -m research_gym.benchmarks.goblin_skill_factory.research_repair --audit /tmp/research-dev
```

This is the first researcher-facing action contract in the Skill Factory.
The next integration should make equivalent bounded adapters for GSF-02
through GSF-05, connect the local model's JSON action protocol, and route
Distill forms to the observable hypothesis/experiment/decision records.
The reference traces are synthetic supervision candidates. They are not
teacher LLM output, confirmed research aptitude, or permission to train
on evaluation roots.
