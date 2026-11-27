# GSF-04: Dual-LDT persona externalization

Two public, synthetic additive policy teachers over 24 audited tags supply
three-lane probability targets. Fit actual additive heads with 300 minibatch
updates, comparing separate external heads (75 parameters each), pooled head
(75), and a shared head with persona-dependent bias (78). The pooled and bias
arms have smaller capacity; comparisons do not isolate architectural separation
from parameter count. Uniform and exact synthetic teacher heads are controls.
The parameter-matched comparison for future work is an equivalent frame-gated
block-diagonal linear head. No private source weights or policy text enter CI.

Use the native `JinnHead`, `sort_with_jinn`, `BeastSnapshot` and
`verify_execution` interfaces. Every public menu has five candidate actions.
Evaluate four phases: initial policy, host tightening (forbidden tag and
authority denial), swap persona while keeping the tightened authorization set,
and a possible HALT. Independent host evidence may have conflicting authority,
missing or ambiguous semantics. The host owns evidence, policy and receipts.
The fitted heads score only preference; they never expand authorization. Verify
every execution, old receipt rejection, same-head surviving odds under tightening,
and independent recomputation of the authorized set. Persona swap does not
change the host-authorized set. This is a finite local contract, not a claim about
real-world control, semantic extraction or authenticated host channels.

Metrics: tag-classification accuracy, mean synthetic teacher log-probability
regret, conditional total variation, viable optimal decisions, raw unauthorized
top choice, unsafe executions, old-receipt rejection. No claim about objective
moral preference. Exact teacher has zero regret by construction. Fit only on
train-partition 4-tag combinations; CAL/dev/trace use different tag sets;
shift uses 5-tag combinations and increased ambiguous/conflicting evidence.
A canonical tag-set hash assigns the partition before menu construction.
Menu roots group every seed/arm/phase; shared tag sets across menus within a
split create additional semantic dependencies. Public tests are development
only. Default: six actual fits, ten policy cells, 1,280 four-phase trace records
on 128 menu roots (5,120 phase records). These are planned counts until CI.

```sh
python -m research_gym.benchmarks.goblin_skill_factory.persona --selftest
python -m research_gym.benchmarks.goblin_skill_factory.persona --smoke --out /tmp/persona-smoke
python -m research_gym.benchmarks.goblin_skill_factory.persona --audit /tmp/persona-smoke
python -m research_gym.benchmarks.goblin_skill_factory.persona --out /tmp/persona-dev
python -m research_gym.benchmarks.goblin_skill_factory.persona --audit /tmp/persona-dev
```

The replay audit regenerates menus, verifies original artifact bytes and
canonical head records, replays all phases, and checks every tag-set split.
Trace records include the public synthetic evidence, selected action, target,
probabilities, receipt, execution status invariants, and regret. Every record
is marked training_ready=false pending DataGoblin intake.

## Private Jinn/Beast lane-head training

Use DataGoblin's existing hash-pinned specialist importer at
[PR #81](https://github.com/MoralityLabAI/DataGoblin/pull/81), including the
source revision and prospective held-root registry. Keep its output private
and outside any Git checkout. This runner consumes the staged train-only
bundle rather than the raw private source.

```sh
python -m research_gym.benchmarks.goblin_skill_factory.persona_private --intake /private/jinn-specialist-v2 --out /private/gsf04-jinn-beast
```

It verifies the intake inventory and known source pins, independently
recomputes exact rational policy labels, checks tag/root custody, excludes
derived views and original held roots, and creates local train/CAL/dev/trace
partitions of the admitted original training roots. Fit the same three
architecture choices to the supplied hard lane labels. Keep all fitted
records, source-derived features, traces and reports private; no CI upload,
public redistribution or corpus admission. The importer receipt is an
operator-supplied authority binding, not a signature. The private runner's
synthetic grammar selftest can run publicly; no actual private fit is claimed.
A full private replay auditor and operator-approved execution remain separate.

The private source policy frames are classification targets. Their learned
lanes must never become the host's execution-authority predicate. Report
critical raw misclassification separately; preserve the native hard floor
in any subsequent deployment.

## GoblinBench integration

Expose the externalization experiment as an actual bounded training action:
select head structure, supervised objective, fit budget, and persona-swap
policy. Compare calibration results and exact fallback before final submission.
Ask Distill for justified experiment choices and analysis of failed
interventions, then grade the proposed artifact on untouched roots.
Owner preference-event storage and authenticated revisions remain separate
extensions of this frozen-head persona swap test.
