# GB-SF-09: unified Skill Factory researcher bench

The eight native/proxy gyms are researcher-facing experiments in a common
bounded session. GSF-01/02/03/05/07 perform new fits with chosen architectures,
GSF-04 fits additive heads, GSF-06 continues a common baseline, GSF-08 reuses a
fit when only the memory policy changes. This is separate from the earlier
skill-execution tournament and DataGoblin Research Verifiers v2. It is a small
public development suite, not a sealed comparison.

## Action protocol

Every task identifies its gym, objective, allowed experiment choices and
control. Four turns; up to three experiment calls. The first must be the named
control. Each subsequent choice is an allowlisted fit or same-weight cache
experiment. Submit a host-assigned artifact ID and CAL claim.

```json
{"action":"run_experiment","choice":"direct"}
{"action":"run_experiment","choice":"refresh"}
{"action":"run_experiment","choice":"rlt"}
{"action":"submit","artifact_id":"a2","claim":"improved"}
```

Example only; use each task's actual control and CAL receipts. The model sees
only the initial task and refreshed compact artifact table. It cannot submit
arbitrary executable code, paths or claimed scores. Invalid actions consume a
turn. Choose highest CAL quality subject to the raw unsafe count guardrail,
then least inference work, train work, earliest artifact. Claim concerns
quality and unsafe-count comparison to the control, not final outcomes.
The final score uses the selected artifact's final-panel quality only when
claim, CAL selection and final guard all pass and no invalid action occurred.
All measurements and failed experiments remain recorded.

Work units are gym-specific, so compare within a gym. Across gyms report an
equal-family macro score and each separate score/ledger; do not compare sums
of heterogeneous work. GSF-04 parameter counts differ. GSF-07 uses native
call count for cost ties; GSF-08 uses cold verdict-issuance counts. Neither
implies measured end-to-end latency. GSF-06's reference session is the common
interface over the underlying matched-continuation implementation.

## Development and trace-train roles

Default one task per gym per role; deterministic public indexes 0..3 are
available. New GSF-01/02/05 worlds use separate derived seeds, GSF-06 uses
research roots with offset 100, GSF-07 uses new CAL/trace group seeds,
GSF-03/08 share the exact original CCA state partition, and GSF-04 uses
disjoint semantic tag pools. Trace research uses separate calibration and
final panels inside designated trace lineages for the finite CCA and persona
tasks; the persona trace panels are not a distribution-shift claim. In
development, final panels remain held from the researcher during each episode
but are public and already source-accessible. No sealed inference claims.

```sh
python -m research_gym.benchmarks.goblin_skill_factory.goblinbench --selftest
python -m research_gym.benchmarks.goblin_skill_factory.goblinbench --out /tmp/bench-dev
python -m research_gym.benchmarks.goblin_skill_factory.goblinbench --audit /tmp/bench-dev
```

Interactive evaluator (one JSON object per stdin line, one response per line):
```sh
python -m research_gym.benchmarks.goblin_skill_factory.goblinbench --session GSF-03 development 0 --out /tmp/gb-session
```

Only the evaluator owns the output directory. The final episode receipt and
selected artifact are written there after submission; stdout contains no final
panel outcomes. Use a trusted subprocess, external RAM and wall caps, and
keep the evaluator filesystem inaccessible to the proposing model. The
session is a bounded protocol, not a sandbox. Fitted parameters and exact
receipt IDs are host generated. Reference records are replayed by refitting;
the default 16 scripted sessions produce eight trace-train research episodes.
No actual model endpoint is invoked by the CPU workflow.

## Cross-gym staging

After all nine CPU jobs succeed, the intake job downloads their artifacts,
reruns each native audit, verifies counts and source records, separates
public observations, observed execution and evaluator-only targets, and
groups lineage through a union-find. The shared CCA state IDs link GSF-03
and GSF-08, semantic tag IDs link persona menus, graph and candidate hashes
catch repeated synthetic inputs, and GB-SF-09 CAL/final lineage binds its
research episodes. Known CAL/dev/shift lineage is excluded. Alternate
views across seeds, arms and policies are not independent roots.

```sh
python -m research_gym.benchmarks.goblin_skill_factory.intake --artifacts /tmp/gsf-artifacts --out /tmp/gsf-intake
```

The nine artifact names are pinned in intake.py. The full fixed development
campaign contains 14,554 native records plus eight scripted GB-SF-09
research episodes (14,562 candidate records). These counts are planned for
the staging job until it passes. Output is compressed normalized envelopes,
a held-lineage registry and a hash-bound intake receipt. Every row remains
training_ready=false and the admitted count is zero. Per-turn rendering must
respect time order; do not place future public observations, evaluator targets
or oracle-only teacher actions in prompts. Oracle-only MCP references are
marked separately. No private Jinn/Beast labels or weights are staged.
Native audits validate the stated synthetic host contracts, not real-world
policy safety. Source IDs/semantic hashes are conservative grouping keys;
they are not a proof that all semantic variants have been discovered.

## Distill research form

Use the actual public task and CAL receipts, with a separate response form:
hypothesis and measurable failure; proposed change and matched control;
expected effect and cost; observed CAL delta and unauthorized-proposal count;
choice of checkpoint with a falsifiable claim; conditions for preferring the
exact solver. Bind each form to the research root, artifact and receipt hash.
Do not include hidden final labels in the teacher's observation. Independently
grade structured actions and keep free-text explanations as unverified
rationales until reviewed. Actual live model rollouts and corpus tokenization
are separate from this staged artifact.
