# GSF-01 — GoblinBench Skill Factory

Public synthetic development campaign. CPU only. This is a small actual PyTorch
learner suite, not a reproduction of MoLT-KV, full TRM, HRM or an LLM agent.
The first stage supplies typed four-candidate skill-template contracts;
integration with existing SKILL.md documents and native service calls is next.

## Gym roster

| Gym | Task | Exact host |
|---|---|---|
| Repair | Pick positive-net-value current verified repair | Authorization, current receipt, veto, gain minus cost |
| Retention | Pick useful current scoped verified evidence | Current epoch, scope, verification and veto |
| Persona | Rank a permitted action by external preference | Independent authorization/scope/veto and preference ordering |
| Meta-repair | Improve a trained refresh controller with equal additional updates | Common checkpoint and fresh optimizer; CAL-selected arm evaluated separately |

All three primary tasks are finite static candidate arrays. Retention is a
receipt-selection proxy: it does not yet simulate sequential invalidation or
KV storage. Persona uses synthetic policy tags, not private Jinn/Beast data.

## Loop tournament

All primary arms have identical parameter count and initialization per seed.
Direct uses one cell visit; residual uses four with encoded input at the first
visit only; refresh uses four with encoded input supplied at each visit.
RLT interpolates the refresh proposal with alpha = 1/(4-i); damp uses the
constant mean of the nonterminal RLT coefficients and the same terminal update.
The 4-visit arms have matched update counts, data order, cell work, width and
supervision. Direct is a lower-work baseline, not a matched-compute claim.
For each fitted recurrent arm, choose a confidence threshold using CAL only,
then measure actual early stopping on dev and shift. This is a MoL-inspired
depth policy, not learned per-token MoLT or a KV cache implementation.

For meta-repair, branch the trained refresh weights into three equal-work
continuations (refresh, RLT, matched damping), reset AdamW in every arm,
and pick the arm by CAL raw accuracy and unsafe rate. Keep the outcome of all
three fits and never select using dev/shift. This tests a repair procedure;
a learned meta-controller that chooses the intervention is a successor task.

The exact solver is a necessary reference; a learned head is worthwhile only
if measured deployment costs or transfer justify it. Record raw accuracy,
guarded accuracy, unsafe proposals, training and inference cell visits, CPU
time, parameter counts and checkpoint hashes separately. The hard LDT-style
gate is host-owned; it can abstain but cannot turn a wrong allowed action
into a correct action.

## Run

From repo root with a CPU PyTorch installation:

```sh
python -m research_gym.benchmarks.goblin_skill_factory.gym --selftest
python -m research_gym.benchmarks.goblin_skill_factory.gym --smoke --out /tmp/gsf-smoke
python -m research_gym.benchmarks.goblin_skill_factory.audit /tmp/gsf-smoke
python -m research_gym.benchmarks.goblin_skill_factory.gym --out /tmp/gsf-dev
python -m research_gym.benchmarks.goblin_skill_factory.audit /tmp/gsf-dev
```

Use a fresh output directory. CPU, one PyTorch thread, default width 32,
two seeds, 100 updates per primary fit, 25 per meta-repair fit, 512 fit cases
and 128 cases in each other partition. The 30 primary and 18 continuation
fits generate 6,144 candidate trace records on separate rollout roots.
These counts are planned until run and audited. A supervisor should impose
machine-specific hard RAM and wall limits for local execution.

Outputs: report.json, actual state_dict artifacts, and train_rollouts.jsonl
with per-visit action/confidence, exact final receipt, repair target, weights
hash and per-record digest. The independent audit recomputes the target,
verifies the artifact and rollout inventory hashes, checks CAL-only selection
and expected root coverage. It is not a proof of model execution integrity
against a malicious producer. Public deterministic splits are development
only, not sealed. Export contains only trace_train roots, disjoint from
fit/CAL/dev/shift, with training_ready=false. Repeated views across arms
must be grouped by root during DataGoblin intake; no raw failures as positives.

## Successor gyms and migration order

| ID | Source | Required next experiment |
|---|---|---|
| GSF-02 | MOLT-HYBRID-01/02; DataGoblin MOLT-03/04 | Sequential receipt bank, epoch invalidation, delayed query, refresh vs carry vs reset |
| GSF-03 | Existing ControlTRMProposer; MCP-08 | Native repair-template execution, multiple attempts, verifier receipts and learned stopping |
| GSF-04 | Dual-LDT / Jinn or Beast | Private specialist import, typed scoped owner preferences, independent Beast authorization and Jinn ranking |
| GSF-05 | HRM / LSA / residual TRM | Matched-clock hierarchy, pass-schedule swaps, recurrent state interventions |
| GSF-06 | RIVET / RLT | Propose-test-repair episodes choosing real fit budgets, with negative outcomes |
| GSF-07 | MCP-06/07/08 | Geometry-aware retrieval, stale/revoked evidence, query costs |
| GSF-08 | DCF-15/18, certificate-directed search | Reuse vs refresh vs exact fallback, certified anytime regret |

For GSF-02 onward, reuse the existing native fixtures and adapters rather
than copying their semantics into new stand-alone proxies. Freeze data and
resource bounds before training, measure symbolic and learned controls,
retain failures, run a fresh external held panel before promotion.

## GoblinBench intake

Use an adapter in DataGoblin Research Verifiers v2 that supplies an existing
skill template, training partition, allowed fit knobs and a bounded fit
tool. The researcher proposes a fit, examines calibration metrics, chooses
or repairs a checkpoint, submits a weight artifact ID, and the host scores
fresh roots. This is distinct from the current neural-skill tournament.
For post-training export, include public observations, chosen experiment,
actual hyperparameters, verified result, work, failure/repair pairs, and
counterfactual comparisons. A Distill research form can ask why the selected
intervention is justified by calibration and when the exact solver is better.
No oracle-only data, evaluation roots, private policy text, or model-written
success claims enter accepted targets without a separate custody audit.
