# FG-09: researcher-facing GSF-10 architecture optimization

Prospective CPU-only public development task. The prior Floodgates campaign
[46103073481](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/46103073481)
passed five jobs, including FG-07's 336 controller and FG-08's 960 serving
rollouts. First-loop single-pass drafting and a cheap repetition heuristic
are strong controls. Serving timings depend on the hardware and batch
fragmentation; the initial research task uses a reproducible structural
single-request tariff, not elapsed time or a claim of serving acceleration.

## Common research interface

Use the existing GB-SF-09 Session action grammar and CAL selection rules.
Task gym is GSF-10, with a separate module so the previously verified
eight-gym source pin and staged corpus remain unchanged. Four turns,
up to three experiments; required first choice fixed_first4, followed by
up to two distinct choices. Submit an artifact ID and improved/equivalent/
regressed claim relative to the required control on CAL. Invalid actions
consume turns. Final metrics are evaluator-only until submission.
A reference policy runs fixed_first4, heuristic and cost_regression, then
chooses by CAL quality. No-op control is a format/action-budget selftest.

Each episode starts with a fresh final-only four-loop width-32 target and
two ordinary rank-four first/deep single-pass draft adapters, fit by
GSF-10, then frozen. Setup fits are the same for all experimental choices
within one root. New generator indices >=50000 with disjoint role/index
blocks; 256 target-fit, 96 adapter-fit, 32 controller-fit, eight CAL,
eight normal and eight shifted final requests per episode. The two
research roles (development/trace_train) use disjoint prefix contents;
four indices per role are supported, default one per role. No sealed
suite; public source and deterministic roots are accessible.

Choices: ordinary serial, fixed_first1/2/4, fixed_deep1/2/4, fixed
first/deep two-pass length four, fixed first/deep cycle-RLT four-pass
length four, heuristic, imitation, cost_regression. Each policy performs
actual greedy decoding on CAL. Learned choices fit the FG-07 width-32 MLP
on distinct controller-fit roots (160 updates, batch 32). Their offline
counterfactual labels use FG-07's exact prefix dynamic program, never
CAL/final answers. Construct the training table only on the first
learned choice and reuse it for the second; record the extra cost and
controller training exposures. Fixed choices incur no fit updates.
No free full-menu CAL sweep. The model gets aggregate measured outcomes
for requested choices only.

## Objective and custody

For each panel measure actual online decoding work:
  C = (4 * serialized_shared_block_calls + token_block_visits)
      / emitted_tokens.
Quality = 20/(20+C), so serial quality is 0.5 and lower cost is strictly
better. This bounded monotonic score is gym-specific. Select highest CAL
quality, then lower tariff, then fewer controller training exposures,
then earlier artifact. Final score is mean panel quality when CAL
selection and claim are valid, every output has exact greedy parity,
and no invalid action occurred. Claim concerns CAL quality only.
Output truth accuracy is a separate diagnostic; the researcher
cannot change target predictions. Controller MACs, reference checking,
common setup and offline label work remain explicit separate ledgers.
No global cost unit sums incompatible optimizer work, structural tariff
and actual time. No measured latency-based selection in this task.

Each experiment yields a host artifact bound to target, adapter and
controller identities. For local interactive execution, save target and
adapter state_dicts, selected policy descriptor and controller weights
as applicable, final receipt and weight inventory outside the checkout.
Trace record contains public task, chronological action/CAL receipts,
exact final grade, source/root lineage and training_ready=false.
Audit refits the reference sessions, reconstructs all requested
experiments, replays exact greedy decoding and checks receipt hashes.
No oracle future IDs in model observations. Dev and trace panel
lineage must be disjoint. Native GSF-10 episode tapes and this researcher
task remain separate from the prior 14,562 staged records until a
new explicit corpus admission review.

Bridge: use the same newline JSON protocol as GB-SF-09 with a separately
pinned GSF-10 evaluator module. Keep the original suite pin and role
unchanged; no model endpoint allocation or inference in CI.
