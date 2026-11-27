# MOLT-GOBLIN-02 preregistered comparison

Registered 2026-11-28 before the discovery campaign started. Status: development study.
Code: `research_gym/molt_goblin_02/` on `research/molt-goblin-02-20261128`.

## Question

Which loop-memory and loop-routing mechanisms improve exact task success of a tiny
recurrent model (width 32, two heads, four loops, 300 AdamW updates of 32 sequences)
at fixed training data, and at what cost in recurrent visits, CPU time and retained
memory? Standard MoLT memory mechanisms (`memory`, `topk`, `rank`, `window`,
`conf_bias`, `loop_keys`) and heterogeneous module routing (`routing`, `n_types`,
`route_hard_eval`, `slow_period`) are separate configuration axes.

## Arms

Matched baselines (`panel.BASELINES`), all with identical data, seeds, optimizer,
update count and loop count unless named otherwise:

| Arm | Definition |
| --- | --- |
| `tied` | Plain tied recurrence: causal self-attention over the current state, input injection |
| `prev` | Previous-loop-only recurrence: query current state, memory previous state at j<i |
| `prefix` | Existing MoLT-Goblin prefix memory: query input embedding, memory previous state at j<i (reference) |
| `prefix_all` | `prefix` with uniform loop-forcing loss over every loop |
| `one_pass` | One loop (data-matched, 1/4 of the forward block calls) |

Candidates come from Inky Mini proposals compiled through `factory.MoltConfig`
(width ≤ 48, loops ≤ 6). Each proposal names a discriminating control that differs in
the claimed mechanism; both are trained.

## Development panel (discovery only)

`panel.PANEL`: pointer_chase (delayed retrieval, depth shift, shuffled-definition control),
revision (obsolete-memory revision, longer horizon, temporal reordering), repair
(misleading candidate repair, more errors, zero-error matched control), proof_deps
(dependency tracking, deeper DAGs). Development roots: `train.DEV_BASE`. Training roots:
`train.TRAIN_BASE`. The proposer sees family descriptions, specs, configurations and
aggregate scores only.

Competence calibration (seed 11, 300 updates, 256 development roots;
`receipts/calibration/calib_panel_s300.jsonl`), exact success:

| Train spec | prefix | tied |
| --- | ---: | ---: |
| pointer_chase L6 D2 X2 | 0.121 | 0.117 |
| revision L16 D1 X2 | 0.906 | 0.879 |
| repair L8 D2 unflagged (dropped: copy floor) | 0.000 | 0.000 |
| repair L6 D1 flagged (registered) | 0.324 | 0.289 |
| proof_deps L14 D3 | 0.578 | 0.598 |

Competence floor: an arm whose panel-mean exact falls below the `one_pass` baseline is
recorded as not competent and cannot be a finalist on that basis alone (it still enters
the archive). Revision is near ceiling in distribution; its shifted specs carry the signal.

Successive halving: rung 0 = seed 11; rung 1 = seeds 11, 23; rung 2 = seeds 11, 23, 37.
Score = mean over panel families of (exact − prefix exact) + 0.5 × (OOD exact − prefix OOD exact).
Adversarial task variants from Inky enter as extra development evaluations only after
`gym.verify_spec` confirms generator/solver agreement, causal labels, sound admissibility
masks and a constant-predictor exact rate below 0.5.

## Freeze and confirmation-style readout

At the campaign deadline `finals.freeze` writes `freeze.json` (with SHA-256) naming the
top three candidates by (highest rung, score), every baseline, and a parameter-matched
single-block `prefix` control for the top finalist. Only then are arms trained and
evaluated on hidden roots (`train.HIDDEN_BASE`, never used by the campaign) for:

* the four panel families,
* withheld families never shown to the proposer: bounded_walk, experiment_select,
* controls: shallow_copy (recurrence should give little benefit) and cumsum (v1 fixture task).

Seeds: 11, 23, 37 (FINAL_SEEDS[:3]); 256 hidden roots per evaluation.

Amendment (2026-11-28, during discovery, before any freeze or hidden-root run): the
campaign frontier moved to soft HRM routing over four typed modules, which evaluates
every module on every loop. `finals.freeze` therefore also adds a visit-matched control,
`prefix_all` with loops = module visits of the top finalist (capped at 16). The
validation range for `loops` was raised to 16 for this arm; the discovery loop keeps
its own limit of 6. Because every rung-2 candidate uses loop forcing (`loss=all`), the
parameter-matched control is built from `prefix_all` (arm `prefix_all_param_matched`)
so it differs from the top finalist in routing and width only.

## Measures

Exact task success (primary); success per recurrent visit; success per train CPU-second;
retained memory bytes per example; peak process working set; routing entropy and
routing/loop mutual information; useful (wrong→right), harmful (right→wrong) and
redundant loop transitions; halting Brier score and ECE for continue decisions; depth
extrapolation (OOD specs, and OOD specs at 2× loops); robustness under memory reset,
shuffled loop history, stale memory swapped from another example, and Gaussian memory
noise; BPD-style subspace interventions (loop updates projected onto the fitted top-k
subspace vs a random k-subspace vs the complement).

## Primary contrast and inference

Primary: each frozen finalist minus `prefix`, exact on hidden roots, per panel family,
paired by training seed. Independent unit for claims about the architecture: training
seed (n = 3). Root-level paired bootstrap intervals are reported as conditional on the
trained checkpoints. Meaningful effect floor: 0.05 exact. Comparisons: 3 finalists ×
4 panel families primary; withheld families, controls and interventions are secondary.
No multiplicity correction is applied; with n = 3 seeds the study is a development
screen and its results set the next tournament's preregistration, not a promotion.

## Resource envelope

CPU only (`CUDA_VISIBLE_DEVICES=-1`), one worker process, torch threads = 2, incremental
RAM cap 3 GiB (Windows Job Object process-memory limit plus RSS watchdog). Inky Mini via
the free `thinkingmachines/Inkling-2-Mini:peft:65536:free` id only; 12k-token default
context, 16k ceiling, completion cap 3,072 tokens, three bounded retries per request.
Campaign wall-clock budget 5.5 h; finals budget 2.0 h; total under the 8 h deadline.
