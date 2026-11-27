# MOLT-GOBLIN-02 results (2026-11-28)

Branch `research/molt-goblin-02-20261128`. Registered design: [`PREREG.md`](PREREG.md)
(two amendments, both written before the freeze). Everything here ran on one Windows
laptop CPU (torch 2.15.0+cpu, 2 threads per worker, no GPU, no cloud), with Inkling-2-Mini
on the free Tinker id as the only language model. Paid spend: $0.

The evidence falls into three kinds, reported separately:

| Kind | What it covers |
| --- | --- |
| Implementation, verified by tests | Factory, gym, campaign loop, finals harness, analysis, export: `python -m pytest -q research_gym/molt_goblin_02`, giving 53 gym + 37 factory + 4 campaign + 2 analysis tests passing |
| Smoke and development evidence | Stage A reproduction, calibration, and the 400 development-root runs of the discovery campaign (selection-exposed) |
| Completed experimental evidence | 400 frozen-arm runs on hidden roots and withheld families, 5 training seeds, after freeze `2ac88dff` |

## Stage A: baseline reproduction

**Upstream fixture (CPU).** The LoopedTransformers `experiments/molt_goblin_v1` tests pass
(6/6) at `c42044c5`. Study v2 was rerun at 256 updates with 3 seeds
([`receipts/stage_a`](receipts/stage_a/study_v2_summary.md)). Append-only prefix parity
is `max_abs_vs_sequential = 0.0`. At this budget the cumulative-sum fixture stays near
zero sequence accuracy for every arm (token accuracy 0.29–0.56). That fixture is plumbing,
not a discriminating task.

**Parity of the new factory.** `MoltConfig(memory="prefix"|"prev", heads=1, rel_bias=False,
inject=False)` loaded with the pilot's weights reproduces `PrefixReasoner` logits and the
q/k/v/o and embedding gradients to 1e-5. This covers the `prefix_kv`, `residual` and
`one_pass` arms (`test_factory.test_pilot_parity`, `test_one_pass_parity`). The strict-prefix
arm stabilizes at loop count = sequence length and is causal (`test_prefix_stabilizes_and_is_causal`).
The factory is not upstream MoLT: it has no special first pass and no upstream kernels.

**Matched baselines on the new gym** (hidden roots, 5 seeds; exact task success):

| arm | params | train CPU s | visits/ex | panel mean | all-family mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| tied (plain tied recurrence) | 9,474 | 7.9 | 4 | 0.447 | 0.498 |
| prev (previous-loop-only) | 9,474 | 7.9 | 4 | 0.469 | 0.506 |
| prefix (MOLT-GOBLIN-01 prefix memory, final-loop loss) | 9,474 | 7.8 | 4 | 0.467 | 0.510 |
| prefix_all (prefix + loop forcing) | 9,474 | 8.4 | 4 | 0.466 | 0.510 |
| one_pass | 9,474 | 3.1 | 1 | 0.401 | 0.415 |

None of tied, prev and prefix_all differs reliably from prefix in the mean (per-family
seed CIs in [`analysis.md`](receipts/finals_20261128/analysis.md)). Recurrence beats one
pass by 0.10 overall, by up to 0.39 on bounded_walk. Loop forcing changes *how* the loops are
used rather than the final score. Revision exact after loop 1 is 0.23 for prefix and 0.78
for prefix_all. Under memory reset, prefix drops from 0.915 to 0.232 and prefix_all from
0.945 to 0.777. At 2× loops, prefix falls on revision ood_long (0.484 → 0.276) and prefix_all
holds (0.616 → 0.602).

## Gym extensions (Stage C)

Eight exact families in [`gym.py`](gym.py). Each has an independent solver, sound input-only
admissibility masks, a causal-label test and `verify_spec`:

| Requirement | Family / control |
| --- | --- |
| Delayed retrieval of earlier deductions | pointer_chase (k-hop function composition after filler delay) |
| Revision of an accepted hypothesis; obsolete memory | revision (register overwrite; `reordered` temporal control) |
| Recovery from misleading intermediates; repair after verifier rejection | repair (`flagged` = verifier-marked candidate, `unflagged` = misleading candidate, depth 0 = matched copy control) |
| Multi-stage proof dependency tracking | proof_deps (DAG of lemmas, depth cap) |
| Long-horizon planning under bounded memory | bounded_walk (withheld) |
| Selection among expensive experiments | experiment_select (knapsack, withheld) |
| Recurrence beyond training depth | `ood_*` specs, also evaluated at 2× loops |
| Irrelevant history | `distract` filler tokens |
| Corrupted / obsolete memory | model-side `stale_swap`, `noise`, `reset_history`, `shuffle_history` |
| Little expected benefit from recurrence | shallow_copy; repair depth 0 |

Calibration ([`receipts/calibration`](receipts/calibration/calib_panel_s300.jsonl)) moved
repair to the flagged variant: unflagged repair sat at the copy floor (0.000). shallow_copy
saturated at 1.000 for every arm, including one_pass, so it cannot show a difference in
either direction and is uninformative as a control. cumsum stays near floor.

## Discovery campaign (Stage D, development evidence)

The archive is in [`receipts/campaign_20261128/ARCHIVE.md`](receipts/campaign_20261128/ARCHIVE.md).
The campaign ran 4.33 h and 63 proposal iterations. 55 were trained with their
discriminating controls. 8 were recorded failures: 1 empty reply, 7 duplicates of tested
configs. 11 adversarial task requests produced 10 verified variants. The run used 400 CPU
training runs and 93 Inky requests (0 failed; 21 hit the completion cap and were retried at
low effort), with 0.90 M prompt and 0.24 M completion tokens. It was stopped 70 min early
after five consecutive duplicate proposals.

What the loop found, by mechanism (rung 0 = 1 seed, rung 2 = 3 seeds; score relative to prefix):

| Research item | Best evidence | Development outcome |
| --- | --- | --- |
| 1 full loop memory | it0 control, it15, it23, it39, it50 | ≤ 0 alone; on HRM bases it lowers proof_deps by up to 0.27 vs control |
| 2 sparse top-k | it0, it11, it19, it44, it52 | −0.09 to +0.02 |
| 3 learned retention / eviction | it4, it16, it24, it33, it41, it49 | ≈ 0 alone; hurts proof_deps on HRM bases |
| 4 low-rank loop memory | it18, it28, it47, it55 | −0.02 to −0.09 |
| 5 confidence-weighted memory | it1, it21, it34 | small positive on HRM base at rung 2 (+0.098 vs +0.093) |
| 6 error-conditioned recurrence | it14, it42 | +0.005 and +0.026 at rung 0; flat against controls |
| 7 typed loops, fixed cycle (untied depth) | it7, it22, it32, it51 | +0.02 to +0.06 |
| 8 HRM routing among typed loops | it9 | +0.131 rung 0, +0.093 rung 2; hard top-1 routing at eval (it10, it56) collapses to −0.30 / −0.21 |
| 9 LDT-certified pruning | it6, it35, it43, it48 | +0.04 rung 0, −0.015 rung 2 on prefix; no gain on HRM |
| 10 utility halting (training head) | it8, it26, it40 | +0.108 rung 2 when stacked on HRM+conf (best development score) |
| 11 damped / RLT updates | it5, it13, it17, it53 | damped and RLT on HRM: +0.05 to +0.08 rung 2, below plain HRM |
| 12 low-rank update subspace (`update_rank`) | it2, it12, it20, it38, it46, it54 | −0.01 to −0.10 |

Memory-axis mechanisms (1–5) and routing (7–8) stayed separate: every memory change was
tested against a control differing only in that mechanism, on a fixed routing base.

## Frozen comparison on hidden roots (completed experimental evidence)

Freeze `2ac88dff` (written 06:51, before any hidden-root run). The finalists nest:
f2 = HRM routing over 4 typed modules + loop forcing; f1 = f2 + confidence-weighted memory;
f0 = f1 + utility halting head. Controls: `prefix_all_visit_matched`, a single block at
16 loops (= 4 loops × 4 modules), and `prefix_all_param_matched`, width 74 with 46.7 k
params against HRM's 50.6 k. Seeds 11, 23, 37, 41, 53; 256 hidden roots per evaluation;
withheld families never reached the proposer.

| arm | pointer_chase | revision | repair | proof_deps | bounded_walk* | experiment_select* | cumsum | mean (8) | train CPU s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f2 HRM | 0.142 | 0.954 | 0.315 | 0.652 | **0.829** | 0.659 | 0.117 | **0.583** | 33.2 |
| f1 HRM+conf | 0.136 | 0.953 | 0.321 | 0.653 | 0.809 | 0.662 | 0.108 | 0.580 | 34.6 |
| f0 HRM+conf+halt | 0.148 | 0.967 | 0.321 | 0.654 | 0.766 | 0.656 | 0.105 | 0.577 | 36.9 |
| visit-matched | 0.134 | 0.952 | 0.277 | 0.540 | 0.764 | 0.568 | 0.067 | 0.538 | 28.0 |
| param-matched | **0.182** | **0.981** | 0.314 | 0.525 | 0.595 | 0.597 | 0.040 | 0.529 | 12.1 |
| prefix | 0.137 | 0.915 | 0.284 | 0.532 | 0.640 | 0.561 | 0.011 | 0.510 | 7.8 |

\* withheld families. shallow_copy = 1.000 for all arms (omitted).

Paired seed-level differences, f2 minus control (mean [t 95% CI], n = 5 seeds):

| f2 − | proof_deps | repair | experiment_select* | bounded_walk* | revision | pointer_chase |
| --- | --- | --- | --- | --- | --- | --- |
| prefix | +0.120 [+0.024, +0.215] | +0.030 [−0.004, +0.065] | +0.098 [+0.039, +0.158] | +0.189 [+0.028, +0.351] | +0.039 [+0.001, +0.077] | +0.005 [−0.006, +0.017] |
| visit-matched | +0.112 [+0.007, +0.216] | +0.037 [+0.009, +0.066] | +0.091 [+0.029, +0.154] | +0.065 [−0.199, +0.329] | +0.002 [−0.013, +0.018] | +0.008 [−0.010, +0.026] |
| param-matched | +0.127 [+0.011, +0.242] | +0.001 [−0.020, +0.022] | +0.062 [−0.002, +0.127] | +0.234 [−0.087, +0.556] | −0.027 [−0.048, −0.006] | −0.040 [−0.078, −0.002] |

Reading against the registered 0.05 effect floor:

* **HRM routing over typed modules beats prefix memory on dependency and selection tasks,
  and neither extra compute nor extra parameters explains it.** On proof_deps the gain is
  +0.11 to +0.13 over all three references, with every CI excluding zero. On the withheld
  experiment_select family it is +0.09 over prefix and the visit-matched control.
* **On retrieval and revision, width wins.** The parameter-matched single block beats HRM on
  pointer_chase (−0.040) and revision (−0.027), both CIs excluding zero. The HRM advantage is
  family-specific, not general.
* **Confidence-weighted memory and utility halting add nothing at the frozen readout.**
  f2 ≈ f1 ≈ f0 everywhere, so the development edge of f0 (+0.108 vs +0.093) was selection noise.
* The withheld bounded_walk gain over prefix (+0.19) is real, but its seed variance is too
  large to separate HRM from the matched controls.

Cost: HRM uses 4.3× the CPU of prefix and 4× the module visits. Success per visit is lower
(revision 0.060 vs 0.229). Retained memory is 3,296 B per example for every prefix-memory
arm; the param-matched arm retains 7,622 B. Peak working set was 366 MB per worker. Pareto
front over (mean exact, CPU s, bytes): f2, one_pass, prefix, param-matched and
visit-matched. f2 is the quality end of the front.

## Mechanistic results

Full tables are in [`analysis.md`](receipts/finals_20261128/analysis.md).

* **The HRM manager does not specialize.** Routing entropy is 1.04–1.23 nats against a
  maximum of 1.386, and routing/loop mutual information is 0.004–0.013 nats. The learned
  router gives a near-uniform, loop-independent mixture of the four modules. Forcing top-1
  routing at evaluation collapses accuracy (proof_deps 0.652 → 0.336, revision 0.954 → 0.334).
  The gain comes from a soft parallel mixture of typed blocks, not from discrete typed
  deduce/construct/repair/verify roles. The uniform-mix control was tested only at rung 0
  (it62: +0.079 against +0.108 for the matched HRM config) and needs a frozen comparison.
* **Loop memory is causally used.** Swapping in another example's memory (`stale_swap`)
  sends every recurrent arm to ≤ 0.18 exact. Gaussian memory noise costs 0.05–0.78.
  `shuffle_history` is a no-op for these single-state memories, so that control has no power
  here. Under `reset_history`, HRM keeps 0.78 of 0.95 on revision, while final-loop prefix
  falls from 0.915 to 0.232.
* **Extra recurrence is mostly redundant past 4–5 loops.** The 16-loop control plateaus by
  loop 5, and redundant transitions dominate (proof_deps: 1,974 redundant vs 77 fixes).
  Running 2× loops on OOD specs gave no depth extrapolation for any arm: pointer_chase
  ood_depth stays ≈ 0.02 and proof_deps ood_deep moves ≤ 0.014.
* **Halting is poorly calibrated.** f0 continue-decisions have Brier 0.007–0.23 and ECE
  0.05–0.30. Halting at evaluation saves 48–75% of module visits but costs 0.07–0.18 exact
  on revision, proof_deps and bounded_walk.
* **BPD-style subspace test: HRM loop updates live in a small reusable subspace.** On
  proof_deps (f2, seed 11), projecting every loop update onto the rank-4 subspace fitted on
  disjoint roots keeps 0.742 of 0.762. A random rank-4 subspace gives 0.066 and the
  complement 0.016. On revision, rank 8 keeps 0.840 of 0.949 (random 0.445). The prefix arm
  is less compressible: rank 4 keeps 0.42 of 0.51 on proof_deps and 0.27 of 0.93 on revision.
  In contrast, constraining updates to a learned rank-4 subspace during training
  (`update_rank`) hurt in development, so the subspace appears once training is done; it is
  not a useful constraint from initialization.

## Recommendation for the Baby Goblin architecture tournament

1. **Primary candidate:** a soft mixture over 4 typed loop modules with loop forcing (f2).
   Enter it against a uniform-mix control (manager value), a width-matched single block, and
   a CPU-matched single block. Use dependency-tracking and selection skills (proof_deps-like,
   experiment_select-like), where the effect cleared all matched controls.
2. **Retrieval and revision skill modules:** use a wider single-block prefix recurrence with
   loop forcing instead of typed routing. Width beat routing there at equal parameters.
3. **Default to loop forcing.** It front-loads accuracy, survives memory reset and stays
   stable at extra loops, with no loss in mean exact.
4. **Drop at this scale:** full / top-k / low-rank / retention loop memory, error feedback,
   confidence bias, eval-time hard routing and the current halting head. None cleared its
   control.
5. **Next experiments, each with a frozen control:** (a) train-time hard or straight-through
   routing with a load-balance term, to get real specialization and a 4× compute saving;
   (b) halting trained with an explicit compute penalty and calibration target; (c)
   initializing a low-rank update path from the fitted BPD subspace; (d) depth-extrapolation
   tasks where loop count must grow with depth, since no arm extrapolated here.

## Limitations

n = 5 training seeds, one model scale (width 32, 300 updates) and synthetic tasks. Finalists
were selected on development roots by a single proposer, and no multiplicity correction was
applied across the 3 × 8 contrasts. shallow_copy saturated and cannot act as a
no-benefit control. History-mode memories (full, top-k, retention) were not finalists, so
the shuffle_history and retention-eviction interventions were not measured under freeze.
The cumsum v1 fixture remains near floor. CPU timings were taken on a shared, intermittently
loaded machine: wall times vary, while process CPU seconds and visit counts are the stable cost measures.

## Artifacts

* Campaign: `receipts/campaign_20261128/` (`state.json`, `runs.jsonl.gz`, `inky_requests.jsonl`, `ARCHIVE.md`, `datagoblin/`)
* Finals: `receipts/finals_20261128/` (`freeze.json`, `final_runs.shard*.jsonl.gz`, `analysis*.{md,json}`, `checkpoints/` with 32 seed-11 state dicts for the finalists and prefix)
* DataGoblin export: `receipts/campaign_20261128/datagoblin/trajectories.jsonl`. It holds 63 architecture trajectories (proposal, rationale, predicted effect, observed outcome, diagnosis, revision, decision), 11 adversarial task proposals and 1 final-decision record bound to the freeze and the hidden-root analysis.
