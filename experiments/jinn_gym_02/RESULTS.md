# JINN-GYM-02 results

Live campaign `outputs/run_20261128`, 2026-11-28, 11:47 to 15:04 local time: 3.23 h of the 8 h budget, plus the BUILD restarts recorded in [DEVIATIONS.md](DEVIATIONS.md). Model: `thinkingmachines/Inkling-2-Mini:peft:65536:free`, weights frozen; Lean 4.35.0.

* **Calls:** 1,156 successful Inky calls (840 neutral, 128 patch, 180 routing, 8 self-prediction), 744k prompt and 683k completion tokens. 8 calls hit the cap and 3 returned empty text. $0 paid spend.
* **Resources:** peak process-tree RAM increment 301 MB, CPU only.
* **Records:** the tables come from `outputs/run_20261128/results.json`, which `python -m research_gym.jinn_gym_02.report outputs/run_20261128` regenerates.

Rules are in [PREREGISTRATION.md](PREREGISTRATION.md). The GYM-01 audit is in [audit_gym01/AUDIT.md](audit_gym01/AUDIT.md).

## Answer to the main question

**No.** On untouched held-out families with genuine verifiers (180 tasks), routing a population of evolved external LDT personas did not improve the verified research competence of frozen Inkling-2-Mini.

* **Routing lost to neutral Inky.** The compact TRM router reached normalized verified utility 0.766, against 0.804 for neutral Inky. TRM − neutral = −0.038 [−0.076, −0.001] at task level, and −0.038 [−0.077, −0.002] at template (family) level.
* **No other router beat it either.** Static-best, per-domain-best, the hand rule, Inky's own persona choice and the K = 1 TRM all scored below neutral or tied with it.
* **Headroom existed.** An oracle that picked the best module per task would have gained +0.070 [0.047, 0.096] over neutral Inky. No router found it from observable state.

The personas did keep their behavior on held-out. Fidelity was 0.445, against 0.210 for neutral Inky, a gain of +0.236 [0.205, 0.267]. But every frozen persona failed the preregistered 2-point utility noninferiority margin on held-out:

* pooled FINAL − A = −0.065 [−0.093, −0.037]
* per persona from −0.035 (research strategist) to −0.073

On the selection family, all eight frozen personas met the margin: panel gaps between −0.018 and +0.020.

**The utility constraint was satisfied where it was enforced and failed to transfer across families.** On the held-out families, neutral Inky's own judgment was much stronger: 0.963 on held-out Lean and 0.818 on held-out Python repair, against 0.69 on the selection panel. Fixed persona preferences that cost nothing on selection tasks cost 6 to 7 points there.

## Hypotheses

| | Contrast (held-out family 1) | Estimate [95% CI, task level] | Rule | Verdict |
|---|---|---|---|---|
| H1 | TRM (K = 4) − neutral, normalized utility | −0.038 [−0.076, −0.001] | lower bound > 0 | **not supported: significantly worse** |
| H2 | FINAL − A utility / fidelity | −0.065 [−0.093, −0.037] / +0.236 [0.205, 0.267] | utility lower bound > −0.02; fidelity lower bound > 0 | **noninferiority fails**; fidelity supported |
| H3 | FINAL − ROOT (GYM-01 personas), utility | +0.014 [0.005, 0.023] | lower bound > 0 | supported (small); fidelity +0.024 [0.009, 0.041] |
| H4 | inky arm − random arm, fidelity / utility | −0.030 [−0.048, −0.012] / +0.014 [0.002, 0.026] | reported | Inky patches traded fidelity for utility; no arm beat random on both |
| H5 | TRM − static-best / TRM − per-domain-best | +0.032 [−0.005, 0.069] / −0.016 [−0.046, 0.012] | lower bound > 0 | not supported |
| H6 | TRM K = 4 − K = 1 | −0.006 [−0.038, 0.024] | reported | null: recurrence added nothing |
| H7 | learned self-model, Brier skill for success | +0.41 pooled (0.35 to 0.56 per persona) | > 0 | supported |
| H8 | smallest rank with mean behavioral agreement ≥ 0.95 | rank 12 of 108 (0.957) | reported | see compactification |
| H9 | boundary | 0 unauthorized executions; freeze audit passed (1,347 actions, 22 modules and composites); forced boundary-loving probe contained and rejected by the validator; 0 mutation boundary-audit failures; host code hash unchanged since GYM-01 | no violation | supported |

Template-level intervals (24 clusters) agree with the task-level ones in direction and significance for H1, H2 and H3. See `results.json`.

## Routing detail (held-out, normalized verified utility)

| | design_sim | lean_proof | python_repair | all |
|---|---|---|---|---|
| neutral Inky | 0.631 | 0.963 | 0.818 | **0.804** |
| TRM K = 4 | 0.607 | 0.903 | 0.787 | 0.766 |
| TRM K = 1 | 0.614 | 0.912 | 0.790 | 0.772 |
| per-domain best | 0.592 | 0.902 | 0.852 | 0.782 |
| Inky self-routing | 0.596 | 0.845 | 0.846 | 0.762 |
| hand rule | 0.611 | 0.766 | 0.846 | 0.741 |
| static best (skeptic) | 0.587 | 0.902 | 0.712 | 0.733 |
| oracle | 0.688 | 0.995 | 0.939 | 0.874 |

Per-domain best beat neutral on held-out Python repair (+0.034, with the research strategist), but nowhere else.

* **TRM choices.** The TRM spread its choices over 14 options, picking neutral Inky only 15 times out of 180.
* **Oracle choices.** The oracle mostly picked the constructivist (114 of 180).
* **Inky self-routing.** Inky mostly picked the repair specialist (72) and the research strategist (53).
* **Validation vs held-out.** On development-family validation the TRM scored 0.825, against 0.766 on held-out: the router learned selection- and development-family regularities that did not carry over.

## Utility-constrained evolution

There were 16 rounds (the whole fresh-batch pool), giving 640 candidates in 5 arms of 128.

| arm | accepted | accepted for fidelity gain | Pareto utility | feasibility restoration | violates margin | parent infeasible, no restoration |
|---|---|---|---|---|---|---|
| random | 18 | 9 | 4 | 5 | 20 | 44 |
| abstention repair | 17 | 7 | 3 | 7 | 17 | 44 |
| heuristic | 22 | 11 | 6 | 5 | 10 | 47 |
| Inky patch | 16 | 6 | 3 | 7 | 23 | 41 |
| learned selector | 23 | 8 | 5 | 10 | 13 | 47 |

On held-out, each arm's final lineage scored:

| arm | fidelity | utility |
|---|---|---|
| random | 0.454 | 0.724 |
| abstention | 0.441 | 0.736 |
| heuristic | 0.440 | 0.722 |
| Inky | 0.424 | 0.737 |
| learned | 0.413 | 0.739 |

Paired against random:

* Inky and learned lost fidelity (−0.030 and −0.040) and gained a little utility (+0.014 and +0.016).
* Abstention repair gained +0.012 utility at −0.012 fidelity.
* The heuristic was indistinguishable from random.

No operator dominated random. Inky's patches gave a valid candidate in 120 of 128 cases. The 6 invalid ones were all in round 0, before the D2 prompt fix stated the numeric bounds, and 2 responses could not be parsed.

### Abstention-aware diagnostics

On the genuine domains the dominant abstention diagnostic flipped relative to GYM-01. The deterministic operator fired these rules:

| rule | times |
|---|---|
| tighten: threshold up (unsupported action dominated) | 88 |
| relax: threshold down | 6 |
| relax: no-ranked abstention off | 2 |
| recovery rules | 29 |
| no-op | 3 |

In GYM-01 the frozen personas were over-cautious. On verifier-backed tasks with genuine missing-input and underpowered cases, they mostly acted when abstaining was best. Held-out abstain F1 was 0.859 for FINAL, against 0.814 for ROOT and 0.867 for neutral Inky.

### Recombination

There were 36 recombinants (6 pairs × 2 directions × 3 operators, all valid), measured on the 60-task recombination reserve.

| operator | transfer of donor signature | interference with own signature | complementarity (utility vs better parent) | share with positive complementarity |
|---|---|---|---|---|
| rubric feature | +0.007 | 0.040 | −0.019 | 42% |
| action ranking | +0.005 | 0.082 | −0.006 | 33% |
| LDT parameters | **+0.081** | 0.051 | −0.013 | 42% |

* **Transfer needs parameters.** Exchanging head parameters is the only operator that transfers behavior; exchanging typed rubric components or rankings barely moves the donor's signature.
* **Interference.** All three interfere with the recipient's own signature, and ranking exchange interferes most.
* **No average complementarity.** Mean complementarity was negative for every operator.

### Pareto archive and freeze

There were 488 variants. Pareto fronts held 11 to 29 variants per persona. Every frozen persona came from a mutation lineage, none from a recombinant:

| arm | personas frozen from it |
|---|---|
| random | 3 |
| heuristic | 2 |
| learned | 2 |
| Inky | 1 |

## Self-model

Metric definitions were fixed before elicitation (PREREGISTRATION, item 8).

| target | learned self-model skill vs base rate, held-out (pooled mean) | Inky elicited self-prediction, mean absolute error (24 persona × domain cells) |
|---|---|---|
| success (optimal action) | Brier skill +0.41 | 0.19 |
| failure (normalized utility < 0.5) | Brier skill +0.06 | 0.09 |
| abstain | Brier skill +0.82 | 0.07 |
| compute (cost units 1 to 3) | MAE skill +0.46 | 0.43 |

* **Learned.** A per-persona model fit on selection rows predicts its own held-out success, abstention and compute well above base rate, but barely predicts failure.
* **Elicited.** Inky's self-predictions are reasonable for abstention and failure rates but miss success by 19 points and compute by 0.43 units. Self-predictions off by more than 0.15 are exported as failed hypotheses.

## Population compactification

The population is 488 heads × 108 parameters.

* **Spectrum.** Effective rank is 16.7 and the participation ratio 8.6. The first 8 components explain 78% of the variance.
* **Persona subspaces.** Each persona's own variants span an effective rank of 13 to 34. The mean first principal angle between two personas' top-3 subspaces is 63° (range 34° to 80°), so the personas are not sharing one low-dimensional direction.
* **Low-rank reconstruction of the frozen heads, on held-out:**

| rank | behavioral agreement | min over personas | fidelity retention | utility retention |
|---|---|---|---|---|
| 1 | 0.843 | 0.728 | 0.673 | 1.040 |
| 4 | 0.900 | 0.856 | 0.845 | 1.017 |
| 8 | 0.944 | 0.867 | 0.946 | 1.006 |
| 12 | 0.957 | 0.922 | 0.938 | 1.003 |
| 24 | 0.983 | 0.967 | 0.986 | 1.002 |

Low-rank heads keep utility (and slightly raise it), but specialized competence, meaning signature fidelity, needs about 8 to 12 shared dimensions to reach 94–95% retention. A rank-1 population keeps 67% of fidelity and drifts toward neutral-like choices, which is why utility rises.

## What JINN-GYM-01 looked like under audit

The audit is in [audit_gym01/AUDIT.md](audit_gym01/AUDIT.md). Every one of the 4,392 GYM-01 rows reproduced from raw records. The whole GYM-01 held-out utility gain came from abstention repair, and accepted mutations overstated their gains about threefold. GYM-02 confirms the second point at family scale: noninferiority on the selection family did not survive the move to held-out families.

## Limitations

* **One family per role per domain.** Each domain has one family per role, so the family-level bootstrap has 24 template clusters on held-out. A different held-out family could favor the personas. The point is that the evolution loop has no way to know which family it will face.
* **Synthetic menus over genuine verifiers.** Lean goals, Python bugs and simulated worlds are generated from templates, and utilities apply a fixed cost schedule on top of verifier outcomes. The verifier outcomes themselves are genuine: Lean accepted or rejected each proof, and Python tests and seeded simulations actually ran.
* **Python repair stays a choice among concrete candidates.** Inky chooses among candidate patches rather than writing them, and the Lean domain is restricted to core Lean.
* **Selection-panel reuse.** The Pareto panel (45 selection tasks) was reused for every archived variant and for the freeze choice, so the frozen personas are selected on it. That is selection-family overfitting by design; held-out was untouched.
* **Deviations.**
  * **D1:** the verifier timeout was raised during BUILD, and 15 tasks were rebuilt.
  * **D1a:** CPU was capped at your request.
  * **D2:** the patch prompt now states the numeric bounds, from round 1 on, and the partial round 1 was discarded and rerun (3 records in `mutations_aborted.jsonl`).
  * **Scope:** round 0 of the Inky arm used the old prompt. None of these touched held-out data.
* **Stale metadata.** `error_launch1_build_timeout.txt` is the traceback from the first launch (D1). The completed run ended normally (exit 0, phase REPORT).
* **Single run.** One campaign, one model, CPU-only training. The TRM router is small (64 hidden units) and was trained on 600 selection tasks.

## Deliverables

All of these are in `outputs/run_20261128/`:

* **Raw receipts.** `bank/*.jsonl` holds every task with its Lean, Python and simulation verifier receipts; `responses.jsonl` holds every raw Inky response.
* **Mutation histories.** `mutations.jsonl` holds all 640 candidates, accepted and rejected, with diagnostics and batch statistics; `mutations_aborted.jsonl` holds the partial round 1 discarded under D2.
* **Archive and freeze.** `archive.jsonl`, `pareto_front.json`, `recombination.jsonl`, `frozen/`, and `modules/specs/` (every spec by hash).
* **Router and self-model.** `router.json`, `router_trm_k{1,4}.pt`, `selfmodel_elicited.json`, `selfmodel_eval.json`, `compact.json`, `cca_audit_freeze.json`.
* **Held-out.** `heldout_rows.jsonl` (persona-level) and `routing_rows.jsonl` (system-level).
* **DataGoblin export** in `export/`:

| file | rows |
|---|---|
| `mutations_accepted.jsonl` | 96 |
| `mutations_rejected.jsonl` | 544 |
| `failed_hypotheses.jsonl` | rejected mutations plus wrong self-predictions |
| `repair_decisions.jsonl` | 128 |
| `verifier_feedback.jsonl` | 840 |
| `routing_decisions.jsonl` | 180 |
| `resource_usage.json` | – |

Reproduce:

```
python -m pytest -q tests/test_jinn_gym_02.py tests/test_jinn_gym_01.py
python -m research_gym.jinn_gym_02.audit01
python -m research_gym.jinn_gym_02.report experiments/jinn_gym_02/outputs/run_20261128
```
