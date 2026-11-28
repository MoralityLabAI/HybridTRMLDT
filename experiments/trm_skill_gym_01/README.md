# TRM-SKILL-GYM-01: a gym for engineering small TRM skill modules

This experiment builds a reusable factory for small trainable recurrent modules (5k–250k parameters) that
augment six agent skills. Inky Mini (the free `thinkingmachines/Inkling-2-Mini` endpoint) designs
architectures through a closed configuration language. Every reported number comes from CPU training and
the host-owned verifier.

- Preregistration: [PREREGISTRATION.md](PREREGISTRATION.md), committed before the first training step.
  It has one amendment (the search tie-break), which is recorded in the file.
- Generated results: [REPORT.md](REPORT.md) and [summary.json](summary.json).
- Post-training records: [goblinbench_records.jsonl](goblinbench_records.jsonl) holds 108 records: every
  search proposal, including raced-out and rejected ones.
- Skill packages: [packages/](packages/) (see "Packages" below for why these are secondary-rule exports).
- Raw run: [run/](run/) contains per-job JSON, `campaign.log`, the search state and archive, the Pareto
  frontier, cached Inky responses, checkpoints, text-only rollouts, and `receipts.json` with source digests
  per (re)start.

## Code

`research_gym/trm_skill_gym/`:

| module | role |
|---|---|
| `tasks.py` | Six deterministic skill generators on one candidate-set interface. Also holds the host verifiers (receipt verdict plus structured diagnostics), sound one-step LDT certificates, and the existing-skill heuristics (arm A). |
| `dsl.py` | Closed architecture language: enumerated or bounded fields, unknown keys rejected, a 5k–250k parameter gate, the meta-RLT edit budget, and the random-mutation control. |
| `modules.py` | Module contract (typed observations, persistent latent `z`, answer state `y`, candidate scores, confidence, value of another recurrence, optional LDT projection, explicit halting). Cores: `trm` (native TRM update rule), `gru` (recurrent proxy) and `ff`. The feedback schedules are ported from DataGoblin `rlt_feedback.feedback_weight`. |
| `train.py` | CPU training with a 3 GiB incremental RSS guard; evaluation of exact success, repair success, ECE, adaptive halting, verifier calls and FLOPs. |
| `inky.py` | Free-tier Tinker client. The credential is read at call time and never logged. Includes the context budget (12k default, 16k ceiling) and bounded backoff on 429/5xx. |
| `campaign.py` | Resumable 8-hour campaign: interleaved search arms with successive-halving racing, freeze, arms A–H, ablations and transfer. |
| `textonly.py` | Inky executes each SKILL.md directly, without a module or verifier. |
| `report.py` | Decision rules, skill packages, GoblinBench records and REPORT.md. |
| `skill_templates.py`, `skills/*/SKILL.md` | Text-only skill templates in DataGoblin's executable SKILL.md layout. |

Reproduce:

```powershell
python -m research_gym.trm_skill_gym.campaign --out experiments/trm_skill_gym_01/run --hours 8
python -m research_gym.trm_skill_gym.textonly --out experiments/trm_skill_gym_01/run/text_only_10k --max-completion-tokens 10000
python -m research_gym.trm_skill_gym.report --run experiments/trm_skill_gym_01/run --out experiments/trm_skill_gym_01
python -m pytest -q tests/test_trm_skill_gym.py
```

## Findings

Frozen evaluation used 512 fresh-root instances per pool, averaged over seeds 0 and 1. Full tables are in
[REPORT.md](REPORT.md).

1. **Trained modules beat the existing skill procedure on 5 of 6 skills.** The best trained arm vs A on
   dev / withheld:

   | skill | best arm | dev (vs A) | withheld (vs A) |
   |---|---|---|---|
   | repair | GRU | 0.94 (0.27) | 0.73 (0.26) |
   | retrieval | TRM | 1.00 (0.41) | 0.86 (0.19) |
   | deduction | GRU | 1.00 (0.11) | 0.98 (0.01) |
   | curation | TRM | 0.75 (0.41) | 0.64 (0.28) |
   | stopping | GRU | 0.99 (0.80) | 0.98 (0.71) |

   On research, the existing variance-of-likelihood heuristic (0.83 / 0.81) beats every module; the best
   module is the GRU at 0.80 dev.
2. **The native TRM update did not beat parameter-matched proxies.** No skill meets the preregistered
   "D or F beats both B and C by 0.02" rule. The GRU proxy is stronger on repair (0.94 vs 0.90 for the
   best TRM), on deduction withheld (0.98 vs 0.94) and on stopping (0.99 vs 0.92). The feedforward head is
   stronger on research. TRM-D is best on the retrieval and curation withheld families, but by less than
   the margin.
3. **RLT and matched damping help the repair skill and depth extrapolation.** On repair, F (RLT) reaches
   0.90 / 0.62 and E 0.89 / 0.60, against D at 0.86 / 0.54. At 2R visits on deduction withheld, F moves
   0.93 → 0.97, E 0.93 → 0.97 and D 0.94 → 0.96. On research, curation and stopping, RLT and damping
   score slightly below direct feedback.
4. **TRM variants degrade less than the GRU when recurrence is doubled, but the GRU is higher at the
   trained depth.** Going from R to 2R, the GRU drops 0.73 → 0.64 on repair withheld, 0.80 → 0.63 on
   research dev and 0.58 → 0.43 on curation withheld. TRM-E stays at 0.60 → 0.60 on repair withheld.
5. **The LDT certificates are sound (0 violations) but hurt withheld generalisation.** G matches D on dev
   but falls on the deeper withheld graphs: retrieval 0.86 → 0.65, deduction 0.94 → 0.63. The module
   leans on one-step certificates whose coverage shrinks as depth grows.
6. **Persistent latent state across visits did not help at this scale.** Resetting `z` every visit
   improved research by 0.10 (F: 0.17), curation by 0.03 and stopping F by 0.08, and was neutral
   elsewhere. Shuffled supervision lands at the label-free chance baseline on every skill, so no task
   leaks labels.
7. **The Inky architecture search produced valid proposals but no clear meta-level advantage.**
   - None of the 108 proposals hit the schema checker; one fell outside the parameter bound.
   - Predictions were optimistic by about 0.10 exact (MAE 0.11–0.12).
   - Promotions to incumbent: Inky direct 7, Inky with the RLT edit budget 6, random mutation 5.
   - H beat D on 2 of 6 skills (curation +0.031, research +0.015), short of the preregistered 4.
   - The development cases saturated (TRM-D at 0.97–1.0), so the tie-break steered H toward cheaper
     configurations. Deduction H (1 visit, no verifier) cut FLOPs 3× at 0.998 dev, but its withheld score
     fell from 0.94 to 0.44. Selection on saturated development cases favours configurations that do not
     extrapolate.
8. **Text-only instructions alone (Inky executing SKILL.md, 16 instances, 10k cap):**
   - On the graph skills, Inky matches the modules and far exceeds A's coded procedure: repair 0.94,
     deduction 1.00, retrieval 0.88.
   - On the numeric skills, it fails because answers truncate: research 0.19 (75% truncated) and stopping
     0.31 (62% truncated).
   - Cost: roughly 1.3k–10k completion tokens per instance, against about 3 MFLOPs for a trained module.
9. **Transfer was negative.** At 300 target steps, shared initialization and a frozen latent core both
   underperformed training from scratch on deduction→retrieval, retrieval→repair and stopping→research.

## Packages

No skill met the preregistered promotion rule, because the best TRM-family arm never beat the GRU and
feedforward baselines by 0.02. `packages/` instead holds the best trained module for each skill that beats
the existing skill A by at least 0.02 on both dev and withheld: repair (GRU), retrieval (TRM-D),
deduction (GRU), curation (TRM-D) and stopping (GRU). These are labelled
`acceptance_basis: secondary_rule_beats_existing_skill_not_preregistered_promotion` in
`module_config.json` and `evaluation_receipts.json`. Research has no package.

## Integration

- DataGoblin: the `direct` / `rlt` / `matched_damping` schedules are a port of
  `src/datagoblin/training_gym/rlt_feedback.py`, and the test pins the coefficients. GoblinBench records
  carry DataGoblin `TrainingCandidate v1` fields (`schemas/training_candidate.schema.json`) plus the
  post-training fields: template, proposal and rationale, prediction, measurement, diagnosis, revision,
  resources and decision. SKILL.md templates follow `training/rlt_gym/hypergame_taste/skills/research_hillclimb/SKILL.md`.
- LoopedTransformers: the endpoint and model string reuse `research_gym/ink_fp_audit_01/tinker_remote.py`
  (free tier, no redirects, no paid fallback).
- HybridTRMLDT: the TRM recursion follows the native update used in `research_gym/neural/trm.py`. The
  certify-then-project pattern mirrors `research_gym/core/certified_repair.py`.
- The two RLT levels are kept separate. In-module answer feedback is arms E and F, using `feedback` in the
  DSL. External meta-repair of the architecture is the `meta_rlt` search arm (edit budget `ceil(6/(K-k))`).
  Every record is labelled with `meta_level_mechanism`.

## Resources

- CPU only, one training worker, torch threads 1.
- Campaign wall clock: 02:22–04:35 (about 2.2 hours of the 8-hour cap), plus about 1.5 hours of
  concurrent text-only rollouts.
- Inky: 192 requests, 258k prompt tokens, 827k completion tokens, 0 failures, 0 retries.
- Peak incremental RSS per training run was about 200 MiB.
- No paid API, GPU or cloud.
