# TRM-SKILL-GYM-01 preregistration

Registered 2026-11-28, before the campaign's first training step. The frozen campaign profile is
`research_gym/trm_skill_gym/campaign.py::PROFILE`, and the source digests are pinned in `run/receipts.json`.

## Question

Can small trainable modules (5k to 250k parameters, CPU only) improve six agent skills beyond the
existing text-only skill procedure? Does the native TRM update rule beat parameter-matched feedforward
and GRU heads under matched training exposure, candidate pools, verifier budget and visit count? Can
Inky Mini (free `thinkingmachines/Inkling-2-Mini`), working only through a closed configuration
language, engineer better modules than a random-mutation control given the same racing budget?

## Skills and task families

All six skills share one candidate-set interface (16 slots, 24 features, 8 globals, one relation
matrix, and a 4-column verifier feedback channel). Instances are pure functions of
`(skill, family, root)`.

| skill | dev families | withheld family | label |
|---|---|---|---|
| repair (bipartite recolouring, min flip set) | tree_region, cycle_region | dense_large_region | multi |
| retrieval (transitive dependencies) | layered_shallow, layered_wide | layered_deep | multi |
| deduction (kernel of a dominance DAG) | kernel_h4, kernel_h5 | kernel_h7 | multi |
| research (max EIG minus cost) | uniform_prior, peaked_prior | sharp_costly | single |
| curation (inconsistent or duplicate) | clusters3, clusters4_dups | noisy_overlap | multi |
| stopping (first action of the finite-horizon optimum) | horizon2, horizon4 | horizon6 | single |

Roots: train `[0, 400000)`; held-out-within-development `[400000, 500000)`, used by the search only;
frozen evaluation `[900000, 1000000)`, never read before `frozen_architectures.json` is written.

## Arms (matched protocol: 3 visits, 1 in-loop verifier call, final receipt, batch 32, same steps)

- A: existing skill procedure (host heuristic plus its local diagnostic repair step), no trainable module.
- B: feedforward head, 3 mixer layers, hidden width chosen to match D's parameter count.
- C: GRU recurrent proxy, parameter-matched to D.
- D: TRM, native update rule, direct answer feedback.
- E: TRM with matched damping (DataGoblin `rlt_feedback` coefficient-sum match).
- F: TRM with the RLT feedback schedule `eta = 1/(D-u)`, terminal replacement.
- G: TRM with D's settings plus sound one-step LDT certificates projected onto logits and exposed to the latent.
- H: the Inky-engineered architecture selected per skill on held-out development cases.
- Ablations (seed 0): D with shuffled supervision; D and F with the latent state reset at every visit.

## Search (external meta-repair; labelled separately from in-module RLT)

Three arms run interleaved for 12 rounds each, every round racing 3 proposals for 300 steps and
promoting the best to 800 steps:

- `meta_direct`: Inky proposals replace any fields.
- `meta_rlt`: the host enforces an edit budget of `ceil(6/(K-k))` fields at round k. This is the RLT
  schedule applied at the meta level: conservative early, full replacement at the terminal round.
- `random_mutation`: 2 random legal fields change per proposal, with no LLM involved.

H for each skill is the best rung-1 Inky configuration on that skill (or, failing that, the best Inky
configuration on any skill, labelled as such).

## Primary outcomes (frozen evaluation, 512 fresh-root instances per pool)

Exact task success on dev families and on the withheld family. Secondary outcomes: repair success
(verifier acceptance), calibration (ECE of the confidence head), adaptive-halting exact and visits,
verifier calls, FLOPs per instance, parameter count, peak incremental RSS, training seconds,
recurrence extrapolation (exact at 1, R and 2R visits), and the text-only Inky arm on 16 instances per
skill.

## Decision rules (fixed in advance)

- **Promotion** of a skill module: the best of H or D–G (by mean frozen-dev exact over seeds) must
  beat max(A, B, C) by at least 0.02 and must not fall below A on the withheld family. LDT
  certificate violations must be 0.
- **TRM beats proxies** on a skill if D or F beats both B and C by at least 0.02 on frozen-dev exact,
  averaged over seeds.
- **Inky adds value** if H beats D on frozen-dev exact in at least 4 of 6 skills, and if
  `meta_direct`/`meta_rlt` promote more incumbents than `random_mutation`. Six rounds per skill give
  little power; this is reported as a development-scale signal.
- **Shuffled supervision** must fall near chance. If it does not, the task leaks its labels and the
  skill is flagged.
- **Recurrence matters** for a skill if exact at R visits beats exact at 1 visit by at least 0.02.

## Budget

CPU only, one worker, `torch.set_num_threads(1)`, a 3 GiB incremental RSS cap enforced per training
run, an 8-hour wall clock with resumable job files, and Inky context at 12k by default (16k ceiling)
with bounded retries. No paid API, GPU or cloud.
