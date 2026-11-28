# JINN-GYM-02 preregistration: utility-constrained persona evolution and compositional routing

Registered 2026-11-28, before the live campaign. Base: `research/jinn-gym-01` at 5ce0517. Code: `research_gym/jinn_gym_02/`; per-file hashes are in `registration.json`. Every change after launch goes in `DEVIATIONS.md`.

## Main question

Can a population of externally evolving LDT research personas improve the verified research competence of a frozen language model (thinkingmachines/Inkling-2-Mini, free tier, weights never modified) under recurrent routing?

## Step 0: audit of JINN-GYM-01 (done before registration)

The audit is in `audit_gym01/audit.json`, produced by `research_gym/jinn_gym_02/audit01.py`. It found:

* The 61 held-out tasks regenerate with 0 digest mismatches.
* Rebuilt heads match the committed weights exactly, and rubrics are equal.
* All 4,392 stored rows reproduce from raw responses.
* An independent NumPy bootstrap confirms the CIs at task level, and domain-level CIs are added.
* Abstention repair (the evidence and abstention fields alone) accounts for 106% of the held-out utility gain and 54% of the fidelity gain. The non-abstention changes raised fidelity by 0.137 and changed utility by −0.018 [−0.038, 0.000].
* Winner's curse: accepted mutations showed mean selection ΔJ 0.079 but held-out ΔJ 0.025. 66% helped, 29% hurt, r = 0.56.

These results motivate the utility constraint, the abstention-aware operators and the fresh selection batches below.

## Tasks: genuine verifiers

There are three domains with three generator families each: family 0 is development, family 2 is selection, family 1 is untouched held-out.

| domain | verifier | dev (f0) | selection (f2) | held-out (f1) |
|---|---|---|---|---|
| `lean_proof` | Lean 4.35 core subprocess: one file per task, each candidate proof is its own theorem, `exact?` first; `#eval` counterexample checks | Nat arithmetic | modular and bounded-quantifier | List and Bool |
| `python_repair` | isolated `python -I` subprocess running a hidden unit suite whose expected values come from the reference implementation; randomized differential test | string functions | list functions | numeric functions |
| `design_sim` | seeded Monte Carlo: each design runs 300 simulated experiments and is scored by decision accuracy minus resources | normal outcomes | heavy-tailed outcomes | clustered binary outcomes |

Sizes per domain: 20 development, 200 selection and 60 held-out tasks, for 840 tasks in total.

Menus, boundary temptations (domain-specific: `sorry`, skipping a test, metric switching) and the abstention, missing-input, scope and ambiguity mechanics are those of GYM-01.

* **Receipts.** Utilities derive from verifier receipts (`domains.utility_from_receipt`), and `verify_task` re-derives every utility. A seeded 5% sample is rebuilt from scratch and must agree.
* **Regression fixtures.** The GYM-01 synthetic domains stay unchanged as regression fixtures (`tests/test_jinn_gym_01.py`, the audit).

## Population and modules

* **Roots.** The roots are the eight frozen GYM-01 specifications.
* **Heads.** Heads use the GYM-01 additive LDT head and veto. Training rows are the audited tag sets of development tasks plus 700 seeded compositional tag sets, labeled by the exact rubric, trained for 30 epochs CPU-only.
* **Decisions.** Decisions use the GYM-01 module projection on neutral Inky's prior.
* **Host.** The host boundary is unchanged: `host_code_hash` must equal GYM-01's registered value at launch, at freeze and at the end.

## Utility-constrained selection (items 1, 3)

The selection pool is split into a fixed Pareto panel (15 per domain), a recombination reserve (20 per domain) and consecutive fresh batches of 30 (10 per domain). Each batch is used for one round only, so there are at most 18 rounds. Mutation rounds stop when the batches run out or at 62% of the budget.

Each round, every persona has one lineage per arm, and each arm proposes at most one candidate. Parent and child are evaluated on the same batch, as is neutral Inky (arm A). Let gap = normalized utility − neutral utility on that batch, and let the margin be **0.02, preregistered**. The acceptance rule is feasibility-first:

* **Parent feasible (gap ≥ −0.02).** Accept if child gap ≥ −0.02 and child fidelity ≥ parent fidelity + 0.02. Also accept a Pareto utility move: child fidelity ≥ parent fidelity and child utility ≥ parent utility + 0.02.
* **Parent infeasible.** Accept if the child's gap is at least 0.02 better and fidelity drops by at most 0.05.
* **Always.** The child's boundary audit on the batch must pass.

There are five matched arms, each with one candidate slot per persona per round:

1. **random:** one uniformly sampled valid op.
2. **abstention:** deterministic abstention-aware repair, using the rules in item 2.
3. **heuristic:** a fixed coordinate schedule toward the registered signature.
4. **inky:** an Inky patch from abstention-aware diagnostics, with the selection rule stated in the prompt.
5. **learned:** an online logistic selector of P(accept), trained on all earlier candidates from every arm. It picks among 10 random, 1 abstention and 1 heuristic candidate.

## Abstention-aware diagnostics and operators (item 2)

Per task:

* **justified abstention:** abstained, and abstain was optimal.
* **excessive caution:** abstained, and acting was better.
* **unsupported action:** acted, and abstain was optimal.
* **successful recovery / failed recovery:** after a reported failure, picked / missed an optimal non-abstain action.
* **missed signature**, **suboptimal utility** (regret > 0.1) and **signature cost**.

The deterministic operator applies these rules in order:

* excessive caution > unsupported action, with at least 2 cases: relax (turn off no-ranked abstention, then lower the evidence minimum, then threshold −0.1).
* unsupported action > excessive caution, with at least 2 cases: tighten (missing-input abstention on, then threshold +0.1).
* failed recovery > successful recovery: protect recovery (add diagnose→repair/test and orient→diagnose transitions, then drop the failure-log requirement).
* otherwise no-op.

## Recombination (item 4)

Recombination uses the 6 registered persona pairs in both directions, starting from each persona's current best feasible variant. There are three operators:

* **feature:** the recipient gains the donor's top-weighted rubric feature.
* **ranking:** interleave the two action rankings.
* **params:** LDT parameter exchange, averaging the donor-signature tag columns of the two heads.

Measured on the recombination reserve:

* transfer = child fidelity on the donor's signature − the recipient's fidelity on it
* interference = the recipient's own fidelity − the child's own fidelity
* complementarity = child utility − the better parent's utility

Recombinants enter the archive.

## Pareto archive and freeze (item 5)

Every evaluated variant is archived with panel metrics: fidelity, utility gap and utility, Brier score, compute cost (mean cost units of executed actions, low 1 / medium 2 / high 3), abstention F1, and the head vector.

The non-dominated front is reported per persona. The frozen persona is the archive variant with maximum panel fidelity subject to panel gap ≥ −0.02, with ties broken by utility; if none is feasible, the one with maximum utility. Per-arm finals are each lineage's current head.

## Router (item 6)

* **Options.** 15 options: the 8 frozen personas, the 6 registered composites, and neutral Inky ("inky").
* **Features.** Observable research-state features only: domain, phase, public evidence flags, the host-audited tag histogram of the permitted menu, permitted share, neutral-prior entropy and maximum, and parse status.
* **TRM.** An embedding, then a latent refined K = 4 times by a residual MLP, with a utility readout after every step under deep supervision. It is trained on selection-family option utilities and early-stopped on development-family routed utility. CPU torch, Adam with learning rate 3e-3, weight decay 1e-4, up to 400 epochs.
* **Baselines:** uniform (the expected utility over options), static-best, per-domain best (learned on selection), a hand rule, TRM with K = 1, Inky self-routing (one call per held-out task: Inky picks a persona or "none"), and an oracle as the upper bound.

## Self-model (item 8)

Metric definitions are fixed here and given verbatim to Inky:

* **success:** P(the executed action is optimal among host-permitted actions, within 1e-9 of the best verified utility).
* **failure:** P(normalized verified utility < 0.5).
* **abstain:** P(the persona abstains).
* **compute:** the expected cost units of the executed action (low 1, medium 2, high 3).

Two self-models are compared:

* **Learned.** A per-persona model (logistic regression for success, failure and abstain; ridge regression for compute) on router features, fit on selection rows. It is scored on held-out by Brier skill and ECE against the base rate, and by MAE skill for compute.
* **Elicited.** One Inky call per frozen persona after freeze and before held-out exposure, giving per-domain predicted rates. These are scored by absolute error against the held-out observed rates.

## Compactification (item 9)

Heads are flattened to 108 numbers. Over all archived variants, the analysis reports:

* the singular spectrum, effective rank (exponential of the spectral entropy) and participation ratio
* principal angles between persona subspaces
* rank-r reconstructions of each frozen head (r ∈ {1, 2, 3, 4, 6, 8, 12, 16, 24}), scored by executed-action agreement with the full head and by fidelity and utility retention on held-out

## Held-out evaluation and leakage control

* **Freeze before exposure.** Held-out tasks are used only after freeze; `Campaign.heldout` refuses to run before it, and a test checks this.
* **What held-out may receive.** Neutral priors for held-out tasks may be prefetched, since they are persona-independent and never read by selection code.
* **Fresh selection batches.** Selection uses fresh batches, so no batch drives two rounds.
* **Panel reuse.** The panel is reused for archiving and freeze. It belongs to the selection family, never the held-out family.
* **Bootstrap.** Clusters are roots (tasks), and families are (domain, generator template), with 4,000 resamples.

## Hypotheses (held-out family 1)

* **H1 (main).** TRM-routed (K = 4) normalized utility − neutral Inky: root-level CI lower bound > 0. Template-level CI also reported.
* **H2 (noninferiority and fidelity).** Pooled frozen persona (FINAL) − A, normalized utility: CI lower bound > −0.02. FINAL − A fidelity: CI lower bound > 0.
* **H3.** FINAL − ROOT (GYM-01 personas) normalized utility: CI lower bound > 0.
* **H4 (operators).** Each arm against random, for fidelity and utility at the persona level on held-out. Reported with CIs; the registered contrast is inky − random.
* **H5 (routing beyond fixed choices).** TRM − static-best and TRM − domain-best: CI lower bound > 0.
* **H6 (recurrence).** TRM K = 4 − K = 1: reported.
* **H7 (self-model).** Learned self-model Brier skill > 0 for success, pooled mean over personas. Elicited absolute errors reported.
* **H8 (compactification).** The smallest rank with mean behavioral agreement ≥ 0.95 is reported, with fidelity and utility retention.
* **H9 (boundary).** 0 unauthorized executions in every held-out row, routed decision and mutation audit; the freeze audit passes; the forced boundary-loving probe is contained; the host code hash is unchanged.

Negative and null results are reported as such.

## Resources

* All LLM calls go to the free `thinkingmachines/Inkling-2-Mini:peft:65536:free` route through one worker thread, with one request in flight.
* Decision calls are capped at 1,200 tokens, patch calls at 2,500, and routing and self-prediction calls at 1,200, all at low reasoning effort.
* Context is 12,288 tokens with a 16,384 ceiling.
* Rate limits back off; there is no paid fallback.
* Work is CPU-only. Lean runs sequentially and Python verification uses at most 4 subprocesses.
* The process-tree RAM increment cap is 2 GiB.
* The campaign budget is at most 8 hours.
* Checkpoints are atomic, the JSONL is append-only, and the response cache makes resumes free.
