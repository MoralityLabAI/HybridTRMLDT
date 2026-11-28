# JINN-GYM-01 results

Live run `outputs/run_20261128`, 2026-11-28, 02:47 to 09:52 local time, 7.39 h of an 8 h budget. Model: `thinkingmachines/Inkling-2-Mini:peft:65536:free` with weights frozen. 3,147 successful calls, 0 HTTP failures, 0 rate limits, $0 paid spend, peak process-tree RAM increment 15 MiB. One clean stop and resume (deviation D1). The machine-generated tables are in `outputs/run_20261128/results.md` and `results.json`. Rules are in [PREREGISTRATION.md](PREREGISTRATION.md) and deviations in [DEVIATIONS.md](DEVIATIONS.md).

## Summary

On 61 held-out tasks from held-out generator families, the evolved external LDT personas behaved distinctly and the host boundary held without exception:

* **Fidelity.** The learned-head persona (arm D) showed its signature behavior on 44.8% of the cells where it was available. That compares with 20.7% for neutral Inky (A) and 24.9% for the text persona card (B).
* **Utility.** Verified utility stayed within noise of neutral: D − A = −0.029, 95% CI [−0.074, 0.014].
* **Evolution.** Overnight evolution moved held-out fidelity by +0.24 and normalized utility by +0.20 compared with Inky's first-shot specifications. Most of that gain came from repairing a shared over-abstention defect that Inky wrote into all eight initial specifications.
* **Null results.** The text card, bounded memory and card-plus-LDT added little or nothing beyond the LDT head.
* **Boundary.** There were 0 unauthorized executions in all 5,400 evaluated cells (Stage B and final) and in all 465 evolution steps, including from a deliberately boundary-loving persona that was compiled on purpose.

## Hypotheses (held-out final, preregistered rules)

| | Contrast | Estimate [95% CI] | Rule | Verdict |
|---|---|---|---|---|
| H1 | D − A fidelity | +0.241 [0.180, 0.298] | lower bound > 0 | supported |
| H2 | D − C fidelity | +0.017 [−0.013, 0.052] | CI within ±0.10 | supported |
| H3 | D − B fidelity | +0.199 [0.140, 0.262] | lower bound > 0 | supported |
| H4 | D − G fidelity | +0.328 [0.251, 0.403] | lower bound > 0 | supported |
| H5 | D − A normalized utility | −0.029 [−0.074, 0.014] | lower bound > −0.05 | not supported: cannot exclude a utility cost of up to 7.4 points |
| H6 | D − D0 fidelity / utility | +0.237 [0.178, 0.302] / +0.203 [0.127, 0.286] | either lower bound > 0, other estimate ≥ −0.02 | supported |
| H7 | F − D normalized utility | −0.015 [−0.041, 0.004] | lower bound > 0 | not supported (null) |
| H8 | boundary | 0 unauthorized executions; freeze audit passed over 457 held-out actions and 22 modules or composites; host code hash unchanged; adversarial probe rejected by the validator and contained by the host | no violation | supported |
| H9 | compositions cooperating | 26 / 46 = 57% | ≥ 50% | supported, narrowly |

Cells are matched (task, persona) pairs: 241 cells where a signature action was available for fidelity, 488 for utility. Intervals come from a task-cluster bootstrap with 2,000 resamples.

## Held-out arm table (61 tasks × 8 personas)

| arm | fidelity | norm. utility | optimal | Brier | abstain F1 | persona divergence (disagreement) | info retention | unauthorized exec. |
|---|---|---|---|---|---|---|---|---|
| A neutral | 0.207 | 0.799 | 0.492 | 0.207 | 0.903 | 0.000 (persona-independent by design) | – | 0 |
| B text card | 0.249 | 0.787 | 0.504 | 0.216 | 0.932 | 0.207 | – | 0 |
| C exact rubric | 0.432 | 0.748 | 0.475 | 0.351 | 0.862 | 0.466 | 0.937 | 0 |
| D learned head | 0.448 | 0.770 | 0.486 | 0.292 | 0.933 | 0.415 | 0.959 | 0 |
| E card + head | 0.440 | 0.764 | 0.492 | 0.304 | 0.929 | 0.431 | 0.961 | 0 |
| F head + memory | 0.436 | 0.754 | 0.457 | 0.308 | 0.933 | 0.438 | 0.958 | 0 |
| G shuffled head | 0.120 | 0.770 | 0.486 | 0.292 | 0.933 | 0.415 | 0.959 | 0 |
| C0 initial rubric | 0.245 | 0.554 | 0.311 | 0.585 | 0.517 | 0.334 | 0.939 | 0 |
| D0 initial head | 0.212 | 0.566 | 0.322 | 0.562 | 0.521 | 0.301 | 0.947 | 0 |

Per-persona fidelity (A / B / C / D0 / D / G), with the number of signature-available cells:

| persona | A | B | C | D0 | D | G | n |
|---|---|---|---|---|---|---|---|
| constructivist | 0.08 | 0.12 | 0.12 | 0.00 | 0.27 | 0.15 | 26 |
| skeptic | 0.27 | 0.39 | 0.61 | 0.29 | 0.66 | 0.05 | 41 |
| experimentalist | 0.15 | 0.15 | 0.85 | 0.08 | 0.69 | 0.08 | 13 |
| formalist | 0.15 | 0.15 | 0.79 | 0.06 | 0.62 | 0.18 | 34 |
| compressionist | 0.17 | 0.17 | 0.17 | 0.15 | 0.29 | 0.08 | 48 |
| mechanistic interpreter | 0.12 | 0.32 | 0.56 | 0.40 | 0.52 | 0.20 | 25 |
| research strategist | 0.31 | 0.31 | 0.21 | 0.31 | 0.24 | 0.03 | 29 |
| repair specialist | 0.40 | 0.36 | 0.40 | 0.40 | 0.40 | 0.24 | 25 |

The skeptic, experimentalist, formalist and mechanistic interpreter are clearly externalized. The constructivist and compressionist improved but remain weak. The research strategist and repair specialist were not distinguishable from neutral Inky on held-out tasks, a null for those two personas.

Utility by domain, A vs D: counterexample 0.94 vs 0.89, evidence curation 0.85 vs 0.87, experimental design 0.61 vs 0.64, Lean tactic 0.91 vs 0.80, research planning 0.59 vs 0.51, Storyworld routing 0.88 vs 0.90. The utility cost that H5 could not rule out sits in Lean tactic selection and research planning.

## Stage A: Inky engineers the personas

All eight specifications passed independent validation, compilation and the boundary audit. Six passed on the first attempt. Two passed on the second, after an invalid feature name and a malformed object were fed back. Ten spec calls in total.

The specifications were not the problem; their defaults were. All eight shared the same cautious settings: `min_evidence_items = 2`, `abstain_if_no_ranked_action = true`, and abstention thresholds of 0.35 to 0.6. On the Stage B development panel these made the module arms abstain on 85% of tasks, against 21% abstain-optimal. Stage B fidelity for D was 0.147 vs 0.206 for A, and utility was −0.22. Taken at face value, Inky's first-shot persona engineering produced personas worse than no persona: a negative result for zero-shot engineering.

## Stage C: overnight evolution

There were 465 iterations: 73 revisions accepted (15.7%), 330 not better on 16 fresh cases, 59 invalid patches (26 of them empty responses at the old patch cap before D1), and 3 with no change. Final generations by persona:

| persona | generation |
|---|---|
| formalist | 14 |
| constructivist | 10 |
| skeptic | 10 |
| research strategist | 10 |
| compressionist | 9 |
| mechanistic interpreter | 9 |
| experimentalist | 8 |
| repair specialist | 3 |

The most common diagnostic codes were overconfident (275), suboptimal utility (227), lattice blocked the best action (173), missed signature (121) and wrong abstain (103). The ops in accepted patches were: set abstention threshold 40, set weight 31, add lattice transition 26, add tag 20, set evidence 8, turn off the no-ranked-action abstention rule 7, set ranking 3, remove tag 1.

Inky found the shared defect itself. Every frozen persona has `min_evidence_items = 1`, and seven of eight dropped `abstain_if_no_ranked_action`; the repair specialist kept it and has the highest held-out abstain rate (0.48). Held-out abstain F1 went from 0.52 (D0) to 0.93 (D).

Repair cost:

* 465 patch calls, about 1.2 M patch-prompt tokens.
* About 6.4 patch calls per accepted revision, plus 16 neutral calls per comparison. Neutral calls are shared across personas: 943 neutral calls in all.

Lineage, with parent and child spec hashes for all 473 nodes, is in `lineage.jsonl`. Accepted revisions are DataGoblin rows in `training_records.jsonl` (73). Rejected specs and patches are kept separately in `failed_trajectories.jsonl` (394).

### Compositions

There were 46 composition tests over 6 pairs; 26 met the cooperation criterion. Every one passed the CCA-06 commutation check and the containment check.

| pair | cooperated | mean signature retention | mean utility vs best solo |
|---|---|---|---|
| research strategist + repair specialist | 6 / 8 | 0.85 / 0.92 | −0.001 |
| skeptic + formalist | 5 / 8 | 0.78 / 0.77 | −0.060 |
| compressionist + mechanistic interpreter | 4 / 8 | 1.30 / 0.40 | −0.022 |
| skeptic + experimentalist | 4 / 7 | 1.08 / 0.69 | −0.063 |
| formalist + compressionist | 4 / 7 | 1.19 / 0.58 | −0.029 |
| constructivist + experimentalist | 3 / 8 | 0.27 / 0.78 | −0.009 |

Retention above 1 means the composite showed that persona's signature more than the persona did alone. Two personas usually keep their signatures under a product-of-experts composition. A composite costs utility relative to the better solo persona, and the constructivist is usually absorbed by the experimentalist.

## Secondary measures

* **Calibration.** Neutral Inky's stated confidence was best calibrated (Brier 0.207). Module probabilities are overconfident (Brier 0.29 to 0.35), and evolution did not fix this even though "overconfident" was the most frequent diagnostic code. The patch vocabulary has no op that targets confidence.
* **Divergence.** Externalized personas disagree with each other on 42–47% of tasks, against 21% under text cards. The persona card barely individuates Inky.
* **Information retention.** The CCA-06 Fisher ratio after host repair was 0.94 to 0.96. The boundary rarely had to discard persona preference, because the host repaired only 1–2.5% of proposals.
* **Boundary temptations.** Unconstrained proposals for unauthorized actions: A 8, B 5, C 12, D 6, E 5, F 12, G 6, out of 488 each. All were repaired to abstain by the host. The actor's claimed tags were recorded and never consulted.
* **Falsifiable predictions.** Each persona's own predictions were checked on frozen arm C held-out rows (32 predictions). 17 held and 15 failed. Signature-rate predictions held for the skeptic (0.61 vs ≥ 0.6), experimentalist (0.85 vs ≥ 0.6) and formalist (0.79 vs ≥ 0.7). They failed for the constructivist, compressionist, mechanistic interpreter (0.56 vs ≥ 0.6), research strategist and repair specialist. All 4 `intervention_rate` predictions are uninterpretable; see limitations.

## Limitations

* **Selection.** Stage C accepts a child on a 0.02 J margin over 16 fresh development cases, so some accepted revisions are noise. The held-out D − D0 contrast is the unbiased check, and it is large, but it mixes real persona shaping with the one-off abstention repair.
* **Lean domain.** Lean tactic selection uses a fixed rule table of which tactic closes which goal class, not a Lean kernel. The other five checkers compute utilities exactly from hidden state, but the domains are synthetic.
* **Exact-rubric arm.** Arm C uses the exact lane mapped to train 0.9 / hold 0.1 / reject 0.01, registered before launch. The raw `score_policy` score was judged too compressed during the dry run. H2 compares the head with this lane mapping.
* **Deviation D1.** The decision cap rose from 700 to 1200 tokens mid-run. Stage B ran at 700 with 46 truncated decision calls, so the Stage B and held-out absolute levels are not comparable. Held-out arms all ran at 1200 and are matched. 111 calls (3.5%) still ended at a cap, 68 with empty text. Those fall back to a uniform prior (module arms) or abstain (A/B) and are counted as parse failures: A 16, B 4, C/D/G 8, E 4, F 3.
* **Prediction metrics.** The prediction schema named metrics without defining them. Inky read `intervention_rate` as its own use of causal interventions, while the metric is the host repair rate, so those 4 predictions are uninterpretable as written. Definitions are now in the schema prompt, after this run.
* **Run metadata.** `state.json` reports `stop_reason: stop_file` from the D1 stop. The resumed run finished normally (exit 0, phase REPORT). The client usage counters in `state.json` cover only the resumed process; `responses.jsonl` is the complete call record (3,147 calls).
* **Single run.** One seed, one model, one night. Held-out task counts per domain are about 10.
* **Inky's role.** Inky wrote every specification and patch, and Claude wrote none. The persona briefs, tag vocabulary, signatures and checkers are Claude-authored, so fidelity is fidelity to the registered signatures.

## Reproduce

```
python -m pytest -q tests/test_jinn_gym_01.py
python -m research_gym.jinn_gym_01.report experiments/jinn_gym_01/outputs/run_20261128
```

Modules for rejected children are not committed; `build_modules` regenerates them deterministically from `specs/<hash>.json`, since head seeds derive from the spec hash.
