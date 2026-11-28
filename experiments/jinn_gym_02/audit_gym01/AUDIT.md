# Independent audit of JINN-GYM-01

Reproduce: `python -m research_gym.jinn_gym_02.audit01` (about 2 minutes, no model calls). Full output: `audit.json`.

## Reconstruction from raw records

* The 61 held-out tasks regenerate from their ids with 0 digest mismatches against the DataGoblin decision records.
* Every frozen and initial module was rebuilt from its spec with the GYM-01 recipe. The maximum absolute difference from the committed head weights is 0.0, and every compiled rubric equals the committed one (version label excepted): True.
* All arms were recomputed from the raw response texts: 4392 of 4392 rows match on executed action and utility, with mismatches none.
* The boundary audit was rerun and passed: True, with 0 unauthorized executions. The host code hash equals the registered one: True.

## Paired intervals, recomputed with independent code

These come from a vectorized cluster bootstrap with 4,000 resamples and seed 7, at the root (task) level and at the domain (task family) level, which has only 6 clusters.

| contrast | metric | root level | domain level |
|---|---|---|---|
| D − A | fidelity | +0.241 [+0.184, +0.301] | +0.241 [+0.164, +0.311] |
| D − A | utility | -0.029 [-0.076, +0.015] | -0.029 [-0.073, +0.010] |
| D − B | fidelity | +0.199 [+0.140, +0.262] | +0.199 [+0.110, +0.281] |
| D − C | fidelity | +0.017 [-0.016, +0.049] | +0.017 [-0.012, +0.042] |
| D − G | fidelity | +0.328 [+0.252, +0.407] | +0.328 [+0.179, +0.484] |
| D − D0 | fidelity | +0.237 [+0.176, +0.299] | +0.237 [+0.131, +0.345] |
| D − D0 | utility | +0.203 [+0.127, +0.286] | +0.203 [+0.053, +0.351] |
| F − D | utility | -0.015 [-0.042, +0.004] | -0.015 [-0.030, -0.005] |
| E − D | fidelity | -0.008 [-0.038, +0.020] | -0.008 [-0.021, +0.009] |
| C − C0 | utility | +0.194 [+0.122, +0.271] | +0.194 [+0.049, +0.336] |

The root-level intervals agree with the GYM-01 report to about ±0.005. They differ from it only through the bootstrap seed and implementation.

## How much of the improvement was abstention repair

The counterfactual modules are D0+abst (the initial spec carrying the frozen spec's `evidence_requirements` and `abstention` fields) and D−abst (the frozen spec carrying the initial ones). The gain is split by the two-way Shapley average:

| metric | total D − D0 | abstention fields | everything else | abstention share |
|---|---|---|---|---|
| fidelity | +0.237 | +0.129 | +0.108 | 54% |
| normalized_utility | +0.203 | +0.216 | -0.012 | 106% |
| abstain_f1 | +0.412 | +0.407 | +0.005 | 99% |

* Abstention repair alone (D0+abst − D0): utility +0.221 [+0.146, +0.304], fidelity +0.100 [+0.046, +0.160].
* Everything else on top of it (D − D0+abst): fidelity +0.137 [+0.085, +0.190], utility -0.018 [-0.038, +0.000].

So the whole held-out utility gain of the overnight loop was abstention repair. The remaining persona shaping raised fidelity by about 0.14 but cost a little utility, and the interval for that cost reaches zero only at its upper end.

## Accepted mutation lineages

* 73 revisions were accepted and 392 rejected.
* Accepted ops: {'set_evidence': 8, 'set_abstain_threshold': 40, 'add_tag': 20, 'set_weight': 31, 'add_transition': 26, 'set_abstain_if_no_ranked_action': 7, 'remove_tag': 1, 'set_ranking': 3}.
* Revision kinds: {'abstention': 13, 'abstention+preference': 19, 'abstention+lattice+preference': 6, 'abstention+lattice': 11, 'lattice+preference': 6, 'preference': 15, 'lattice': 3}.
* Stage C iteration of each persona's first abstention-related accepted revision: {'constructivist': 0, 'research_strategist': 3, 'mechanistic_interpreter': 6, 'formalist': 9, 'compressionist': 16, 'experimentalist': 31, 'skeptic': 43, 'repair_specialist': 191}.

## Winner's curse

Each accepted (parent, child) pair was re-evaluated on the 61 held-out tasks (n = 73):

* Mean ΔJ was 0.079 on its selection batch and 0.025 on held-out.
* 66% of accepted revisions improved held-out J and 29% reduced it.
* Selection ΔJ and held-out ΔJ correlate at r = 0.56.

Selection on 16 fresh cases with a 0.02 margin overstated gains by about 3×.
