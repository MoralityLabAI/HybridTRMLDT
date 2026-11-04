# CCA-12: six-checkpoint neural robustness evidence

Study date: 2026-11-04. CCA-12A retrospective data analysis, CCA-W
post-hoc exploratory preference analysis. CCA-12B live-checkpoint probe remains
unrun because the six original 67,381-byte .pt files are stored outside Git.

- [CI run (passing, CCA-T/CWA)](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/41826423713)
- [Full JSON evidence artifact](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/41826423713/artifacts/13319377511)
- Exact JSON SHA-256: e07f309fe68ef0d97bf7a335134e1db9762c8d9cdc3f28f77340684266985c1f
- Original tape-manifest SHA-256: 4fff437b16d6c5e8ac0c57a0d0eea08f8a5dd21ef061781022c6a5bb117665f1c
- Code: research_gym/core/cca_neural_margins.py
- Runner: experiments/cca12_neural_robustness/run_tapes.py
- Tests: tests/test_cca_neural_margins.py (7 passing)
- Dataset: each of six SHA-verified tapes holds passes 1–8 for 2,100 inputs.
  The evaluation selects 1,750 nonterminal decisions per checkpoint.

## Registered CCA-12A primary endpoint, pass four

The CCA-05 verifier (modeled rival, horizon 6) certifies at least one action
on 793 of 1,750 input pairs. It certifies zero actions on 957 pairs.
The zero-action states are appropriately abstained by the shield.
An empty certified set is NOT evidence that the model selected incorrectly.

| Tape | Raw uncertified / 1750 | Raw certified / 793 viable | Certified even under arbitrary logit L-inf eps 0.1 / 793 | Adversarial uncertified / 1750 at eps 0.1 | Shield uncertified |
|---|---:|---:|---:|---:|---:|
| R211 | 1002 | 748 | 746 | 1004 | 0 |
| R223 | 1002 | 748 | 744 | 1006 | 0 |
| R227 | 992 | 758 | 755 | 995 | 0 |
| X211 | 1001 | 749 | 745 | 1005 | 0 |
| X223 | 1003 | 747 | 744 | 1006 | 0 |
| X227 | 1001 | 749 | 742 | 1008 | 0 |

Across the 6 checkpoint replicates, 4,499/4,758 viable
checkpoint-state decisions have a raw certified winner (94.56%).
4,476/4,758 remain guaranteed safe for any L-infinity
logit noise <= 0.1 (94.07%) under the strict max gap rule.
The adversarial witness produces 23 additional uncertified raw outputs at
eps=0.1 across the six repetitions. These are *not* independent examples;
the same environmental states are evaluated for every checkpoint.

### Region sensitivity

Each checkpoint has its seed's original 320 proposer-training examples,
a fixed 408-decision untouched hash-bucket 8–9 set, and 1,022 other-unseen
decisions. After excluding states with no certified action, the rates across
all six checkpoint replicates are:

| Split | Viable checkpoint-state decisions | Raw uncertified, viable | Raw uncertified rate | Certified under eps=0.1 |
|---|---:|---:|---:|---:|
| Proposer train | 926 | 13 | 1.40% | 98.27% |
| Other unseen | 2884 | 168 | 5.83% | 93.62% |
| Untouched buckets 8–9 | 948 | 78 | 8.23% | 91.35% |

These are descriptive population counts for the selected regions, not a
cross-seed significance test. Region families have different baseline
environment-state distributions; this is not a causal isolation of the
training effect.

### Recurrence depth

All eight passes were checked (1–4 trained regime, 5–8 beyond trained
depth). The shape is nonmonotone: e.g., R227 has 992 raw uncertified
at pass 4, 991 at pass 5, and 999 at pass 8. X211 has 1001 at pass 4
and 1009 at pass 8. Additional GRU passes do not carry a proven
monotone safety benefit. Full per-pass figures appear in the JSON artifact.

## Exploratory CCA-W preference utility bounds

This question was added after examining the primary CCA-12A results.
The uniform safe-down/unsafe-up witness preserves relative certified
scores by construction, so it cannot test preference degradation inside
the safe set. CCA-W instead enumerates the candidate safe winners possible
under arbitrary per-logit L-infinity bounded noise, then takes the minimum
CCA-05 utility among those candidates.

| Tape | Unperturbed shield utility | Worst-case lower bound, eps=0.1 | Worst-case lower bound, eps=1.0 | Certified oracle |
|---|---:|---:|---:|---:|
| R211 | 0.844629 | 0.844453 | 0.841051 | 0.846594 |
| R223 | 0.845098 | 0.844834 | 0.841227 | 0.846594 |
| R227 | 0.843456 | 0.843163 | 0.840875 | 0.846594 |
| X211 | 0.843339 | 0.842987 | 0.840553 | 0.846594 |
| X223 | 0.843690 | 0.843133 | 0.840758 | 0.846594 |
| X227 | 0.843339 | 0.842811 | 0.839819 | 0.846594 |

The six-replicate means are 0.843925 unperturbed, 0.843563 at eps=0.1,
and 0.840714 at eps=1.0, versus the fixed certified oracle 0.846594.
These are conditional averages over 793 viable decisions per checkpoint.
The lower bound includes all possible shielded winners, and may be
conservative at exact ties. It measures the registered symbolic utility,
not separately learned persona fidelity or a real-world intervention.

## Proof and deployment limitations

The CCA-T gap criterion (best certified minus best uncertified logit
greater than twice the noise budget) is a sharp analytic robust-selection
condition for independently bounded per-action additive logit noise.
The implementation uses Python floating-point arithmetic on serialized
float32 logits and does not currently furnish interval or exact-rational
numerical error certificates. Ties are handled by vocabulary ordering.

The CCA-W possible-winner set yields a valid finite-domain lower bound
on certified utility under the same logit adversary; it does not assert
that every boundary tie is attainable with the fixed winner order.

None of this bounds perturbations *inside* the GRU, identifies an SAE
feature, or establishes a correspondence between neural latent distance
and the environment's Manhattan viability radius. CCA-U's final-hidden
L2 bound is ready to evaluate when the original registered checkpoints
are present; its optional runner performs both source-file and
state-dict SHA checks and compares pass-four outputs to the saved tapes.

The CCA-05 outer execution checker retains authorization authority;
its modeled-rival and environment-coverage assumptions remain active.
