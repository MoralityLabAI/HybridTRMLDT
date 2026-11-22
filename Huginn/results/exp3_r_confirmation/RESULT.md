# HUG-CAL-02 result: Silico experiment #3, R-only confirmation (exported 2026-11-09)

Verdict: **no temporal role specialization at a meaningful scale.** Matched read-rotation controls (family R,
four per factor) passed the registered gate in both windows. On 240 clean-correct sealed programs × 3 recurrent
noise seeds, the control-adjusted late-minus-early effect is within ±0.0065 nats for all eight frozen factors:

| factor | nomination | late − early (nats), 95% CI | Holm |
|---|---|---|---|
| core1_factor19 | early_only | +0.0065 [+0.0036, +0.0095] | rejects 0 |
| core1_factor26 | | −0.0014 [−0.0046, +0.0018] | |
| core1_factor31 | | −0.0005 [−0.0032, +0.0023] | |
| core1_factor58 | | −0.0007 [−0.0039, +0.0023] | |
| core3_factor1 | | +0.0007 [−0.0028, +0.0044] | |
| core3_factor17 | | −0.0064 [−0.0103, −0.0027] | rejects 0 |
| core3_factor44 | | +0.0057 [+0.0025, +0.0090] | rejects 0 |
| core3_factor60 | | −0.0008 [−0.0038, +0.0024] | |

- None of the four early-only / late-only nominations meets the registered active-harm criterion
  (< −0.05 nats, 95% CI excluding zero).
- The 0.1–0.77-nat late-iteration ablation effects in the 2026-11-03 local SVD heat map
  (LoopedTransformers `rmp-v0-phase2`, `huginn_heatmap/`) used controls norm-matched to a random component.
  Under activation-matched controls they shrink to the ≤0.0065-nat interactions above.
- Family W (write rotation) passed matching but had no causal run. Factors are restricted-SVD, not full VPD.
  The original inconclusive closeout (experiment #2) stands as recorded.

Files: Silico's raw export (`confirmation-analysis.json`, `confirmation-receipt.json`, gate and audit receipts,
`claim_evidence_ledger.csv`, `item-effects.jsonl`) and `huginn_pr8.patch`. The patch is applied to `Huginn/`
in the same commit; `pytest Huginn/tests` gives 17 passed.
