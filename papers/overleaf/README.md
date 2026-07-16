# Overleaf Bundle

Paper: `Trade Offs between TRM/LDT Hybrids`

Upload `hybrid_ldt_trm_overleaf.zip` to Overleaf, or upload this folder directly.

Entry point:

```text
main.tex
```

Figures are TikZ source files under:

```text
figures/
```

The paper intentionally excludes VPD from the benchmark argument.

The RSITopology-aware HRM training-review section includes a synthetic receipt-contract matrix. It does not
claim that a neural model was trained or promoted.

The sequencer-control section reports a registered 667-episode deterministic replay with paired inference and an
error-budget sensitivity table. It measures controller selection over fixed skills, not neural weight infusion.

The AIRIS/DAS section reports paired deterministic receipt fault injection and includes a TikZ integrity-flow
figure. It does not report training or adversarial robustness of an AIRIS learner.
