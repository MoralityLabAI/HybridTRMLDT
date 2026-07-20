# Loop Schedule Algebra v0.1 Empirical Note

Entry point: `main.tex`.

The bundle is self-contained. Figures under `figures/` are generated from the
sealed LSA v0.1 JSON artifacts with:

```text
python -m research_gym.scripts.report_lsa_v0_1 \
  --output papers/loop_schedule_alignment_v0_1/figures

python scripts/convert_lsa_figures.py \
  --source papers/loop_schedule_alignment_v0_1/figures \
  --output papers/loop_schedule_alignment_v0_1/figures
```

`main.tex` uses the generated vector PDFs and does not require shell escape or
an external SVG converter on Overleaf.

The paper is scoped to direct visit-alignment and short-horizon trainability
measurements at small scale. The labeled saturation addendum rejects both
registered smooth high-loop extrapolators; it does not establish asymptotic
saturation. The note makes no language-model, task-performance, or large-scale
transfer claim.
