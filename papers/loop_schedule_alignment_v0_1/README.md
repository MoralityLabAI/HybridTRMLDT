# Visit Alignment Note

Entry point: `main.tex`.

The bundle is self-contained. Figures under `figures/` are generated from the
sealed LSA v0.1 JSON artifacts with:

```text
python -m research_gym.scripts.report_lsa_v0_1 \
  --output papers/loop_schedule_alignment_v0_1/figures
```

The paper is intentionally scoped to direct visit-alignment and short-horizon
trainability measurements at small scale. It makes no language-model,
task-performance, or large-scale transfer claim.
