# Sequencer Control-Math Benchmark Notes

## Registration

- registered on: `2026-07-13`
- primary endpoint: `macro_utility_delta_control_math_vs_global_signed`
- protocol SHA-256: `a80d6c729c0005d26f0eccac24e27f564c21cfad4b97417c2bcbc66f6f1007b8`
- calibration SHA-256: `815eec35d4910b70c6492cd468e9762d7f18f860ab56f13799ed70ad427f5758`
- evaluation SHA-256: `d27f92280afe80a46718b934ff01d7c451d24c9ba2d792a4c8afe6d81d9f3db5`

## Design

- Candidate skill sequences are proposal-only TRM, deduction-only LDT, and typed propose/certify hybrid.
- Sequence scores are fitted on calibration episodes only and frozen before evaluation is passed in.
- Two calibration folds define independent transport paths; loop disagreement and orientation feed the
  conservative RSITopology control bound.
- Full control uses global signed transfer only inside the frozen bound; otherwise it invokes a local
  calibrated section. Lineage-only deliberately omits the loop/orientation term.
- Families are macro-weighted so routing volume cannot dominate Sudoku, ARC, or storyworld objectives.
- Confidence intervals use paired family-stratified hierarchical context-cluster bootstrap resampling.
  P-values use context-cluster sign flips and Holm correction across three registered controls.

## Primary result

- global signed macro utility: `0.893142`
- full control-math macro utility: `0.900085`
- paired delta: `+0.006943`
- 95% bootstrap CI: `[+0.004909, +0.009232]`
- paired sign-flip p: `0.00029997`
- Holm-adjusted p: `0.00059994`
- registered context section rate: `1.000000`

## Negative result

- control math versus lineage-only delta: `+0.000310`
- clustered sign-flip p: `0.50235`
- This run does not establish incremental utility from holonomy/orientation beyond lineage-only control.

## Scope

- Sudoku uses rule-preserving automorphisms of the local 4x4 puzzles.
- ARC-1/ARC-2 are balanced procedural tasks over the local primitive rule library; these are ARC-style,
  not official ARC leaderboard tasks.
- ARC tasks are retained only when all candidate sequences solve them, isolating efficiency rather
  than solver coverage.
- Routing uses real normalized Tesseract trajectories with model-train, sequencer-calibration, and eval
  partitions.
- Storyworld uses disjoint finite-state starts under secret-ending and moral-optimization objectives.
- No neural model was trained, infused, promoted, or edited. The measured object is the sequencer.
- Error-budget sweeps are post-registration sensitivity analysis, not alternate primary endpoints.

## Verifiers v1

The saved eval rows are exported into the local Verifiers 0.1.14 v1 Taskset/Harness package as an
LLM-free replay environment. Package contract tests validate its shape; exact runtime status is
recorded separately and is not claimed as a native model evaluation.

- target Verifiers: `0.1.14`
- installed Verifiers: `0.1.11.dev1`
- installed Verifiers compatible: `False`
- minimum uv: `0.11.1`
- installed uv: `0.8.17`
- installed uv compatible: `False`
- native Windows Prime blocked: `True`
