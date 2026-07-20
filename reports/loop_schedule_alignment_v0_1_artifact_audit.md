# Loop Schedule Alignment v0.1: artifact audit

## Frozen inputs

- Registered config SHA-256:
  `c6cd78a09909956962a2c6d585270807088a9b8b0dc5e1e69fc8b151d45882cc`
- Sealed v0 gamma records SHA-256:
  `d614837dff4dc7fab220c69fb49d5ccc05b19da9a33f26be9247217fa9c3482d`
- Source checkpoint manifest SHA-256:
  `3442528a33ccbf409107e998ab501d92a59d29ca2c0e8f1742967ea22f112473`
- Source replay: 18/18 terminal cells exact, maximum absolute error `0.0`.

## Valid result corpus

| Phase | Records | Records SHA-256 | Result SHA-256 |
|---|---:|---|---|
| source replay | 42 | `0e2736313dd85bef5d9c697a3178ec672a1ccc35e5bfdad06eb94ffb5f2e8d7d` | `9453b966fd4cbcbac2f6d9d6b6664321c9b946e3f2a5d63b8da5a8b47989f176` |
| primary | 72 | `1fc75ed2c26d110d9b5a3205a12683dbbb0c11cd1dee0a18b2866466b9ea090e` | `f3a1f5c81058c673c84a5ae49b71526758c936fe198f67bc0277bedf737ea40e` |
| attention+MLP | 24 | `6fea7187bc85f08ef2c969362c87376f609ee1b2843786dc87363877e534c991` | `6cf4e3e0281970734b79ad8ff6a668934abe73f1e7d40b1ec5372b207f75ab95` |
| LR ladder | 96 | `89e14fd2c7b0b1ae56c8cfddf0e798e5d138be04b16c29b4acbef021ac84da82` | `411cbdfedb6f11f9ecae0b36eb599940cd1ba17ed6e2880a2fc9e52657459bf6` |

The machine-readable final receipt is byte-identical at
`data/benchmarks/lsa_v0_1_receipt.json` and
`experiments/loop_schedule_algebra_v0_1/result_receipt.json`. SHA-256:
`3fa8c1e6f360035a8d1fc17932d18b2c6fe0249e6653c6e5aa9992a5cdf57414`.

## Invalid attempts retained

1. Primary attempt 1 completed but changed the stochastic kappa probe
   direction at every checkpoint. Its records and result are under
   `invalid_primary_attempt_1/`, with `outcomes_used=false`; 72 local
   checkpoints are retained and ignored by Git.
2. Ladder attempt 1 failed before the first optimizer step because the initial
   trainer incorrectly required exact exposure-budget divisibility. It
   produced no records or outcomes. Its failed resource receipt is retained.
3. Validation attempt 1 stopped before child launch on a transient GPU-idle
   preflight sample. Its restored receipt records the wrapper overwrite defect
   that was fixed before scientific phases.

## Paper artifacts

- Markdown note SHA-256:
  `c813c51b6384b3261d1c5f9b24cd303fc2b6e7746a7671f61a3698696bcb8572`
- TeX source SHA-256:
  `01b7be52fc88ea933932f1122ae607ac8c425319f50a89a6fca46f39e3eaeea5`
- BibTeX SHA-256:
  `12cb36532055244548f566ead3c03642c3c3c11fa38b4a6ca68a216666f3595a`
- Gamma trajectory SVG SHA-256:
  `ef3a8462b5654db9dacc8a0229c262b0d03cbf1f63493954d1a3251c62df3c8f`
- Kappa scaling SVG SHA-256:
  `e6499d1feb52f6efcf2116c2f5869c3268481176265826d317bd72e032fb2238`
- Boundary ladder SVG SHA-256:
  `9b494a9227b143a59126e0a65329f5b73cc47c0a321badb565984757893493bd`
- Overleaf ZIP SHA-256:
  `4ea8fd8471789e2a338bb4f1597bec587e93544299ebf99ed75108d3d9352c3c`

The paper-local SVGs are byte-identical to the report copies and regenerate
deterministically from the sealed JSON artifacts. The 9,241-byte ZIP contains
exactly `main.tex`, `references.bib`, `README.md`, and the three SVG figures.

## Verification

- Focused suite: `18 passed`.
- Deterministic figure regeneration: passed.
- SVG XML parse: passed for all three figures.
- Final receipt record/result rehash: passed for all four valid phases.
- PowerShell wrapper parse: passed.
- TeX compile: not run; no TeX engine or `chktex` is installed locally. The
  self-contained source uses Overleaf's `svg` package for figure conversion.
- Repository-wide suite baseline: `260 passed, 8 failed`; all eight failures
  are inherited Windows CRLF rehash failures in older sealed artifacts. No old
  frozen artifact was rewritten for v0.1.
