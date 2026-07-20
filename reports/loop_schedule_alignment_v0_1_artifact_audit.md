# Loop Schedule Algebra v0.1: artifact audit

## Frozen inputs

- LSA v0.1 config SHA-256:
  `c6cd78a09909956962a2c6d585270807088a9b8b0dc5e1e69fc8b151d45882cc`
- Sealed v0 gamma records SHA-256:
  `d614837dff4dc7fab220c69fb49d5ccc05b19da9a33f26be9247217fa9c3482d`
- Source checkpoint manifest SHA-256:
  `3442528a33ccbf409107e998ab501d92a59d29ca2c0e8f1742967ea22f112473`
- Source replay: 18/18 terminal cells exact, maximum absolute error `0.0`.
- Saturation-addendum config SHA-256:
  `f4eb1cd7d1df7a18f523498ed48a159ccc2f123c80957913301c51cc1cc30894`
- Saturation-addendum registration commit: `911fbc3`; capped validation receipt
  commit: `8bb27ff`.

The addendum is explicitly post-v0.1 and pre-`R=32`/`R=64`. Initial
registration commit `ade595e` contained a mistyped parent receipt hash.
Validation rejected it before outcomes; commit `911fbc3` superseded it with the
correct hash and unchanged scientific fields. The construction correction is
recorded in the registration checkpoint.

## Valid result corpus

| Phase | Records | Records SHA-256 | Result SHA-256 |
|---|---:|---|---|
| source replay | 42 | `0e2736313dd85bef5d9c697a3178ec672a1ccc35e5bfdad06eb94ffb5f2e8d7d` | `9453b966fd4cbcbac2f6d9d6b6664321c9b946e3f2a5d63b8da5a8b47989f176` |
| primary | 72 | `1fc75ed2c26d110d9b5a3205a12683dbbb0c11cd1dee0a18b2866466b9ea090e` | `f3a1f5c81058c673c84a5ae49b71526758c936fe198f67bc0277bedf737ea40e` |
| attention+MLP | 24 | `6fea7187bc85f08ef2c969362c87376f609ee1b2843786dc87363877e534c991` | `6cf4e3e0281970734b79ad8ff6a668934abe73f1e7d40b1ec5372b207f75ab95` |
| LR ladder | 96 | `89e14fd2c7b0b1ae56c8cfddf0e798e5d138be04b16c29b4acbef021ac84da82` | `411cbdfedb6f11f9ecae0b36eb599940cd1ba17ed6e2880a2fc9e52657459bf6` |
| saturation fit `R=32` | 6 | `574ef9e44e7e4aa2912872c03d794734f7c1b5262ae8d0f5c9b8372ab552eb1a` | `234c85a2bb7fa5bf38b8bfd54257092804e819aa8d33066f66db66c861f35f73` |
| untouched holdout `R=64` | 6 | `04840e2e4586578559a847b49d34e66c53092341bab0be4deeb6d6614f1241fe` | `bb5a3223a7529b698692863bbcb44899b8220dcd44a202760934507abb3e9648` |

The LSA v0.1 receipt is byte-identical at its data and experiment paths,
SHA-256
`3fa8c1e6f360035a8d1fc17932d18b2c6fe0249e6653c6e5aa9992a5cdf57414`.
The saturation-addendum receipt is likewise byte-identical, SHA-256
`425e97fde47eb40b5baac381118ae95c77531b09083d2f847ee689951860b557`.

## Prediction integrity

The `R=64` predictions were generated from `R={2,4,8,16,32}`, committed, and
pushed before the holdout phase:

- prediction SHA-256:
  `910f4d994be6d24afb8bb370b9b6e42a2b907df7b4b82245516286295119ed42`
- prediction commit: `d91b745007c25bc01403aa12b67de73acb969c77`
- sealed saturation prediction: `2.551038`
- sealed logarithmic prediction: `2.876785`
- measured `R=64` tied geometric mean: `1.509898`
- registered classification: `form_unresolved`

The holdout code verified that the prediction file was tracked, clean,
byte-identical to its commit, reachable from `HEAD`, and contained in the
remote branch before reading or generating any `R=64` record.

## Post-hoc gate diagnostic

The mini-transformer untied-flatness analysis is labeled post hoc and does not
alter the failed registered composite gate. It rehashes source external records
`6fea7187...c991` and reports 20,000-draw gamma interval
`[-0.092064,0.077037]`. Receipt SHA-256:
`f50badf7f496229598314d3dcf2ebe340f3f9eaa1d660f8a16c310495ea6bbae`.

## Invalid attempts retained

1. Primary attempt 1 changed the stochastic kappa direction at every
   checkpoint. It is retained with `outcomes_used=false` and excluded.
2. Ladder attempt 1 failed before the first optimizer step on the original
   exact-divisibility check. It produced no outcome.
3. V0.1 validation attempt 1 stopped before child launch on transient GPU-idle
   preflight and exposed a wrapper receipt-overwrite defect fixed before all
   scientific phases.
4. Saturation registration `ade595e` was rejected by source-hash validation and
   superseded before any new outcome, as described above.

## Resource audit

| Phase | Elapsed seconds | Peak RAM MB | Peak I/O MB/s | Cleanup |
|---|---:|---:|---:|---|
| `R=32` fit | 119.130 | 910.781 | 20.761 | pass |
| `R=64` holdout | 199.224 | 929.957 | 41.910 | pass |

Aggregate addendum runtime was `318.354 s` under a one-hour cap. The Windows
Job Object enforced 2,048 MB process memory and 50% CPU, with a 50 MB/s
sustained-I/O abort and 1,500 MB VRAM abort. All 12 new cells were finite,
owned processes exited, and no owned GPU application remained.

## Paper artifacts

- Markdown note SHA-256:
  `246358364d275c1918f1bdaf9aa233d7598826f8a520a036e1db0b3b18bc154d`
- TeX source SHA-256:
  `e2eee98846d3a08d43f03b84c7974005c77160e4612929b0527b20229f14536e`
- BibTeX SHA-256:
  `7614129bc7f3e83233c85cf7d28db4efe80beb697ddba82ec43522cb74ef3b22`
- Compiled six-page PDF SHA-256:
  `6152a0f14097751b65ec7b3da8484d269bdaac8c386f079f12825a395a0b39a1`
- Gamma trajectory PDF SHA-256:
  `edf06a4d8d2f05f72ea877a1a05cc24683a2a2ce2ec01c46e11e17416362a5b4`
- Kappa scaling PDF SHA-256:
  `3cd15ee13d2ae4982908ea944e74e213062867756d3b8979cd4038f9192d18a3`
- High-loop PDF SHA-256:
  `93c52279d82c0139bb0156c02d1f8e7ae27a5f8a4f29a1721a6faf57743abf3c`
- Boundary ladder PDF SHA-256:
  `6b2d4e97de8b78eb0b1d5bf12d20b20822b4692568c30f10441f34e89baa0c13`
- Overleaf ZIP SHA-256:
  `3510ae2ae57495d0583b174252850102172854a1e672874621867e5123fdc09a`

The report and paper SVGs regenerate deterministically from sealed JSON. Their
vector-PDF conversions are byte-stable. The 149,334-byte ZIP contains
`main.tex`, `references.bib`, `README.md`, and four figures in both PDF and SVG
form; TeX uses the PDFs and requires no shell escape.

## Verification

- Tectonic `0.16.9` compile: pass; bibliography and references resolved.
- Compile log: no overfull/underfull boxes, LaTeX warnings, or undefined
  references.
- Visual inspection: all six rendered pages pass for clipping, overlap,
  ordering, legibility, captions, page numbering, and section transitions.
- Deterministic figure regeneration and SVG XML parse: pass for four figures.
- Deterministic vector-PDF conversion: pass for four figures.
- Final receipt and phase record/result rehash: pass.
- Focused suite: `22 passed`.
- Repository-wide suite: `273 passed, 8 failed`. The eight failures are the
  inherited Windows CRLF/working-tree rehash mismatches in older frozen
  attestation, LSA v0, architecture-discovery, and LSPG artifacts. No older
  frozen artifact was rewritten to make checkout bytes match canonical hashes.
