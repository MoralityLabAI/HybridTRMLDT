# Loop-Schedule Kappa State/Moment v0.2.3 Artifact Audit

Audit time: `2026-07-21T04:41:20.3520430Z`

Sealed source commit before reporting: `afa755833b913535676b05a1ae784c6a2ac37d4e`

## Integrity

| Artifact | Canonical SHA-256 |
|---|---|
| Frozen config | `806b93a029445bf9db707912ee9734c06c7fb74f4d8afaa7e6b9e823f158af8b` |
| Canonical result receipt | `d5299f930ebb67ce8cb882cc24a56a29f31072d7734b0152886fb5e934c3ea4a` |
| Recovery result | `2b87aad725f5555da15d1534870b4f149ff8de72ad00e82763de790eb7d6e581` |
| Recovery records | `ee1d17c1d2d66df080f453a20f75d7ee22f1b90b02ab043cc7bf995b6ebd7ce1` |
| Run resource receipt | `418f75853f797e2873e7f259ce4d44991dee047b1679d4bbf9976f11ac03307e` |
| Recovery conformance receipt | `964397b554c58c9943cb40b7f31e0fab32f23b0d560cee60239ab84b665aa72c` |
| Deterministic SVG | `6a24aaeb147537d3aa8042333e0b62d759bf8c285e3c73aea284c97ae6997ffd` |

The canonical and output-local result receipts are byte-identical. The receipt re-verifies the recovery result, records, and completed resource receipt. All parent records, receipts, trainer config, and E2048 checkpoints are hash-gated by the frozen config.

## Interrupted Attempt

Attempt 1 completed the science child but lacked a wrapper resource receipt after the controlling tool session was interrupted. It remains explicitly unsealed. The recovery charged `620 s` of prior elapsed time and did not overwrite attempt 1.

The recovery's records are byte-identical to attempt 1. After normalizing output-directory paths, the science summaries are exactly equal. Each of the four terminal checkpoints is byte-identical across attempts; model tensors and complete AdamW state are therefore exactly reproduced.

## Resource And Cleanup

The accepted run completed in `662.195 s`, with `1282.195 s` aggregate charged time. Peak process RAM was `934.559 MB`, peak I/O was `9.201 MB/s`, and the wrapper reported no owned-process VRAM. Cleanup passed with no lingering owned process or GPU compute application.

## Scientific Assertions

- All four new component gates passed tensor-exactly before continuation.
- The result contains 16 new measurement records and no failed cell.
- The registered classification is `factorial_unresolved`.
- The three-way effect is `+0.5280058034`; the model-weight main effect is `+0.5031478989`.
- Their absolute-effect ratio is `1.0494047666`, below the frozen `1.5` gate.
- Under `M211`, swapping AdamW state reverses the binary suffix-recovery pattern.
- The fit is saturated and has zero residual degrees of freedom.

## Verification

Targeted report and seal tests: `7 passed`.

Full repository suite: `338 passed in 52.83s`.
