# Loop Schedule Algebra v0.1: source replay checkpoint

The source replay hashed all 42 ignored LSA v0 gamma checkpoints before loading
any tensor. The sorted manifest SHA-256 was
`3442528a33ccbf409107e998ab501d92a59d29ca2c0e8f1742967ea22f112473`
both before and after replay.

All 18 terminal cells reproduced their sealed v0 kappa values exactly. The
maximum absolute error was `0.0` against the registered tolerance of `1e-5`.
This establishes that the local ignored checkpoints correspond to the
committed v0 gamma records; it does not make the uneven old checkpoint cadence
a valid `gamma(t)` design.

Artifact hashes:

- replay records:
  `0e2736313dd85bef5d9c697a3178ec672a1ccc35e5bfdad06eb94ffb5f2e8d7d`
- replay result:
  `9453b966fd4cbcbac2f6d9d6b6664321c9b946e3f2a5d63b8da5a8b47989f176`
- resource receipt:
  `0beaebb317049f9c621f4915106c9bb870ef21b8a6468bb04f0f74b3e44ec1d9`

The capped phase completed in 20.022 seconds with 747.496 MB peak RAM and
30.483 MB/s peak observed I/O. No abort fired and cleanup passed. GPU telemetry
did not catch an allocated sample during this short phase, so the resource
receipt reports 0 MB observed peak VRAM rather than claiming zero actual CUDA
allocation.
