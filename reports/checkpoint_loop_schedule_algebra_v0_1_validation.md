# Loop Schedule Algebra v0.1: validation checkpoint

The v0.1 protocol was frozen at config SHA-256
`c6cd78a09909956962a2c6d585270807088a9b8b0dc5e1e69fc8b151d45882cc`
before outcome execution. Validation rehashed the three sealed LSA v0 Git
blobs, including gamma records SHA-256
`d614837dff4dc7fab220c69fb49d5ccc05b19da9a33f26be9247217fa9c3482d`.

The first capped validation attempt stopped before child launch because one GPU
telemetry sample exceeded the 20% idle threshold. A subsequent check found 0%
utilization, 0 MB allocated VRAM, and no compute application. Attempt 2 then
completed in 9.221 seconds with 309.121 MB peak RAM, no measured I/O or VRAM
pressure, successful cleanup, and no lingering owned process.

The first wrapper version wrote only a stable phase receipt, so attempt 2
overwrote attempt 1. The captured attempt-1 receipt has been restored with that
fact recorded, and the wrapper now writes immutable attempt receipts as well as
the stable per-phase alias used by finalization. Scientific phases will use
explicit attempt numbers.

Focused implementation tests: `14 passed`. Repository-wide tests:
`260 passed, 8 failed`; all eight failures are inherited Windows CRLF rehashes
of older frozen artifacts. V0.1 does not rewrite those artifacts and verifies
its v0 inputs from the committed Git blobs instead.
