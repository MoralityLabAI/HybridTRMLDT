# Checkpoint: LSPG Architecture Harness v1

## Completed

- Added versioned proposals spanning discovery, reserve, controls, nulls, and contextual baselines.
- Added exact candidate/control parameter and compute matching at 5M, 12M, and 30M.
- Added provisional, distinct-winner, one-extension, and final statistical gates.
- Added deterministic balanced-task training cells with AMP, activation checkpointing, strict gradient/loss stops, and locked evaluation.
- Added paced, atomic, SHA-256-attested checkpoints and exact cell resume identities.
- Added a Windows Job Object wrapper using the explicitly approved 3,800 MB RAM, 50% CPU, 50 MB/s I/O, and 2,500 MB VRAM envelope.

## Safety

- One architecture/scale/seed cell runs per owned child process.
- Each child is capped at 1,800 seconds and may be resumed from a verified checkpoint.
- GPU contention prevents launch; unrelated processes are never terminated.
- CUDA objects and allocator state are released in `finally`; wrapper cleanup is PID-scoped.

## Tests

```text
21 focused architecture tests passed
PowerShell wrapper parse passed
Python module compilation passed
```

## Next Step

Commit this construction, freeze its hashes, then generate and seal both proposal batches before any calibration or task outcome.
