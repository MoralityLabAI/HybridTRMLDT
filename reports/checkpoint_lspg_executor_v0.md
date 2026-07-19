# LSPG-v0 Executor Checkpoint

## Completed

- Added causal LoopedDecoderLM with tied embedding/output projection.
- Added schedule dispatch for tied, alternating/grouped, and untied visits.
- Added independent parameter freezing and state-edge detachment.
- Added final and intermediate supervision paths.
- Added reset and detached carry behavior.
- Added bounded Stage-A trainer with five checkpoint percentages.
- Added a Windows Job Object wrapper with RAM/CPU hard caps, I/O and VRAM
  telemetry aborts, timeout, PID ownership, and resource receipt.

## Tests

The combined LSPG-focused suite passes 29 tests. Actual PyTorch parameter
counts match the scale solver, split gradient masks produce distinct autograd
graphs, and a tiny screening run emits all registered checkpoint percentages.

## Resource Plan

- Process RAM hard cap: 2,048 MB.
- CPU hard cap: 50%.
- Sustained I/O abort: 50 MB/s for three consecutive samples.
- Sampled VRAM abort: 1,500 MB.
- Timeout: 1,800 seconds.
- Chunking: one proposal at a time; no parallel model residency.
- Checkpointing: 1%, 3%, 10%, 30%, and 100% exposure, with one resumable
  `latest.pt` per proposal and metric records for every checkpoint.
- Cleanup: explicit model/optimizer/CUDA release per proposal, Job Object
  process ownership, then the standard post-run memory audit.

## Open Work

Freeze the implementation/config hashes, generate and commit the twelve
proposal batch, then execute only Stage A under the wrapper.
