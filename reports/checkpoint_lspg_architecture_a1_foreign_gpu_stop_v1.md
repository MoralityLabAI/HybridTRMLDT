# Checkpoint: LSPG A1 Foreign-GPU Stop v1

## Event

A1 completed and sealed seven of eighteen cells before a separately owned
`llama-server.exe` process acquired the GPU. The eighth cell failed during the
wrapper preflight with `foreign_gpu_compute_process_present`; no model process
was launched for that cell. The failed stage receipt and all seven completed
cell receipts were committed at `f256a5a` before any retry.

## Operational correction

A new stage-runner invocation previously restarted its per-cell attempt number
at one. That would overwrite the working-tree path of the already committed
foreign-GPU receipt. The runner now derives the next attempt number from sealed
resource-receipt filenames, so the next launch of the eighth cell writes
`attempt-2` and preserves `attempt-1` directly on disk as well as in Git.

This changes receipt bookkeeping only. The sealed A1 manifest, proposal set,
model precision, task data, seed, token-visit budget, resource caps, and
promotion rules are unchanged. Completed cells remain content-addressed and are
skipped on resume.

After cell eight completed on `attempt-2`, a different foreign
`llama-server.exe` process acquired the GPU before cell nine. The second failed
stage receipt was committed at `7a56b32`. This exposed a separate accounting
issue: repeated preflight refusals could exhaust the maximum-resumption counter
without launching a model. The counter now limits receipts with an owned model
PID; preflight refusals retain monotonically increasing attempt numbers but do
not consume a training attempt. This is also an orchestration-only correction.

The second process was subsequently confirmed to be `llama-server.exe` with
the literal command-line pair `--n-gpu-layers 0`, zero sampled GPU utilization,
and no remaining client connection. Execution amendment v1.3 was frozen at
canonical SHA-256 `c74391f0b3072022424016e4bfa0872c2f1674ad248d7e26065d33aa56fe5855`
before implementation. The wrapper may ignore only that exact CPU-only
llama.cpp classification, records every ignored PID, and still rejects missing
command lines, other binaries, nonzero GPU layer counts, or utilization above
the registered threshold.
