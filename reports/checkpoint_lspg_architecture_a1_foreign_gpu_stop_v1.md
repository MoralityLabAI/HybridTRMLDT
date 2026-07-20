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
