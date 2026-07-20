# Kappa Transient v0.2 Recovery 1: Attempt-1 Construction Failure

Recovery attempt 1 failed before the replay gate emitted an event and before any new kappa or gradient-control outcome was produced.

The deterministic replay reached the optimizer-state comparison, but `torch.equal` was called on one CPU tensor and one CUDA tensor. PyTorch rejected the cross-device comparison before testing values. The registered gate requires tensor-exact values and shapes, not identical storage devices, so the implementation must compare device-normalized tensors.

Resource receipt:

- status: `failed`
- reason: `process_exit_1`
- elapsed: `12.988 s`
- prior debit: `1800.713 s`
- aggregate debit: `1813.701 s`
- peak RAM: `790.410 MB`
- peak I/O: `18.964 MB/s`
- lingering owned process/GPU app: `false/false`
- cleanup passed: `true`

No `recovery_events.jsonl`, combined records, result, or final receipt exists. Attempt 2 may retain the frozen recovery protocol and normalize both compared tensors to CPU before an exact dtype/shape/value comparison. The wrapper must retain attempt 1's debit automatically.
