# Kappa Surface Recovery 1: Untied I/O Abort

The registered `untied_gradients` phase aborted after three consecutive I/O samples above 50 MB/s. Peak sampled I/O was 80.99 MB/s. The wrapper terminated only its owned PID, reported no lingering process or GPU entry, and passed cleanup.

The failure occurred because gradient-only untied cells complete rapidly, causing their unchanged 2048- and 4096-exposure checkpoints to be written back-to-back. The expensive tied kappa probes had incidentally paced the same writes in the preceding phase.

The partial event ledger is preserved at `experiments/loop_schedule_kappa_surface_v1_recovery1/untied_gradients_events.jsonl` with SHA-256 `28b77a2cbb2718b420642bea9ff8a9e106facf5357326912a6dee065e5d00995`. It contains 59 events, 30 gradient measurements, and seven complete cells. None are admitted to the final analysis because the phase did not produce canonical records or a passing resource receipt.

The abort consumed 16.875 seconds, bringing conservative aggregate execution to 1909.626 seconds. A subsequent recovery must retain the 50 MB/s cap, bind to this event and receipt hash, rerun all nine untied cells, and pace checkpoint writes rather than reuse partial outcomes.
