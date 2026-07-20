# Kappa Transient v0.2: Registered Timeout Checkpoint

## Outcome

The first registered `R=128` attempt ended in `phase_timeout` after `1800.713 s`. This is a resource outcome, not a timing classification. The wrapper stopped only owned PID `21144`, cleanup passed, and no result JSON or normalized records file was emitted.

## Admitted Partial Evidence

The append-only event stream contains ten tied-kappa measurements:

| Seed | E=1024 / step 1 | E=2048 / step 2 | E=4096 / step 4 | E=8192 / step 8 | Cell status |
|---:|---:|---:|---:|---:|---|
| 101 | 1.067760 | 0.037271 | 0.494096 | 0.700279 | complete, finite |
| 103 | 0.193642 | 0.548293 | 0.713820 | 0.466505 | complete, finite |
| 107 | 0.595911 | 0.299740 | unobserved | unobserved | incomplete |

Seed 101 individually matches the exposure-pinned rule and recovers. Seed 103 has no registered trough. These signs are descriptive only: the frozen endpoint is the three-seed geometric trajectory, and seed 107 is incomplete.

The seed-107 `E=4096/step-4` checkpoint was written before the timed-out kappa probe. It contains model and AdamW state after exactly four updates and has SHA-256 `a762f8bf26545d3bde924ec4e6d85fb3dfe613a173883c1a1b3a0c2d39762236`. No untied control cell started.

## Resource Receipt

- status: `aborted`
- reason: `phase_timeout`
- elapsed: `1800.713 s`
- peak RAM: `947.539 MB` under the `2048 MB` hard cap
- average RAM: `907.130 MB`
- peak I/O: `13.376 MB/s` under the `50 MB/s` abort threshold
- peak reported VRAM: `0 MB`
- average CPU: `7.843%` under the `50%` hard cap
- lingering owned process: `false`
- lingering owned GPU app: `false`
- cleanup passed: `true`

## Recovery Boundary

A labeled recovery may retain the two complete seed-isolated cells, resume seed 107 only from the sealed step-4 model-plus-optimizer checkpoint, and run all three untied gradient controls from initialization. It must:

1. freeze the admitted event IDs and hashes before new outcomes;
2. verify the checkpoint identity, step, exposure, and hash before load;
3. measure seed 107 at step 4 before any additional optimizer update;
4. continue deterministic streams 4 through 7 to step 8;
5. preserve the original timing rules, seeds, task, optimizer, and practical threshold;
6. debit this attempt's `1800.713 s` against the one-hour aggregate cap;
7. label the final result as a post-timeout deterministic recovery.

The recovery may not classify from the two complete seeds, drop seed 107, reduce the registered probe power, or treat the checkpoint as a new initialization.

## Artifact Hashes

- partial event stream: `09f54528a6deed960d08f7acd01b66edd897f6e9454b20740b57652c374b213e`
- resource receipt: `dc82c934997eb6789b6f38afa688366368c95778d1d99bf95f26542d22cb06e5`
- seed-101 step-4 checkpoint: `f84965b652ea55987e83baa38d2997cf8dc514717f7b99c473c9865694b11100`
- seed-101 step-8 checkpoint: `1807f1d5c13594354dbe35b322e0b5221854c95323e0f0e32a81d1903f7888e0`
- seed-103 step-4 checkpoint: `c0fc6e2eed888d50d2c3ce20d77ae9f987cbac6f0a6d3141f4cf9655359a1a24`
- seed-103 step-8 checkpoint: `d401132947d6292de4bfe1186357292c30b462af6a239dc4a5bcca6e5b7d5aa5`
- seed-107 step-4 checkpoint: `a762f8bf26545d3bde924ec4e6d85fb3dfe613a173883c1a1b3a0c2d39762236`
