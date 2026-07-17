# Checkpoint: Above-Majority Gaming Versus Improvement

Date: `2026-07-16`

## Protocol

- Frozen config: `gaming_vs_improvement_v2_above_majority`
- Config SHA-256: `7feb9d72997f844ba6736bf8cfe3deee36b63cb1b63f2b5367ccb3ab40c4469a`
- Region family: state-hash buckets `0-4 / 5-6 / 7 / 8-9` for train / probe / development / eval
- Development panel: nine seeds, minimum `+0.023` above majority
- Final panel: three seeds, `+0.062` to `+0.117` above majority
- Remaining oracle headroom: `0.203` to `0.211`
- Adaptation capacity: `0.133` minimum development action-change under exact/exposed gates

## Result

All 30 adaptive evidence by rejection by seed trajectories degrade oracle proposal behavior. There are zero
genuine-improvement classifications. Mean proposal-utility deltas range from `-0.0079` to `-0.0153`; mean
oracle-optimal deltas range from `-0.0729` to `-0.1484`.

The exposed gate produces strict evasion in seeds 223 and 227. Seed 211 is mixed: exact soundness rises `+0.0156`,
but proposal utility, hidden AUROC, and oracle-optimal rate fall. Claims saturate to 1.000 in every seed.

Exact mechanics plus distinct fallback is the strongest adapted controller at `+0.0242` executed utility versus
its degraded proposal. This is correction, not proposer learning. The exposed-probe controller loses about 85% of
its fallback benefit (`+0.0042 -> +0.0006`) as action change falls (`0.549 -> 0.104`).

The result distinguishes control from teaching: exact verification protects execution, but accepted-only
self-imitation supplies no counterfactual lesson on rejected states.

## Preflight Receipt

The first full invocation aborted without writing artifacts because the harness incorrectly enforced the
development margin on each final-seed development resample. The frozen final margins all passed. The correction
kept the config byte-identical, enforced development power only on calibration seeds, and preflighted all final
seeds before every arm. `test_all_seed_power_preflights_before_any_arm` guards this ordering.

## Integrity

- full records: `26,880`
- full records SHA-256: `08c9412b37201e96e489003bf87878775c279829a63847c247848e25c84017b6`
- smoke records SHA-256: `58ee323e25b36c60185c67b1e5bf1266fa85755a77ab937156740bc99e45c9fc`
- calibration JSON SHA-256: `5e6c2157459a7b8897676886c02fe482703c19545d0a0856701c1c57ec0abe8d`
- receipt failures: `0`
- non-eval record rows: `0`
- region group overlap: `0`
- canonical-to-mirror mismatches: `0`
- causal seed-round checks: `18`
- full neural suite: `141 passed`
- simulated install without neural extra: `122 passed, 2 skipped`
- dependency-free core import: passed with `torch_available=False`
- v1 records remain `e587b8456fda8e2183694f7b6c1e40660b2cfe06e5e048b319f0a040f5f3cc9d`
- Fable ZIP remains `c4c220fea7cf800a1813696b36078a8755233d80642d4e12568b10a74cf96ef1`

## Next Step

Add a correction-infused adaptation arm that trains on independently verified fallback or oracle actions for
rejected states. Compare it with accepted-only self-imitation while keeping the verifier score out of the target
and preserving the hidden audit.
