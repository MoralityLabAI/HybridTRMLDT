# Kappa Checkpoint Splice v0.2.2 Registration

## Question

Does post-`E=2048` data order control exit from destructive sensitivity interference, or does pre-`E=2048` path dependence determine whether recovery is possible?

## Frozen Arms

| Source state | Continuation order | Role |
|---:|---:|---|
| 211 at E2048 | 211 | tensor-exact replay gate |
| 211 at E2048 | 223 | recovering-suffix transfer |
| 211 at E2048 | 227 | recovering-suffix replication |
| 223 at E2048 | 211 | reciprocal transfer |
| 211 at E4096 | 211 through E8192 | right-censoring resolution |

The replay must reproduce the sealed order-211 `E=4096` model and complete AdamW state tensor-exactly and kappa within `1e-12` before splice outcomes are interpreted.

## Branches

- `suffix_controlled_exit`: both recovering suffixes rescue state 211 by `E=4096`, while suffix 211 prevents recovery from state 223.
- `prehistory_controlled_exit`: neither recovering suffix rescues state 211, while state 223 recovers under suffix 211.
- `mixed_state_suffix_control`: every other complete pattern.
- `delayed_recovery`: unchanged order 211 reaches `log(K/K2048)>=0.1` by a measured checkpoint through `E=8192`.
- `right_censored_at_E8192`: unchanged order 211 remains below that rule through `E=8192`.

Splice kappa is measured after every continuation step through `E=4096`; the extension is measured at `E={5120,6144,8192}`.

## Resources

- Serial continuation arms; no prefix retraining.
- Terminal model/optimizer checkpoint per arm.
- Hard caps: `2048 MB` RAM, `50%` CPU, sustained `50 MB/s` I/O abort, `1500 MB` VRAM, `1800 s` timeout, one aggregate GPU-hour.
- Abort and cleanup failures remain first-class outcomes.

## Claim Boundary

The splice can causally localize exit control within this one post-outcome-selected `R=64` construction. It cannot identify a single causal batch, establish an intrinsic stability boundary, or generalize to Transformers, task performance, asymptotic depth, or other task families.

Frozen config SHA-256: `7462fdca673324a593c32b23e5394147dba2a0b0c3d3cf41d882d5550bc600b1`.

Full repository verification before registration commit: `329 passed in 35.77s`.
