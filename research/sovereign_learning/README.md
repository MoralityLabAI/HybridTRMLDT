# Sovereign learning research

A frozen low-bit base, an external typed state, specialist adapters, and a learned routing policy are a hypothesis to test against deterministic routing, ordinary retrieval, and equal-budget dense model alternatives.

The authority contract inherits HybridTRMLDT's explicit LDT state and TRM proposer separation. Proposers, router and adapters have no commit authority; only an external environment receipt and independent acceptor can authorize or promote changes. BitAgent remains the benchmark source rather than a forked harness.

Start in [s0_trainability](s0_trainability/). It isolates a low-bit model's train/deploy identity before investing in controller work. It does not change the existing HybridTRMLDT PR #1 calibration work, and calibration success is not interpreted as causal routing advantage.
