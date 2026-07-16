# Checkpoint: Gaming Versus Oversight Leverage

Historical study ID: `gaming_vs_improvement_v1`. The improvement comparison is power-limited in this run.

Date: `2026-07-16`

## Completed

- Added an optional provenance-verifier seam while preserving default membrane trace shape and behavior.
- Added an optional recurrent proposer, deterministic expert-iteration rollout, grouped probes, drift, and
  matched random/orthogonal controls.
- Froze and ran the full evidence by rejection by adaptation cross over three seeds and five rounds.
- Enforced contiguous held-out regions, under-training margin, config hashes, arm alias collapse, and canonical
  decision receipts.
- Saved 26,880 full decision records plus a 1,152-record smoke and mirrored experiment notes.
- Canonicalized JSONL bytes so the embedded records digest matches the on-disk and mirrored artifact exactly.
- Added true probe-component ablation at all 18 full seed-round checkpoints; it changes acceptance in 9/18 checks,
  while orthogonal additions change 0/72.

## Quantitative Result

Exact mechanics plus state-conditioned fallback changes 24.0% of decisions and improves mean utility by
`+0.0407`. This is controller utility supplied by the gate-plus-fallback, not proposer learning. Identical fallback
has exactly zero utility delta in every evidence arm and round.

Claim-only adaptation raises false environment-sound claims to 100% in all three seeds without improving exact
soundness. Exposed-probe pass rate reaches 100% while exact soundness is flat; hidden-audit AUROC falls in two
seeds and rises in one, yielding two evasion classifications and one retained counterexample.

The hidden rotating probe rejects every final proposal and has a roughly 79% false-rejection share. Dual channel
aliases exact mechanics at the final checkpoint because all adapted claims are environment-sound.

The exposed-probe plus distinct-fallback arm changes 48.7% of actions and supplies `+0.0404` utility with a frozen
proposer. After expert iteration it changes 0.0% and supplies `+0.0000`: full probe passage silently removes the
entire observed oversight benefit. The hidden rotating arm instead changes 96.1% of actions, but roughly 79% of
its rejections are false rejections; its `+0.0376` depends on this gym's near-oracle fallback.

Round-0 oracle accuracy equals the majority baseline in every seed, and exact soundness never moves. The run
cleanly measures gaming and controller-leverage loss but has no power to compare them with genuine proposer
improvement.

## Boundary

This is a toy deterministic storyworld control experiment. It does not establish general alignment, neural AIRIS
performance, production robustness, mechanistic verification of a constitution, or how a proposer capable of
above-majority learning trades improvement against evasion.

## Verification

- full records SHA-256: `e587b8456fda8e2183694f7b6c1e40660b2cfe06e5e048b319f0a040f5f3cc9d`
- full records / receipts / non-eval rows: `26880 / 26880 / 0`
- receipt integrity failures: `0`
- canonical-to-mirror byte mismatches: `0`
- dependency-free import under `python -S`: passed with `torch_available=False`
- full suite: `135 passed`
- simulated install without the neural extra: `122 passed, 2 skipped`

## Next Step

Repeat the exposed-versus-hidden comparison on a non-enumerable storyworld or region family with an action model
that exceeds its majority baseline, then test whether probe evasion persists when proposal behavior itself has
room to improve.
