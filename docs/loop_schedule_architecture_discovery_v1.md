# Loop Schedule Architecture Discovery v1

## Question

Can the ordering of repeated physical Transformer blocks improve held-out
algorithmic and routing performance when unique parameters, applied visits,
optimizer exposure, data, and inference budget are matched?

The campaign seeks at most two schedule topologies. A topology is called novel
only when its canonical module word is not equivalent to a registered reference
inside the frozen finite grammar. This is not a global architecture-novelty
claim.

## Intervention

For physical modules `K in {2, 3, 4}` and expanded visits `L in {6, 8}`, the
search enumerates canonical restricted-growth words. It retains words with
balanced module usage and primitive period `L`, then removes the periodic
module-order templates used as controls.

Each candidate is paired with the periodic control having the same `(K, L)`.
The pair has identical physical blocks, hidden width, unique parameter count,
expanded visit count, estimated FLOPs, optimizer exposure, normalization,
residual scaling, gradient policy, supervision, and task batches. Only the
module dispatch order changes.

Twelve candidates are selected without outcomes by deterministic farthest-point
sampling over seven schedule descriptors. Twelve additional candidates are
sealed as the only permitted reserve batch. One balanced null is retained per
stratum, with fully tied and fully untied models used only as contextual
baselines.

## Task Contract

All models use the same 2,048-token vocabulary and 64-token examples. The five
families are pointer chasing, modular recurrence, rewrite normalization, 4x4
Sudoku query completion, and deterministic routing examples. Synthetic task
depth is held out: train depths are 1-4, calibration depths are 5-6, and locked
evaluation depths are 7-8. Sudoku full-puzzle correctness and routing macro
accuracy are retained alongside answer-token accuracy.

The dataset bundle is materialized once and hash checked. Architecture search
cannot consume the locked evaluation split before Stage D.

## Funnel

The execution ladder is:

1. Stage A1 screens 12 candidates and matched controls at 5M parameters.
2. Stage A2 replicates at most six survivors over three seeds at 5M.
3. Stage B tests at most four candidates on Sudoku and routing at 5M.
4. Stage C transfers at most four candidates to 12M over all five tasks.
5. Stage D confirms at most two structurally distinct finalists at 30M on the
   locked split.

If fewer than two candidates survive Stage B, the sealed reserve may be opened
once. Scales 72M, 170M, and 400M are materialized and costed but cannot execute
in v1. A candidate can be promoted only if at least two seeds are positive, no
task regresses by more than 0.03, step time is within 1.10x of its matched
control, and the finite-run guards pass. Final claims use paired hierarchical
bootstrap intervals and Holm correction.

## Resource Calibration

Before task outcomes, a fixed-periodic control at 5M, 12M, and 30M runs for 20
warm-up and 30 measured optimizer steps. The measured mandatory-path forecast
selects the largest of the full, medium, or minimum token-visit budgets fitting
within 24 GPU-hours, leaving 12 GPU-hours for the one-time extension. If the
minimum mandatory path does not fit, the sealed outcome is construction
failure rather than a changed protocol.

Every training child is isolated under the approved limits: 3,800 MB RAM, 50%
CPU, 50 MB/s I/O telemetry, 2,500 MB VRAM, and 1,800 seconds per invocation.
Checkpoints are atomic, SHA-256 attested, paced to 40 MB/s, and written every
100 optimizer steps or 120 seconds. Resume count is capped at four per cell.

## Literature Boundary

`architecture_literature_registry_v1.json` records the primary sources checked
before proposal generation. Published mechanisms involving adaptive halting,
residual scaling, multi-resolution recursion, context anchors, depth
conditioning, external memory, latent-explicit interfaces, role
specialization, or hierarchical timescales are not claimed as discoveries by
this grammar. Fully tied, fully untied, and periodic repeated-stack schedules
are controls or contextual references, not candidates.

## Claim Boundary

A positive result means that a sealed schedule word transferred better than its
matched periodic control under this task bundle and resource policy. It does
not establish general language-model superiority, global novelty, mechanistic
causality beyond dispatch order, or transfer beyond the measured scales and
tasks. Fewer than two qualifying winners, including zero, is a valid result.
