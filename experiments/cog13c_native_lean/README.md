# COG-13C — native Mathlib tactic portfolios

Stacked on [COG-13B #35](https://github.com/MoralityLabAI/HybridTRMLDT/pull/35). Both its replay and native Lean jobs passed in [45850214483](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/45850214483).

## Registered engineering pilot

24 deterministically generated **Mathlib-backed theorem statements**, eight train, eight ordinary, eight deeper. Four families: Nat inequalities, rational polynomial equalities, Finset membership, and mixed propositional/arithmetic goals. These are new synthetic statements, not original Mathlib theorem bodies, Cogentic discoveries, or unseen theorem families. The deeper partition has four/five conjunction clauses vs one/two in train/ordinary. Distinct statements within cohort; no theorem-isomorphism, semantic novelty, or external leakage audit. No reference proof bodies enter the controller.

Use Lean 4.35.0 and Mathlib commit 56a005855ea17e0b0958f73ec801ae140b0a1679. The original RAM15 ProbeServer at LoopedTransformers commit c25fe023b40eb4a802f4bb733f777d6d68067c88 has Git blob eb6885674e0ce25da78833b64a8a65c3752346a6 and is checked before compilation. The probe server's internal sorry on shutdown is quarantined; it cannot certify a result. Every success requires two independent full-source compiler processes **without** importing ProbeServer, identical proof-source SHA-256, exit zero, no error/sorry, and an axiom inventory limited to propext, Classical.choice, Quot.sound (or empty). Both must finish within the portfolio's wall budget. Compiler rejection/timeout is not a successful theorem.

Fixed 16-tactic menu. Training explores the menu at each of at most five teacher-reached states per training root. The environment supplies accepted/changed/closed labels; unknown/timeouts do not produce negative training labels. Each known progress label is based on printed-goal change, not semantic necessity. The teacher chooses a closed candidate or first changed candidate only after evaluating training candidates. Training labels have explicit source roots.

The recurrent proposal controller has 553 parameters, 32 public goal/tactic features, hidden width 12 and three tied tanh passes; trained using backpropagation through recurrence, 20 epochs, seed 1701, SGD. No held-out tuning. Its input W is approximated by four orthogonal power-iteration factors learned from W. The BPD-inspired packet arms transmit four components using float32 or packed 4-bit quantized coordinates; quantizer scales come solely from training feature activations. Receiver decodes the actual packet before applying shared recurrent weights and readout. Dense reference transmits 32 float32 features. Packet has four-byte header plus packed data and one-byte reply. Bytes count the declared in-process protocol, not network I/O or model setup. It is a new factorized channel, not native BPD TiedComponentLinear or a theorem about log-rank.

The LDT precursor is a five-feature root-weighted ridge predictor of remaining probe calls, trained from two separate dense static-block portfolios on train roots; unfinished trajectories are censored and counted. Requires at least two independently certified training roots before evaluation. The trained neural tactic proposer and learned scheduler are distinct. A full neural LDT is not implemented.

## Primary comparison

28 paired portfolios (seven arms × two four-root portfolios × two held-out splits). Same 32 selected native probe cap and 90-second whole-portfolio wall budget; deterministic rotated arm order. Independent final certification calls are counted separately, with their wall cost inside the same deadline. Up to ten accepted script actions and eight goals. All root outcomes are retained. Arms:

- native round robin with one or two resident Lean sessions
- native static statement-length block scheduling, one session
- dense recurrent static block, one session
- dense recurrent adaptive scheduler, one session
- rank-four float32 adaptive scheduler, one session
- rank-four packed 4-bit adaptive scheduler, one session

The native ProbeServer resets to the theorem's initial saved tactic state **on every query**, including queries to a resident server. Each submitted prefix is replayed internally. On session eviction the host retains exact script prefixes, displayed goal states and tried-action records. Reopening checks the initial goal display. This is an exact external script ledger and session-residency study; true native resumable intermediate MetaM checkpoints remain future work. Prefix-command counts are submitted counts, not exact number executed after failures or timeouts. A printed-goal signature is only a heuristic for search-loop suppression.

Receipt includes probe and certificate counts, every native response, source and log hashes, wall time, peak sampled resident process RSS, UTF-8 ledger metadata size and per-process GNU time user/system seconds and max RSS. CPU records may be absent for killed process groups; report the number with complete timing rather than imputing zero. Sampled RSS can double-count shared pages and miss startup spikes; it excludes parent Node, filesystem cache and other host memory. No physical-RAM limit or matched actual CPU time has yet been imposed. A whole portfolio may overrun during cleanup; certificates completing after deadline get no credit.

## Run

Fixture: `node experiments/cog13c_native_lean/run.mjs --fixture /tmp/cog13c-fixture.json`

Native workflow installs pinned Mathlib cache, compiles the pinned probe server and runs the native matrix. Native outputs include result.json, every probe and certificate file, process_cost_by_phase.json, and DataGoblin-compatible training-only exports with manifest (no changes to DataGoblin repository).

## Current evidence

The portable JS fixture executed all 28 portfolios with a deterministic fake backend: 128 toy tactic labels, eight toy residual labels, a forged-certificate rejection and exact 11-byte encoding for three rank-four four-bit candidates. This is a software-contract test, **zero native Mathlib measurements**. Native runs require the CI runtime; the current execution sandbox has neither Lean nor external network access. Native workflow results will determine whether the training gate and benchmark succeed. No native learned advantage, transfer or RAM result is claimed. Source pins and all development choices precede native scoring.

Further work after this pilot: original MathCan targets; independent source replay and policy auditing; three training seeds; calibrated heldout residuals and censor-aware learning; genuine saved-state checkpoints, native BPD component API and learned LDT router; matched CPU-time and cgroup memory caps. Only then compare theorems per CPU-hour across memory tiers.
