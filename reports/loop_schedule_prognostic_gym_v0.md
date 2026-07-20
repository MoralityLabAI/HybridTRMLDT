# Loop Schedule Prognostic Gym v0

## Contract

LSPG-v0 maps a base algebra, bounded mutations, model scale, task context,
hardware budget, and prior receipts to ranked training proposals. It keeps
three channels separate:

1. Theory contains deterministic algebraic invariants and the unmodified
   `p*=(1+gamma)/4` prediction.
2. Empirical prediction contains receipt-calibrated distributions and native
   censored observations.
3. Decision combines information gain, falsification value, diversity, and
   resource cost, then applies explicit exclusion and promotion rules.

The existing P2 cells enter as `left_censored, upper=0.15`. They are never
treated as exact observations at 0.15.

## Implementation

- `lsa/gradient_policy.py` separates parameter-visible applications from
  retained state-transition edges.
- `lsa/canonical.py` provides algebra, model, and run identities.
- `lsa/scale.py` solves the 5M, 12M, 30M, 72M, 170M, and 400M rungs.
- `lsa/cost.py` makes Transformer cost depend on sequence, microbatch,
  precision, checkpointing, and expanded visits.
- `lsa/normalization.py` rejects residual double normalization unless the
  record is a known-invalid control, which remains unrankable.
- `research_gym/prognostics/` implements censoring, features, theory,
  empirical calibration, acquisition, planning, promotion, and reporting.
- `research_gym/neural/looped_decoder.py` is a real causal decoder with a tied
  embedding/head and schedule-dispatched physical Transformer blocks.
- `research_gym/neural/schedule_executor.py` independently freezes parameters
  and detaches state edges.

The initial scale family holds the visit graph and block architecture fixed and
solves width to target unique parameters. Tying controls alter the physical
parameter equivalence classes and re-solve width to remain within 3% of the
same unique-parameter target.

## Sealed Proposals

Twelve S0 proposals were generated and committed before execution. The leading
two are lower-p grid extensions at `p=0.10` and `p=0.05`, as required by the
edge-censoring policy. The remaining batch covers replication, fully untied and
alternating tying, short and interpolated schedules, independent parameter and
state masks, intermediate supervision, detached carry, and post-normalization.

- Proposal table SHA-256:
  `dad24032c40bc8e10648db50554a6277e06e06e265da26530c379b0a5f97e34a`.
- Proposal receipt SHA-256:
  `4ebb010474f048d8347957c98a2ec4d524473252d507dbb715996c39a0a5704f`.
- Known-invalid controls in ranked batch: zero.
- S1-S5: materialized and costed, not executable in Stage A.

## Stage A

The first launch was invalidated before a proposal receipt because WDDM
reported `[N/A]` for per-process VRAM. The invalid receipt is retained. The
numeric-only telemetry correction did not change any scientific input or
proposal hash.

The corrected run reached the registered 1,800-second timeout. Three proposal
receipts were produced:

| Proposal | Outcome | Exposures | Max gradient | Loss ratio | Mean step |
|---|---|---:|---:|---:|---:|
| p-grid extension 0.10 | stopped on first backward | 0 | 235.404 | n/a | n/a |
| p-grid extension 0.05 | stopped on first backward | 0 | 278.823 | n/a | n/a |
| untied R=4 control | completed | 8,192 | 81.435 | 0.311 | 10.593 s |

The alternating R=4 proposal reached its 10% checkpoint with gradient norm
38.575 and loss 404.438, but it has no final receipt and is excluded from the
posterior. Nine proposals remain pending.

The raw timeout trace used Python non-finite JSON values for immediate stops.
It is retained byte-identically. The canonical normalized records replace
those values with JSON `null`, and future execution rejects non-finite JSON at
serialization time.

## Resources

- Process RAM hard cap: 2,048 MB; observed peak 960.488 MB.
- CPU hard cap: 50%; observed average 6.735% of logical capacity.
- I/O abort threshold: 50 MB/s sustained; observed peak 45.703 MB/s.
- Sampled VRAM threshold: 1,500 MB; WDDM did not expose numeric per-process
  memory, while PyTorch recorded 121.374 MB allocated for the completed cell.
- Timeout: 1,800 seconds; run aborted at 1,800.897 seconds.
- Cleanup: passed, no lingering owned PID and no reported compute app.

The cleanup audit observed two unrelated Research_Engine processes and 100%
GPU utilization before and after LSPG cleanup. The measured 27,034x
step-forecast error is therefore confounded. It is evidence that the original
resource model is unusable on this local state, but not a clean estimate of
uncontended device throughput.

## Decision

Promotion is prohibited for four independent reasons:

- Stage A is incomplete.
- The original p boundary remains left-censored.
- The run reached its registered timeout.
- The local resource forecast is badly miscalibrated and externally
  confounded.

No 12M-400M training was launched. The next admissible experiment is another
sealed S0 completion or shorter throughput-calibration batch after the external
GPU workload is cleared; it is not a scale promotion.

No prediction about task accuracy, sample efficiency, or reasoning quality is made. LSPG-v0 remains scoped to trainability boundaries, cost, and experiment selection.
