# Loop-Schedule Kappa State/Moment Decomposition v0.2.3: Registration

## Question

At the sealed `R=64`, `E=2048` cancellation checkpoint, is post-checkpoint kappa rebound controlled primarily by model weights, AdamW moments, the data suffix, or their interactions?

## Frozen design

The experiment crosses model weights `{211, 223}`, complete AdamW state `{211, 223}`, and deterministic suffixes `{211, 223}`. Four unswapped cells are imported from hash-sealed v0.2.1/v0.2.2 artifacts. Only the four missing weight/moment swaps are executed, each from `E=2048` through `E=4096`, with measurements at every 512 state-visit exposures.

Before continuation, every new cell must reproduce the selected model state tensor-exactly and the selected complete optimizer state tensor-exactly, including AdamW first moments, second moments, and step counters. A failed component gate stops that cell before an update.

## Endpoints

The primary response is

\[
y_{m,o,d}=\log\frac{\kappa_{4096}(m,o,d)}{\kappa_{2048}(m)}.
\]

The seven standard two-level factorial effects are computed with `211=-1` and `223=+1`. A term is descriptively dominant only if its absolute effect is at least `1.5x` the runner-up; otherwise the result is `factorial_unresolved`. Recovery, `y >= 0.1`, is a secondary binary diagnostic. Endpoint sensitivity and gradient interference ratios are retained for mechanism interpretation.

This is a saturated deterministic eight-cell design with zero residual degrees of freedom. It cannot support uncertainty-calibrated dominance.

## Safety plan

- Wrapper: `scripts/run_lsa_kappa_surface_phase.ps1`
- Hard caps: 2048 MB process RAM, 50% CPU, 50 MB/s sustained-I/O abort, 1500 MB VRAM
- Time cap: 1800 seconds per phase and one aggregate GPU-hour
- Chunking: one continuation cell at a time
- Checkpointing: one terminal checkpoint per new cell
- Cleanup: explicit model/optimizer release, Python garbage collection, CUDA cache cleanup, owned-process and GPU-app audit in the wrapper receipt

## Claim boundary

The result is scoped to one post-outcome-selected R64 residual-loop construction. It cannot identify individual causal batches or moment tensors and does not generalize to Transformers, task performance, asymptotic depth, other optimizers, or other task families.
