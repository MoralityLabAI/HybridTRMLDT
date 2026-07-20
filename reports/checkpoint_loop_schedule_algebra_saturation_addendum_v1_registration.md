# Loop Schedule Algebra saturation addendum v1: registration checkpoint

## Decision timing

This addendum was authored after the sealed Loop Schedule Algebra v0.1 result
showed `kappa(16)=2.550934` against the preregistered power-law prediction
`3.315221`, and before any `R=32` or `R=64` outcome was generated. It is a
labeled addendum, not part of the original v0.1 preregistration.

## Frozen question

Does terminal tied visit alignment in the unchanged small residual-loop family
favor bounded saturation or continued logarithmic growth? The addendum trains
matched tied and untied `R=32` cells, fits a saturating exponential and a
logarithmic model on `R={2,4,8,16,32}`, commits and pushes their untouched
`R=64` predictions, and only then permits the `R=64` cells to run.

The saturating form is

```text
kappa(R) = K - A exp(-R/tau), A >= 0.
```

The unbounded slow-growth alternative is

```text
kappa(R) = a + b log(R), b >= 0.
```

The holdout winner needs both a `0.05` absolute-error advantage and a winner to
loser error ratio no greater than `0.75`. Calling saturation additionally
requires fit `R^2 >= 0.9`, `kappa(64) <= 3.0`, and registered tail-growth
bounds. Anything weaker is reported as logarithmic or unresolved, never folded
into a saturation claim.

## Resource and integrity envelope

- Windows Job Object: `2,048 MB` process memory and `50%` CPU hard caps.
- Sustained I/O abort: `50 MB/s` for three samples.
- VRAM abort: `1,500 MB`; Torch allocator fraction: `0.35`.
- Phase timeout: `1,800 s`; aggregate addendum budget: one GPU-hour.
- Chunking: one rounds-by-regime-by-seed cell at a time.
- Checkpoint: terminal `4,096` state-visit exposures for every new cell.
- Foreign GPU processes cause a preflight stop and are never terminated.
- The `R=64` phase must verify that the prediction artifact is tracked, clean,
  committed, reachable from `HEAD`, and present on the remote branch before it
  reads or produces any holdout record.

## Claim boundary

If alignment saturates, its local effective exponent tends to zero and the
DeepLoop threshold expression tends toward `p=1/4`. This does not erase the
explicit `R_g` factor in `B(A)`, so the addendum cannot claim that every
depth-dependent stability cost vanishes. The family remains a small synthetic
residual-loop measurement, not a task-performance or general-Transformer test.

The config SHA-256 and registration commit are recorded in
`configs/loop_schedule_algebra_saturation_addendum_v1_registration.json`.

## Construction correction

The first registration commit `ade595e` transcribed the sealed parent receipt
hash as `3fa8c165...7414`. Pre-outcome validation rejected it; the actual
SHA-256 is `3fa8c1e6...57414`. No `R=32` or `R=64` outcome directory existed.
The superseding registration changes only that source hash and its registration
timestamp; the scientific design, decision thresholds, seeds, and resource caps
are byte-for-byte unchanged.

## Validation receipt

Committed runner `911fbc3` passed capped validation before any new training:

- config SHA-256: `f4eb1cd7d1df7a18f523498ed48a159ccc2f123c80957913301c51cc1cc30894`
- parent primary records SHA-256: `1fc75ed2c26d110d9b5a3205a12683dbbb0c11cd1dee0a18b2866466b9ea090e`
- resource receipt SHA-256: `236e1174afd9dfaaaf2eef5ac43b607582bda69321d6de88c9c7339b933091dc`
- peak RAM: `380.148 MB`; peak I/O: `6.544 MB/s`; peak VRAM: `0 MB`
- cleanup: passed; `R=32`/`R=64` records present: false
