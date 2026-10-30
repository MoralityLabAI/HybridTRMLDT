# Source-specificity mathematics for the measured-stalk bridge

Status: **offline algebraic audit**, separate from the registered bridge study.
Parent: `master` at `f162153f1c397a93d4750a4cce3edb1d5fa1cab7`.
Original protocol and outcome: [controller_mesh_measured_stalk_bridge.md](controller_mesh_measured_stalk_bridge.md).
No existing endpoint, task split, baseline, source receipt, or observed statistic was changed.

## Frozen empirical constraint

The registered source-weight null rejects attribution: observed combined rho
`0.7523`, typed N0 shuffle mean `0.7582`, and N0 p `0.9380`
(120/128 nulls at or above observation, with finite-sample correction).
The categorical bootstrap lower bound is `-0.0855`; top-16 uplift trails the
best baseline by `-0.0112`. These are negative source-specificity findings.

## Exact spectral identities

Let `C` be real symmetric n-by-n PSD and `Q` be real symmetric m-by-m PSD,
and write `B=C⊗I_m+I_n⊗Q`. The spectral theorem gives eigenvectors
`u_i⊗v_j` with eigenvalues `c_i+q_j`. Equivalently the two summands
commute, so `exp(-tB)=exp(-tC)⊗exp(-tQ)`. Hence for `t≥0`:

`Tr exp(-tB) = (Tr exp(-tC))(Tr exp(-tQ))`.

With `h_t(A)=Tr exp(-tA)/dim(A)`, the dimension-normalized identity is
`h_t(B)=h_t(C)h_t(Q)`. This is the **unnormalized matrix** identity, despite
the dimension average. It applies before spectral maximum normalization.

## Product maximum normalization

Let `a=max spec(C)` and `b=max spec(Q)`. Since both are PSD,
`max spec(B)=a+b`. When `a+b>0`, the actual implementation normalizes
pairwise sums **once**, giving `Bhat=B/(a+b)` and

`h_t(Bhat)=h_{t/(a+b)}(C) h_{t/(a+b)}(Q)`.

Equivalently for `a,b>0`, with `Chat=C/a`, `Qhat=Q/b`, and
`alpha=a/(a+b)`, this is
`h_{t alpha}(Chat) h_{t(1-alpha)}(Qhat)`.
A product of individually normalized heat traces at unscaled `t` is in
general false. For `C=diag(0,1)`, `Q=diag(0,3)`, `t=1`, the correct
trace is `(1+exp(-1/4)+exp(-3/4)+exp(-1))/4`; the individually normalized
product instead is `((1+exp(-1))/2)^2`, which differs.

If `a+b=0`, PSD implies both matrices are zero; normalization by the
maximum is undefined mathematically. The tiny reference defines a
zero-spectrum convention of all zeros, yielding heat trace 1. A successor
must register its zero case and verify exact agreement with production
handling before applying the reference as an implementation oracle.

## Information and identification limits

For a fixed source `Q`, every spectral product feature is a deterministic
function of `spec(C)` and the constant multiset `spec(Q)`. In particular,
the bridge's gap, spectral band counts, and heat traces depend only on the
sorted pairwise sums and their maximum. A model receiving a coarse six-feature
summary of `C` can improve using these richer nonlinear features even if
`Q` provides zero source-identity information. This is **feature
redundancy relative to the full controller spectrum**, not necessarily
redundancy relative to six lossy controller features.

Any isospectral symmetric PSD sources `Q` and `Q'` give identical bridge
features for every controller, including after maximum normalization.
Orthogonally conjugate operators share spectra; distinct graphs may also be
Laplacian-isospectral. Thus spectral features cannot identify which
underlying source graph or state/context labeling generated them.
Even a successful predictive gain would require an identity-sensitive null
to attribute value to a measured source. Fixed source identity cannot be
estimated from one repeated source across all controller cases.

## Offline CPU certificates

Run `python -m unittest tests.test_stalk_source_specificity_math -v`
from the repository root. This uses only the standard library. The
independent explicit Kronecker matrix is diagonalized with a tiny symmetric
Jacobi solver, then compared with the pairwise sums. Heat traces are
checked independently through both routes. Fixtures include diagonal and
nondiagonal PSD matrices, the all-zero limit, isospectral path-graph
relabelings, and the individually-normalized-product counterexample.
Numerical tolerances are test-local; these certificates prove no statement
about real activation geometry or predictive performance.

## Separately registered successor proposal (not executed)

Register a **new study ID and protocol hash** before outcomes are visible.
Recruit multiple independently measured source stalks with externally
justified source-to-task-family associations, using frozen extraction and
controller representation. Reserve entire source identities and complete
task families for evaluation; split discovery, validation, and evaluation by
identity rather than by repeated rows. Freeze mapping, preprocessing,
normalization, features, ridge settings, selection budget, null draws,
metrics, uncertainty intervals, and gate thresholds in advance.

Compare correctly matched sources with (i) misassigned real source
identities, (ii) topology- and edge-type-constrained weight shuffles,
(iii) synthetic symmetric PSD sources with **matched eigenvalue multisets**,
and (iv) controller-full-spectrum flexible baselines under matched model
capacity. The isospectral control is an exact invariance test: a purely
spectral bridge must tie it. Identity discrimination requires
identity-sensitive features and evidence beyond this spectral map.
Compute uncertainty across independent source/task-family clusters,
report leakage audits and all tests, and use a registered selection metric.
When there are too few independent sources or task families to assess that
metric, mark the gate `INSUFFICIENT_EVIDENCE`. Preserve all existing
results and hashes, with no retuning or retrospective gate change.
