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

For **raw matrices** `C` and `Q`, let `a=max spec(C)` and
`b=max spec(Q)`. The maximum of the unnormalized Kronecker sum is `a+b`.
The *single-stage* normalization `Bhat=B/(a+b)` (when `a+b>0`) yields

`h_t(Bhat)=h_{t/(a+b)}(C) h_{t/(a+b)}(Q)`.

This follows from the unnormalized factorization with a shared time
rescaling. It is **not** the complete repository computation.

### Actual frozen v1 implementation: normalize factors, then product

Source inspection on `master` at `f162153f1c397a93d4750a4cce3edb1d5fa1cab7`
found the following exact sequence:

1. `research_gym/analysis/controller_mesh_sheaf.py:219-230` computes
   controller `laplacian_spectrum` with `eigvalsh(...).clamp_min(0)`
   and divides by its strictly positive maximum.
2. `research_gym/analysis/controller_mesh_stalk_bridge.py:124-151`
   independently computes and maximum-normalizes the measured
   restriction graph spectrum under the same `maximum > 0.0` rule.
3. `controller_mesh_stalk_bridge.py:172-186` constructs
   `controller + coupling * measured` for each spectral pair and
   again divides by the pairwise maximum if positive.
   Registered `configs/controller_mesh_measured_stalk_bridge_v1.json`
   freezes `coupling=1`.

Formally, define `Cbar=C/a` for `a>0`, otherwise `Cbar=C=0`;
similarly `Qbar=Q/b` for `b>0`, otherwise `Qbar=0`. Let
`kappa>=0` denote the coupling, and let `abar=max spec(Cbar)` and
`bbar=max spec(Qbar)`, each in `{0,1}`.
Then the production-style pre-final sum is
`D=Cbar⊗I + kappa I⊗Qbar` with maximum
`M=abar+kappa*bbar`. For `M>0`, the actual heat trace is

`h_t(D/M) = h_{t/M}(Cbar) * h_{kappa*t/M}(Qbar)`.

When both factors are nonzero and `kappa=1`, `M=2` exactly:
`h_t(D/2)=h_{t/2}(Cbar) h_{t/2}(Qbar)`.
Both raw factor scale magnitudes have been discarded before the product.
A naive `h_t(Cbar)h_t(Qbar)` misses the final shared division.

For `C=diag(0,1)`, `Q=diag(0,3)`, `t=1`:
- Single-stage raw normalized spectrum: `(0,1/4,3/4,1)`,
  heat trace `(1+exp(-1/4)+exp(-3/4)+exp(-1))/4` ≈ `0.6547616942`.
- **Production two-stage normalized spectrum:** `(0,1/2,1/2,1)`,
  heat trace `(1+2exp(-1/2)+exp(-1))/4` ≈ `0.6452351901`.
- Incorrect product at unscaled `t=1`: `((1+exp(-1))/2)^2`
  ≈ `0.4677735414`.

If both factors are exactly zero, `M=0` and division is skipped in
production, yielding a zero spectrum and normalized heat trace 1.
With one zero factor, `M` is the positive contribution from the other.
**Any strictly positive maximum is normalized**, even at `1e-20`;
the reference now follows this behavior. The registered zero tolerance
`1e-8` determines spectral-gap and band features, not whether spectral
maximum normalization executes.

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
Orthogonally conjugate operators share spectra. The CPU tests contain two
**non-isomorphic connected six-vertex graphs** with equal Laplacian
spectra: their degree multisets differ, which certifies non-isomorphism.
Thus spectral features cannot identify which underlying source graph or
state/context labeling generated them. Separately normalizing a fixed
source also erases any positive global scaling of its Laplacian.
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
relabelings, a non-isomorphic Laplacian-cospectral pair, strict positive
near-zero scaling, and the production two-stage-normalization counterexample.

**Local execution receipt, 2026-10-31:** Python 3.13.5, 9/9 committed
standard-library unittests passed in 0.004 seconds. The executed source
and test files were matched byte-for-byte by Git blob SHA
`738cb83bdaf161d67266ea484e509a25dbd30bdd` and
`29af4e48fcb97c2be4cde399c9abd4fd22145291`.
An independent deterministic NumPy `eigvalsh` check over 90 tiny PSD
matrix pairs (dimensions 1–4 per factor) found maximum absolute spectral
discrepancy `1.7763568394002505e-14`. NumPy is used for this separate
sanity check; the committed tests require only the standard library.
No GitHub Actions CI run was present when checked.

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
