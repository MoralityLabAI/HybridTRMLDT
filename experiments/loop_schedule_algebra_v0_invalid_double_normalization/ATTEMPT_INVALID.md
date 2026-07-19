# Invalid gamma attempt

This pre-prediction run is retained as a negative implementation receipt. The
probe divided the observed directional-alignment normalization by
`(beta/alpha)^2`, while the stability functional also applied that factor.
That double normalization produced `kappa=16` for a one-visible-visit mask at
`beta=0.25`, violating the registered `kappa in [0,R_g]` invariant.

No P2 prediction or boundary outcome was generated from these records. Commit
`56716025fff67b3edb0a7ce40b363b6a99b3bd20` produced the invalid run; the
subsequent estimator correction leaves residual scaling solely in `B(A)` and
adds a one-visit regression test.
