# Loop Schedule Algebra v0.1: external-validity checkpoint

The mini causal attention+MLP loop completed all 24 registered cells on
deterministic prefix-context regression. Every run was finite. The tied fit was
`gamma=0.501246`, `R^2=0.854021`; the visit-untied point estimate was
`gamma=0.017859`, `R^2=0.050919`. Their gamma contrast was `0.483387`.

The frozen composite replication rule is **not confirmed** because it required
both fits to have `R^2>=0.8`. The untied negative control is nearly flat rather
than power-law shaped: its geometric-mean kappas over `R={2,4,8,16}` were
`{0.7338,0.7849,0.8571,0.7426}`. A log-linear fit explains little variance in
that flat sequence. The criterion was poorly chosen for a null exponent, but it
is not relaxed after outcomes.

The licensed interpretation is narrower. The tied attention+MLP loop shows a
finite, good-quality positive exponent on a second task family, while the
untied control has a near-zero point estimate and poor power-law fit. This is
positive mechanistic evidence against the result being unique to two linear
layers, but not a pass under the preregistered external-validity gate and not a
language-model claim.

Artifact hashes:

- records:
  `6fea7187bc85f08ef2c969362c87376f609ee1b2843786dc87363877e534c991`
- result:
  `6cf4e3e0281970734b79ad8ff6a668934abe73f1e7d40b1ec5372b207f75ab95`
- resource receipt:
  `a46e51b58d3e0fab59b9c1b5c6d69561c5bd12194798d811e4e543cb8f64faad`

The phase completed in 166.129 seconds with 1,076.234 MB peak RAM and 14.257
MB/s peak observed I/O. Cleanup passed with no lingering owned process. The
one-second monitor again observed no nonzero VRAM sample; this is a telemetry
limit, not evidence that CUDA was unused, since cell receipts identify their
device as `cuda`.
