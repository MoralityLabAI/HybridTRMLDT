# Loop Schedule Algebra v0.1: learning-rate ladder checkpoint

Ladder attempt 2 completed all 96 registered cells. Every seed-level run was
stable across `R={6,12}`, `p={0,0.05,0.10,0.15}`, and learning rates
`{0.0003,0.001,0.003,0.01}`.

All eight round-by-learning-rate boundaries are left-censored at `p<=0`. The
machine grid endpoint is `0`, but no observed instability lies below it, so the
endpoint is not promoted to an exact boundary. Every registered depth-ordering
comparison is consequently `nonidentifying_censoring`.

The most stressed cell was `ladder_LR0.0003_R12_P0.00_S107`, with maximum
gradient norm `66.0719`, below the registered cutoff of 100. The largest
final/initial loss ratio was `0.6855`; no run approached the registered ratio
cutoff of 10. The learning-rate ladder therefore does not rescue the predicted
threshold as an ordering in this regime. It strengthens the v0 conclusion that
the boundary fails to bind over the tested horizon.

Artifact hashes:

- records:
  `89e14fd2c7b0b1ae56c8cfddf0e798e5d138be04b16c29b4acbef021ac84da82`
- result:
  `411cbdfedb6f11f9ecae0b36eb599940cd1ba17ed6e2880a2fc9e52657459bf6`
- resource receipt:
  `51e651800d144110e6ad5ab9a8d645992016c3ef56c996e1995da64928c8749f`

The valid phase completed in 72.347 seconds with 778.004 MB peak RAM and
15.579 MB/s peak observed I/O. Cleanup passed with no lingering owned process.
