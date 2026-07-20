# Loop Schedule Algebra v0.1: invalid primary attempt 1

Primary attempt 1 completed under the registered resource caps but is excluded
from all scientific interpretation. The trajectory implementation seeded the
kappa power iteration as `seed+50000+exposure`, so each checkpoint used a
different stochastic probe direction. That violates the intended fixed-probe
comparison across training time and makes the terminal cells differ from the
sealed v0 estimator, which used `seed+50000`.

The failure was visible in the built-in replication diagnostic before any
headline was accepted: terminal v0.1/v0 kappa ratios at `R={2,4,8}` were
`{0.972942,0.947023,0.937465}` rather than the exact replay established in the
preceding phase. The attempt's records, result, event log, resource receipts,
and local checkpoints are retained. `INVALIDATION.json` marks
`outcomes_used=false`.

The repair fixes one probe seed per training seed at every exposure and adds a
regression test for the seed rule. No registered config, endpoint, threshold,
or outcome branch changes. Primary attempt 2 must reproduce the v0 terminal
cells before its `R=16` residual or `gamma(t)` label is interpreted.
