# Loop Schedule Algebra v0.1: ladder attempt 1 construction failure

Ladder attempt 1 exited before the first optimizer step. The generalized
trainer required the 4,096 state-visit exposure target to be exactly divisible
by `batch_size*R`. That is true for the primary `R={2,4,8,16}` cells but false
for the registered ladder's `R=6` cell.

LSA v0 used ceiling budget semantics: `ceil(4096/(8*6))=86` steps and 4,128
realized exposures; `R=12` similarly uses 43 steps and 4,128 exposures. The
repair restores that behavior while retaining exact reachability checks for
requested measurement and checkpoint exposures. A regression test locks the
`R={6,12}` step counts.

No ladder record or result file was produced, and no ladder outcome was
observed. The failed attempt's resource receipt is retained; attempt 2 is the
first eligible ladder run.
