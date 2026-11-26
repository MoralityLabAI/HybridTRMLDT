# MMS in MCP06–09 stacked spatial series

MCP06 spatial retrieval trains fast and slow recurrent GRUs for 16 visits.
MCP07 geometry routing trains fast GRU and projection-memory gate across
variable-length verifier-labelled traces. MCP08 uses the same sites across
its two-lookup truncated BPTT chunks; capture only the first lookup in
each chunk and count later calls without modifying their gradients directly.
The teacher/DAGGER row order is preserved by native concatenation, so the
same training row IDs remain valid. Fixed stride keys encode row and visit
even when total batch visits change. All positive-budget arms use identical
explicit GRU equations; parameters and native checkpoint schema survive
the temporary wrapper. Other clocks, routing losses, verification and
calibration are unchanged. Phase-3 calibration runs without MMS.

Run one arm per process to avoid multiplying RAM by the existing worker count:
```sh
python -m MCP.run_spatial_mms --gym 07 --mms-mib 16 --mms-mode bpd --epochs 4 --roots 512 --out runs/mcp07_mms
python -m MCP.run_spatial_mms --gym 08 --mms-mib 32 --mms-mode bpd --epochs 4 --roots 512 --out runs/mcp08_mms
```
Use observe, dense, sham, fixed-round-robin and zero-budget controls. Training
is fixed exposure; measure actual CPU externally and keep the 16K training
world distinct from physical capacity extrapolation. The 16K/256K physical
record store is not the gradient bank. Bank capacity counts only tensors and
tags, not overall RAM. Very small fixed training sets may not fill it.
The default map is SVD; optional hash-pinned maps are per physical site.
Adaptive score is feedback admission, not verified improvement.

MCP08 solve_domain_episode now accepts transposition_mib, converting to
entries using the existing nominal 64-byte charge. Zero disables the
transposition and decoded-option cache; the default reproduces 1024 entries.
Decoded option storage and Python-object overhead are outside this cap,
so measure actual RSS and add a separate cap before claiming a RAM ceiling.
MCP09 is a three-valued speculative planner with charged native previews;
no learned optimizer loop. Leave its evidence rules intact. A future
memoization study must bind route, verifier version and budget semantics,
especially UNKNOWN and budget exhaustion. No historical test was opened.
