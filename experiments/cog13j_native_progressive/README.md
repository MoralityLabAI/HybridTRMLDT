# COG-13J: native progressive transport, CPU-aware comparison

Protocol registered before native scoring in commit 7cb4445d956036b223dfb5215d6e90daf348c134. Base master 433ad5e30123f5ee80ffd18fabe67357547a263d integrates COG-13, B, C, D and I (#34, #35, #37, #38, #43). All five original CI runs green. Sibling COG-13E/F/G/H PRs remain separate. The COG-13G physical-pressure study found 13 block-scheduled completions per tier and zero interleaved completions; COG-13H found recurrence gate failure, compression and cache pass. These guide the single-session/frequency controls here.

## Executed archive timing (local CPU, one run)

Pinned COG-13C archive SHA-256 d99bc6a95d4ea31a132b408cd653644e9680a167fa87aca3b722522af0ab5aea. Freeze original fitted checkpoint and train-only frequency probabilities. All 449 archived evaluation menus processed by each policy, 2,245 decisions total. Four dense-derived policies exactly agree with the canonical Python dense replay at every menu. Clear the Decimal tanh interval endpoint cache between decisions. Timing is process CPU, not wall time; the offline dense comparison has its own auditNs counter. The current implementation executes that comparison inline before each native request, so it contributes to whole-root wall and host CPU budgets even though excluded from routing CPU.

| Mode | Recurring bytes | Sender CPU s | Receiver CPU s | Total routing CPU s |
|---|---:|---:|---:|---:|
| Frequency | 449 | 0.061095 | 0 | 0.061095 |
| Local dense | 449 | 0.394389 | 0 | 0.394389 |
| Shared dense | 30981 | 0.037165 | 0.365404 | 0.402569 |
| Binary16 / exact | 13021 | 0.086845 | 11.522444 | 11.609289 |
| Progressive | 10212 | 0.161710 | 12.151771 | 12.313481 |

Progressive sends 449 int8 packets and 27 binary16 refinements, zero exact fallbacks or sender abstentions. Binary16 alone certifies all 449. Compression is 3.034x vs shared dense, but routing CPU is 30.6x higher in this sequential timing pilot. The control with local full weights needs only one action byte per decision. Frequency uses smoothed training progress counts, not dense scores. No new training. Installed binary64 arrays counted once: frequency 128B; local/shared dense 4424B; half/progressive 5960B including receiver's full dense fallback model and a duplicate sender-side Wg. Header and each accept/refine reply charged. In-process model, no measured network latency. Caching between repeated decisions is deliberately disabled. Real arithmetic bounds depend on trusted sender containment and finite IEEE/Decimal assumptions, plus 1e-10 acceptance guard; this is not formal equivalence of all floating point executions. Every actual selected action is independently checked against dense.

The exact transport.py and native.py blobs match local files (see archive_timing.json). Mock backend completed five arms and 69 requests using the pinned checkpoint; forged cache, certificate and byte counts rejected. Verified 16 new theorem statement strings distinct from 24 COG-13C plus 16 COG-13H statements (H artifact 15139420524 SHA-256 f20b8ce3b6f1fe6d8e12816899024027c921bcb2065ddefd6c578c7d4ec80f8d). Ordinary depth 2–3, deeper depth 6–8; four generated families, including multivariate rational square expansion, distributive Finset membership, Nat inequalities and nested logic. These are generated development exercises, not an original-theorem discovery corpus.

## Native run

Five arms x 16 roots, 48 selected probes and 60s total per-root wall budget, one native process and 16 exact LRU snapshots, rotated policy order. Replays same 16-tactic grammar and validates raw mailbox data, longest-prefix cache transitions and recurring packets. Separate full-source compilers with allowed axioms provide two independent in-budget certificates for credit. Native child CPU from waited processes via RUSAGE_CHILDREN; total host CPU and routing component times also recorded. Complete paired roster, unknowns, cross-arm common-script verdicts and certified root sets reported. Post-score audit recomputes ranking, packets and cache transition using the same policy/codec routines; it is not an independent numeric implementation. Native CI still pending; neither local fixture nor archived timing establishes fresh Lean completion or actual network-speed benefit.

Run the mock and timing from this directory:
```
python native.py /path/to/cog13c.zip --fixture
python transport.py /path/to/cog13c.zip /tmp/cog13j-timing.json
```
For native execution prepare the pinned Mathlib/Lean toolchain and compiled StateCacheJ.olean, then run native.py with --old-h, --lean, --cwd and --out. The registered workflow does this and preserves all output on failure. Next research gate is cheaper safe interval propagation or a sender-local control with constrained weight availability, before claiming a systems advantage for quantized transport.
