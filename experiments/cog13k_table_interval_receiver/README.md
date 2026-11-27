# COG-13K: Table-backed certified interval receiver

Protocol frozen at 0fed4b370764a22a126392e178eb64f12966be14. Extends COG-13J on its source-pinned C/J archives. No model training or new native Lean search occurs in this experiment.

## Method

The receiver generates 2,049 endpoint enclosures for tanh on [0, 8], spaced by h=1/256, with the same directed Decimal endpoint routine used by J. Store the lower and upper endpoints as binary64. Use odd symmetry for negative arguments, monotone saturation beyond the table, and outward-rounded linear interpolation widened by h^2/4 and a rounding allowance. Since |tanh''| <= 2, the linear interpolation error is bounded by 2h^2/8 = h^2/4. C uses nextafter for interval addition, multiplication and score subtraction. Build with -fno-fast-math and -ffp-contract=off. Require IEEE binary64, round-to-nearest, finite moderate inputs, correctly rounded Decimal endpoint generation and trusted sender containment, as in J. The C receiver selects a winner only on strict separation (>1e-10) of its score interval from all competitors; otherwise the existing progressive refinement and exact fallback are used. No cross-decision endpoint memoization except the immutable table. The real-network interval calculation plus finite audit tolerance is not a formal proof about every floating-point runtime. The discrete dense replay remains an independent action check.

The 32,784-byte table is model-independent installed data, separate from 5,960 bytes of installed model arrays. Generation took 90.861 ms CPU locally. Raw packet accounting includes all refinement replies and excludes setup. There is no measured network latency.

## Local execution

Two frozen archives: 449 C reached evaluation menus and 189 J exact shared-dense menus from 16 certified roots. C artifact 15136527625; J artifact 15160808431. Three rotated-order repetitions for each of four arms: local dense, shared dense, Decimal progressive, table progressive (7,656 route decisions). The table arm checked 2,016 compressed intervals over the repetitions, including refinements. No false certified actions or exact fallback; every final action agrees with canonical dense. Tested 1,004 table endpoints against Decimal bounds, four malformed packets, an infinite endpoint and an unsupported preactivation range. All passed.

Median component CPU in seconds, calculated separately per component; routing CPU excludes the separate dense audit:

| Archive | Mode | Recurring bytes | Sender CPU | Receiver CPU | Routing CPU |
|---|---|---:|---:|---:|---:|
| C | Local dense | 449 | .375499 | 0 | .375499 |
| C | Shared dense | 30,981 | .034137 | .334595 | .368731 |
| C | Decimal progressive | 10,212 | .176913 | 12.130856 | 12.307769 |
| C | Table progressive | 10,212 | .086007 | .098901 | .184909 |
| J | Local dense | 189 | .158066 | 0 | .158066 |
| J | Shared dense | 13,041 | .019976 | .141193 | .161170 |
| J | Decimal progressive | 4,172 | .080352 | 4.836301 | 4.916654 |
| J | Table progressive | 4,172 | .040357 | .039313 | .079669 |

All registered gates pass locally. C and J respectively give 122.66x and 123.02x receiver speedups, 66.56x and 61.71x routing speedups over Decimal progressive, and about 2x lower routing CPU than shared dense in the observed replay. C retains 27 refinements and J retains seven; the recurring bytes are identical to the Decimal policy. Setup generation and dense audit cost are additional. Local action-only retains the fewest recurring bytes when the sender owns the complete model. No hardware-independent or proof-throughput speedup is inferred from these component timings.

`results.json` preserves every repetition's sender, receiver and audit CPU times, discrete results, archive hashes and exact source blobs. The source hashes match the executed local files. The workflow replays both archives on hosted CPU and enforces the registered thresholds without requiring identical timing numbers. Reproduce:

```
python experiments/cog13k_table_interval_receiver/benchmark.py C.zip J.zip /tmp/cog13k-new
python experiments/cog13k_table_interval_receiver/verify.py /tmp/cog13k-new/result.json
```

Next: integrate the table receiver into the native snapshot-cache search, charge table installation, run the same budgets with matched local, shared dense, Decimal and table arms on newly held-out harder goals, and audit full proof trajectories. The 638 archived menus are a retrospective reached-state sample; repeated timings are not independent theorem evaluations.
