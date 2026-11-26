# COG-13I — shared-goal packets and interval-certified refinement

## Evidence custody

Retrospective analysis of the successful COG-13C native run [45852214025](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/45852214025), artifact 15136527625, SHA-256 d99bc6a95d4ea31a132b408cd653644e9680a167fa87aca3b722522af0ab5aea. The analysis reconstructs the exact original JS features from displayed goals, validates source archive member digests, and independently checks all 493 archived learned packets (262 dense, 108 rank-four float32, 123 rank-four 4-bit), their encoded bytes, decoded scores, ranking and action selection. Train-only goal scale fit uses 14 distinct archived teacher states. Readout comprises 449 evaluation decisions (193 ordinary, 256 deeper), including repeated visits and different reached-state distributions across original policies. The certified study has 88 distinct numerical feature/menu pairs; the repeated visits are weighted as observed, not treated as independent samples. No heldout policy rollout or new Lean action is executed. This is posthoc protocol development.

## Shared-goal baseline

Each candidate input is a 16-way one-hot action plus a common 16-vector g. For a linear input map W=[Wa,Wg], Wx_a = Wa[:,a] + Wg g. A four-byte header contains version, mode and the 16-bit available-action mask. All original menus are in canonical order. Charge one additional byte for the selected menu index or a refinement request.

| Mode | Recurring bytes, 449 menus | Dense top agreement |
|---|---:|---:|
| Naive repeated dense features | 864965 | 449 |
| Shared dense goal features | 30981 | 449 |
| Shared rank-four goal, fp32 | 9429 | 429 |
| Shared rank-four goal, q4 | 3143 | 429 |
| Exact action column, rank-four goal fp32 | 9429 | 445 |
| Exact action column, rank-four goal q4 | 3143 | 445 |

Naive dense is a counterfactual dense packet on every observed menu; actual original mixed-policy recurring bytes total 450327. Shared dense reproduces the original dense score exactly in the independent Python replay. The low-rank versions decode actual bytes, including clipping. With exact action columns, rank-four q4 matches all 256 deeper decisions but misses four ordinary decisions. Q4 clips 133 evaluation menus (183 coordinate clipping events). The original residual-norm interval screen certifies zero compressed menus. This failed bound is retained. Posterior score error/margin tests have access to dense scores, and are not a deployable guarantee.

The explicit setup array byte counts are in audit.py. Sender/receiver must install consistent model/menu/version and scales; recurring traffic excludes setup. Local action selection would require only an action reply if the sender possessed the entire scoring model. The comparison is specific to this split of computation.

## COG-13I-B: bound the common preactivation

Keep Wa exact. Define the 12-dimensional common preactivation c = Wg g. Receiver uses p_a=Wa[:,a]+c and the same three tied tanh passes. Three new packet types:
- q4_envelope: two bytes of quantized rank-four goal coordinates plus a four-byte upward-rounded radius for the actual preactivation residual, 11 bytes including header/reply.
- pre_i8: one fp32 positive scale and twelve signed int8 preactivation coordinates, 21 bytes including header/reply.
- pre_f16: twelve half-precision preactivation coordinates, 29 bytes including header/reply.

These move the dense goal projection to the sender: the sender requires Wg (192 binary64 weights; 1536 B), while the receiver needs Wa, R, b, v and o (361 binary64 weights; 2888 B); the rank-four envelope path also needs matching factors and scales. The direct-preactivation stages do not need rank-four factors. The quoted traffic does not include installed arrays or CPU overhead.

For each packet the trusted sender independently encloses the exact common preactivation with outward-rounded products and sums, and checks that the receiver's envelope contains it. The receiver decodes the packet, propagates interval bounds through three recurrent passes, using interval affine arithmetic and monotonic tanh, and accepts only if the selected candidate's score lower bound exceeds every other upper bound. The real-arithmetic sufficient test follows by induction on the recurrence. The implementation widens each binary64 basic operation by one ULP, assumes IEEE rounding with finite moderate inputs, and uses 48-digit correctly rounded Decimal.exp surrounded by adjacent decimal values, directed decimal arithmetic, then outward binary64 conversion for tanh. These are implementation-dependent numeric enclosures of the real network, not a formal Lean proof or a universal certificate for the archived JavaScript floating-point runtime. The audit separately checks every observed dense score against the enclosing intervals with 1e-10 replay tolerance, and every issued certificate against the dense replay argmax. Trusted sender containment and installed weight identity are explicit premises; no adversarial sender assurance.

| Mode | Certified decisions | Empirical dense agreement | Total bytes |
|---|---:|---:|---:|
| q4 + actual residual envelope | 0/449 | 445/449 | 4939 |
| Shared preactivation int8 | 422/449 | 449/449 | 9429 |
| Shared preactivation binary16 | 449/449 | 449/449 | 13021 |

Minimum positive certified score gap among these reached decisions: int8 0.005235059273403619; binary16 0.01161270623029109. Int8 certifies 171/193 ordinary and 251/256 deeper decisions. Half precision certifies all. The q4 envelope has no useful sufficient certificates even with actual sender residual; therefore do not include it by default in a guaranteed policy.

Progressive int8 -> binary16 -> exact shared dense uses 449 int8 packets and 27 binary16 refinement packets, total 10212 recurring bytes (including a one-byte accept/refine reply at each stage), all final actions agree with archived dense, zero exact fallbacks. Full protocol fallback is modeled from the executed stage decisions and known exact shared-dense decoder; no live network exchange was run. This is 3.034x smaller than exact shared dense, 84.70x smaller than repeated naive dense, 44.10x smaller than the actual mixed original traffic (the latter was a different policy; compare the matched exact baseline first). Binary16-only is 13021 bytes; q4-first increases the progressive cost to 15151 bytes because every q4 certificate abstains.

Reproduce with the SHA-pinned original ZIP and Python 3.13:
```
python audit.py ARCHIVE.zip --out /tmp/cog13i/shared
python certified.py ARCHIVE.zip --out /tmp/cog13i/certified
python verify.py /tmp/cog13i
```
Create /tmp/cog13i first. Source and original local summary hashes are in results.json. Both scripts executed against the original archive; Git blob hashes match locally executed source exactly. CI checks discrete fields and archives full summaries and per-decision packets. CI pending. No new model fitting or native Lean results from COG-13I. In particular, the 88 numerical feature/menu pairs are a small posthoc sample; do not train on the readout or claim general exactness.

## Next native experiment

Integrate the exact shared-goal baseline and progressive preactivation protocol into the native search path with receiver scoring cost charged. Pin sender/receiver implementations and setup, check same dense argmax or abstain at every reached state. Include a local-action-only baseline, same fixed candidate menu, and a cheap frequency scorer. Hold out fresh, more difficult theorem families. Independently certify whole proofs and retain every unknown, fallback and failure. The interval checker may be much more expensive than the original forward pass; bytes alone do not establish end-to-end benefit.
