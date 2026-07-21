# RLM Hybrid ControlTRM Training Results v1

All three registered CPU-only training cells completed 200 steps and emitted checkpoints every 50 steps. Final-step checkpoints are retained for seeds `211`, `223`, and `227`; no outcome-dependent checkpoint selection occurred.

| Seed | Parameters | Final loss | Train accuracy | Calibration accuracy |
|---:|---:|---:|---:|---:|
| 211 | 17,203 | 0.238001 | 0.8958 | 0.8125 |
| 223 | 17,203 | 0.448856 | 0.9271 | 0.8125 |
| 227 | 17,203 | 0.285152 | 0.9062 | 0.8125 |

The proposal table contains 456 rows: one frozen proposal ranking for every task and checkpoint seed. Evaluation rankings are sealed for replay, but evaluation accuracy was not computed or used during training.

The capped wrapper completed in 12.983 seconds with peak process RAM 443.328 MB, average CPU 18.588%, peak I/O 20.537 MB/s, zero VRAM, no lingering owned process, and cleanup passed.

Claim boundary: these are gym ControlTRM analogues trained on compact public task features, not official TinyRecursiveModels checkpoints and not long-context language models.
