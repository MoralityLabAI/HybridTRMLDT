# ARC-2 Benchmark

Two-rule grid transformation benchmark over ordered primitive compositions.

| Solver | Solved | Solve Rate | Steps | Proposals | Rejected |
|---|---:|---:|---:|---:|---:|
| `hybrid` | 3/3 | 1.000 | 32 | 32 | 29 |
| `ldt` | 3/3 | 1.000 | 120 | 0 | 0 |
| `trm` | 3/3 | 1.000 | 38 | 38 | 35 |

## Per Task

- `arc2_flip_then_color` `ldt` solved=True steps=40 proposals=0 rejected=0
- `arc2_flip_then_color` `trm` solved=True steps=5 proposals=5 rejected=4
- `arc2_flip_then_color` `hybrid` solved=True steps=13 proposals=13 rejected=12
- `arc2_fill_then_rotate` `ldt` solved=True steps=40 proposals=0 rejected=0
- `arc2_fill_then_rotate` `trm` solved=True steps=21 proposals=21 rejected=20
- `arc2_fill_then_rotate` `hybrid` solved=True steps=5 proposals=5 rejected=4
- `arc2_color_then_flip` `ldt` solved=True steps=40 proposals=0 rejected=0
- `arc2_color_then_flip` `trm` solved=True steps=12 proposals=12 rejected=11
- `arc2_color_then_flip` `hybrid` solved=True steps=14 proposals=14 rejected=13
