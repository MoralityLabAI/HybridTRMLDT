# ARC-1 Benchmark

Single-rule grid transformation benchmark comparing LDT, TRM, and hybrid rule selection.

| Solver | Solved | Solve Rate | Steps | Proposals | Rejected |
|---|---:|---:|---:|---:|---:|
| `hybrid` | 3/3 | 1.000 | 5 | 5 | 2 |
| `ldt` | 3/3 | 1.000 | 11 | 0 | 0 |
| `trm` | 3/3 | 1.000 | 9 | 9 | 6 |

## Per Task

- `arc1_color_map` `ldt` solved=True steps=4 proposals=0 rejected=0
- `arc1_color_map` `trm` solved=True steps=1 proposals=1 rejected=0
- `arc1_color_map` `hybrid` solved=True steps=1 proposals=1 rejected=0
- `arc1_horizontal_flip` `ldt` solved=True steps=3 proposals=0 rejected=0
- `arc1_horizontal_flip` `trm` solved=True steps=5 proposals=5 rejected=4
- `arc1_horizontal_flip` `hybrid` solved=True steps=1 proposals=1 rejected=0
- `arc1_fill_background` `ldt` solved=True steps=4 proposals=0 rejected=0
- `arc1_fill_background` `trm` solved=True steps=3 proposals=3 rejected=2
- `arc1_fill_background` `hybrid` solved=True steps=3 proposals=3 rejected=2
