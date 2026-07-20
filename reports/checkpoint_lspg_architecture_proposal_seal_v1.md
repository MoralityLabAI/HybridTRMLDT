# Checkpoint: LSPG Architecture Proposal Seal v1

## Seal

- Registration commit: `4860e1681cfc4111dc8bd2d2842ced37791dd0da`.
- Executable proposal commit: `cf837046f47e6ab370e17750a7a000a6a89e991d`.
- Proposal count: 38.
- First batch: 12 candidates.
- Sealed reserve: 12 candidates.
- Matched periodic controls: 6.
- Balanced schedule nulls: 6.
- Context models: 2.
- Proposal table SHA-256: `19e52be60a6296323549dc08c88fe75c5af6c550930c43dca5e13ee208c891e7`.
- Proposal manifest SHA-256: `8cdc824a9a39ac7bf6800fd5bd4b84bfdb4cc78843e28d0602c79f757ea660c8`.
- Proposal receipt SHA-256: `8ca17035e25422cf5d63ad954377d9dbe0d8372ed38c3666398c6d5ca965cd2e`.
- Task bundle hash: `f6f5b87a8fd9861e51fc34d75f6f918080d030a88577ceedf6f606f417f01255`.
- Outcomes observed at seal: false.

## First Batch

| Proposal | Canonical word | Matched control |
| --- | --- | --- |
| `LSAD-B1-K2L6-01` | `000111` | `LSAD-C-K2L6` |
| `LSAD-B1-K2L6-02` | `010011` | `LSAD-C-K2L6` |
| `LSAD-B1-K2L8-01` | `00001111` | `LSAD-C-K2L8` |
| `LSAD-B1-K2L8-02` | `00101101` | `LSAD-C-K2L8` |
| `LSAD-B1-K3L6-01` | `012021` | `LSAD-C-K3L6` |
| `LSAD-B1-K3L6-02` | `001122` | `LSAD-C-K3L6` |
| `LSAD-B1-K3L8-01` | `00011221` | `LSAD-C-K3L8` |
| `LSAD-B1-K3L8-02` | `01020212` | `LSAD-C-K3L8` |
| `LSAD-B1-K4L6-01` | `012233` | `LSAD-C-K4L6` |
| `LSAD-B1-K4L6-02` | `012032` | `LSAD-C-K4L6` |
| `LSAD-B1-K4L8-01` | `00112233` | `LSAD-C-K4L8` |
| `LSAD-B1-K4L8-02` | `01231023` | `LSAD-C-K4L8` |

These are opaque schedule interventions, not named architectures. Names and
mechanistic interpretations are assigned only to final candidates that pass the
registered matched-control and held-out transfer gates.

## Integrity

The generator verified every preregistered input hash before writing. It then
re-read the proposal table, checked the manifest and all derived artifact hashes,
and recomputed each proposal hash. All paths in the manifest are repository
relative. A second generation pass produced byte-identical files.

## Next Step

Run resource-only calibration on `LSAD-C-K2L8` at S0, S1, and S2 under the hard
caps. Select one frozen budget profile from measured step times before any task
metric is interpreted.
