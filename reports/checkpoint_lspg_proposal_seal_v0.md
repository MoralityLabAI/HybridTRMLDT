# LSPG-v0 Proposal Seal

## Seal

- Proposal count: 12.
- Proposal table SHA-256:
  `dad24032c40bc8e10648db50554a6277e06e06e265da26530c379b0a5f97e34a`.
- Proposal receipt SHA-256:
  `4ebb010474f048d8347957c98a2ec4d524473252d507dbb715996c39a0a5704f`.
- Generation commit: `71e033dde2cff082becac640e6016f59938f3b84`.
- Outcomes observed at seal: false.

## Ranking

The first two decisions are `p=0.10` and `p=0.05` grid extensions because the
prior contains two left-censored boundary observations at `upper=0.15`.
Remaining proposals cover replication, fully untied and alternating tying,
short and interpolating schedules, split parameter/state masks, intermediate
supervision, detached carry, and post-normalization.

The historical double-normalization control is present in the bounded mutation
space but is excluded from the rankable batch. All twelve proposals materialize
and cost S0-S5. Only S0 is executable under this sealed stage.
