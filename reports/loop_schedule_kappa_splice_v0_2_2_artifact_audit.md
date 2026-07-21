# Kappa Checkpoint Splice v0.2.2 Artifact Audit

## Provenance Chain

1. `bebae96`: right-censoring correction committed before splice design.
2. `df4cbee`: five-arm splice protocol and branches frozen before outcomes.
3. `5e0ff12`: no-outcome capped preflight sealed.
4. `295961a`: complete unsealed splice records, checkpoints, result, and resource receipt committed before finalization.

## Sealed Identity

- config: `7462fdca673324a593c32b23e5394147dba2a0b0c3d3cf41d882d5550bc600b1`
- records: `62ed09f3b62fda62e1373e4bb16a1ed2f113aa3125c9ef8834f9bebfc97549e7`
- result: `7996681f19c1960a19fc783f0fa47291accbb92e1f6507580aa386b75c740904`
- run resource: `5dc88e641cf2aef1615dafb2eac3590b6d7db7ae7b2b508c03cd761595db8c2f`
- result receipt: `8f05d7befc860225e9d568064ae0ca1433025d2df4a714ccc0b9722ec95afb1a`
- finalizer resource: `8620f5640d4e34f41bb50e865d98fde4e04eb0f5442e48bfcb1e2a6b3ead8a6b`
- report figure: `2a2ed935796d7c4a1d3bc627580092e74017d8f4607e7bfad39065acee9a162a`

## Integrity

- exact replay: model and complete AdamW state tensor-exact, endpoint kappa difference `0.0`;
- source prefixes retrained: `0`;
- continuation arms completed: `5/5`;
- records: `16/16` unique;
- nonfinite cells: `0`;
- peak RAM: `931.668 MB` under `2048 MB`;
- peak I/O: `16.061 MB/s` under the sustained `50 MB/s` abort threshold;
- training cleanup: passed, no lingering owned process or GPU application;
- finalizer cleanup: passed;
- v0.1 alignment note: untouched.

Full repository verification after packaging: `331 passed in 34.36s`.
