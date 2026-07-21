# Kappa Data-Order Intervention v0.2.1 Construction Failure

## Status

All four registered cells completed, all 20 measurement records were written, and resource cleanup passed. Result assembly then failed before classification because the baseline replay checker required equality between the new and parent kappa exposure sets.

The sets are legitimately different:

- new baseline kappa: `E={0,512,1024,2048,4096}`
- sealed parent kappa: `E={0,1024,2048,4096}`
- sealed parent `E=512`: gradient-only, not a kappa checkpoint

The protocol text requires replay of every **parent checkpoint kappa**. It does not require discarding the newly registered `E=512` measurement. The implementation overconstrained the gate.

## Integrity

- wrapper status: `failed`, `abort_reason=process_exit_1`
- elapsed time: `767.664 s`
- peak RAM: `931.625 MB`
- peak I/O: `14.494 MB/s`
- peak observed VRAM: `0 MB`
- cleanup: passed, no lingering owned process or GPU application
- complete cells: `4/4`
- unique records: `20/20`
- checkpoints: `8`, totaling `3,221,776` bytes

Hashes:

- records: `84419b04bbb59ceba0a87f91b67f8e143bcf3f2378b71ce9f8a47b86cb3518b3`
- events: `804ec89fd6c06c10c0aeaf18c3bdbc26f1ea4efd88856856c4b75cadf259e2af`
- resource receipt: `bbf30a39d96d4382efb6d472125a691175937aa4a2a246e529d5fa96cb2afcca`
- stderr: `d487312592fd269ebaf4eafba4b695789f1d1285db964370646b75a06681334f`
- stdout: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## Recovery Rule

Do not rerun training and do not rewrite the records. A labeled post-completion recovery may admit the 20 records by hash, require exact cell/exposure identities, compare the baseline against the four sealed parent kappa checkpoints with the unchanged `1e-12` tolerance, retain `E=512` as a new non-replay measurement, and apply the already frozen branch rules without modification.

The failed attempt consumed outcomes, so the recovery must be registered and pushed before any aggregate branch is computed.
