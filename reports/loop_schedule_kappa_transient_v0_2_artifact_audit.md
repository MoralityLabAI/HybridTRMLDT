# Kappa Transient v0.2 Artifact Audit

## Provenance Chain

1. `9d2ba6f`: frozen R128 protocol and divergent timing predictions, before outcomes.
2. `89fbc9c`: successful no-outcome preflight.
3. `aef5ce8`: registered timeout evidence, partial event stream, and checkpoints.
4. `39d076c`: deterministic recovery frozen before recovery outcomes.
5. `517095b`: recovery attempt-1 construction failure sealed before its fix.
6. `bdd36d2`: device-normalized exact comparator fix.
7. `6effcaf`: complete recovered outcome and resources.
8. `1f3e329`: deterministic final receipt.

## Sealed Artifacts

- recovery config: `a28fe5c68429cab6721de0725a1422a97bd8f2c2602164035bb6ee76dec2d2ae`
- combined records: `1bdd2c5ae4f42ab6a66a9760300bc0ec5dc93bf5a9b9d7eb4fcee7f470f91653`
- recovery events: `a8ee13a4c71c9bee4426b0b09a36dbe216dc8cda978e1444b2f0b60ffe20d344`
- recovery result: `e80e183f3dbd4c620b7c9b568eec24a9086b72eb88d8b222b1e16b30fff9ac2f`
- run resource receipt: `5bb9e31f8851a85c67a3f2b4e0979cdd30605b6715668deefe1ab297e339a27e`
- finalizer resource receipt: `af79283b72acf9781a5e392f4b825909d7d82511d6c9537a52f11aa1b9cc897d`
- benchmark/result receipt: `163ce7216f784426be7005b8a38d8edbbb4f819d6248f5bf8f84468b2b5993e7`
- report figure: `36efaab2159901893d384d55b4397357732aa29ed31c3d1c01ed6fef45e71104`

The copy under `data/benchmarks` is JSON-identical to the result receipt in the recovery directory. All text hashes use canonical LF verification with legacy raw-byte compatibility; the checkpoint remains raw-byte exact.

## Integrity Checks

- registration hash matches the frozen recovery config;
- parent timeout resource, events, and resume checkpoint hashes match;
- ten and only ten parent measurements were admitted;
- model and complete AdamW state replayed tensor-exact at seed 107 step 4;
- all 24 combined record IDs are unique;
- all six registered seed-regime cells are represented across parent and recovery evidence;
- three untied controls completed 8192 exposures and remained finite;
- result re-verifies the canonical records hash;
- successful run and finalizer report no lingering owned process or GPU app;
- no sealed v0.1 alignment-note artifact was modified.

Full repository verification after packaging: `314 passed in 80.63s`.
