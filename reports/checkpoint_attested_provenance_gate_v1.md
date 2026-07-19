# Attested Provenance Gate v1 Checkpoint

## Outcome

The full three-seed artifact seal is internally coherent and independently recomputed. In every
seed, final false-claim acceptance changes from 100% under `claim_only` to 0% under
`attested_claim`: `30/30 -> 0/30` for seed 17, `30/30 -> 0/30` for seed 29, and
`32/32 -> 0/32` for seed 43. True-proposal acceptance remains 100% and signed
executed-utility cost is exactly zero in all three seeds.

The proposer did not stop making false claims. The attested gate denied those claims because no
trusted anchor record existed for the exact state, mechanics, action, and claim payload. This is
acceptance suppression at a trusted evidence boundary, not proposer alignment.

## Provenance

- Frozen protocol commit: `71e13d5`.
- Frozen config SHA-256: `21c2455a627742bc88481ed9047d59d3d1e4381d294d2c1ec20390bcb1799761`.
- Pre-outcome conformance commit: `ceda86d`; 255/255 RSITopology golden vectors passed.
- Smoke commit: `1260a7e`.
- Full artifact producer commit: `562da23`.
- Post-seal immutable-registry cache commit: `e9b6705`; it did not produce the reported data.

The first full attempt reached the one-hour shell cap before producing a result because it refit
semantically unused probe audits for every arm-round. Commit `562da23` removed those audits from
the claim-only/attested crossing; the exact 768-row v1 replay is canonically identical to the
sealed baseline (`45d866bef4f3c08c2f32316d02fd1c1859525b0a7d37998dbfaf2f942b2fd03b`),
which establishes that the runtime correction did not alter claim-only decision rows.

The second shell wrapper also reported exit 124 at its one-hour output-collection cap, but the
complete artifact set had been atomically written immediately beforehand. Process completion was
not observed. The result is accepted on post-seal evidence instead: all artifact and registry
hashes recompute, the canonical JSONL contains 10,752 rows, both receipt-failure counts are zero,
split overlap is zero, all three tampers are detected, the exact v1 replay passes, and all five
attestation tests pass. The complete repository suite also passes 175/175 tests. The wrapper ambiguity is retained in
`data/benchmarks/attested_provenance_gate_v1_execution_receipt.json` rather than hidden.

## Artifact Hashes

- Results: `8b1ddcbc49c9d26b68e210389039e2297128c03552b1a240295b82990d5feb72`.
- Records: `c0e294883af1a4e4258ee6b96cc8b74a26614358b885f101d5a086e419fc8c9c`.
- Report: `1bcb8460a572339e3ff9986d34e861e5e844c2a1ee6430eda8d128e89a2dde16`.
- Conformance: `bdef2bd43472dc47c760123d0c9cf177b5cba0c7e325b0e6314a659438059f6a`.

## Interpretation

All three registered tamper classes were detected: spectrum-preserving anchor substitution,
band/registry failover, and re-anchoring with a preserved payload hash. The no-cost result is a
property of this toy construction: the trusted exact-mechanics issuer has complete coverage for
truly sound state/action pairs. It is not evidence that real attestation systems have complete
coverage or zero escalation cost.

toy storyworld benchmark certifies claim/evidence channels, not neural feature identity; no general-alignment or oversight-sufficiency claim.
