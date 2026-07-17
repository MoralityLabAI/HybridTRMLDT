# Checkpoint: Controller-Mesh Sheaf Forward v1

Date: `2026-07-17`

## Completed

- froze 64 evidence/threshold/fallback genomes before outcome reveal;
- held out one hash-selected threshold in every evidence/fallback cell;
- fitted a bounded correction-infused fallback on calibration hard cases;
- separated intervention and stability Laplacians;
- sealed geometry before discovery outcomes and predictions before held-out outcomes;
- evaluated 128 matched N0 predictors;
- retained categorical and split-Laplacian comparisons as post-hoc diagnostics.

## Registered Result

- held-out Spearman rho: `+0.7049`
- matched N0 mean rho: `+0.2757`
- matched N0 one-sided p: `0.0078`
- predicted top-four uplift: `+0.1476`
- predicted/actual top-four overlap: `2/4`
- registered forward gate: passed

The actual best genome uses exact evidence, threshold `0.35`, and correction infusion. Its objective is `1.1531`,
utility is `0.9703`, and unsafe rate is zero. Three of the actual top four genomes use correction infusion.

## Post-Hoc Audit

The categorical architecture baseline reaches `rho=+0.6726` and `+0.0950` top-four uplift. The spectral predictor
is directionally better but the 16-genome panel cannot establish unique spectral information. Intervention-only
spectra are weak (`rho=+0.0839`); stability-only reaches `+0.5666`; the combined predictor reaches `+0.7049`.

Phase-boundary proximity does not identify high utility: the four closest genomes average below the held-out
panel. It remains a negative control, not an acquisition result.

## Receipts

- semantic config SHA-256: `828f13f7eca5161327a765b10032f798f5d5a4d38c2dfc4eafe0f2cf2a96697a`
- calibration geometry SHA-256: `3e563b5623a017fc3d844223423a03fc2d353283c4e1f9098e5441b662b2c8bf`
- discovery outcomes SHA-256: `96674f5ec1cb6ceb39ad243078b819a6817275c9a77f97168bf46d08b30aec5f`
- held-out predictions SHA-256: `863d2676c1dd4943f8243c40fb83d0140348eb748911edabb04fa46f8956b38e`
- held-out outcomes SHA-256: `1a3b0e19ccd6ee3dcb4b73fd17270b4229119539929e8438cb6ee8bb38615965`
- analysis receipt SHA-256: `fc8bc845de436dde0248c365429f47b095cf9c29b8e6f92fecb555fd8dfe2f8a`
- result file SHA-256: `928f721ca44adc80f3eea3d616e5622a55986c001fefb89c730eca66961cba0d`
- canonical/experiment mirror mismatch: `0`

## Tests

- full suite: `153 passed`
- simulated install without neural extra: `122 passed, 4 skipped`
- focused forward suite: `6 passed`
- deterministic replay: byte-identical
- compileall: passed

## Recommended Next Step

Run an independently seeded replication with the categorical baseline preregistered and whole-family holdouts.
Require a positive spectral-minus-categorical delta before claiming predictive information beyond genome labels.
