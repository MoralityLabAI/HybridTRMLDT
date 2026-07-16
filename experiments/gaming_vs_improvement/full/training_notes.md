# Gaming Versus Oversight Leverage Training Notes

Historical study ID: `gaming_vs_improvement_v1`; proposer-improvement power is absent in this run.

## Registration

- config SHA-256: `056c03cbc54db642aae75b6587ef57885187fd050a76af38181d86325fa30c38`
- records SHA-256: `e587b8456fda8e2183694f7b6c1e40660b2cfe06e5e048b319f0a040f5f3cc9d`
- mode: `full`
- proposer seeds: `[17, 29, 43]`
- TRM latent / recurrence: `48 / 4`
- round-0 / adaptation steps: `3 / 6`
- expert-iteration rounds: `5`
- proposer training region: `trust <= 0`
- probe calibration region: `trust == 1`
- held-out evaluation region: `trust >= 2`
- state-hash overlap across regions: `0`
- accepted expert traces train the proposer; no gradient passes through a verifier

## Controls

- arm distinctness assertion exercised: `True`
- behaviorally distinct pairs: `1676/1770`
- effective policies: `31`
- identical fallback has zero utility delta: `True`
- round-indexed causal probe checks: `18`
- exposed-probe evasion classifications: `2/3`

## Boundary

Toy storyworld evidence, gate, adaptation, and fallback interactions; not general alignment, neural AIRIS performance, or production robustness.
