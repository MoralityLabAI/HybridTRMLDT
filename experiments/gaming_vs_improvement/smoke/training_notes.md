# Gaming Versus Improvement Training Notes

## Registration

- config SHA-256: `056c03cbc54db642aae75b6587ef57885187fd050a76af38181d86325fa30c38`
- records SHA-256: `58b17a050b6541c610b288226b34406c7d1314a22700afd88e24dd83384dcd83`
- mode: `smoke`
- proposer seeds: `[17]`
- TRM latent / recurrence: `32 / 3`
- round-0 / adaptation steps: `2 / 3`
- expert-iteration rounds: `2`
- proposer training region: `trust <= 0`
- probe calibration region: `trust == 1`
- held-out evaluation region: `trust >= 2`
- state-hash overlap across regions: `0`
- accepted expert traces train the proposer; no gradient passes through a verifier

## Controls

- arm distinctness assertion exercised: `True`
- behaviorally distinct pairs: `63/66`
- effective policies: `9`
- identical fallback has zero utility delta: `True`
- round-indexed causal probe checks: `3`
- exposed-probe evasion classifications: `0/1`

## Boundary

Toy storyworld evidence, gate, adaptation, and fallback interactions; not general alignment, neural AIRIS performance, or production robustness.
