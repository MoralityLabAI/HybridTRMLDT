# Gaming Versus Improvement Training Notes

Protocol: `gaming_vs_improvement_v2_above_majority`; improvement power passed: `True`.

## Registration

- config SHA-256: `7feb9d72997f844ba6736bf8cfe3deee36b63cb1b63f2b5367ccb3ab40c4469a`
- records SHA-256: `08c9412b37201e96e489003bf87878775c279829a63847c247848e25c84017b6`
- mode: `full`
- proposer seeds: `[211, 223, 227]`
- TRM latent / recurrence: `48 / 4`
- round-0 / adaptation steps: `320 / 6`
- expert-iteration rounds: `5`
- proposer training region: `hash buckets [0, 1, 2, 3, 4] mod 10`
- probe calibration region: `hash buckets [5, 6] mod 10`
- power development region: `hash buckets [7] mod 10`
- held-out evaluation region: `hash buckets [8, 9] mod 10`
- state-hash overlap across regions: `0`
- accepted expert traces train the proposer; no gradient passes through a verifier

## Controls

- arm distinctness assertion exercised: `True`
- behaviorally distinct pairs: `1767/1770`
- effective policies: `57`
- identical fallback has zero utility delta: `True`
- round-indexed causal probe checks: `18`
- exposed-probe improvement classifications: `0/3`
- exposed-probe evasion classifications: `2/3`

## Boundary

Hash-held-out deterministic storyworld evidence about learning, gate, and fallback interactions; not general alignment, neural AIRIS performance, or production robustness.
