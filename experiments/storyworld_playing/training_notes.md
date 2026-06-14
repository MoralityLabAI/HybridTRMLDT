# Storyworld Playing Training Notes

Task: play the coupled storyworld to reach the secret ending predicate.

Models:

- `ldt`: exact finite-horizon reachability planner under the modeled rival policy.
- `trm`: heuristic policy over persistent local deficits: heat, evidence, trust, scene.
- `hybrid`: TRM proposes actions; LDT checks modeled reachability and overrides unsafe proposals.

Data:

- sampled viable starts: `64`
- horizon: `6`
- seed: `7`

This run uses exact environment transitions and does not involve VPD.

- `hybrid` success_rate=1.000 successes=64/64 avg_steps=3.28 overrides=124
- `ldt` success_rate=1.000 successes=64/64 avg_steps=3.50 overrides=0
- `trm` success_rate=0.172 successes=11/64 avg_steps=2.81 overrides=0
