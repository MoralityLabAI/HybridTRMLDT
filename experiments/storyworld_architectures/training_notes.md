# Storyworld Architecture Training Notes

Task: compare typed membrane, hard gate, and confidence arbitration on storyworld control.

Scenarios:

- `secret_ending`: exact secret-ending reachability is environment-sound.
- `moral_optimization`: terminal moral score is a soft preference surface, not a hard ending gate.

Policies:

- `trm`: greedy local heuristic.
- `typed_membrane`: TRM proposal plus typed reachability override.
- `hard_gate`: always use LDT reachability action when available.
- `confidence_arbitration`: use soft moral-score action when margin is high; otherwise defer to LDT.

- episodes per scenario/policy: `64`
- horizon: `6`
- seed: `7`
- confidence gamma: `2.00`

## moral_optimization
- `confidence_arbitration` success_rate=1.000 avg_score=11.06 overrides=134 avg_margin=0.23
- `hard_gate` success_rate=1.000 avg_score=10.80 overrides=0 avg_margin=0.00
- `trm` success_rate=0.188 avg_score=3.45 overrides=0 avg_margin=0.00
- `typed_membrane` success_rate=1.000 avg_score=9.95 overrides=124 avg_margin=0.00

## secret_ending
- `confidence_arbitration` success_rate=0.938 avg_score=0.94 overrides=134 avg_margin=0.23
- `hard_gate` success_rate=1.000 avg_score=1.00 overrides=0 avg_margin=0.00
- `trm` success_rate=0.172 avg_score=0.17 overrides=0 avg_margin=0.00
- `typed_membrane` success_rate=1.000 avg_score=1.00 overrides=124 avg_margin=0.00
