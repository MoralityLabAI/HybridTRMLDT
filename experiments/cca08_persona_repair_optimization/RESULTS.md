# CCA-08 exact finite repair optimization: CPU receipt

- Enumerated the complete |S|^(|A|-|S|) family of finite retractions on the registered small domains. Every repair fixes the certified set and keeps every output in it.
- The three-action Pareto example has two nondominated repairs: c->a costs 1/3 intervention and loses 2/3 scalar Fisher information; c->b costs 1 and loses 1/6.
- The weighted optimizer selects c->a with no information-loss penalty and c->b with penalty weight four.
- Joint repair to the single certified action 2 replaces an alternating repair cycle with a single idempotent map.
- Explicit caps and invalid-input checks prevent unbounded search and invalid cost calculations.

Four CCA-08 Python tests passed locally. The full combined CCA-06 through CCA-08 local suite passed 19 tests prior to uploading. These are CPU mathematics checks, not neural or real-world deployment outcomes. Underlying soundness depends on the trusted admissible-action set.
