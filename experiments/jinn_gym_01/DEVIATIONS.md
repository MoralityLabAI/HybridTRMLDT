# JINN-GYM-01 deviations from the preregistration

## D1: completion caps raised mid-run (decision 700 to 1200, patch 1000 to 2500)

At Stage C iteration 50 (about 1.2 h into the run), the raw responses showed that the registered caps were too small for Inkling-2-Mini at low reasoning effort. Reasoning tokens count toward `max_completion_tokens`, and when they used up the cap the endpoint returned `finish_reason=length` with empty content:

* stageC_patch: 27 of 50 calls hit the cap, 26 of them with empty content. More than half of all Stage C iterations failed as `invalid_patch` with no ops for that reason alone, not because of anything Inky proposed.
* Decision calls: 46 of 421 hit the cap (card 20, memory 14, neutral 12). These produce parse failures, which fall back to a uniform prior or to abstain.

The run was stopped with the registered `STOP` file, `DECISION_MAX_TOKENS` was raised to 1200 and `PATCH_MAX_TOKENS` to 2500, and the run resumed in the same output directory. The deadline (started_at + 8 h) is unchanged, so total runtime stays at or under 8 hours.

Scope:

* Stage A ran at the registered spec cap of 4500. One attempt hit the cap and was retried under the registered protocol.
* Stage B (18 tasks) ran entirely at 700. Its parse failures stay in the data as recorded.
* Stage C iterations 0 to 49 used the old caps; iterations from 50 on use the new caps. Cached neutral responses for development-pool tasks already fetched keep their 700-token text.
* The final held-out evaluation runs entirely at 1200 for every arm, so the arms stay matched there.

Every prompt still fits the 12,288-token context: the largest decision prompt is under 2,000 tokens, and the largest patch prompt is under 4,000 tokens.
