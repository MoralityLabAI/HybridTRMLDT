# Hybrid Control Harness Experiment Notes

## Scope

The project scan is real and local. The benchmark cases are deterministic source-inspired proxies
built from the control interfaces found during the scan; they do not execute or score the neighboring
projects themselves. This keeps the experiment dependency-free and prevents proxy results from being
misreported as native game performance.

- projects scanned: `130`
- relevant candidates retained: `67`
- train cases per application: `48`
- held-out cases per application: `48`
- random seed: `23`
- selected confidence margin: `0.10`
- selected counterfactual beam width: `2`

## Source-inspired applications

- `GPTStoryworld`: exact secret-route reachability versus soft morality and opponent-model scores.
- `AI_Diplomacy`: exact order legality versus conditional coalition/betrayal forecasts.
- `SmallControlHarness`: attested provenance versus confident but semantically re-anchored evidence.
- `BitVMArena`: exact protocol compatibility versus replay-derived control-profile rankings.
- `StoryForge`: exact choice gates versus model-sound prisoner-dilemma best responses.
- `TheySing`: exact channel/campaign permissions versus modeled persuasion and treaty response.
- `Blighted Galaxy`: exact strategic-tick legality versus replay-derived action value.
- `CodexGameStudio`: exact agent scope versus learned task-to-specialist routing.

## Structures tested

- `trm`: Latent proposal only; cheapest path and no explicit certification.
- `ldt`: Explicit scorer only; robust on exact mechanics but proxy-bound on preferences.
- `hard_gate`: Treat all LDT/model candidates as mandatory filters, regardless of provenance.
- `confidence_arbitration`: Use TRM above a calibrated margin and otherwise defer to LDT.
- `typed_membrane`: Use TRM unless an environment-sound constraint rejects it, then use LDT.
- `typed_confidence`: Apply environment constraints first, then confidence arbitration inside the safe set.
- `counterfactual_beam`: Certify a bounded TRM beam and rank surviving branches by combined TRM/LDT score.
- `skill_router`: Select a hybrid structure per skill using only train-split control utility.

## Held-out results

- `trm` accuracy=0.725 utility=0.833 unsafe_rate=0.157 cost=1.00
- `ldt` accuracy=0.528 utility=0.756 unsafe_rate=0.000 cost=2.00
- `hard_gate` accuracy=0.465 utility=0.729 unsafe_rate=0.146 cost=2.00
- `confidence_arbitration` accuracy=0.808 utility=0.887 unsafe_rate=0.111 cost=1.22
- `typed_membrane` accuracy=0.845 utility=0.932 unsafe_rate=0.000 cost=1.36
- `typed_confidence` accuracy=0.928 utility=0.967 unsafe_rate=0.000 cost=1.57
- `counterfactual_beam` accuracy=0.907 utility=0.965 unsafe_rate=0.000 cost=3.50
- `skill_router` accuracy=0.935 utility=0.972 unsafe_rate=0.000 cost=1.55

## Learned skill routing

- `coalition_planning` -> `typed_membrane`
- `control_profile_selection` -> `confidence_arbitration`
- `moral_optimization` -> `trm`
- `opponent_modeling` -> `trm`
- `provenance_control` -> `typed_confidence`
- `reachability` -> `typed_confidence`
- `skill_orchestration` -> `ldt`
- `strategic_tick_control` -> `typed_confidence`
- `treaty_channel_control` -> `typed_confidence`

## Design decisions

- Environment-derived candidate exclusions are the only hard constraints.
- Model- and experience-derived candidates are visible to hard-gate ablations but not promoted by typed policies.
- Hyperparameters and skill routes are selected on train cases and reported on a disjoint deterministic eval split.
- The selection objective penalizes unsafe actions and deliberation cost in addition to rewarding utility and exact choice.
- Oracle utility is used only for calibration/evaluation, never as an input to an architecture at decision time.

## Native follow-ups

- Wrap `GPTStoryworld/storyworld/env/diplomacy_env.py` action normalization and turn traces directly.
- Add a controller adapter around `SmallControlHarness` attestation decisions and shared registry memory.
- Route `BitVMArena` settlement profiles by trajectory phase, then certify protocol compatibility before funding.
- Use `StoryworldTRM` SWMD PICK traces to train the skill router instead of selecting from synthetic score cards.
- Apply typed delegation to `CodexGameStudio` skill scopes, then optimize consultation cost on real task traces.
- Insert the membrane into `TheySing` bridge policies and compare hard/soft/graduated treaty scenarios.
- Wrap `Blighted Galaxy` replay ticks to test whether beam certification survives longer horizons.
