# Hybrid Control and Game Application Map

This shortlist follows a broad automated scan with targeted inspection of the highest-value native interfaces.
Priority A means a deterministic adapter can be built from an existing action/trace/controller surface.
Priority B means the application is useful but is either routing-oriented or needs a narrower fixture first.

| Priority | Project | Native surface | Hybrid application | Current status |
|---|---|---|---|---|
| A | `GPTStoryworld` | storyworld/env/diplomacy_env.py; verifier symbolic routes and turn traces | certify legal/reachable choices while keeping morality and opponent forecasts soft | proxy benchmarked; native adapter next |
| A | `TheySing` | src/harness policies, bridge policy, campaign clock, tournaments, replay traces | typed channel permissions plus confidence/beam arbitration under treaty pressure | native adapter and matched enforcement benchmark complete |
| A | `StoryworldTRM` | SWMD PICK traces, latent traces, rollout commitments, controller eval | train skill-conditioned membranes from real proposal traces and certified transitions | inventory mapped; direct training-data follow-up |
| A | `SmallControlHarness` | oracle-control attestation gates, registry memory, intervention receipts | typed provenance membrane around untrusted high-confidence proposals | proxy benchmarked; native controller adapter next |
| A | `AI_Diplomacy` | possible-order context, relationship memory, negotiations, phase summaries | hard order legality with soft coalition/betrayal forecasts and recurrent memory | proxy benchmarked; native order wrapper next |
| A | `BitVMArena` | settlement trajectories, control profiles, compatibility and regression catalogs | certify protocol compatibility before selecting replay-ranked control bundles | proxy benchmarked; trajectory adapter next |
| A | `Blighted Galaxy` | deterministic world simulation, strategic policy, replay, benchmark harness | bounded lookahead with exact tick/resource legality and latent strategic ranking | proxy benchmarked; replay adapter next |
| B | `StoryForge` | choice gates, variables, strategy instrumentation, prisoner-dilemma example | separate exact gate evaluation from model-sound best-response prediction | proxy benchmarked |
| B | `CodexGameStudio` | agent scopes, escalation paths, skill tests, and quality gates | optimize specialist routing while enforcing typed scope and review constraints | proxy benchmarked |
| B | `prime_intellect_research_environments` | versioned verifier environments and benchmark adapters | environment-pointer routing followed by typed action/output verification | inventory mapped; INTELLECT logic adapter already present |
| B | `metta-storyworld` | MeTTa/AIRIS gyms, navmesh, gate cartography, cross-domain pilot | symbolic skill routing and rule closure without introducing VPD into this study | inventory mapped |
| B | `Tesseract` | environment router and TRM training scaffold | calibrated route confidence with typed environment capability constraints | routing benchmark already integrated |

## New architecture hypotheses

- `typed-confidence`: certify only environment-derived exclusions, then arbitrate uncertain safe actions by margin.
- `counterfactual-beam`: keep a small latent proposal beam, certify each branch, then rank survivors jointly.
- `skill-router`: learn which control structure to invoke for each skill rather than using one global membrane.
- `dual-clock`: run cheap latent policy every tick and exact deduction only on state-boundary or confidence events.
- `receipt-carrying action`: require durable actions to carry the mechanic/provenance receipt that authorized them.
- `opponent-model quarantine`: retain coalition and morality forecasts in soft state until an observed transition promotes evidence.

## Recommended native experiment order

1. TheySing bridge-policy adapter: strongest ready-made game harness and hard/soft/graduated scenarios.
2. SmallControlHarness attestation adapter: clean provenance stress test for typed-confidence.
3. Blighted Galaxy replay adapter: longer-horizon test for counterfactual beam and dual-clock control.
4. StoryworldTRM trace adapter: replace synthetic skill routing with trained routing from committed PICK traces.
