from __future__ import annotations


APPLICATION_SHORTLIST = (
    {
        "project": "GPTStoryworld",
        "priority": "A",
        "native_surface": "storyworld/env/diplomacy_env.py; verifier symbolic routes and turn traces",
        "hybrid_application": "certify legal/reachable choices while keeping morality and opponent forecasts soft",
        "status": "proxy benchmarked; native adapter next",
    },
    {
        "project": "TheySing",
        "priority": "A",
        "native_surface": "src/harness policies, bridge policy, campaign clock, tournaments, replay traces",
        "hybrid_application": "typed channel permissions plus confidence/beam arbitration under treaty pressure",
        "status": "proxy benchmarked; native adapter next",
    },
    {
        "project": "StoryworldTRM",
        "priority": "A",
        "native_surface": "SWMD PICK traces, latent traces, rollout commitments, controller eval",
        "hybrid_application": "train skill-conditioned membranes from real proposal traces and certified transitions",
        "status": "inventory mapped; direct training-data follow-up",
    },
    {
        "project": "SmallControlHarness",
        "priority": "A",
        "native_surface": "oracle-control attestation gates, registry memory, intervention receipts",
        "hybrid_application": "typed provenance membrane around untrusted high-confidence proposals",
        "status": "proxy benchmarked; native controller adapter next",
    },
    {
        "project": "AI_Diplomacy",
        "priority": "A",
        "native_surface": "possible-order context, relationship memory, negotiations, phase summaries",
        "hybrid_application": "hard order legality with soft coalition/betrayal forecasts and recurrent memory",
        "status": "proxy benchmarked; native order wrapper next",
    },
    {
        "project": "BitVMArena",
        "priority": "A",
        "native_surface": "settlement trajectories, control profiles, compatibility and regression catalogs",
        "hybrid_application": "certify protocol compatibility before selecting replay-ranked control bundles",
        "status": "proxy benchmarked; trajectory adapter next",
    },
    {
        "project": "Blighted Galaxy",
        "priority": "A",
        "native_surface": "deterministic world simulation, strategic policy, replay, benchmark harness",
        "hybrid_application": "bounded lookahead with exact tick/resource legality and latent strategic ranking",
        "status": "proxy benchmarked; replay adapter next",
    },
    {
        "project": "StoryForge",
        "priority": "B",
        "native_surface": "choice gates, variables, strategy instrumentation, prisoner-dilemma example",
        "hybrid_application": "separate exact gate evaluation from model-sound best-response prediction",
        "status": "proxy benchmarked",
    },
    {
        "project": "CodexGameStudio",
        "priority": "B",
        "native_surface": "agent scopes, escalation paths, skill tests, and quality gates",
        "hybrid_application": "optimize specialist routing while enforcing typed scope and review constraints",
        "status": "proxy benchmarked",
    },
    {
        "project": "prime_intellect_research_environments",
        "priority": "B",
        "native_surface": "versioned verifier environments and benchmark adapters",
        "hybrid_application": "environment-pointer routing followed by typed action/output verification",
        "status": "inventory mapped; INTELLECT logic adapter already present",
    },
    {
        "project": "metta-storyworld",
        "priority": "B",
        "native_surface": "MeTTa/AIRIS gyms, navmesh, gate cartography, cross-domain pilot",
        "hybrid_application": "symbolic skill routing and rule closure without introducing VPD into this study",
        "status": "inventory mapped",
    },
    {
        "project": "Tesseract",
        "priority": "B",
        "native_surface": "environment router and TRM training scaffold",
        "hybrid_application": "calibrated route confidence with typed environment capability constraints",
        "status": "routing benchmark already integrated",
    },
)


def application_map_markdown() -> str:
    lines = [
        "# Hybrid Control and Game Application Map",
        "",
        "This shortlist follows a broad automated scan with targeted inspection of the highest-value native interfaces.",
        "Priority A means a deterministic adapter can be built from an existing action/trace/controller surface.",
        "Priority B means the application is useful but is either routing-oriented or needs a narrower fixture first.",
        "",
        "| Priority | Project | Native surface | Hybrid application | Current status |",
        "|---|---|---|---|---|",
    ]
    for row in APPLICATION_SHORTLIST:
        lines.append(
            f"| {row['priority']} | `{row['project']}` | {row['native_surface']} | "
            f"{row['hybrid_application']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## New architecture hypotheses",
            "",
            "- `typed-confidence`: certify only environment-derived exclusions, then arbitrate uncertain safe actions by margin.",
            "- `counterfactual-beam`: keep a small latent proposal beam, certify each branch, then rank survivors jointly.",
            "- `skill-router`: learn which control structure to invoke for each skill rather than using one global membrane.",
            "- `dual-clock`: run cheap latent policy every tick and exact deduction only on state-boundary or confidence events.",
            "- `receipt-carrying action`: require durable actions to carry the mechanic/provenance receipt that authorized them.",
            "- `opponent-model quarantine`: retain coalition and morality forecasts in soft state until an observed transition promotes evidence.",
            "",
            "## Recommended native experiment order",
            "",
            "1. TheySing bridge-policy adapter: strongest ready-made game harness and hard/soft/graduated scenarios.",
            "2. SmallControlHarness attestation adapter: clean provenance stress test for typed-confidence.",
            "3. Blighted Galaxy replay adapter: longer-horizon test for counterfactual beam and dual-clock control.",
            "4. StoryworldTRM trace adapter: replace synthetic skill routing with trained routing from committed PICK traces.",
            "",
        ]
    )
    return "\n".join(lines)
