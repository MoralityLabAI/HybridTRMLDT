from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from research_gym.adapters.theysing import (
    PLAYABLE_FACTIONS,
    TheySingHarnessClient,
    TheySingNativeRun,
    TheySingRunSpec,
)


DEFAULT_SCENARIOS = (
    ("high_pressure_detente", "playtest/scenarios/high-pressure-detente.json"),
    ("diplomacy_ladder", "playtest/scenarios/asi2-asi3-diplomacy-question-ladder.json"),
    ("babel_compact", "playtest/scenarios/the-babel-compact-seven-asi.json"),
)
ENFORCEMENT_MODES = ("soft", "hard", "graduated")


@dataclass(frozen=True)
class TheySingRunMetrics:
    run_id: str
    scenario: str
    enforcement_mode: str
    proposal_source: str
    seed: int
    requested_orders: int
    accepted_orders: int
    rejected_orders: int
    order_acceptance_rate: float
    breach_attempts: int
    breach_blocked: int
    breach_executed: int
    breach_sanctions: int
    breach_prevention_rate: float | None
    trace_events: int
    state_change_events: int
    final_turn: int
    status: str
    winner: str | None
    active_pacts: int
    recent_messages: int
    mean_trust: float
    tas: float
    kessler: float
    pax_jenkins_authority: float
    trace_path: str

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


def summarize_native_run(run: TheySingNativeRun) -> TheySingRunMetrics:
    requested = 0
    accepted = 0
    rejected = 0
    blocked = 0
    executed = 0
    sanctions = 0
    state_changes = 0
    for entry in run.trace:
        event_type = str(entry.get("type", ""))
        data = entry.get("data", {})
        if not isinstance(data, dict):
            data = {}
        if event_type == "orders_submitted":
            requested += int(data.get("requestedOrderCount", 0) or 0)
            accepted += int(data.get("acceptedOrderCount", 0) or 0)
            rejected += int(data.get("rejectedOrderCount", 0) or 0)
        elif event_type == "pact_breach_blocked":
            blocked += 1
        elif event_type == "pact_breach_executed":
            executed += 1
        elif event_type == "pact_breach_sanctioned":
            sanctions += 1

        trace = entry.get("trace", {})
        if isinstance(trace, dict) and trace.get("pre_state_hash") != trace.get("post_state_hash"):
            state_changes += 1

    breach_attempts = blocked + executed
    snapshot = run.snapshot
    state = snapshot.get("state", {})
    if not isinstance(state, dict):
        state = {}
    counters = state.get("counters", {})
    if not isinstance(counters, dict):
        counters = {}
    active_pacts = snapshot.get("activePacts", [])
    recent_messages = snapshot.get("recentMessages", [])
    return TheySingRunMetrics(
        run_id=run.spec.run_id,
        scenario=run.spec.scenario,
        enforcement_mode=run.spec.enforcement_mode,
        proposal_source=run.spec.proposal_source,
        seed=run.spec.seed,
        requested_orders=requested,
        accepted_orders=accepted,
        rejected_orders=rejected,
        order_acceptance_rate=accepted / requested if requested else 0.0,
        breach_attempts=breach_attempts,
        breach_blocked=blocked,
        breach_executed=executed,
        breach_sanctions=sanctions,
        breach_prevention_rate=blocked / breach_attempts if breach_attempts else None,
        trace_events=len(run.trace),
        state_change_events=state_changes,
        final_turn=int(snapshot.get("turn", 0) or 0),
        status=str(snapshot.get("status", "unknown")),
        winner=str(snapshot["winner"]) if snapshot.get("winner") else None,
        active_pacts=len(active_pacts) if isinstance(active_pacts, list) else 0,
        recent_messages=len(recent_messages) if isinstance(recent_messages, list) else 0,
        mean_trust=_mean_trust(snapshot.get("trustMatrix", {})),
        tas=float(counters.get("tas", 0.0) or 0.0),
        kessler=float(counters.get("kessler", 0.0) or 0.0),
        pax_jenkins_authority=float(counters.get("paxJenkinsAuthority", 0.0) or 0.0),
        trace_path=str(run.trace_path),
    )


def _mean_trust(value: object) -> float:
    if not isinstance(value, dict):
        return 0.0
    trust: list[float] = []
    for faction, row in value.items():
        if not isinstance(row, dict):
            continue
        for counterparty, score in row.items():
            if faction == counterparty or not isinstance(score, (int, float)):
                continue
            trust.append(float(score))
    return sum(trust) / len(trust) if trust else 0.0


def _aggregate(rows: list[TheySingRunMetrics]) -> dict[str, float | int | None]:
    requested = sum(row.requested_orders for row in rows)
    accepted = sum(row.accepted_orders for row in rows)
    attempts = sum(row.breach_attempts for row in rows)
    blocked = sum(row.breach_blocked for row in rows)
    return {
        "runs": len(rows),
        "requested_orders": requested,
        "accepted_orders": accepted,
        "order_acceptance_rate": accepted / requested if requested else 0.0,
        "breach_attempts": attempts,
        "breach_blocked": blocked,
        "breach_executed": sum(row.breach_executed for row in rows),
        "breach_sanctions": sum(row.breach_sanctions for row in rows),
        "breach_prevention_rate": blocked / attempts if attempts else None,
        "mean_trust": sum(row.mean_trust for row in rows) / len(rows) if rows else 0.0,
        "mean_tas": sum(row.tas for row in rows) / len(rows) if rows else 0.0,
        "mean_kessler": sum(row.kessler for row in rows) / len(rows) if rows else 0.0,
        "mean_pax_jenkins_authority": (
            sum(row.pax_jenkins_authority for row in rows) / len(rows) if rows else 0.0
        ),
        "mean_state_change_events": (
            sum(row.state_change_events for row in rows) / len(rows) if rows else 0.0
        ),
    }


def run_theysing_native_benchmark(
    repo_root: Path,
    artifact_root: Path,
    *,
    seeds: Iterable[int] = (400, 401),
    turns: int = 6,
    scenarios: Iterable[tuple[str, str]] = DEFAULT_SCENARIOS,
) -> dict[str, object]:
    seed_values = tuple(seeds)
    scenario_values = tuple(scenarios)
    specs = [
        TheySingRunSpec(
            run_id=f"{scenario}-{mode}-seed{seed}",
            scenario=scenario,
            scenario_path=scenario_path,
            enforcement_mode=mode,
            seed=seed,
            turns=turns,
        )
        for scenario, scenario_path in scenario_values
        for seed in seed_values
        for mode in ENFORCEMENT_MODES
    ]
    rows: list[TheySingRunMetrics] = []
    with TheySingHarnessClient(repo_root, artifact_root) as client:
        for spec in specs:
            rows.append(summarize_native_run(client.run(spec)))
        for seed in seed_values:
            for mode in ENFORCEMENT_MODES:
                spec = TheySingRunSpec(
                    run_id=f"bilateral_probe-{mode}-seed{seed}",
                    scenario="bilateral_probe",
                    scenario_path="playtest/scenarios/high-pressure-detente.json",
                    enforcement_mode=mode,
                    seed=seed,
                    turns=1,
                    proposal_source="forced_bilateral_probe",
                )
                rows.append(summarize_native_run(client.run_manual(spec, _bilateral_probe_plan())))

    grouped_by_mode: dict[str, list[TheySingRunMetrics]] = defaultdict(list)
    grouped_by_scenario: dict[str, dict[str, list[TheySingRunMetrics]]] = defaultdict(lambda: defaultdict(list))
    grouped_by_source: dict[str, list[TheySingRunMetrics]] = defaultdict(list)
    for row in rows:
        grouped_by_mode[row.enforcement_mode].append(row)
        grouped_by_scenario[row.scenario][row.enforcement_mode].append(row)
        grouped_by_source[row.proposal_source].append(row)
    return {
        "schema": "theysing_native_hybrid_benchmark_v1",
        "source_repo": str(repo_root.resolve()),
        "turns_requested": turns,
        "seeds": list(seed_values),
        "scenarios": [scenario for scenario, _ in scenario_values] + ["bilateral_probe"],
        "interpretation": {
            "soft": "TRM-like proposal path: legal orders execute despite pact conflicts and receive sanctions.",
            "hard": "LDT-like hard gate: every detected active-pact conflict is blocked.",
            "graduated": "Typed hybrid: destructive institutional conflicts are blocked; bilateral conflicts execute with sanctions.",
        },
        "summary_by_mode": {
            mode: _aggregate(grouped_by_mode[mode]) for mode in ENFORCEMENT_MODES
        },
        "summary_by_scenario": {
            scenario: {
                mode: _aggregate(mode_rows)
                for mode, mode_rows in sorted(mode_groups.items())
            }
            for scenario, mode_groups in sorted(grouped_by_scenario.items())
        },
        "summary_by_proposal_source": {
            source: _aggregate(source_rows)
            for source, source_rows in sorted(grouped_by_source.items())
        },
        "runs": [row.to_jsonable() for row in rows],
    }


def _bilateral_probe_plan() -> dict[str, object]:
    plan: dict[str, object] = {}
    for faction in PLAYABLE_FACTIONS:
        pacts = []
        if faction == "HEGEMON":
            pacts = [{"type": "NON_AGGRESSION", "counterpartyIds": ["STATE"], "durationTurns": 2}]
        elif faction == "STATE":
            pacts = [{"type": "NON_AGGRESSION", "counterpartyIds": ["HEGEMON"], "durationTurns": 2}]
        action_orders = []
        if faction == "HEGEMON":
            action_orders = [
                {
                    "type": "ATTACK",
                    "unitId": "H_SAT_1",
                    "targetNodeId": "SAT_GUOWANG",
                }
            ]
        plan[faction] = {
            "negotiationRounds": [
                {
                    "reasoning": "Matched bilateral non-aggression enforcement probe.",
                    "messages": [],
                    "pacts": pacts,
                }
            ],
            "allocation": {"reasoning": "No allocation in enforcement probe.", "orders": []},
            "action": {"reasoning": "Submit matched pact-conflicting action.", "orders": action_orders},
        }
    return plan


def summary_markdown(payload: dict[str, object]) -> str:
    by_mode = payload["summary_by_mode"]
    assert isinstance(by_mode, dict)
    lines = [
        "# TheySing Native Hybrid Enforcement Benchmark",
        "",
        "This benchmark runs the compiled TheySing headless engine directly.",
        "The enforcement modes are native game mechanics, not source-inspired proxy task cards.",
        "The three game scenarios use native heuristic policies; `bilateral_probe` is a matched forced-pressure turn.",
        "",
        f"Scenarios: {', '.join(f'`{name}`' for name in payload['scenarios'])}",
        f"Seeds: {', '.join(str(seed) for seed in payload['seeds'])}",
        f"Requested turns per run: `{payload['turns_requested']}`",
        "",
        "| Mode | Runs | Order Acceptance | Breach Attempts | Blocked | Executed | Sanctions | Prevention | Mean Trust |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ENFORCEMENT_MODES:
        metrics = by_mode[mode]
        assert isinstance(metrics, dict)
        prevention = metrics["breach_prevention_rate"]
        prevention_text = "n/a" if prevention is None else f"{float(prevention):.3f}"
        lines.append(
            f"| `{mode}` | {metrics['runs']} | {float(metrics['order_acceptance_rate']):.3f} | "
            f"{metrics['breach_attempts']} | {metrics['breach_blocked']} | {metrics['breach_executed']} | "
            f"{metrics['breach_sanctions']} | {prevention_text} | {float(metrics['mean_trust']):.2f} |"
        )

    by_scenario = payload["summary_by_scenario"]
    assert isinstance(by_scenario, dict)
    lines.extend(["", "## Scenario splits", ""])
    for scenario, modes in by_scenario.items():
        assert isinstance(modes, dict)
        lines.extend(
            [
                f"### {scenario}",
                "",
                "| Mode | Acceptance | Attempts | Blocked | Executed | Mean TAS | Mean Kessler |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for mode in ENFORCEMENT_MODES:
            metrics = modes[mode]
            assert isinstance(metrics, dict)
            lines.append(
                f"| `{mode}` | {float(metrics['order_acceptance_rate']):.3f} | {metrics['breach_attempts']} | "
                f"{metrics['breach_blocked']} | {metrics['breach_executed']} | "
                f"{float(metrics['mean_tas']):.2f} | {float(metrics['mean_kessler']):.2f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Control interpretation",
            "",
            "- The institutional breach tests destructive action under a cislunar common-carrier pact.",
            "- The bilateral probe tests the same action channel under a non-aggression pact.",
            "- Graduated enforcement should block the institutional breach but execute and sanction the bilateral breach.",
            "- Hard enforcement should block both; soft enforcement should execute and sanction both.",
            "",
        ]
    )
    return "\n".join(lines)


def write_results(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
