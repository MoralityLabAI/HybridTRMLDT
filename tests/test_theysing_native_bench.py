from pathlib import Path

from research_gym.adapters.theysing import TheySingNativeRun, TheySingRunSpec
from research_gym.benchmarks.theysing_native_bench import _bilateral_probe_plan, summarize_native_run


def test_native_trace_summary_separates_blocking_execution_and_usefulness(tmp_path: Path):
    spec = TheySingRunSpec(
        run_id="fixture-hard",
        scenario="fixture",
        scenario_path="fixture.json",
        enforcement_mode="hard",
        seed=7,
        turns=2,
    )
    trace = [
        {
            "type": "orders_submitted",
            "data": {"requestedOrderCount": 3, "acceptedOrderCount": 2, "rejectedOrderCount": 1},
            "trace": {"pre_state_hash": "a", "post_state_hash": "b"},
        },
        {"type": "pact_breach_blocked", "data": {}, "trace": {"pre_state_hash": "b", "post_state_hash": "b"}},
        {"type": "pact_breach_executed", "data": {}, "trace": {"pre_state_hash": "b", "post_state_hash": "c"}},
        {"type": "pact_breach_sanctioned", "data": {}, "trace": {"pre_state_hash": "c", "post_state_hash": "d"}},
    ]
    snapshot = {
        "turn": 3,
        "status": "running",
        "winner": None,
        "activePacts": [{"id": "p1"}],
        "recentMessages": [{"content": "hold"}],
        "trustMatrix": {"A": {"A": 100, "B": 40}, "B": {"A": 60, "B": 100}},
        "state": {"counters": {"tas": 11, "kessler": 12, "paxJenkinsAuthority": 13}},
    }
    run = TheySingNativeRun(spec, "session", snapshot, tmp_path / "trace.jsonl", trace)

    metrics = summarize_native_run(run)

    assert metrics.order_acceptance_rate == 2 / 3
    assert metrics.breach_attempts == 2
    assert metrics.breach_prevention_rate == 0.5
    assert metrics.breach_sanctions == 1
    assert metrics.state_change_events == 3
    assert metrics.mean_trust == 50.0


def test_bilateral_probe_binds_both_parties_and_targets_state_asset():
    plan = _bilateral_probe_plan()
    hegemon = plan["HEGEMON"]
    state = plan["STATE"]

    assert hegemon["negotiationRounds"][0]["pacts"][0]["counterpartyIds"] == ["STATE"]
    assert state["negotiationRounds"][0]["pacts"][0]["counterpartyIds"] == ["HEGEMON"]
    assert hegemon["action"]["orders"][0]["targetNodeId"] == "SAT_GUOWANG"
