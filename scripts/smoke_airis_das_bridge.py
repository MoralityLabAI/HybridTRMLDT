from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_gym.adapters.airis_das import (
    AirisDasHttpClient,
    airis_observation,
    embedded_forecast,
    load_jsonl,
    resolve_airis_decision,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test hybrid rules against the local AIRIS/DAS HTTP service.")
    parser.add_argument(
        "--metta-apps-root",
        type=Path,
        default=Path(r"C:\projects\metta-storyworld\metta-etc"),
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=Path("data/airis_das/sequencer_rules.json"),
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        default=Path("data/benchmarks/airis_das_bridge_episodes.jsonl"),
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/benchmarks/sequencer_control_results.json"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/airis_das/live_service_smoke.json")
    )
    args = parser.parse_args()

    if not args.metta_apps_root.exists():
        raise FileNotFoundError(f"metta AIRIS implementation not found: {args.metta_apps_root}")
    sys.path.insert(0, str(args.metta_apps_root))
    from metta_apps.airis_das_service import (  # type: ignore[import-not-found]
        AirisDasIndex,
        build_airis_das_server,
        serve_in_background,
    )

    ruleset = json.loads(args.rules.read_text(encoding="utf-8"))
    rules = ruleset["rules"]
    row = load_jsonl(str(args.episodes))[0]
    protocol = json.loads(args.benchmark.read_text(encoding="utf-8"))["protocol_sha256"]
    observation = airis_observation(row, protocol)
    expected = embedded_forecast(observation, rules)

    index = AirisDasIndex.from_rules_path(args.rules)
    server = build_airis_das_server(index, host="127.0.0.1", port=0)
    thread = serve_in_background(server)
    try:
        host, port = server.server_address
        client = AirisDasHttpClient(f"http://{host}:{port}")
        service_summary = client.summary()
        actual = client.forecast(observation)
        decision = resolve_airis_decision(
            row, actual, expected_protocol_sha256=protocol
        )

        stale_observation = deepcopy(observation)
        stale_observation["condition_features"] = [
            "protocol_sha256=stale" if str(feature).startswith("protocol_sha256=") else feature
            for feature in stale_observation["condition_features"]
        ]
        stale_forecast = client.forecast(stale_observation)
        stale_decision = resolve_airis_decision(
            row, stale_forecast, expected_protocol_sha256=protocol
        )
        passed = (
            service_summary.get("schema") == "airis_das_service_v1"
            and service_summary.get("rule_count") == len(rules)
            and actual.get("best_rule_id") == expected.get("best_rule_id")
            and decision.accepted
            and decision.selected_sequence == decision.fallback_sequence
            and not stale_decision.accepted
            and stale_decision.selected_sequence == stale_decision.fallback_sequence
        )
        receipt = {
            "schema": "hybrid_airis_das_live_smoke_v1",
            "service_summary": service_summary,
            "episode_id": row["episode_id"],
            "embedded_best_rule_id": expected.get("best_rule_id"),
            "http_best_rule_id": actual.get("best_rule_id"),
            "forecast_parity": actual.get("best_rule_id") == expected.get("best_rule_id"),
            "decision": decision.to_jsonable(),
            "stale_protocol_decision": stale_decision.to_jsonable(),
            "passed": passed,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
