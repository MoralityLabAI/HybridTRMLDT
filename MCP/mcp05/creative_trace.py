"""Validate and export machine-readable creative benchmark training traces."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .creative_bench import ORDER, digest, save


def journal(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def mcp_exchanges(path: Path) -> list[dict]:
    entries = journal(path)
    pending = {}
    for row in entries:
        wire = row["body"].encode("utf-8")
        if len(wire) != row["bytes"]:
            raise ValueError(f"MCP wire length mismatch: {path.name}")
        payload = json.loads(wire)
        identifier = payload.get("id")
        if row["direction"] == "client_to_mcp" and identifier is not None:
            if identifier in pending:
                raise ValueError(f"duplicate outstanding request: {identifier}")
            pending[identifier] = payload["method"]
        if row["direction"] == "mcp_to_client":
            if identifier not in pending:
                raise ValueError(f"unknown MCP response: {identifier}")
            del pending[identifier]
    if pending:
        raise ValueError(f"unanswered MCP requests: {pending}")
    return entries


def enrich(root: Path, plan: dict, runtime: dict, row: dict) -> dict:
    relative_attempt = row["attempt"].replace("\\", "/")
    attempt = root / relative_attempt
    for name, fingerprint in row["files"].items():
        path = attempt / name
        if path.stat().st_size != fingerprint["bytes"] or digest(path) != fingerprint["sha256"]:
            raise ValueError(f"archive mismatch: {path}")
    events = journal(attempt / "inky-wire.jsonl")
    if [event["event"] for event in events] != ["request", "response"]:
        raise ValueError(f"incomplete Inky exchange: {row['attempt']}")
    complete_request = events[0]["body"]
    request = json.loads(complete_request)
    response = json.loads(events[1]["body"])
    if request.get("cache_prompt") is not False:
        raise ValueError(f"expected a cold request: {row['attempt']}")
    verified = row["verdict"]["verified"]
    verdict = row["verdict"]
    reward = ((verdict.get("passed", 0) / verdict.get("assigned", 1))
              if verdict["status"] != "infrastructure-error" else None)
    receipt = json.loads((attempt / "retrieval.json").read_bytes())
    return {
        "schema": "pocket-goblin-creative-trace-v1",
        "case_group": f"creative-field-tests:{plan['project_commit']}:{row['project']}",
        "project": row["project"], "arm": row["arm"],
        "parent_attempt": (row["parent_attempt"].replace("\\", "/") if "parent_attempt" in row else None),
        "project_commit": plan["project_commit"], "index_revision": plan["index_revision"],
        "checkpoint_sha256": runtime["model_sha256"], "checkpoint_revision": runtime["model_revision"],
        "router_sha256": plan["router_sha256"] if row["arm"] == "assisted" else None,
        "trajectory": {"messages": request["messages"],
                       "assistant": (response.get("choices") or [{}])[0].get("message"),
                       "usage": response.get("usage"), "timings": response.get("timings")},
        "mcp": {"retrieval": mcp_exchanges(attempt / "mcp-wire.jsonl"),
                "goose": mcp_exchanges(attempt / "gdk-mcp-wire.jsonl"), "routes": receipt["routes"],
                "selected_chunks": receipt["chunks"]},
        "outcome": {"verified": verified, "reward": reward, **verdict},
        "lineage": {"attempt": relative_attempt, "request_body_sha256": hashlib.sha256(complete_request.encode()).hexdigest(),
                    "files": row["files"]},
        "split": "public-development", "training_ready": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign.resolve()
    plan = json.loads((root / "plan.json").read_bytes())
    indexed = journal(root / "training-traces.jsonl")
    runtime = json.loads(args.runtime.read_bytes())
    primary = [row for row in indexed if "parent_attempt" not in row]
    if tuple((row["project"], row["arm"]) for row in primary) != ORDER or runtime["context"] != 100096:
        raise ValueError("expected a complete alternating campaign on the pinned slot")
    traces = [enrich(root, plan, runtime, row) for row in indexed]
    path = root / "training-records.jsonl"
    path.write_text("".join(json.dumps(trace, ensure_ascii=False) + "\n" for trace in traces),
                    encoding="utf-8", newline="\n")
    save(root / "training-records.receipt.json", {
        "schema": "pocket-goblin-creative-trace-v1",
        "records": len(traces), "groups": len({trace["case_group"] for trace in traces}),
        "verified": sum(trace["outcome"]["verified"] for trace in traces),
        "raw_index_sha256": digest(root / "training-traces.jsonl"),
        "normalized_records_sha256": digest(path), "runtime_receipt_sha256": digest(args.runtime),
        "checkpoint_revision": runtime["model_revision"], "checkpoint_sha256": runtime["model_sha256"],
        "context": runtime["context"], "cpu_moe_layers": runtime["cpu_moe_layers"],
    })
    print(f"Verified and exported {len(traces)} complete traces")


if __name__ == "__main__":
    main()
