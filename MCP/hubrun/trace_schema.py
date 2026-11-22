"""Non-content schema census for committed public hub-run trace archives.

Tool-call/re-read estimates require a defensible observation boundary. This
probe shows where tool actions are located (or absent) in archived verifiers
traces before a metric extractor is allowed to score them. No message text or
tool argument values are emitted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

SENSITIVE = {"content", "text", "arguments", "prompt", "response", "completion"}
STRUCTURE = {"traces", "messages", "completion", "trajectory", "events", "steps",
             "history", "tool_calls", "tool_call_id", "function", "role", "type",
             "tool_name", "name", "result", "observation", "action"}


def inspect_record(record: Any, *, max_depth: int = 12) -> dict:
    """Count nested keys and explicit OpenAI-style tool message structures."""
    paths: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    tool_call_lists = 0
    tool_call_entries = 0
    tool_result_messages = 0
    message_lists = 0

    def walk(node: Any, path: str, depth: int) -> None:
        nonlocal tool_call_lists, tool_call_entries, tool_result_messages, message_lists
        if depth > max_depth:
            return
        if isinstance(node, dict):
            for key in node:
                if key in STRUCTURE:
                    paths[f"{path}.{key}" if path else key] += 1
            role = node.get("role")
            if isinstance(role, str):
                roles[role] += 1
                if role in ("tool", "function") and "tool_call_id" in node:
                    tool_result_messages += 1
            calls = node.get("tool_calls")
            if isinstance(calls, list):
                tool_call_lists += 1
                tool_call_entries += sum(isinstance(x, dict) for x in calls)
            for key, value in node.items():
                if isinstance(value, (dict, list)) and key not in SENSITIVE - {"completion"}:
                    walk(value, f"{path}.{key}" if path else key, depth + 1)
        elif isinstance(node, list):
            if node and all(isinstance(v, dict) and "role" in v for v in node[:min(len(node), 4)]):
                message_lists += 1
                paths[f"{path}[]:role-sequence"] += 1
            # Visit full arrays so tool-call counters reflect actual sampled rows.
            for value in node:
                if isinstance(value, (dict, list)):
                    walk(value, f"{path}[]", depth + 1)

    walk(record, "", 0)
    return {
        "paths": dict(sorted(paths.items())),
        "roles": dict(sorted(roles.items())),
        "tool_call_lists": tool_call_lists,
        "tool_call_entries": tool_call_entries,
        "tool_result_messages": tool_result_messages,
        "role_sequence_lists": message_lists,
        "top_level_keys": sorted(record) if isinstance(record, dict) else [],
        "first_trace_keys": sorted(record["traces"][0]) if (
            isinstance(record, dict) and isinstance(record.get("traces"), list)
            and record["traces"] and isinstance(record["traces"][0], dict)
        ) else [],
    }


def inspect_archive(path: Path, *, limit: int = 3) -> dict:
    if limit < 1:
        raise ValueError("limit must be positive")
    if not path.is_file() or path.suffix not in (".gz", ".jsonl"):
        raise ValueError("input must be an existing .gz or .jsonl file")
    raw = path.read_bytes()
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(inspect_record(json.loads(line)))
                if len(rows) == limit:
                    break
    paths: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    for row in rows:
        paths.update(row["paths"])
        roles.update(row["roles"])
    return {
        "file": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "compressed_bytes": len(raw),
        "sampled_rows": len(rows),
        "top_level_keys": sorted({k for row in rows for k in row["top_level_keys"]}),
        "first_trace_keys": sorted({k for row in rows for k in row["first_trace_keys"]}),
        "structural_paths": dict(sorted(paths.items())),
        "roles": dict(sorted(roles.items())),
        "tool_call_lists": sum(row["tool_call_lists"] for row in rows),
        "tool_call_entries": sum(row["tool_call_entries"] for row in rows),
        "tool_result_messages": sum(row["tool_result_messages"] for row in rows),
        "role_sequence_lists": sum(row["role_sequence_lists"] for row in rows),
        "tool_actions_observable": any(
            row["tool_call_entries"] > 0 and row["tool_result_messages"] > 0 for row in rows
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    candidates = sorted(args.root.glob("*.jsonl.gz")) + sorted(args.root.glob("*.jsonl"))
    if not candidates:
        ap.error("no trace archives found")
    report = {
        "kind": "non-content-archive-schema-census",
        "archive_count": len(candidates),
        "archives": [inspect_archive(p, limit=args.limit) for p in candidates],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for item in report["archives"]:
        print(json.dumps({key: item[key] for key in ("file", "sampled_rows", "top_level_keys",
                                                   "first_trace_keys", "roles",
                                                   "tool_call_entries", "tool_result_messages",
                                                   "tool_actions_observable")}, sort_keys=True))
    print(f"Full structural receipt: {args.out}")


if __name__ == "__main__":
    main()
