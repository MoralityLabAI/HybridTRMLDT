"""Independent, no-model audit of public hub-run trace tool behavior.

Doc-edit publishes environment counters for reads, edits, stale sites and
submissions. Event-recon publishes structured tool calls in node messages.
This module reports only behavior observable at those respective boundaries.
It cannot grade correctness beyond the environment's recorded reward.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


ARCHIVES = (
    "docedit-raw.results.jsonl.gz",
    "docedit-goblin.results.jsonl.gz",
    "er-t1-raw.traces.jsonl.gz",
    "er-t1-goblin.traces.jsonl.gz",
    "er-t1-late.traces.jsonl.gz",
)
DOCEDIT_COUNTERS = ("read_page_calls", "stale_sites", "edit_text_calls", "submit_calls",
                    "search_doc_calls", "site_recall")
READ_NAMES = {"read_file", "read_page", "open_file", "cat_file", "read_symbol"}


def archive_rows(path: Path) -> tuple[list[dict], dict]:
    if not path.is_file():
        raise ValueError(f"missing archive: {path.name}")
    compressed = path.read_bytes()
    with gzip.open(path, "rb") as fh:
        source = fh.read()
    rows = [json.loads(line) for line in source.decode("utf-8").splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"missing or non-object JSONL rows: {path.name}")
    return rows, {"file": path.name, "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
                  "source_sha256": hashlib.sha256(source).hexdigest(), "rows": len(rows)}


def _number(row: dict, key: str) -> float | None:
    value = row.get(key)
    if type(value) in (int, float) and -1e18 < value < 1e18:
        return float(value)
    return None


def docedit_audit(rows: list[dict]) -> dict:
    """Use environment counters, which have a stronger basis than call guessing."""
    cases = {}
    missing = Counter()
    for row in rows:
        identity = row.get("example_id")
        if type(identity) not in (int, str):
            raise ValueError("doc-edit row missing example_id")
        case = str(identity)
        if case in cases:
            raise ValueError("duplicate doc-edit case ID")
        counters = {key: _number(row, key) for key in DOCEDIT_COUNTERS}
        for key, value in counters.items():
            if value is None:
                missing[key] += 1
        cases[case] = {"counters": counters, "reward": _number(row, "reward"),
                       "submitted": row.get("submitted") if type(row.get("submitted")) is bool else None}
    means = {key: round(sum(case["counters"][key] for case in cases.values()) / len(cases), 4)
             if all(case["counters"][key] is not None for case in cases.values()) else None
             for key in DOCEDIT_COUNTERS}
    return {"basis": "environment-reported-counters", "episodes": len(cases),
            "means": means, "fields_missing_by_episode": dict(sorted(missing.items())),
            "submitted": (sum(case["submitted"] for case in cases.values())
                          if all(case["submitted"] is not None for case in cases.values()) else None),
            "solved_reward_ge_0_999": (sum(case["reward"] >= 0.999 for case in cases.values())
                                      if all(case["reward"] is not None for case in cases.values()) else None),
            "cases": cases}


def _arguments(call: dict) -> dict | None:
    """Recover only explicitly encoded tool arguments. A missing value stays unknown."""
    for key in ("arguments", "args", "input"):
        value = call.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                payload = json.loads(value)
            except (ValueError, TypeError):
                return None
            if isinstance(payload, dict):
                return payload
    return None


def _call_name(call: dict) -> str | None:
    for value in (call.get("name"), (call.get("function") or {}).get("name")
                  if isinstance(call.get("function"), dict) else None):
        if isinstance(value, str) and value:
            return value
    return None


def event_recon_audit(rows: list[dict]) -> dict:
    """Inspect each trace's node messages once and audit exact read-request repeats."""
    cases: dict[str, dict] = {}
    errors: list[str] = []
    for row in rows:
        identity = (row.get("task") or {}).get("data", {}).get("name")
        traces = row.get("traces")
        if not isinstance(identity, str) or not isinstance(traces, list) or not traces:
            raise ValueError("event-recon row missing named task or traces")
        for tr_idx, trace in enumerate(traces):
            if not isinstance(trace, dict) or not isinstance(trace.get("nodes"), list):
                errors.append("trace without nodes for " + identity)
                continue
            key = identity if len(traces) == 1 else f"{identity}#{tr_idx}"
            if key in cases:
                raise ValueError("duplicate event-recon episode ID")
            reads = 0
            valid_read_args = 0
            unparsable_reads = 0
            seen: Counter[str] = Counter()
            paths: Counter[str] = Counter()
            tool_calls = 0
            tool_names: Counter[str] = Counter()
            tool_results = 0
            for node in trace["nodes"]:
                if not isinstance(node, dict):
                    continue
                message = node.get("message")
                if not isinstance(message, dict):
                    continue
                if message.get("role") == "tool":
                    tool_results += 1
                calls = message.get("tool_calls")
                if not isinstance(calls, list):
                    continue
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    tool_calls += 1
                    name = _call_name(call)
                    if name:
                        tool_names[name] += 1
                    if name not in READ_NAMES:
                        continue
                    reads += 1
                    args = _arguments(call)
                    if args is None:
                        unparsable_reads += 1
                        continue
                    valid_read_args += 1
                    # Hash, then count. Tool payloads and paths remain outside
                    # the saved result; exact content is never reported.
                    signature = hashlib.sha256(json.dumps(
                        [name, args], ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"), default=str).encode()).hexdigest()
                    seen[signature] += 1
                    path = args.get("path", args.get("file_path"))
                    if isinstance(path, str):
                        paths[hashlib.sha256(path.encode()).hexdigest()] += 1
            cases[key] = {
                "tool_calls": tool_calls, "tool_result_messages": tool_results,
                "tool_names": dict(sorted(tool_names.items())),
                "read_file_calls": reads, "read_args_parsed": valid_read_args,
                "read_args_missing": unparsable_reads,
                "repeat_exact_requests": (sum(n - 1 for n in seen.values() if n > 1)
                                          if unparsable_reads == 0 else None),
                "repeat_same_path": (sum(n - 1 for n in paths.values() if n > 1)
                                     if unparsable_reads == 0 and sum(paths.values()) == reads else None),
            }
    observed_names: Counter[str] = Counter()
    for case in cases.values():
        observed_names.update(case["tool_names"])
    summary = {"basis": "event-recon-structured-node-tool-calls", "episodes": len(cases),
               "errors": errors, "tool_name_counts": dict(sorted(observed_names.items())),
               "cases": cases}
    for metric in ("tool_calls", "tool_result_messages", "read_file_calls", "read_args_parsed",
                   "read_args_missing", "repeat_exact_requests", "repeat_same_path"):
        values = [case[metric] for case in cases.values()]
        summary[metric + "_total"] = sum(values) if all(v is not None for v in values) else None
    return summary


def build_report(root: Path) -> dict:
    result = {"kind": "historical-public-trace-audit", "sources": {},
              "docedit_pair": {}, "event_recon": {}, "limitations": [
                  "Historical raw/goblin/late runs, not an MCP-04 or MCP-05 evaluation.",
                  "Doc-edit counters originate in the environment; no independent edit-site grader is claimed.",
                  "Event-recon exact read-request repeats require fully observable tool arguments.",
                  "Tool-call counts measure requests; repeated reads may be intentional.",
              ]}
    for filename in ARCHIVES:
        rows, provenance = archive_rows(root / filename)
        result["sources"][filename] = provenance
        if filename.startswith("docedit-"):
            arm = filename.split("-")[1].split(".")[0]
            result["docedit_pair"][arm] = docedit_audit(rows)
        else:
            arm = filename.split("-")[2].split(".")[0]
            result["event_recon"][arm] = event_recon_audit(rows)
    a, b = (result["docedit_pair"][arm]["cases"] for arm in ("raw", "goblin"))
    common = set(a) & set(b)
    result["docedit_pair"]["task_ids_matched"] = len(common) == len(a) == len(b)
    result["docedit_pair"]["n_matched"] = len(common)
    result["docedit_pair"]["goblin_minus_raw"] = {
        key: round(sum(b[c]["counters"][key] - a[c]["counters"][key] for c in common) / len(common), 4)
        if common and all(a[c]["counters"][key] is not None and b[c]["counters"][key] is not None
                          for c in common) else None for key in DOCEDIT_COUNTERS}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_report(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"docedit": {a: {"episodes": v["episodes"], "means": v["means"],
                                   "solved": v["solved_reward_ge_0_999"]}
                                  for a, v in receipt["docedit_pair"].items() if a in ("raw", "goblin")},
                      "docedit_pair": {"matched": receipt["docedit_pair"]["task_ids_matched"],
                                      "goblin_minus_raw": receipt["docedit_pair"]["goblin_minus_raw"]},
                      "event_recon": {a: {"episodes": v["episodes"], "reads": v["read_file_calls_total"],
                                          "exact_repeats": v["repeat_exact_requests_total"],
                                          "unknown_read_args": v["read_args_missing_total"],
                                          "tool_names": v["tool_name_counts"]}
                                      for a, v in receipt["event_recon"].items()},
                      "receipt": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
