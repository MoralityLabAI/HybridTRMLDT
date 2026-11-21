"""Run matched Goose episodes and retain native patch and provider receipts."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .gym import ROOT, clone, dump, source_root, verifier_manifest

ARMS = {
    "A": ("raw", None), "B": ("packet", None), "C": ("typed", "shared"),
    "D": ("typed", "recurrent"), "E": ("typed", "recurrent"), "shuffled": ("typed", "shuffled"),
}


def platformer(stage: Path, python: str) -> list[dict]:
    rows = json.loads((stage / "sealed.json").read_text(encoding="utf-8"))["test"]
    return [{"id": row["instance_id"], "suite": "platformer", "repo": str(source_root(stage, row)),
             "task": row["problem_statement"], "verifier": verifier_manifest(python),
             "gold_symbols": row["_receipt"]["gold_symbols"], "base_commit": row["base_commit"]}
            for row in rows]


def native(verifier: dict, root: Path) -> dict:
    try:
        proc = subprocess.run(verifier["argv"], cwd=root, capture_output=True,
                              timeout=verifier.get("timeout_seconds", 90),
                              env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    except subprocess.TimeoutExpired:
        return {"exit_code": None, "tail": "independent verifier timed out"}
    return {"exit_code": proc.returncode,
            "tail": (proc.stdout + proc.stderr)[-2200:].decode("utf-8", "replace")}


def summarize(rows: list[dict]) -> dict:
    groups = {}
    for key in sorted({(row["suite"], row["arm"]) for row in rows}):
        suite, arm = key
        results = [row for row in rows if (row["suite"], row["arm"]) == key]
        repairs = sum(row["verified"] for row in results)
        tokens = sum(row["input_tokens"] + row["output_tokens"] for row in results)
        recalls = [row["gold_symbol_recall"] for row in results if row["gold_symbol_recall"] is not None]
        groups[f"{suite}/{arm}"] = {
            "assigned": len(results), "verified_repairs": repairs, "provider_tokens": tokens,
            "provider_tokens_per_verified_repair": tokens / repairs if repairs else None,
            "mean_gold_symbol_recall": sum(recalls) / len(recalls) if recalls else None,
            "context_overflows": sum(row["context_overflows"] for row in results),
            "repeated_reads": sum(row["repeated_reads"] for row in results),
            "invalidated_handles": sum(row["invalidated_handles"] for row in results),
            "verifier_calls": sum(row["verifier_calls"] for row in results),
            "controller_seconds": sum(row["controller_seconds"] for row in results),
            "retrieval_seconds": sum(row.get("retrieval_seconds", 0) for row in results),
            "retrieval_postings": sum(row["retrieval_postings"] for row in results),
            "wall_seconds": sum(row["wall_seconds"] for row in results),
            "parity_failures": sum(row["parity_failures"] for row in results),
            "rubric_passes": sum(row["rubric_pass"] is True for row in results),
        }
    return groups


def episode(args, case: dict, arm: str, parent: Path) -> dict:
    import re
    started = time.monotonic()
    memory, controller = ARMS[arm]
    parent.mkdir(parents=True)
    root = parent / "worktree"
    verifier = case.get("verifier")
    if verifier:
        clone(Path(case["repo"]), root)
        before = native(verifier, root)
        dump(parent / "verifier.json", verifier)
    else:
        before = None
        root = Path(case["repo"])
    (parent / "task.txt").write_text(case["task"], encoding="utf-8")
    cmd = [args.binary, "--repo", str(root), "--task-file", str(parent / "task.txt"),
           "--base-url", args.base_url, "--model", args.model, "--budget", str(args.packet_budget),
           "--max-tokens", str(args.completion), "--max-tool-calls", str(args.calls),
           "--max-agent-iterations", str(args.iterations), "--context-window", str(args.context),
           "--timeout-secs", str(args.request_timeout), "--memory-mode", memory,
           "--python", args.python, "--goblin-cwd", str(ROOT),
           "--session-file", str(parent / "session.json")]
    if arm == "A":
        cmd.append("--no-packet")
    if args.json_actions:
        cmd.append("--json-actions")
    if controller:
        cmd += ["--controller-checkpoint", str(args.checkpoints / f"{controller}.pt")]
    if arm == "E":
        cmd.append("--controller-reset")
    if verifier:
        cmd += ["--verifier", str(parent / "verifier.json")]
    try:
        completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace",
                                   timeout=args.episode_timeout)
        (parent / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        payload = (json.loads(completed.stdout) if completed.returncode == 0 else
                   {"error": completed.stderr[-2500:], "exit_code": completed.returncode})
    except (subprocess.TimeoutExpired, ValueError) as exc:
        payload = {"error": str(exc)}
    dump(parent / "gdk.json", payload)
    state = payload.get("state") or {}
    patch = state.get("patch", "")
    after = None
    if verifier and patch:
        (parent / "final.patch").write_bytes(patch.encode())
        after = native(verifier, root)
    preflight = payload.get("preflight", [])
    parity_failures = sum(item.get("provider_preflight_parity") is False for item in preflight)
    seen = set(state.get("read_history") or [])
    for read in (payload.get("packet_stats") or {}).get("reads", []):
        uri = read if isinstance(read, str) else read.get("uri") if isinstance(read, dict) else None
        if uri:
            seen.add(uri)
    gold = set(case.get("gold_symbols", []))
    rubric = case.get("rubric", [])
    rubric_hits = [criterion for criterion in rubric
                   if re.search(criterion, payload.get("answer") or "", re.IGNORECASE | re.MULTILINE)]
    result = {"id": case["id"], "suite": case["suite"], "arm": arm,
              "verified": bool(before and before["exit_code"] != 0 and after and after["exit_code"] == 0
                               and state.get("approved")),
              "before": before, "independent_verifier": after, "patch_sha256": state.get("patch_sha256"),
              "input_tokens": payload.get("provider_input_tokens", 0),
              "output_tokens": payload.get("provider_output_tokens", 0),
              "inference_calls": payload.get("inference_calls", 0),
              "context_overflows": sum("context overflow:" in item.get("error", "") for item in preflight),
              "repeated_reads": state.get("repeated_reads", 0),
              "invalidated_handles": state.get("invalidated_handles", 0),
              "verifier_calls": state.get("verifier_calls", 0),
              "controller_seconds": state.get("controller_seconds", 0),
              "retrieval_seconds": state.get("retrieval_seconds", 0),
              "retrieval_postings": state.get("search_postings_ops", 0),
              "gold_symbol_recall": len(gold & seen)/len(gold) if gold else None,
              "wall_seconds": time.monotonic() - started, "parity_failures": parity_failures,
              "rubric_pass": len(rubric_hits) == len(rubric) if rubric else None,
              "rubric_hits": len(rubric_hits), "rubric_size": len(rubric),
              "error": payload.get("error"), "provider_preflight": preflight}
    dump(parent / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8097/v1/")
    parser.add_argument("--model", default="inky-mini-varloop")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--packet-budget", type=int, default=1024)
    parser.add_argument("--completion", type=int, default=768)
    parser.add_argument("--calls", type=int, default=18)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--episode-timeout", type=int, default=2400)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--json-actions", action="store_true")
    args = parser.parse_args()
    if not args.stage and not args.cases:
        parser.error("supply a frozen platformer stage or external case manifest")
    if any(ARMS[arm][1] for arm in args.arms) and not args.checkpoints:
        parser.error("learned arms require --checkpoints")
    args.binary = str(Path(args.binary).resolve(strict=True))
    args.out = args.out.resolve()
    if args.checkpoints:
        args.checkpoints = args.checkpoints.resolve(strict=True)
    cases = platformer(args.stage.resolve(), args.python) if args.stage else []
    if args.cases:
        cases += json.loads(args.cases.read_text(encoding="utf-8"))
    if args.case_ids:
        selected = set(args.case_ids)
        cases = [case for case in cases if case["id"] in selected]
        if len(cases) != len(selected):
            parser.error("one or more requested case IDs were not staged")
    if args.limit is not None:
        cases = cases[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    plan = {"model": args.model, "context": args.context, "calls": args.calls, "iterations": args.iterations,
            "completion": args.completion, "packet_budget": args.packet_budget, "arms": args.arms,
            "json_actions": args.json_actions,
            "cases": [case["id"] for case in cases]}
    plan_file = args.out / "plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text(encoding="utf-8")) != plan:
        parser.error("output directory contains a different evaluation plan")
    dump(plan_file, plan)
    rows = []
    for case in cases:
        for arm in args.arms:
            if Path(case["id"]).name != case["id"]:
                raise ValueError("case IDs must be single directory names")
            parent = args.out / case["id"] / arm
            if (parent / "result.json").exists():
                result = json.loads((parent / "result.json").read_text(encoding="utf-8"))
            else:
                if parent.exists():
                    parent.rename(parent.with_name(f"{arm}-interrupted-{time.time_ns()}"))
                result = episode(args, case, arm, parent)
            rows.append(result)
            print(json.dumps({key: result[key] for key in ("id", "arm", "verified", "input_tokens", "output_tokens",
                                                           "parity_failures", "rubric_pass", "error")}), flush=True)
            dump(args.out / "summary.json", {"plan": plan, "completed": len(rows),
                                             "assigned": len(cases) * len(args.arms), "summary": summarize(rows)})


if __name__ == "__main__":
    main()
