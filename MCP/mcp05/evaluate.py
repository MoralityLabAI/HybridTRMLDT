"""Run matched Goose episodes and retain native patch and provider receipts."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .context import WorkingSet
from .gym import ROOT, clone, dump, source_root, verifier_manifest
from .service import RepairWorkspace

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
            "episodes_with_final_runtime_metrics": sum(row.get("final_runtime_metrics") is True for row in results),
            "replayed_inference_calls": sum(row.get("replayed_inference_calls", 0) for row in results),
            "controller_seconds": sum(row["controller_seconds"] for row in results),
            "retrieval_seconds": sum(row.get("retrieval_seconds", 0) for row in results),
            "retrieval_postings": sum(row["retrieval_postings"] for row in results),
            "wall_seconds": sum(row["wall_seconds"] for row in results),
            "parity_failures": sum(row["parity_failures"] for row in results),
            "rubric_passes": sum(row["rubric_pass"] is True for row in results),
        }
    return groups


def run_gdk(cmd: list[str], timeout: int, *, env: dict | None = None) -> tuple[int, str, str]:
    windows = os.name == "nt"
    with subprocess.Popen(
        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
        errors="replace", start_new_session=not windows, env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if windows else 0,
    ) as child:
        try:
            stdout, stderr = child.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if windows:
                subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                               capture_output=True, timeout=10, check=False)
            else:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            child.kill()
            child.communicate()
            raise
        return child.returncode, stdout, stderr


def fallback_usage(parent: Path, payload: dict) -> None:
    if not (payload.get("error") or payload.get("exit_code")) or payload.get("preflight"):
        return
    trace = parent / "session.preflight.jsonl"
    if not trace.exists():
        return
    receipts = []
    for line in trace.read_text(encoding="utf-8").splitlines():
        try:
            receipts.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    completed = [row for row in receipts if row.get("provider_status") == 200]
    payload.update(preflight=receipts, inference_calls=len(completed),
                   provider_input_tokens=sum(row.get("provider_prompt_tokens") or 0 for row in completed),
                   provider_output_tokens=sum(row.get("provider_completion_tokens") or 0 for row in completed))


def recover_session(parent: Path, payload: dict) -> list[dict]:
    session = payload.get("session")
    if session is None and (payload.get("error") or payload.get("exit_code")):
        try:
            session = json.loads((parent / "session.json").read_text(encoding="utf-8"))["session"]
        except (OSError, ValueError, KeyError, TypeError):
            return []
        if isinstance(session, dict):
            payload["session"] = session
    receipts = session.get("receipts") if isinstance(session, dict) else None
    return [row for row in receipts if isinstance(row, dict)] if isinstance(receipts, list) else []


def export_swe_predictions(root: Path, rows: list[dict], model: str) -> None:
    official = [row for row in rows if row["suite"] == "swe-bench-lite"]
    for arm in sorted({row["arm"] for row in official}):
        predictions = []
        for row in official:
            if row["arm"] != arm:
                continue
            patch = root / row["id"] / arm / "final.patch"
            predictions.append({"instance_id": row["id"], "model_name_or_path": f"mcp05-{model}-{arm}",
                                "model_patch": patch.read_text(encoding="utf-8") if patch.exists() else ""})
        path = root / f"predictions-{arm}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")


def episode(args, case: dict, arm: str, parent: Path) -> dict:
    import re
    started = time.monotonic()
    memory, controller = ARMS[arm]
    parent.mkdir(parents=True)
    root = parent / "worktree"
    verifier = case.get("verifier")
    workspace = None
    if verifier:
        clone(Path(case["repo"]), root)
        workspace = RepairWorkspace(WorkingSet(root), verifier)
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
           "--python", args.python, "--goblin-cwd", str(args.goblin_cwd),
           "--session-file", str(parent / "session.json")]
    if arm == "A":
        cmd.append("--no-packet")
    if args.json_actions:
        cmd.append("--json-actions")
    if args.deduplicate_reads and memory == "typed":
        cmd.append("--deduplicate-reads")
    if controller:
        cmd += ["--controller-checkpoint", str(args.checkpoints / f"{controller}.pt")]
    if arm == "E":
        cmd.append("--controller-reset")
    if verifier:
        cmd += ["--verifier", str(parent / "verifier.json")]
    env = dict(os.environ)
    env.pop("MCP05_EPISODE_PREFIX_PATH", None)
    if args.provider_prefix_id:
        prefix_path = parent.with_name(f"{parent.name}.provider-prefix.json")
        env["MCP05_EPISODE_PREFIX_PATH"] = str(prefix_path)
    try:
        code, stdout, stderr = run_gdk(cmd, args.episode_timeout, env=env)
        (parent / "stderr.log").write_text(stderr, encoding="utf-8")
        payload = json.loads(stdout) if code == 0 else {"error": stderr[-2500:], "exit_code": code}
    except (subprocess.TimeoutExpired, ValueError) as exc:
        payload = {"error": str(exc)}
    fallback_usage(parent, payload)
    completed_tools = [tool for tool in recover_session(parent, payload) if tool.get("error") is False]
    dump(parent / "gdk.json", payload)
    state = payload.get("state") or {}
    after = None
    patch = workspace.patch() if workspace else {}
    approved = patch.get("approved") and state.get("patch_sha256") in (None, patch["patch_sha256"])
    if approved:
        after = native(verifier, root)
        after_patch = workspace.patch()
        approved = after_patch["approved"] and after_patch["patch_sha256"] == patch["patch_sha256"]
        if approved:
            (parent / "final.patch").write_bytes(patch["patch"].encode())
    preflight = payload.get("preflight", [])
    parity_failures = sum(item.get("provider_preflight_parity") is False for item in preflight)
    seen = set(state.get("read_history") or [])
    if "read_history" not in state:
        seen.update(tool["output"]["uri"] for tool in completed_tools if tool.get("tool") == "context_read"
                    and isinstance(tool.get("output"), dict) and isinstance(tool["output"].get("uri"), str))
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
                               and approved),
              "before": before, "independent_verifier": after, "patch_sha256": patch.get("patch_sha256"),
              "input_tokens": payload.get("provider_input_tokens", 0),
              "output_tokens": payload.get("provider_output_tokens", 0),
              "inference_calls": payload.get("inference_calls", 0),
              "replayed_inference_calls": sum(item.get("provider_replayed") is True for item in preflight),
              "context_overflows": sum("context overflow:" in item.get("error", "") for item in preflight),
              "repeated_reads": state.get("repeated_reads", 0),
              "invalidated_handles": state.get("invalidated_handles", 0),
              "verifier_calls": state.get("verifier_calls", sum(tool.get("tool") == "run_verifier" for tool in completed_tools)),
              "final_runtime_metrics": "controller_seconds" in state and "search_postings_ops" in state,
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
    parser.add_argument("--goblin-cwd", type=Path, default=ROOT, help="Goose service source checkout")
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
    parser.add_argument("--gpu-yield-file", type=Path)
    parser.add_argument("--provider-prefix-id", help="Pin the provider revision and generation settings for optional episode resumption")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--json-actions", action="store_true")
    parser.add_argument("--deduplicate-reads", action="store_true")
    args = parser.parse_args()
    if not args.stage and not args.cases:
        parser.error("supply a frozen platformer stage or external case manifest")
    if any(ARMS[arm][1] for arm in args.arms) and not args.checkpoints:
        parser.error("learned arms require --checkpoints")
    args.binary = str(Path(args.binary).resolve(strict=True))
    args.goblin_cwd = args.goblin_cwd.resolve(strict=True)
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
    if args.goblin_cwd != ROOT:
        plan["goblin_cwd"] = str(args.goblin_cwd)
    if args.deduplicate_reads:
        plan["deduplicate_reads"] = True
    if args.provider_prefix_id:
        plan["provider_prefix_id"] = args.provider_prefix_id
    plan_file = args.out / "plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text(encoding="utf-8")) != plan:
        parser.error("output directory contains a different evaluation plan")
    dump(plan_file, plan)
    yield_file = args.gpu_yield_file
    rows = []
    for case in cases:
        for arm in args.arms:
            if Path(case["id"]).name != case["id"]:
                raise ValueError("case IDs must be single directory names")
            parent = args.out / case["id"] / arm
            if (parent / "result.json").exists():
                result = json.loads((parent / "result.json").read_text(encoding="utf-8"))
            else:
                if yield_file and yield_file.exists():
                    print("GPU yielded before the next episode", file=sys.stderr)
                    return
                if parent.exists():
                    parent.rename(parent.with_name(f"{arm}-interrupted-{time.time_ns()}"))
                result = episode(args, case, arm, parent)
                if yield_file and yield_file.exists():
                    archive = args.out / "interrupted" / "gpu-yield"
                    if not parent.resolve().is_relative_to(args.out) or not archive.resolve().is_relative_to(args.out):
                        raise ValueError("GPU yield attempt and archive must stay in the campaign directory")
                    archive.mkdir(parents=True, exist_ok=True)
                    parent.rename(archive / f"{case['id']}-{arm}-{time.time_ns()}")
                    print("GPU yielded during the episode; saved the unscored attempt", file=sys.stderr)
                    return
            rows.append(result)
            print(json.dumps({key: result[key] for key in ("id", "arm", "verified", "input_tokens", "output_tokens",
                                                           "parity_failures", "rubric_pass", "error")}), flush=True)
            dump(args.out / "summary.json", {"plan": plan, "completed": len(rows),
                                             "assigned": len(cases) * len(args.arms), "summary": summarize(rows)})
            export_swe_predictions(args.out, rows, args.model)


if __name__ == "__main__":
    main()
