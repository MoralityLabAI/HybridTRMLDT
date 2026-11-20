"""goblin-gdk on SWE-bench Lite instances with a real local model, with and without the packet.

    python3 MCP/goose_gdk_app/swebench_demo.py --binary PATH/goblin-gdk --work WORK_DIR --out RECEIPTS.json \
        --base-url http://127.0.0.1:8093/v1/ --model qwen2.5-0.5b-instruct

For each instance it exports the repository at the base commit from the bare clone (``git archive``),
asks the GDK app to name the file and function to edit, and scores the answer against the hidden
gold patch: a gold file path (or its basename) and a gold function or method name must appear in
the answer. The scoring never feeds back into the run. Provider token usage comes from the server.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from MCP.mcp03.repo import gold_symbols, load_instances, SnapshotStore  # noqa: E402

PROMPT = ("You are fixing a bug in a Python repository. Read the issue below. Reply in this format:\n"
          "FILE: <path of the file to edit>\nFUNCTION: <function or method to edit>\nCHANGE: <two sentences>\n\n"
          "ISSUE:\n")


def export_tree(bare: Path, commit: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.mkdir(parents=True)
    data = subprocess.run(["git", "--git-dir", str(bare), "archive", commit], capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest, filter="data")


def score(answer: str, gold: dict) -> dict:
    text = answer.lower()
    file_hit = any(f.lower() in text or Path(f).name.lower() in text for f in gold["files"])
    names = {s.split("::")[1].split(".")[-1].lower() for s in gold["symbols"] if "::" in s}
    func_hit = any(re.search(rf"\b{re.escape(n)}\b", text) for n in names) if names else None
    return {"file_named": file_hit, "function_named": func_hit}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--trees", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8093/v1/")
    p.add_argument("--model", default="qwen2.5-0.5b-instruct")
    p.add_argument("--budget", type=int, default=1536)
    p.add_argument("--repos", nargs="+", default=["psf/requests", "pallets/flask", "mwaskom/seaborn"])
    p.add_argument("--per-repo", type=int, default=4)
    args = p.parse_args()
    rows = [r for r in load_instances(args.work / "data" / "lite_test.jsonl") if r["repo"] in args.repos]
    picked = []
    for repo in args.repos:
        picked += sorted([r for r in rows if r["repo"] == repo], key=lambda r: r["instance_id"])[:args.per_repo]
    store = SnapshotStore(args.work / "repos", args.work / "cache")
    runs = []
    for inst in picked:
        bare = store.bare(inst["repo"])
        gold = gold_symbols(store.snapshot(inst["repo"], inst["base_commit"], with_sources=False), inst["patch"])
        tree = args.trees / inst["instance_id"]
        export_tree(bare, inst["base_commit"], tree)
        task_file = args.trees / f"{inst['instance_id']}.task.txt"
        task_file.write_text(PROMPT + inst["problem_statement"][:6000], encoding="utf-8")
        for use_packet in (True, False):
            cmd = [args.binary, "--task-file", str(task_file), "--base-url", args.base_url, "--model", args.model,
                   "--python", sys.executable, "--goblin-cwd", str(ROOT), "--budget", str(args.budget)]
            cmd += ["--repo", str(tree)] if use_packet else ["--no-packet"]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if out.returncode != 0:
                runs.append({"instance_id": inst["instance_id"], "used_packet": use_packet, "error": out.stderr[-800:]})
                continue
            receipt = json.loads(out.stdout)
            stats = receipt.get("packet_stats") or {}
            runs.append({"instance_id": inst["instance_id"], "repo": inst["repo"], "used_packet": use_packet,
                         "label": f"{inst['instance_id']} {'packet' if use_packet else 'no packet'}",
                         "packet_tokens": stats.get("packet_tokens"), "local_result_tokens": stats.get("local_result_tokens"),
                         "router_tool_calls": stats.get("tool_calls"),
                         "provider_input_tokens": receipt["provider_input_tokens"],
                         "provider_output_tokens": receipt["provider_output_tokens"],
                         "retrieval_seconds": receipt["retrieval_seconds"], "inference_seconds": receipt["inference_seconds"],
                         "answer": receipt["answer"], "score": score(receipt["answer"], gold), "gold": gold})
            print(json.dumps({k: runs[-1].get(k) for k in ("label", "provider_input_tokens", "score")}), flush=True)
            args.out.write_text(json.dumps({"runs": runs}, indent=1), encoding="utf-8")
    summary = {}
    for flag in (True, False):
        sel = [r for r in runs if r.get("used_packet") is flag and "score" in r]
        summary["packet" if flag else "no_packet"] = {
            "runs": len(sel), "file_named": sum(r["score"]["file_named"] for r in sel),
            "function_named": sum(bool(r["score"]["function_named"]) for r in sel),
            "provider_input_tokens_mean": sum(r["provider_input_tokens"] for r in sel) / max(1, len(sel)),
            "provider_output_tokens_mean": sum(r["provider_output_tokens"] for r in sel) / max(1, len(sel))}
    args.out.write_text(json.dumps({"model": args.model, "budget": args.budget, "summary": summary, "runs": runs},
                                   indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
