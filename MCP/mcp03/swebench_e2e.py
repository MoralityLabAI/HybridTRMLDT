"""End-to-end SWE-bench runs through the goblin-gdk Goose GDK app, scored by the official harness.

    python MCP/mcp03/swebench_e2e.py --binary goblin-gdk --work WORK --trees TREES --out OUT \
        --base-url http://127.0.0.1:8095/v1/ --model qwen3-4b-instruct --conditions none bm25_13k goblin

For each instance and context condition the GDK app produces search/replace edits, which are applied to the
repository at the base commit and written as a git diff into ``predictions_<condition>.jsonl`` for
``swebench.harness.run_evaluation``. Conditions:

* ``none``: the issue only.
* ``bm25_13k``: SWE-bench's BM25 retrieval, whole files ranked by BM25 of the issue, packed to 13,000 tokens.
* ``goblin``: the Data Goblin packet. The app calls ``compact_context`` (24 local MCP calls, composed packet,
  4,096-token budget) and the GDK operation adds it as a prompt part.

Provider token usage, packet accounting and edit-application outcomes go to ``receipts_<condition>.json``.
"""
from __future__ import annotations

import argparse
import difflib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from MCP.mcp03.env import CodeEnv  # noqa: E402
from MCP.mcp03.repo import SnapshotStore, gold_symbols, load_instances  # noqa: E402
from MCP.mcp03.tokens import count_tokens, describe  # noqa: E402

INSTRUCTIONS = """You are fixing an issue in a Python repository. Reply only with one or more edits in exactly this format:

FILE: <path relative to the repository root>
<<<<<<< SEARCH
<existing lines to replace, copied exactly from the file>
=======
<replacement lines>
>>>>>>> REPLACE

Rules:
- The SEARCH block must be copied character for character from code shown to you, including indentation.
- Use the shortest SEARCH block that pins the location, usually 1 to 6 lines. Never write code you have not been
  shown into a SEARCH block.
- Keep the change minimal, edit library code only, and do not modify tests.
"""
EDIT = re.compile(r"FILE:\s*(?P<path>\S+)\s*\n+<{7}\s*SEARCH\s*\n(?P<search>.*?)\n={7}\s*\n(?P<replace>.*?)\n?>{7}\s*REPLACE",
                  re.S)


def export_tree(bare: Path, commit: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.mkdir(parents=True)
    data = subprocess.run(["git", "--git-dir", str(bare), "archive", commit], capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest, filter="data")


def bm25_context(env: CodeEnv, issue: str, budget: int = 13000) -> tuple[str, list[str]]:
    """SWE-bench-style BM25 context: whole files in BM25 order, skipping any that would overflow the budget."""
    files, used, parts = [], 0, []
    for path in ranked_files(env, issue):
        text = env.snapshot.sources[path]
        block = f"[start of {path}]\n{text}\n[end of {path}]\n"
        cost = count_tokens(block)
        if used + cost > budget:
            continue
        parts.append(block)
        files.append(path)
        used += cost
    return "\n".join(parts), files


def ranked_files(env: CodeEnv, issue: str) -> list[str]:
    from MCP.mcp02.textutil import BM25
    from MCP.mcp03.env import code_tokens
    paths = sorted(env.snapshot.sources)
    index = BM25([" ".join(code_tokens(env.snapshot.sources[p])) for p in paths])
    scores = index.scores(" ".join(code_tokens(issue)))
    return [p for _s, p in sorted(zip(scores, paths), key=lambda x: (-x[0], x[1]))][:50]


def edits_to_diff(answer: str, tree: Path) -> tuple[str, dict]:
    """Apply search/replace edits to the base files and return a git-style unified diff."""
    text = answer.replace("```python", "").replace("```diff", "").replace("```", "")
    edits = [m.groupdict() for m in EDIT.finditer(text)]
    stats = {"edits_parsed": len(edits), "edits_applied": 0, "edits_failed": 0}
    new: dict[str, str] = {}
    old: dict[str, str] = {}
    for e in edits:
        path = e["path"].strip().lstrip("./")
        f = tree / path
        if not f.is_file():
            stats["edits_failed"] += 1
            continue
        if path not in new:
            old[path] = f.read_text(encoding="utf-8", errors="replace")
            new[path] = old[path]
        search, replace = e["search"], e["replace"]
        if search and search in new[path]:
            new[path] = new[path].replace(search, replace, 1)
            stats["edits_applied"] += 1
            continue
        # Tolerate trailing-whitespace differences line by line.
        lines = new[path].split("\n")
        target = [l.rstrip() for l in search.split("\n")]
        hit = next((i for i in range(len(lines) - len(target) + 1)
                    if [l.rstrip() for l in lines[i:i + len(target)]] == target), None) if target else None
        if hit is None:
            stats["edits_failed"] += 1
            continue
        lines[hit:hit + len(target)] = replace.split("\n")
        new[path] = "\n".join(lines)
        stats["edits_applied"] += 1
    chunks = []
    for path in sorted(new):
        if new[path] == old[path]:
            continue
        a, b = old[path].splitlines(keepends=True), new[path].splitlines(keepends=True)
        if a and not a[-1].endswith("\n"):
            a[-1] += "\n"
        if b and not b[-1].endswith("\n"):
            b[-1] += "\n"
        body = "".join(difflib.unified_diff(a, b, fromfile=f"a/{path}", tofile=f"b/{path}"))
        chunks.append(f"diff --git a/{path} b/{path}\n{body}")
    return "".join(chunks), stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--trees", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8095/v1/")
    p.add_argument("--model", default="qwen3-4b-instruct")
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--conditions", nargs="+", default=["none", "bm25_13k", "goblin"])
    p.add_argument("--packet-budget", type=int, default=4096)
    p.add_argument("--max-tokens", type=int, default=1536)
    args = p.parse_args()
    ids = set(args.ids.read_text().split())
    rows = sorted([r for r in load_instances(args.work / "data" / "lite_test.jsonl") if r["instance_id"] in ids],
                  key=lambda r: r["instance_id"])
    store = SnapshotStore(args.work / "repos", args.work / "cache")
    args.out.mkdir(parents=True, exist_ok=True)
    receipts = {c: json.loads((args.out / f"receipts_{c}.json").read_text()) if (args.out / f"receipts_{c}.json").is_file()
                else {"condition": c, "runs": {}} for c in args.conditions}
    for inst in rows:
        iid = inst["instance_id"]
        tree = args.trees / iid
        export_tree(store.bare(inst["repo"]), inst["base_commit"], tree)
        snap = store.snapshot(inst["repo"], inst["base_commit"])
        gold = gold_symbols(snap, inst["patch"])
        env = CodeEnv(snap, {}, gold, "search")
        issue = inst["problem_statement"][:8000]
        for cond in args.conditions:
            if iid in receipts[cond]["runs"]:
                continue
            task = INSTRUCTIONS + "\nISSUE:\n" + issue
            context_files = []
            if cond == "bm25_13k":
                ctx, context_files = bm25_context(env, issue)
                task += "\n\nCODE CONTEXT (BM25 retrieval, whole files):\n" + ctx
            task_file = args.out / "tasks" / f"{iid}.{cond}.txt"
            task_file.parent.mkdir(parents=True, exist_ok=True)
            task_file.write_text(task, encoding="utf-8")
            cmd = [args.binary, "--task-file", str(task_file), "--base-url", args.base_url, "--model", args.model,
                   "--python", sys.executable, "--goblin-cwd", str(ROOT), "--budget", str(args.packet_budget),
                   "--max-tokens", str(args.max_tokens), "--timeout-secs", "1800"]
            cmd += ["--repo", str(tree)] if cond == "goblin" else ["--no-packet"]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=7200,
                                 env=dict(os.environ, MCP03_COMPOSED="1"))
            run = {"instance_id": iid, "repo": inst["repo"], "condition": cond, "context_files": context_files,
                   "task_tokens": count_tokens(task)}
            if out.returncode != 0:
                run.update(error=out.stderr[-1500:], patch="")
            else:
                receipt = json.loads(out.stdout)
                patch, stats = edits_to_diff(receipt["answer"], tree)
                run.update(stats, patch=patch, answer=receipt["answer"],
                           provider_input_tokens=receipt["provider_input_tokens"],
                           provider_output_tokens=receipt["provider_output_tokens"],
                           packet=receipt.get("packet_stats"), inference_seconds=receipt["inference_seconds"],
                           retrieval_seconds=receipt["retrieval_seconds"])
            receipts[cond]["runs"][iid] = run
            receipts[cond]["tokenizer"] = describe()
            (args.out / f"receipts_{cond}.json").write_text(json.dumps(receipts[cond], indent=1), encoding="utf-8")
            print(json.dumps({"id": iid, "cond": cond, "input": run.get("provider_input_tokens"),
                              "applied": run.get("edits_applied"), "patch_bytes": len(run.get("patch", ""))}), flush=True)
    for cond in args.conditions:
        with open(args.out / f"predictions_{cond}.jsonl", "w", encoding="utf-8") as f:
            for iid, run in sorted(receipts[cond]["runs"].items()):
                f.write(json.dumps({"instance_id": iid, "model_name_or_path": f"goblin-gdk/{args.model}/{cond}",
                                    "model_patch": run.get("patch", "")}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
