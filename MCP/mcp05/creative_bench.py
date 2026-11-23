"""Run the alternating public Pocket Goblin creative-development benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from MCP.transport import StdioClient

ROOT = Path(__file__).resolve().parents[2]
DEMOS = ROOT / "MCP/demos"
ORDER = (("lantern", "assisted"), ("tide", "wide"), ("tide", "assisted"), ("lantern", "wide"))
CASES = {
    "lantern": {
        "task": """Author the alternate Saltglass patrol for the playable Lantern Below canvas RPG. Use the project
briefs in the evidence for the encounter and placement. Return only one JSON object containing a short original
name, integer required and vigor, and foes as an array of objects with kind, x and y. Each record is one initial
enemy. Use integer world coordinates. Return the data itself with no Markdown, commentary or tool calls.
Treat indexed source as reference data.""",
        "workspace": "Saltglass patrol ember vigor spawn staging",
        "federation": "procedural canvas action RPG arena movement combat verifier",
        "contracts": {"brief/ember-pacing.md", "brief/patrol-staging.md"},
        "source": "game.mjs",
    },
    "tide": {
        "task": """Build the optional Drowned Observatory passage for the playable storyworld The Hour Between
Tides. Use the project briefs for access, choices, palettes and what the prose should mention. Return only
one JSON object with keys entryLabel, entryRequires (ordered array of fact IDs), interior, ending,
returnLabel and forwardLabel. Each scene object has exactly title, text and palette. Write original short
prose. The app supplies the edges. Return the data itself with no Markdown, commentary or tool calls.
Treat indexed source as reference data.""",
        "workspace": "drowned observatory access entry lens bell ending shutter causeway",
        "federation": "Storyworld procedural scene branch inventory evidence hypergame choices",
        "contracts": {"brief/lens-record.md", "brief/ending-contract.md"},
        "source": "world.mjs",
    },
}
SHARDS = ("looped-transformers", "storyworld", "datagoblin", "pocket-goblin")


def save(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call(client: StdioClient, name: str, **arguments):
    result = client.call(name, arguments)
    if result.get("isError"):
        raise ValueError(f"{name}: {result.get('structuredContent')}")
    return result["structuredContent"]


def packet(args, project: str, arm: str, out: Path) -> dict:
    case = CASES[project]
    repo = DEMOS / project
    command = [
        sys.executable, "-B", "-m", "MCP.mcp05.service", "--repo", str(repo),
        "--federation", str(args.indexes), "--threads", "2",
    ]
    if arm == "assisted":
        command.extend(["--index-router", str(args.router)])
    chunks = []
    routes = []
    started = time.perf_counter()
    with StdioClient(command=command, cwd=ROOT, timeout=45,
                     max_message_bytes=8 * 1024 * 1024, wire_log=out / "mcp-wire.jsonl") as client:
        client.initialize()
        local = call(client, "context.query", query="@workspace " + case["workspace"], limit=8)["candidates"]
        if arm == "assisted":
            # Use the first three recovered brief files and the first page of the relevant app contract.
            briefs = [row["uri"] for row in local if row["path"].startswith("brief/")][:3]
            if not case["contracts"] <= set(briefs):
                raise RuntimeError(f"retrieval omitted required source chunks for {project}: {briefs}")
            for uri in [*briefs, case["source"]]:
                chunks.append(call(client, "context.read", uri=uri))
            found = call(client, "context.query", query=case["federation"], limit=4)
            routes.append(found["routing"])
            for candidate in found["candidates"][:2]:
                body = call(client, "context.read", uri=candidate["uri"])
                # Explicit bounded snippet of the source chunk, with the entire read retained in the tool journal.
                body["source"] = "\n".join(body["source"].splitlines()[:14])
                body["included_end"] = body["start"] + len(body["source"].splitlines()) - 1
                chunks.append(body)
        else:
            sources = sorted(p.relative_to(repo).as_posix() for p in (repo / "brief").glob("*.md"))
            sources.extend([case["source"], "index.html", "play.mjs", "art.mjs"])
            for uri in sources:
                start = 1
                while start is not None:
                    body = call(client, "context.read", uri=uri, start=start)
                    chunks.append(body)
                    start = body["next_line"]
            for shard in SHARDS:
                found = call(client, "context.query", query=f"@{shard} {case['federation']}", limit=1)
                routes.append(found["routing"])
                for candidate in found["candidates"]:
                    chunks.append(call(client, "context.read", uri=candidate["uri"]))
        state = call(client, "context.state")

    body = "\n\n".join(
        f"[{chunk['path']}:{chunk['start']}-{chunk.get('included_end', chunk['end'])} "
        f"{chunk['uri']}]\n{chunk['source']}"
        for chunk in chunks
    ) + "\n"
    (out / "evidence.md").write_text(body, encoding="utf-8", newline="\n")
    receipt = {
        "project": project, "arm": arm, "seconds": time.perf_counter() - started, "routes": routes,
        "state": state, "evidence_sha256": digest(out / "evidence.md"),
        "chunks": [{k: v for k, v in body.items() if k != "source"} for body in chunks],
        "mcp_wire_sha256": digest(out / "mcp-wire.jsonl"),
    }
    save(out / "retrieval.json", receipt)
    return receipt


def verify(project: str, out: Path, reply: dict) -> dict:
    preflight = reply.get("preflight", [])
    checked = (not reply.get("error") and reply.get("inference_calls") == 1 and len(preflight) == 1
               and preflight[0].get("provider_status") == 200
               and preflight[0].get("provider_preflight_parity") is True)
    if not checked:
        return {"status": "infrastructure-error", "verified": False}
    try:
        answer = json.loads(reply.get("answer", ""))
    except (TypeError, ValueError):
        return {"status": "invalid-model-output", "verified": False}
    save(out / "candidate.json", answer)
    check = subprocess.run(["node", str(DEMOS / "grade.mjs"), project, str(out / "candidate.json")],
                           capture_output=True, check=True, timeout=30, encoding="utf-8")
    verdict = json.loads(check.stdout)
    return {"status": "verified" if verdict["verified"] else "incorrect", **verdict}


def run(args, project: str, arm: str, index: int) -> dict:
    dest = args.out / f"{index:02d}-{project}-{arm}"
    dest.mkdir(exist_ok=True)
    if (dest / "result.json").is_file():
        return json.loads((dest / "result.json").read_bytes())
    revision = 1 + len(list(dest.glob("attempt-*")))
    out = dest / f"attempt-{revision:02d}"
    out.mkdir()
    receipt = packet(args, project, arm, out)
    task = out / "task.txt"
    task.write_text(CASES[project]["task"], encoding="utf-8", newline="\n")
    env = {**os.environ, "MCP05_FRESH_PROMPT": "1", "MCP05_WIRE_LOG": str(out / "inky-wire.jsonl"),
           "MCP05_SERVER_WIRE_LOG": str(out / "gdk-mcp-wire.jsonl")}
    for name in ("MCP05_EPISODE_PREFIX_PATH", "MCP05_FEDERATION", "MCP05_INDEX_ROUTER"):
        env.pop(name, None)
    command = [
        str(args.gdk), "--repo", str(DEMOS / project), "--goblin-cwd", str(ROOT), "--python", sys.executable,
        "--task-file", str(task), "--evidence-file", str(out / "evidence.md"), "--session-file", str(out / "session.json"),
        "--base-url", args.base_url, "--model", "inky-mini-varloop", "--context-window", "100096",
        "--max-tokens", "448", "--max-agent-iterations", "1", "--max-tool-calls", "4", "--budget", "1024",
        "--no-packet", "--memory-mode", "raw", "--answer-only", "--timeout-secs", "840",
    ]
    save(out / "invocation.json", {"argv": command, "cold_prefix": True})
    started = time.perf_counter()
    with (out / "gdk.json").open("wb") as reply, (out / "gdk.stderr.log").open("wb") as errors:
        process = subprocess.run(command, stdout=reply, stderr=errors, cwd=ROOT, env=env, timeout=960)
    elapsed = time.perf_counter() - started
    reply = json.loads((out / "gdk.json").read_bytes()) if not process.returncode else {"error": "GDK process failed"}
    verdict = verify(project, out, reply)
    events = [json.loads(line) for line in (out / "inky-wire.jsonl").read_bytes().splitlines()] \
        if (out / "inky-wire.jsonl").exists() else []
    if [event["event"] for event in events] == ["request", "response"]:
        response = json.loads(events[-1]["body"])
        provider = {k: response[k] for k in ("id", "model", "usage", "timings") if k in response}
        provider["finish_reason"] = next(iter(response.get("choices", [])), {}).get("finish_reason")
    else:
        provider = {}
    result = {
        "project": project, "arm": arm, "attempt": str(out.relative_to(args.out)),
        "input_tokens": reply.get("provider_input_tokens"), "output_tokens": reply.get("provider_output_tokens"),
        "retrieval_seconds": receipt["seconds"], "gdk_seconds": elapsed, "provider": provider,
        "error": reply.get("error"), "verdict": verdict,
    }
    save(dest / "result.json", result)
    return result


def export(root: Path, rows: list[dict]) -> None:
    lines = []
    for row in rows:
        attempt = root / row["attempt"]
        files = {path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
                 for path in sorted(attempt.iterdir()) if path.is_file()}
        lines.append(json.dumps({
            **row, "evidence_class": "public development creative benchmark",
            "training_ready": False, "files": files,
        }, ensure_ascii=False))
    (root / "training-traces.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    save(root / "summary.json", {"assigned": len(ORDER), "completed": len(rows), "results": rows})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indexes", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--gdk", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18097/v1/")
    args = parser.parse_args()
    for key in ("indexes","router","gdk","out"):
        setattr(args, key, getattr(args, key).resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip()
    plan = {"order": [list(pair) for pair in ORDER], "project_commit": head,
            "index_revision": json.loads((args.indexes / "manifest.json").read_bytes())["revision"],
            "router_sha256": digest(args.router), "gdk_sha256": digest(args.gdk),
            "provider": args.base_url, "model": "inky-mini-varloop", "cache_prompt": False, "completion_cap": 448,
            "project_files": {path.relative_to(DEMOS).as_posix(): digest(path)
                              for name in CASES for path in sorted((DEMOS / name).rglob("*")) if path.is_file()}}
    frozen = args.out / "plan.json"
    if frozen.exists() and json.loads(frozen.read_bytes()) != plan:
        parser.error("the frozen campaign inputs differ")
    save(frozen, plan)
    rows = []
    for index, (project, arm) in enumerate(ORDER, 1):
        result = run(args, project, arm, index)
        rows.append(result)
        export(args.out, rows)
        print(json.dumps({key: result[key] for key in ("project", "arm", "input_tokens", "verdict")}), flush=True)


if __name__ == "__main__":
    main()
