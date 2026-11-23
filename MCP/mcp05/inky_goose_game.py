"""Launch a fresh Inkling game workspace with Goose-owned tool execution."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from .context import sha
from .creative_bench import save
from .inky_web_game import MODEL, ROOT

BRIEFS = {
    "story": """Invent an original playable Canvas story with a choice and at least four distinct scenes.
Use an inline script and styles in index.html. Expose window.gameState() returning {scene,clue,ending}.
Start at scene 'start' without a clue or ending. Provide visible data-choice buttons named 'study', 'cross',
and 'rush' by attribute. Cross starts disabled. Study finds the clue and advances to an intermediate scene.
Cross then leads to ending 'good'. Rush at the start leads to ending 'bad'. Both endings must visibly
differ on the canvas. A button with id 'restart' restores the starting scene and clears progress.
You decide the title, setting, art, interface, story, and code.""",
    "rpg": """Invent an original playable Canvas action RPG. Use inline script and styles in index.html.
Include a clear objective, visible health and controls, moving enemies, combat feedback, win and loss,
and a button with id 'restart'. Implement WASD and arrow movement and Space to attack. Hold D and Space
at the start to reach and strike an enemy during browser verification. Expose window.gameState()
returning {x,y,hp,hits,kills,status}. Start with status 'playing'. Count damage dealt as hits and enemies
defeated as kills. The restart must restore the original health and hit count.
You decide the title, setting, art, interface, rules, and code.""",
}

WORKFLOW = """Use the create_file tool now. Set path to index.html. Put all of your first runnable draft
in content. Aim for under 3,500 characters of source on this turn. Focus on the playable core and one
visual motif. Use inline Canvas drawings and avoid external assets. You can refine your draft in later
tool turns. You are the sole author. Work one tool action at a time until you pass the browser checks."""


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True,
                          encoding="utf-8", timeout=20).stdout.strip()


def read_json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def provider_actions(replies: list[dict]):
    for reply in replies:
        message = reply["choices"][0]["message"]
        if message.get("content"):
            yield message["content"].strip()
        for call in message.get("tool_calls") or []:
            function = call["function"]
            yield function["name"], json.loads(function["arguments"])


def replay_writes(root: Path, paths: list[str], *, require_verified: bool = True) -> dict:
    """Replay tool writes and match their receipts to the raw model and MCP exchanges."""
    invocation = read_json(root / "invocation.json")
    verify_config = root / "verifier.json"
    brief = root / "brief.txt"
    require(sha(brief.read_bytes()) == invocation["brief_sha256"], "brief hash changed")
    if "verifier_sha256" in invocation:
        require(sha(verify_config.read_bytes()) == invocation["verifier_sha256"], "verifier hash changed")
    workspace = root / "workspace"
    require(not git(workspace, "ls-tree", "-r", invocation["base_commit"]), "base workspace was populated")
    final_report = root / "gdk.json"
    gdk = read_json(final_report if final_report.stat().st_size else root / "session.json")
    wire = root / "inky-wire.jsonl"
    mcp = root / "mcp-wire.jsonl"
    replies = [json.loads(event["body"]) for event in events(wire)
               if event["event"] == "response" and event["status"] == 200]
    model_actions = provider_actions(replies)
    frames = [json.loads(event["body"]) for event in events(mcp)]
    mcp_responses = {frame["id"]: frame for frame in frames if "result" in frame or "error" in frame}
    names = {"create_file": "host.create", "apply_edit": "host.edit", "run_verifier": "host.verify"}
    calls = iter(frame for frame in frames if frame.get("method") == "tools/call"
                 and frame["params"]["name"] in names.values())
    frame = next(calls, None)
    sources = {}
    writes, checks = [], []
    for receipt in gdk["session"]["receipts"]:
        require(receipt["origin"] == "solver", "a tool was called by an actor other than Inkling")
        raw = receipt.get("raw_assistant_action")
        proposed = raw.strip() if raw else (receipt["tool"], receipt["arguments"])
        require(any(action == proposed for action in model_actions), "solver receipt is missing its provider reply")
        name = receipt["tool"]
        if name not in names:
            continue
        args, output = receipt["arguments"], receipt["output"]
        response = None
        if frame and frame["params"] == {"name": names[name], "arguments": args}:
            received = mcp_responses[frame["id"]]
            response = received.get("result")
            frame = next(calls, None)
            require((response is None or response["isError"]) == receipt["error"], "MCP error status differs")
        if receipt["error"]:
            continue
        require(response is not None, "successful solver action has no matching MCP call")
        mcp_output = {key: value for key, value in output.items() if key != "next_action_hint"}
        actual = response["structuredContent"].copy()
        if name == "run_verifier":
            require(math.isclose(actual.pop("seconds"), mcp_output.pop("seconds"), abs_tol=1e-12, rel_tol=1e-12),
                    "MCP verifier duration differs from solver receipt")
        require(actual == mcp_output, "MCP result differs from solver receipt")
        if name == "run_verifier":
            checks.append({"action": receipt["index"], "passed": output["passed"], "seconds": output["seconds"]})
            continue
        path = output["path"]
        require(path in paths, "unexpected file write")
        source = sources.get(path)
        require(output["before"] == (sha(source.encode("utf-8")) if source is not None else None),
                "before hash does not match replay")
        if name == "create_file":
            require(source is None and args["path"] == path, "unexpected create target")
            source = args["content"]
        else:
            old, new = args["old"], args["new"]
            if "\r\n" in source:
                old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
            require(source.count(old) == 1, "replacement is not unique in replay")
            source = source.replace(old, new, 1)
        require(sha(source.encode("utf-8")) == output["after"], "after hash does not match replay")
        sources[path] = source
        writes.append({"action": receipt["index"], "path": path, "sha256": output["after"]})
    require(frame is None, "unattributed MCP operation")
    require(set(sources) == set(paths), "replay did not write each requested source file")
    for path, source in sources.items():
        require(source.encode("utf-8") == (workspace / path).read_bytes(),
                f"{path} does not match the replayed model writes")
    verified = bool(gdk.get("state", {}).get("verified") and checks and checks[-1]["passed"]
                    and checks[-1]["action"] > writes[-1]["action"])
    require(verified or not require_verified, "no solver-triggered passing verification of the final write")
    files = [brief, verify_config, wire, mcp, root / "invocation.json", final_report,
             root / "session.json", root / "session.preflight.jsonl", *(workspace / p for p in paths)]
    return {"model": gdk.get("model", MODEL), "base_commit": invocation["base_commit"],
            "gdk_sha256": invocation["gdk_sha256"], "writes": writes, "verifications": checks,
            "solver_verified": verified, "provider_calls": len(replies),
            "provider_timings": [reply.get("timings") for reply in replies], "wall_seconds": gdk.get("wall_seconds"),
            "sha256": {file.relative_to(root).as_posix(): sha(file.read_bytes()) for file in files}}


def audit(root: Path, mode: str) -> dict:
    receipt = replay_writes(root, ["index.html"])
    browser = read_json(root / "browser/browser-receipt.json")
    require(browser["game"] == mode and browser["status"] == "verified", "missing matching browser verification")
    receipt["sha256"].update({file.relative_to(root).as_posix(): sha(file.read_bytes())
                              for file in sorted((root / "browser").iterdir())})
    return {**receipt, "title": browser["title"], "game": mode, "browser_checks": browser["checks"]}


def run(root: Path, mode: str, gdk: Path, node: Path, chrome: Path, base_url: str) -> int:
    root.mkdir(parents=True)
    game = root / "workspace"
    game.mkdir()
    git(game, "init", "--quiet")
    git(game, "-c", "user.name=Goose workspace", "-c", "user.email=goose@example.invalid",
        "commit", "--quiet", "--allow-empty", "-m", "Begin with an empty workspace")
    brief = root / "brief.txt"
    brief.write_text(BRIEFS[mode] + "\n\n" + WORKFLOW + "\n", encoding="utf-8", newline="\n")
    verifier = root / "verifier.json"
    save(verifier, {"argv": [str(node), str(ROOT / "MCP/demos/browser-smoke.mjs"), str(chrome),
                            str(root / "browser"), mode, str(game)],
                    "editable": ["index.html"], "creatable": ["index.html"], "timeout_seconds": 50})
    command = [str(gdk), "--repo", str(game), "--goblin-cwd", str(ROOT), "--python", sys.executable,
               "--task-file", str(brief), "--session-file", str(root / "session.json"),
               "--base-url", base_url, "--model", MODEL, "--context-window", "100096",
               "--max-tokens", "4608", "--max-agent-iterations", "12", "--max-tool-calls", "20",
               "--budget", "4096", "--no-packet", "--memory-mode", "raw", "--json-actions",
               "--verifier", str(verifier), "--allow-create", "--timeout-secs", "840"]
    save(root / "invocation.json", {"argv": command, "base_commit": git(game, "rev-parse", "HEAD"),
                                    "base_tree": git(game, "ls-tree", "HEAD"),
                                    "verifier_sha256": sha(verifier.read_bytes()),
                                    "brief_sha256": sha(brief.read_bytes()),
                                    "gdk_sha256": sha(gdk.read_bytes())})
    env = {**os.environ, "MCP05_WIRE_LOG": str(root / "inky-wire.jsonl"),
           "MCP05_SERVER_WIRE_LOG": str(root / "mcp-wire.jsonl")}
    for key in ("MCP05_FRESH_PROMPT", "MCP05_EPISODE_PREFIX_PATH", "MCP05_FEDERATION", "MCP05_INDEX_ROUTER"):
        env.pop(key, None)
    print(f"Launching {mode} in {game}", flush=True)
    with (root / "gdk.json").open("wb") as stdout, (root / "gdk.stderr.log").open("wb") as stderr:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
    if result.returncode:
        return result.returncode
    if not read_json(root / "gdk.json")["state"].get("verified"):
        print(f"Goose ended without a passing check. Traces are in {root}.", file=sys.stderr)
        return 1
    return record_audit(root, mode)


def record_audit(root: Path, mode: str) -> int:
    receipt = audit(root, mode)
    path = root / "source-receipt.json"
    save(path, receipt)
    print(f"Verified {receipt['title']}. Replay receipt saved to {path}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=BRIEFS)
    parser.add_argument("out", type=Path)
    parser.add_argument("--audit", action="store_true", help="verify the finished rollout traces")
    parser.add_argument("--gdk", type=Path)
    parser.add_argument("--node", type=Path)
    parser.add_argument("--chrome", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:18097/v1/")
    args = parser.parse_args()
    if args.audit:
        return record_audit(args.out.resolve(), args.mode)
    if not (args.gdk and args.node and args.chrome):
        parser.error("a rollout requires --gdk, --node, and --chrome")
    return run(args.out.resolve(), args.mode, args.gdk.resolve(strict=True), args.node.resolve(strict=True),
               args.chrome.resolve(strict=True), args.base_url)


if __name__ == "__main__":
    raise SystemExit(main())
