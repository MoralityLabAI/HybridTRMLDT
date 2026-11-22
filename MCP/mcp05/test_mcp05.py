"""Protocol and deterministic provider integration checks for MCP-05."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from MCP.mcp03.goblin_server import GoblinService
from MCP.serve import Session
from .context import WorkingSet
from .evaluate import episode
from .gym import ROOT, clone, dump, verifier_manifest
from .preflight import prepare

TASK = "The player falls upward. Correct vertical gravity in physics."
SOURCE = "def fall(speed, gravity):\n    return speed - gravity\n"


def commit_fixture(root: Path) -> None:
    (root / "engine").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "engine/physics.py").write_bytes(SOURCE.encode())
    (root / "tests/test_fall.py").write_bytes(b"""
import unittest
from engine.physics import fall

class PhysicsTest(unittest.TestCase):
    def test_gravity(self):
        self.assertEqual(fall(1., 2.), 3.)
""")
    for cmd in (["init", "-q"], ["add", "."], ["-c", "user.email=ci@example.invalid", "-c", "user.name=MCP",
                                              "commit", "-qm", "frozen regression"]):
        subprocess.run(["git", *cmd], cwd=root, check=True, capture_output=True)


def template(request: dict) -> str:
    return json.dumps({"messages": request["messages"], "tools": request.get("tools", [])}, sort_keys=True)


class Provider(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template":
            return self.reply({"prompt": template(request)})
        if self.path == "/tokenize":
            return self.reply({"tokens": list(range(len(request["content"]) // 4))})
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        history = request["messages"]
        tools = [m for m in history if m["role"] == "tool"]
        index = len(tools)
        research = request["model"] == "stub-research"
        if index == 0:
            call = ("context_query", {"query": "fall gravity"})
        elif index == 1:
            call = ("context_read", {"uri": "engine/physics.py::fall"})
        elif research:
            call = (None if "Tool use has ended." in json.dumps(history) else
                    ("context_query", {"query": "fall gravity"}))
        elif index == 2:
            handle = json.loads(tools[1]["content"])["handle"]
            call = ("apply_edit", {"handle": handle, "old": "return speed - gravity",
                                   "new": "return speed + gravity"})
        elif index == 3:
            handle = json.loads(tools[1]["content"])["handle"]
            call = ("context_read", {"handle": handle})
        elif index == 4:
            call = ("run_verifier", {})
        else:
            call = None
        answer = "The return subtracts gravity in engine/physics.py." if research else "Native regression passed."
        message = {"role": "assistant", "content": answer if call is None else None}
        reason = "stop"
        if call:
            name, arguments = call
            if request.get("tools"):
                message["tool_calls"] = [{"id": f"trace_{index}", "type": "function",
                                          "function": {"name": name, "arguments": json.dumps(arguments)}}]
                reason = "tool_calls"
            else:
                action = json.dumps({"name": name, "args": {"arguments": arguments}})[:-1]
                message["content"] = (f"<|message_model|><|content_invoke_tool_json|>{action}<|end_message|>")
        self.reply({"id": f"turn_{index}", "object": "chat.completion", "model": "stub",
                    "choices": [{"index": 0, "finish_reason": reason, "message": message}],
                    "usage": {"prompt_tokens": len(template(request)) // 4,
                              "completion_tokens": 23, "total_tokens": len(template(request)) // 4 + 23}})

    def reply(self, value):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Contracts(unittest.TestCase):
    def test_episode_grades_authorized_patch_after_provider_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "source"
            source.mkdir()
            commit_fixture(source)
            args = SimpleNamespace(
                binary="stub", base_url="http://localhost", model="stub", packet_budget=512,
                completion=128, calls=10, iterations=4, context=1024, request_timeout=1,
                episode_timeout=1, python=sys.executable, json_actions=False,
            )
            case = {"id": "fixture", "suite": "test", "repo": source, "task": TASK,
                    "verifier": verifier_manifest()}
            for filename, expected in (("engine/physics.py", True), ("tests/test_fall.py", False)):
                with self.subTest(filename=filename):
                    def timeout(cmd, limit):
                        root = Path(cmd[cmd.index("--repo") + 1])
                        path = root / filename
                        replacement = SOURCE.replace("-", "+") if expected else "# out of scope\n"
                        path.write_text(replacement, encoding="utf-8")
                        raise subprocess.TimeoutExpired(cmd[0], limit)

                    parent = base / filename.split("/")[0]
                    with patch("MCP.mcp05.evaluate.run_gdk", side_effect=timeout):
                        outcome = episode(args, case, "A", parent)
                    self.assertIn("timed out", outcome["error"])
                    self.assertEqual(outcome["verified"], expected)
                    self.assertEqual((parent / "final.patch").is_file(), expected)
                    if expected:
                        self.assertEqual(outcome["independent_verifier"]["exit_code"], 0)
                    else:
                        self.assertIsNone(outcome["independent_verifier"])

    def test_json_sections_use_current_exact_source_lines(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "world.json"
            source = json.dumps({
                "nodes": [{"id": "accord_end"}],
                "rules": {"outcomes": {"victory": "accord_end"}, "event/name~alias": {"enabled": True}},
            }, indent=2)
            path.write_bytes(source.replace("\n", "\r\n").encode())
            working = WorkingSet(root)
            uri = "world.json::/rules/outcomes"
            found = working.query("rules outcomes victory")["candidates"]
            self.assertIn(uri, {row["uri"] for row in found})
            expanded = working.expand("s:world.json")["children"]
            self.assertIn("world.json::/rules/event~1name~0alias", {row["uri"] for row in expanded})
            current = working.read(uri=uri)
            self.assertEqual(current["source"], '    "outcomes": {\n      "victory": "accord_end"\n    },')
            self.assertEqual(current["source"], "\n".join(source.splitlines()[current["start"] - 1:current["end"]]))
            self.assertEqual(working.read(uri="world.json")["start"], 1)
            path.write_bytes(b'{"rules":')
            with self.assertRaisesRegex(ValueError, "stale"):
                working.read(handle=current["handle"])
            self.assertEqual(working.read(uri="world.json")["source"], '{"rules":')
            self.assertNotIn(uri, {row["uri"] for row in working.query("rules outcomes")["candidates"]})

    def test_exact_compaction_preserves_recent_tool_pair(self):
        def forward(_, path, raw):
            request = json.loads(raw)
            response = ({"prompt": template(request)} if path == "/apply-template" else
                        {"tokens": list(range(len(request["content"]) // 4))})
            return 200, json.dumps(response).encode()

        def pair(index, body):
            return [{"role": "assistant", "tool_calls": [{"id": str(index),
                     "function": {"name": "context_read", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": str(index), "content": body}]

        original = {"role": "user", "content": "Fix gravity and preserve all existing behavior."}
        latest = pair(3, '{"source":"current source"}')
        raw = {"messages": [original, *pair(1, "old text " * 1000), *pair(2, "old result " * 1000), *latest]}
        forwarded, receipt = prepare(forward, raw, window=850, reserve=128, memory="typed", budget=500)
        self.assertTrue(receipt["compacted"])
        self.assertTrue(receipt["preflight"]["fits"])
        self.assertEqual(forwarded["messages"][0], original)
        self.assertEqual(forwarded["messages"][-2:], latest)

    def test_mcp_protocol_and_source_revision(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            commit_fixture(root)
            service = Session(GoblinService())

            def request(method, params, i=1):
                return service.dispatch({"jsonrpc": "2.0", "id": i, "method": method, "params": params})

            init = request("initialize", {"capabilities": {}, "protocolVersion": "2025-06-18",
                                          "clientInfo": {"name": "test", "version": "1"}})
            self.assertEqual(init["result"]["protocolVersion"], "2025-06-18")
            service.dispatch({"jsonrpc": "2.0", "method": "notifications/initialized"})
            names = {tool["name"] for tool in request("tools/list", {})["result"]["tools"]}
            self.assertIn("compact_context", names)
            self.assertTrue({"context.query", "context.expand", "context.read", "context.state"} <= names)

            def call(name, **args):
                return request("tools/call", {"name": name,
                               "arguments": {"repo_path": str(root), **args}})["result"]

            first = call("context.read", uri="engine/physics.py::fall")["structuredContent"]
            recovered = call("context.read", handle=first["handle"])["structuredContent"]
            self.assertEqual(first, recovered)
            call("context.expand", node_id="d:", predicate="test_path", exclude=["d:tests", "d:engine"])
            certificate = call("context.state")["structuredContent"]["exclusions"][0]
            self.assertEqual(certificate["excluded"], ["d:tests"])
            self.assertEqual(certificate["rejected"], ["d:engine"])
            (root / "engine/physics.py").write_bytes(SOURCE.replace("-", "+").encode())
            self.assertTrue(call("context.read", handle=first["handle"])["isError"])
            second = call("context.read", uri=first["uri"])["structuredContent"]
            self.assertNotEqual(first["source_digest"], second["source_digest"])
            self.assertNotEqual(first["revision"], second["revision"])
            self.assertEqual(call("context.state")["structuredContent"]["invalidated_handles"], 1)
            self.assertEqual(call("context.state")["structuredContent"]["exclusions"], [])

    def test_deterministic_gdk_replay(self):
        binary = os.environ.get("GOBLIN_GDK_BINARY")
        if not binary:
            self.skipTest("GOBLIN_GDK_BINARY is not set")
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            stage = base / "fixture"
            commit_fixture(stage)
            dump(base / "verifier.json", verifier_manifest())
            server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            receipts = []
            try:
                for variant in ("native", "marker", "research"):
                    root = base / variant
                    clone(stage, root)
                    research = variant == "research"
                    task = "Which expression causes upward fall? Cite the source." if research else TASK
                    action = [str(Path(binary).resolve()), "--task", task, "--repo", str(root), "--no-packet",
                              "--model", "stub-research" if research else "stub",
                              "--python", sys.executable, "--goblin-cwd", str(ROOT),
                              "--base-url", f"http://127.0.0.1:{server.server_port}/v1/",
                              "--max-agent-iterations", "8", "--memory-mode", "typed",
                              "--max-tool-calls", "10"]
                    if not research:
                        action.extend(["--verifier", str(base / "verifier.json")])
                    if variant != "native":
                        action.append("--json-actions")
                    result = subprocess.run(action, capture_output=True, text=True, encoding="utf-8",
                                            timeout=120, check=True)
                    observations = json.loads(result.stdout)
                    self.assertIsNone(observations["error"], observations)
                    if research:
                        self.assertEqual(observations["inference_calls"], 4)
                        self.assertTrue(observations["session"]["answer_only"])
                        self.assertEqual(observations["state"]["read_history"], ["engine/physics.py::fall"])
                        self.assertTrue(observations["session"]["receipts"][-1]["error"])
                        self.assertEqual(observations["answer"], "The return subtracts gravity in engine/physics.py.")
                        continue
                    self.assertTrue(observations["state"]["verified"], observations)
                    self.assertEqual(observations["state"]["invalidated_handles"], 1)
                    self.assertEqual(observations["state"]["verifier_calls"], 1)
                    self.assertTrue(all(x["provider_preflight_parity"] for x in observations["preflight"]))
                    self.assertEqual(observations["provider_output_tokens"], 115)
                    observed = observations["session"]["receipts"]
                    self.assertTrue(observed[3]["error"])
                    receipts.append([(row["tool"], row["error"]) for row in observed])
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
            self.assertEqual(receipts[0], receipts[1])


if __name__ == "__main__":
    unittest.main()
