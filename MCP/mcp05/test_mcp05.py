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
from .context import WorkingSet, sha
from .evaluate import episode, main as evaluate
from .gym import ROOT, clone, dump, verifier_manifest
from .preflight import compact_create_history, deduplicate_read_sources, prepare
from .service import RepairWorkspace

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
        if float(self.headers.get("X-Inkling-Timeout-Seconds", 0)) <= 0:
            self.send_error(400, "Missing upstream request deadline")
            return
        history = request["messages"]
        tools = [m for m in history if m["role"] == "tool"]
        index = len(tools)
        research = request["model"].startswith("stub-research")
        if request["model"] == "stub-research-expand":
            uri = "engine/physics.py::fall"
            lookup = ("context_expand", {"node_id": f"s:{uri}", "limit": 8})
            trace = [
                ("context_query", {"query": "fall gravity"}),
                lookup,
                ("context_read", {"uri": uri}),
                ("context_expand", {**lookup[1], "cursor": 1}),
                lookup,
            ]
            call = None if "Tool use has ended." in json.dumps(history) else trace[min(index, len(trace) - 1)]
        elif index == 0:
            call = ("context_query", {"query": "fall gravity"})
        elif index == 1:
            call = ("context_read", {"uri": "engine/physics.py::fall"})
        elif request["model"] == "stub-research-duplicates":
            if index == 2:
                call = ("context_read", {"uri": "engine/physics.py::fall"})
            else:
                earlier, later = (json.loads(tool["content"]) for tool in tools[1:])
                if (not earlier.get("source_in_later_result") or "source" in earlier
                        or earlier["handle"] != later["handle"] or "return speed - gravity" not in later["source"]):
                    self.send_error(400, "Repeated source history was not reduced correctly")
                    return
                call = None
        elif request["model"] == "stub-research-repeat-read":
            if index in (2, 3):
                call = ("context_read", {"uri": "engine/physics.py::fall"})
            else:
                rendered = json.dumps(history)
                if (request.get("tools") or "Tool use has ended." not in rendered
                        or "Which expression causes upward fall?" not in rendered
                        or "return speed - gravity" not in rendered):
                    self.send_error(400, "Expected an answer turn after three identical reads")
                    return
                call = None
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
    def test_create_prompt_compaction_preserves_receipts_and_pending_source(self):
        content = "å" * 200
        def action(id_):
            return {"id": id_, "function": {"name": "create_file", "arguments": json.dumps({
                "path": "fixture.js", "content": content})}}

        receipt = {"role": "tool", "tool_call_id": "done", "content": '{"path":"fixture.js","after":"digest"}'}
        messages = [{"role": "assistant", "tool_calls": [action("done")]}, receipt,
                    {"role": "assistant", "tool_calls": [action("pending")]}]
        compact, removed = compact_create_history(messages)
        self.assertEqual(removed, 1)
        self.assertIn(sha(content.encode()), compact[0]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(compact[1:], messages[1:])
        self.assertEqual(json.loads(messages[0]["tool_calls"][0]["function"]["arguments"])["content"], content)

    def test_rejected_proposal_leaves_source_and_git_index_clean(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            commit_fixture(root)
            manifest = {"argv": [sys.executable, "-V"], "editable": ["new.js"], "creatable": ["new.js"],
                        "validate_create_argv": [sys.executable, "-c", "import sys; "
                                                 "assert sys.stdin.buffer.read() == b'valid', 'bad source'"]}
            workspace = RepairWorkspace(WorkingSet(root), manifest)
            with self.assertRaisesRegex(ValueError, "bad source"):
                workspace.create("new.js", "invalid")
            self.assertEqual(workspace.git("status", "--porcelain"), "")
            self.assertFalse((root / "new.js").exists())
            workspace.create("new.js", "valid")
            self.assertEqual((root / "new.js").read_bytes(), b"valid")

    def test_new_source_is_exact_readable_and_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            commit_fixture(root)
            working = WorkingSet(root)
            manifest = {"argv": [sys.executable, "-c", "from pathlib import Path; "
                                 "assert Path('new.js').read_bytes().endswith(b'2\\n')"],
                        "editable": ["new.js"], "creatable": ["new.js"]}
            workspace = RepairWorkspace(working, manifest)
            text = "// Crème\nlet value = 1\n"
            with self.assertRaises(ValueError):
                workspace.create("../new.js", text)
            with self.assertRaises(ValueError):
                workspace.create("engine/physics.py", text)
            created = workspace.create("new.js", text)
            self.assertEqual(created["after"], sha(text.encode("utf-8")))
            self.assertEqual((root / "new.js").read_bytes(), text.encode("utf-8"))
            with self.assertRaises(FileExistsError):
                workspace.create("new.js", "replacement")
            self.assertFalse(workspace.verify()["passed"])
            source = working.read(uri="new.js")
            workspace.apply(source["handle"], "value = 1", "value = 2")
            self.assertTrue(workspace.verify()["passed"])
            self.assertEqual(workspace.state()["changed_paths"], ["new.js"])

    def test_gpu_yield_preserves_attempt_but_does_not_score_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = root / "cases.json"
            out = root / "campaign"
            marker = root / "gpu-yield"
            dump(cases, [{"id": "fixture"}])
            argv = ["evaluate", "--cases", str(cases), "--out", str(out),
                    "--binary", sys.executable, "--arms", "A", "--gpu-yield-file", str(marker)]

            def interrupted(_, case, arm, parent):
                parent.mkdir(parents=True)
                dump(parent / "result.json", {"id": case["id"], "arm": arm})
                marker.touch()
                return {"id": case["id"], "arm": arm}

            with patch.object(sys, "argv", argv), \
                    patch("MCP.mcp05.evaluate.episode", side_effect=interrupted):
                evaluate()
            self.assertTrue((out / "plan.json").exists())
            self.assertFalse((out / "summary.json").exists())
            self.assertFalse((out / "fixture" / "A" / "result.json").exists())
            archived, = (out / "interrupted" / "gpu-yield").glob("fixture-A-*/result.json")
            self.assertEqual(json.loads(archived.read_text(encoding="utf-8")), {"id": "fixture", "arm": "A"})

    def test_episode_grades_authorized_patch_after_provider_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "source"
            source.mkdir()
            commit_fixture(source)
            args = SimpleNamespace(
                binary="stub", base_url="http://localhost", model="stub", packet_budget=512,
                completion=128, calls=10, iterations=4, context=1024, request_timeout=1,
                episode_timeout=1, python=sys.executable, json_actions=False, goblin_cwd=source,
                deduplicate_reads=False, provider_prefix_id=None,
            )
            case = {"id": "fixture", "suite": "test", "repo": source, "task": TASK,
                    "verifier": verifier_manifest(), "gold_symbols": ["engine/physics.py::fall"]}
            for filename, expected in (("engine/physics.py", True), ("tests/test_fall.py", False)):
                with self.subTest(filename=filename):
                    def timeout(cmd, limit, *, env):
                        root = Path(cmd[cmd.index("--repo") + 1])
                        self.assertEqual(Path(cmd[cmd.index("--goblin-cwd") + 1]), source)
                        path = root / filename
                        replacement = SOURCE.replace("-", "+") if expected else "# out of scope\n"
                        path.write_text(replacement, encoding="utf-8")
                        dump(root.parent / "session.json", {"session": {"receipts": [
                            {"tool": "context_read", "error": False, "output": {"uri": "engine/physics.py::fall"}},
                            {"tool": "run_verifier", "error": True, "output": {}},
                        ]}})
                        raise subprocess.TimeoutExpired(cmd[0], limit)

                    parent = base / filename.split("/")[0]
                    with patch("MCP.mcp05.evaluate.run_gdk", side_effect=timeout):
                        outcome = episode(args, case, "A", parent)
                    self.assertIn("timed out", outcome["error"])
                    self.assertEqual(outcome["verified"], expected)
                    self.assertEqual((parent / "final.patch").is_file(), expected)
                    self.assertEqual(outcome["gold_symbol_recall"], 1)
                    self.assertEqual(outcome["verifier_calls"], 0)
                    self.assertFalse(outcome["final_runtime_metrics"])
                    if expected:
                        self.assertEqual(outcome["independent_verifier"]["exit_code"], 0)
                    else:
                        self.assertIsNone(outcome["independent_verifier"])

    def test_interrupted_episode_reuses_only_its_successful_provider_prefix(self):
        binary = os.environ.get("GOBLIN_GDK_BINARY")
        if not binary:
            self.skipTest("GOBLIN_GDK_BINARY is not set")

        class InterruptAfterReads(Provider):
            def do_POST(self):
                if self.path == "/v1/chat/completions":
                    if self.server.interrupt and self.server.completed == 2:
                        self.send_response(504)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(b'{"error":{"message":"GPU yielded"}}')
                        return
                    self.server.completed += 1
                super().do_POST()

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            commit_fixture(source)
            case = {"id": "fixture", "suite": "test", "repo": source, "task": TASK,
                    "verifier": verifier_manifest(), "gold_symbols": ["engine/physics.py::fall"]}
            parent = root / "campaign" / case["id"] / "A"
            journal = parent.with_name("A.provider-prefix.json")
            with ThreadingHTTPServer(("127.0.0.1", 0), InterruptAfterReads) as server:
                server.interrupt = True
                server.completed = 0
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                args = SimpleNamespace(
                    binary=str(Path(binary).resolve()), base_url=f"http://127.0.0.1:{server.server_port}/v1/",
                    model="stub", packet_budget=1024, completion=256, calls=10, iterations=8, context=16_384,
                    request_timeout=10, episode_timeout=90, python=sys.executable, json_actions=False,
                    goblin_cwd=ROOT, deduplicate_reads=False, provider_prefix_id="stub-v1",
                )
                try:
                    interrupted = episode(args, case, "A", parent)
                    self.assertFalse(interrupted["verified"])
                    self.assertEqual(interrupted["inference_calls"], 2)
                    self.assertEqual(len(json.loads(journal.read_bytes())), 2)
                    parent.rename(parent.with_name("interrupted"))

                    server.interrupt = False
                    resumed = episode(args, case, "A", parent)
                    self.assertTrue(resumed["verified"], resumed)
                    self.assertEqual(resumed["replayed_inference_calls"], 2)
                    self.assertEqual(resumed["parity_failures"], 0)
                    self.assertEqual(server.completed, resumed["inference_calls"])
                    self.assertTrue(all(x.get("original_provider_seconds", -1) >= 0
                                        for x in resumed["provider_preflight"][:2]))
                    self.assertTrue((parent / "final.patch").exists())
                    records = json.loads(journal.read_bytes())
                    ids = [record["response"]["id"] for record in records]
                    self.assertEqual(len(set(ids)), len(ids))

                    parent.rename(parent.with_name("resumed"))
                    changed = episode(args, dict(case, task=TASK + " Preserve the existing signature."), "A", parent)
                    self.assertTrue(changed["verified"])
                    self.assertEqual(changed["replayed_inference_calls"], 0)
                    self.assertNotEqual(json.loads(journal.read_bytes())[0]["request_key"], records[0]["request_key"])
                finally:
                    server.shutdown()
                    thread.join()

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

    def test_deduplication_preserves_source_reads_from_distinct_revisions(self):
        body = {"handle": "source:preview", "source_digest": "same-digest", "source": "return speed"}
        messages = [{"role": "tool", "tool_call_id": revision, "content": json.dumps(dict(body, revision=revision))}
                    for revision in ("before", "after")]
        self.assertEqual(deduplicate_read_sources(messages), (messages, 0))

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
                for variant in ("native", "marker", "research", "research-expand", "research-duplicates",
                                "research-repeat-read"):
                    root = base / variant
                    clone(stage, root)
                    research = variant.startswith("research")
                    task = "Which expression causes upward fall? Cite the source." if research else TASK
                    action = [str(Path(binary).resolve()), "--task", task, "--repo", str(root), "--no-packet",
                              "--model", f"stub-{variant}" if research else "stub",
                              "--python", sys.executable, "--goblin-cwd", str(ROOT),
                              "--base-url", f"http://127.0.0.1:{server.server_port}/v1/",
                              "--max-agent-iterations", "8", "--memory-mode", "typed",
                              "--max-tool-calls", "10"]
                    if not research:
                        action.extend(["--verifier", str(base / "verifier.json")])
                    if variant in ("marker", "research"):
                        action.append("--json-actions")
                    if variant == "research-duplicates":
                        docstring = "Read this expression to check the vertical gravity behavior. " * 8
                        (root / "engine/physics.py").write_text(
                            SOURCE.replace("    return", f'    """{docstring}"""\n    return'), encoding="utf-8",
                        )
                        action.append("--deduplicate-reads")
                    result = subprocess.run(action, capture_output=True, text=True, encoding="utf-8",
                                            timeout=120, check=True)
                    observations = json.loads(result.stdout)
                    self.assertIsNone(observations["error"], observations)
                    if research:
                        expected_calls = {"research-expand": 6, "research-repeat-read": 5}
                        self.assertEqual(observations["inference_calls"], expected_calls.get(variant, 4))
                        self.assertEqual(observations["answer"], "The return subtracts gravity in engine/physics.py.")
                        if variant == "research-duplicates":
                            last_request = observations["preflight"][-1]
                            self.assertTrue(last_request["provider_preflight_parity"])
                            self.assertEqual(last_request["duplicate_sources_removed"], 1)
                            self.assertLess(last_request["provider_prompt_tokens"], last_request["raw_prompt_tokens"])
                            source_receipts = [row["output"]["source"] for row in observations["session"]["receipts"]
                                               if row["tool"] == "context_read"]
                            self.assertEqual(len(source_receipts), 2)
                            self.assertEqual(source_receipts[0], source_receipts[1])
                            continue
                        self.assertTrue(observations["session"]["answer_only"])
                        if variant == "research-repeat-read":
                            reads = [row for row in observations["session"]["receipts"]
                                     if row["tool"] == "context_read" and not row["error"]]
                            self.assertEqual(len(reads), 3)
                            self.assertEqual(observations["state"]["read_history"], ["engine/physics.py::fall"] * 3)
                            self.assertEqual(len({row["output"]["handle"] for row in reads}), 1)
                            self.assertIn("unchanged", reads[-1]["output"]["next_action_hint"])
                            continue
                        self.assertEqual(observations["state"]["read_history"], ["engine/physics.py::fall"])
                        self.assertTrue(observations["session"]["receipts"][-1]["error"])
                        if variant == "research-expand":
                            errors = [row["output"]["error"] for row in observations["session"]["receipts"]
                                      if row["error"]]
                            self.assertEqual(len(errors), 3)
                            self.assertIn("unknown", errors[0])
                            self.assertIn("unknown", errors[1])
                            self.assertIn("already unavailable", errors[2])
                        continue
                    self.assertTrue(observations["state"]["verified"], observations)
                    self.assertEqual(observations["state"]["invalidated_handles"], 1)
                    self.assertEqual(observations["state"]["verifier_calls"], 1)
                    self.assertTrue(all(x["provider_preflight_parity"] for x in observations["preflight"]))
                    for call in observations["preflight"]:
                        self.assertGreaterEqual(call["provider_seconds"], 0)
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
