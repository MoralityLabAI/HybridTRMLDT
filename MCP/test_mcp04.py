"""MCP-04 exact preflight and typed state preservation contracts (stdlib)."""
from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from MCP.hubrun.mcp04 import (PreflightError, WorkingSetError, adaptive_preflight,
                              exact_preflight, typed_packet)
from MCP.hubrun.proxy import Relay, make_handler


def episode(n: int = 7) -> list[dict]:
    out = [
        {"role": "system", "content": "Update platformer engine safely."},
        {"role": "user", "content": "Repair collision.py and preserve existing player controls."},
    ]
    for i in range(n):
        edited = i == 1
        args = {"path": "engine/collision.py", "start_line": 12, "old_str": "if x > width:",
                "new_str": "if x >= width:"} if edited else {"path": f"logs/trace_{i}.txt"}
        method = "edit_file" if edited else "read_file"
        out.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"t{i}", "type": "function",
             "function": {"name": method, "arguments": json.dumps(args)}}]})
        text = "Successful engine edit." if edited else (f"engine telemetry trace_{i}\n" + "0123456789" * 60)
        out.append({"role": "tool", "tool_call_id": f"t{i}", "content": text})
        if i == 3:
            out.append({"role": "user", "content": "Keep the existing collision invariant and save the final patch."})
    return out


class FakeBackend:
    def __init__(self):
        self.paths = []
        self.last_chat = None
        self.valid = True

    def forward(self, method: str, path: str, body: bytes):
        self.paths.append(path)
        req = json.loads(body)
        if path == "/apply-template":
            if not self.valid:
                return 404, b'{"error":"template unavailable"}'
            return 200, json.dumps({"prompt": json.dumps({
                "messages": req["messages"], "tools": req["tools"]}, sort_keys=True)}).encode()
        if path == "/tokenize":
            return 200, json.dumps({"tokens": list(req["content"].encode("utf-8"))}).encode()
        if path == "/v1/chat/completions":
            self.last_chat = req
            return 200, b'{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":1}}'
        return 404, b'{}'


class PreflightTests(unittest.TestCase):
    def test_raw_exact_template_includes_tool_schemas(self):
        backend = FakeBackend()
        request = {"messages": episode(1), "tools": [{"name": "edit_file", "parameters": {"path": "string"}}]}
        count = exact_preflight(backend.forward, request, context_window=5000,
                                completion_reserve=100, safety_margin=200)
        assert count["fits"] and count["tools_counted"] == 1 and count["prompt_tokens"] > 0
        out, receipt = adaptive_preflight(request, backend.forward, context_window=5000,
                                           completion_reserve=100, safety_margin=200)
        self.assertEqual(request, out)
        self.assertFalse(receipt["compacted"])
        self.assertEqual(receipt["preflight"]["prompt_tokens"], count["prompt_tokens"])

    def test_compaction_retains_edits_constraints_and_tool_pairing(self):
        backend = FakeBackend()
        messages = episode()
        handles = {}
        out, rec = typed_packet(messages, budget_tokens=1800, keep_recent=2, store=handles)
        self.assertEqual(out[:2], messages[:2])
        self.assertEqual(out[-4:], messages[-4:])
        packet = out[2]["content"]
        self.assertIn("start_line=12", packet)
        self.assertIn("old_str=if x > width:", packet)
        self.assertIn("Keep the existing collision invariant", packet)
        self.assertIn("engine/collision.py", packet)
        self.assertEqual(rec["archive_edit_calls"], 1)
        self.assertTrue(handles)
        self.assertTrue(any("0123456789" in body for body in handles.values()))

        # Full-history raw count exceeds the limit; the final request is
        # proved to fit by the same upstream template/tokenizer.
        raw = exact_preflight(backend.forward, {"messages": messages, "tools": []},
                              context_window=3000, completion_reserve=100, safety_margin=200)
        self.assertFalse(raw["fits"])
        adapted, measured = adaptive_preflight({"messages": messages, "tools": []},
                                                backend.forward, context_window=3000,
                                                completion_reserve=100, safety_margin=200,
                                                packet_budget=1800, keep_recent=2, store=handles)
        self.assertTrue(measured["compacted"])
        self.assertTrue(measured["preflight"]["fits"])
        self.assertEqual(adapted["messages"][-4:], messages[-4:])

    def test_unpaired_tool_or_mutable_history_fails_closed(self):
        messages = episode(5)
        del messages[5]   # tool result of t1 is now missing
        with self.assertRaises(WorkingSetError):
            typed_packet(messages, budget_tokens=1800, keep_recent=1)

    def test_missing_upstream_template_endpoint_rejects(self):
        backend = FakeBackend()
        backend.valid = False
        with self.assertRaisesRegex(PreflightError, "/apply-template"):
            exact_preflight(backend.forward, {"messages": episode(1)},
                            context_window=32768, completion_reserve=4096, safety_margin=1024)


class HttpUpstream(BaseHTTPRequestHandler):
    chats = []

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template":
            payload = {"prompt": json.dumps({"messages": req["messages"], "tools": req["tools"]}, sort_keys=True)}
        elif self.path == "/tokenize":
            payload = {"tokens": list(req["content"].encode("utf-8"))}
        elif self.path == "/v1/chat/completions":
            HttpUpstream.chats.append(req)
            prepared = json.dumps({"messages": req["messages"], "tools": req.get("tools", [])}, sort_keys=True)
            payload = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": len(prepared.encode()), "completion_tokens": 1}}
        else:
            self.send_response(404)
            self.end_headers()
            return
        text = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(text)))
        self.end_headers()
        self.wfile.write(text)

    def log_message(self, *_args):
        pass


class RelayIntegrationTests(unittest.TestCase):
    def test_mcp04_proxy_compacts_and_exposes_stable_read_handles(self):
        HttpUpstream.chats.clear()
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), HttpUpstream)
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "calls.jsonl"
            relay = Relay(f"http://127.0.0.1:{upstream.server_port}", log, None,
                          512, 1000, 1200, 2, 6000, 30,
                          context_window=3000, completion_reserve=100, safety_margin=200)
            proxy = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(relay))
            threading.Thread(target=proxy.serve_forever, daemon=True).start()
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{proxy.server_port}/mcp04/t1/v1/chat/completions",
                    data=json.dumps({"model": "m", "messages": episode(), "tools": []}).encode(),
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=30) as response:
                    self.assertEqual(json.load(response)["choices"][0]["message"]["content"], "ok")
                logs = [json.loads(s) for s in log.read_text().splitlines() if s.strip()]
                self.assertEqual(len(logs), 1)
                self.assertTrue(logs[0]["compaction"]["compacted"])
                self.assertTrue(logs[0]["preflight_matches_provider"])
                self.assertTrue(HttpUpstream.chats)
                self.assertLess(len(HttpUpstream.chats[-1]["messages"]), len(episode()))
                handle = next(iter(relay.mcp04_store))
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{proxy.server_port}/mcp04/t1/v1/working-set/{handle}",
                    timeout=30) as response:
                    self.assertEqual(json.load(response)["handle"], handle)
            finally:
                proxy.shutdown()
        upstream.shutdown()


if __name__ == "__main__":
    unittest.main()
