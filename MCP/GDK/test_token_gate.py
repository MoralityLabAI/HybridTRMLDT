"""Mocked-backend tests for the three exact preflight model profiles.

These tests establish protocol and request enforcement, not live model
tokenization parity. That requires real-model backend integration runs.
"""
from __future__ import annotations

import http.server
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from MCP.GDK.profiles import PROFILES, get_profile
from MCP.GDK.token_gate import (
    BackendHTTP, ExactGate, GateError, check_local_url, handler_for, probe,
)


class FakeBackend:
    def __init__(self, count: int = 140):
        self.count = count
        self.requests = []
        self.completions = 0

    def post_json(self, path, payload):
        self.requests.append((path, payload))
        if path == "/v1/chat/completions/render":
            return {"token_ids": list(range(self.count))}
        if path == "/v1/chat/completions/input_tokens":
            return {"input_tokens": self.count}
        raise AssertionError(f"unknown counter route {path}")

    def forward(self, body):
        self.completions += 1
        return 200, json.dumps({"ok": True}).encode(), "application/json"

    def models(self):
        return 200, b'{"data":[]}', "application/json"


def request(model="local-model", *, size=32, maximum=2048):
    return {
        "model": model,
        "messages": [{"role": "system", "content": "instructions"},
                     {"role": "user", "content": "x" * size}],
        "tools": [{"type": "function", "function": {
            "name": "read_symbol", "parameters": {"type": "object", "properties": {}}}}],
        "max_tokens": maximum,
        "stream": False,
    }


def make_gate(profile, *, count=140, backend=None, context=12288, receipt_file=None):
    fake = backend or FakeBackend(count)
    return ExactGate(
        get_profile(profile), served_model_id="local-model",
        backend=fake, served_context_tokens=context,
        receipts_path=receipt_file,
    ), fake


class ProfileTests(unittest.TestCase):
    def test_all_three_profile_ids_and_explicit_backend_counters(self):
        self.assertEqual(set(PROFILES), {"inkling-2-mini", "bonsai-8b", "bonsai-27b"})
        self.assertEqual(get_profile("inkling-2-mini").default_counter, "vllm_render")
        for name in ("bonsai-8b", "bonsai-27b"):
            self.assertEqual(get_profile(name).default_counter, "llama_cpp_input_tokens")
            self.assertIn("prism-ml/", get_profile(name).hf_reference)
        self.assertEqual(get_profile("inkling-2-mini").hf_reference, "thinkingmachines/Inkling-2-Mini")

    def test_each_profile_uses_backend_owned_counter(self):
        for name in PROFILES:
            with self.subTest(profile=name):
                gate, fake = make_gate(name)
                obj = request()
                raw = json.dumps(obj).encode()
                report = gate.preflight(obj, raw)
                self.assertTrue(report.accepted)
                self.assertEqual(report.input_tokens, 140)
                self.assertEqual(report.output_reserve_tokens, 2048)
                self.assertEqual(gate.input_limit, 8192)
                self.assertEqual(fake.requests[0][0],
                    "/v1/chat/completions/render" if name == "inkling-2-mini"
                    else "/v1/chat/completions/input_tokens")
                gate.forward(raw, report)
                self.assertEqual(fake.completions, 1)

    def test_prompt_overflow_and_malformed_counter_fail_closed(self):
        gate, fake = make_gate("bonsai-27b", count=8193)
        with self.assertRaisesRegex(GateError, "exceeds exact input"):
            gate.preflight(request(), b"input")
        self.assertFalse(gate.last_receipt.accepted)
        self.assertEqual(fake.completions, 0)
        fake.count = -1
        with self.assertRaisesRegex(GateError, "valid exact"):
            gate.preflight(request(), b"input")
        self.assertEqual(fake.completions, 0)

    def test_vllm_ids_must_be_valid(self):
        class Invalid(FakeBackend):
            def post_json(self, *args):
                return {"token_ids": [1, "two"]}
        gate, fake = make_gate("inkling-2-mini", backend=Invalid())
        with self.assertRaises(GateError):
            gate.preflight(request(), b"input")
        self.assertEqual(fake.completions, 0)

    def test_reject_model_mismatch_stream_missing_output_and_multimodal(self):
        gate, fake = make_gate("bonsai-8b")
        samples = [
            {**request(), "model": "different"},
            {**request(), "stream": True},
            {k: v for k, v in request().items() if k != "max_tokens"},
            {**request(), "max_tokens": 2049},
            {**request(), "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abcd"}}]}]},
            {**request(), "n": 2},
            {**request(), "truncate_prompt_tokens": 20},
        ]
        for sample in samples:
            with self.subTest(sample=str(sample)[:80]), self.assertRaises(GateError):
                gate.preflight(sample, b"sample")
        self.assertFalse(fake.requests)
        self.assertEqual(fake.completions, 0)

    def test_backend_context_clamps_12k_envelope(self):
        gate, _fake = make_gate("bonsai-8b", context=9216, count=5120)
        self.assertEqual(gate.input_limit, 5120)
        gate.preflight(request(), b"input")
        gate2, _ = make_gate("bonsai-8b", context=9216, count=5121)
        with self.assertRaises(GateError):
            gate2.preflight(request(), b"input")

    def test_local_only_url(self):
        for bad in ("http://example.com:8080", "https://127.0.0.1:8080",
                    "http://127.0.0.1:8080@evil.com", "http://localhost:8080",
                    "http://127.0.0.1:8080/v1/models", "http://127.0.0.1:8080?q=x"):
            with self.subTest(url=bad), self.assertRaises(ValueError):
                check_local_url(bad)
        self.assertEqual(check_local_url("http://127.0.0.1:8080/v1/"), "http://127.0.0.1:8080")

    def test_receipt_excludes_prompt_and_is_persisted(self):
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "nested" / "receipts.jsonl"
            gate, fake = make_gate("inkling-2-mini", receipt_file=path)
            obj = request(size=55)
            raw = json.dumps(obj).encode()
            report = gate.preflight(obj, raw)
            gate.forward(raw, report)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertTrue(records[0]["accepted"])
            self.assertTrue(records[1]["forwarded"])
            self.assertEqual(records[0]["request_sha256"], records[1]["request_sha256"])
            self.assertNotIn("xxxxx", path.read_text())


class BackendFixture(http.server.BaseHTTPRequestHandler):
    seen = []
    token_count = 120

    def log_message(self, *args):
        pass

    def do_POST(self):
        size = int(self.headers.get("Content-Length", 0))
        blob = self.rfile.read(size)
        self.seen.append((self.path, blob))
        if self.path == "/v1/chat/completions/input_tokens":
            payload = {"input_tokens": self.token_count}
        elif self.path == "/v1/chat/completions/render":
            payload = {"token_ids": list(range(self.token_count))}
        elif self.path == "/v1/chat/completions":
            payload = {"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                       "usage": {"prompt_tokens": self.token_count, "completion_tokens": 3}}
        else:
            self.send_error(404)
            return
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ProxyHTTPTests(unittest.TestCase):
    def setUp(self):
        BackendFixture.seen = []
        BackendFixture.token_count = 120
        self.backend = http.server.ThreadingHTTPServer(("127.0.0.1", 0), BackendFixture)
        self.backend_thread = threading.Thread(target=self.backend.serve_forever, daemon=True)
        self.backend_thread.start()
        backend_url = f"http://127.0.0.1:{self.backend.server_port}"
        self.gate = ExactGate(
            get_profile("bonsai-8b"), served_model_id="local-model",
            backend=BackendHTTP(backend_url), served_context_tokens=12288,
        )
        self.proxy = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.gate))
        self.proxy_thread = threading.Thread(target=self.proxy.serve_forever, daemon=True)
        self.proxy_thread.start()
        self.origin = f"http://127.0.0.1:{self.proxy.server_port}"

    def tearDown(self):
        self.proxy.shutdown()
        self.proxy.server_close()
        self.backend.shutdown()
        self.backend.server_close()

    def call(self, body):
        req = urllib.request.Request(
            self.origin + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, json.load(response)

    def test_preflight_then_exact_wire_forward(self):
        self.assertEqual(
            probe(self.origin + "/v1/", "bonsai-8b", "local-model")["counter"],
            "llama_cpp_input_tokens",
        )
        original = json.dumps(request()).encode()
        status, obj = self.call(original)
        self.assertEqual(status, 200)
        self.assertIn("choices", obj)
        self.assertEqual([item[0] for item in BackendFixture.seen],
                         ["/v1/chat/completions/input_tokens", "/v1/chat/completions"])
        self.assertEqual(BackendFixture.seen[1][1], original)

    def test_oversize_never_forwards_inference(self):
        BackendFixture.token_count = 9000
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.call(json.dumps(request()).encode())
        self.assertEqual(cm.exception.code, 422)
        self.assertEqual(len(BackendFixture.seen), 1)
        self.assertEqual(self.gate.last_receipt.input_tokens, 9000)
        self.assertFalse(self.gate.last_receipt.forwarded)

    def test_missing_counter_fails_closed(self):
        self.gate.counter = "vllm_render"
        BackendFixture.token_count = 9999
        with self.assertRaises(urllib.error.HTTPError):
            self.call(json.dumps(request()).encode())
        self.assertEqual(len(BackendFixture.seen), 1)
        self.assertEqual(BackendFixture.seen[0][0], "/v1/chat/completions/render")

    def test_probe_rejects_wrong_profile(self):
        with self.assertRaises(GateError):
            probe(self.origin, "bonsai-27b", "local-model")


if __name__ == "__main__":
    unittest.main()
