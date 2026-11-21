"""Exercise the adapter against a real loopback HTTP server, without a model."""

from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
import unittest
from unittest.mock import patch

from .local_model import LocalModel, LocalModelError, normalize_usage, parse_action, strict_json_loads


def completion(text='{"tool":"finish","arguments":{"answer":7}}', usage=None):
    result = {"model": "fixture-model", "choices": [{"message": {"role": "assistant", "content": text},
                                                     "finish_reason": "stop"}]}
    if usage is not None:
        result["usage"] = usage
    return result


@contextmanager
def local_server(replies, *, delay=0.0):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append({"path": self.path, "headers": dict(self.headers), "body": json.loads(data)})
            item = replies[min(len(requests) - 1, len(replies) - 1)]
            status, payload, headers = item if isinstance(item, tuple) else (200, item, {})
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
            if delay:
                time.sleep(delay)
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                for key, value in headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


MESSAGES = [{"role": "system", "content": "Return one JSON action."},
            {"role": "user", "content": "Read the value."}]


class LocalModelTests(unittest.TestCase):
    def test_loopback_request_and_observed_usage(self):
        usage = {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27,
                 "prompt_tokens_details": {"cached_tokens": 8}}
        with local_server([completion(usage=usage)]) as (url, requests):
            model = LocalModel(url, "requested", max_tokens=20, checkpoint="sha256:example", parameter_count=800000000)
            receipt = model.step(MESSAGES)
        self.assertEqual(receipt["action"], {"tool": "finish", "arguments": {"answer": 7}})
        self.assertEqual(receipt["usage"]["cached_tokens"], 8)
        self.assertEqual(receipt["usage"]["prompt_tokens"], 20)
        self.assertEqual(receipt["usage"]["completion_tokens"], 7)
        self.assertEqual(receipt["response_model"], "fixture-model")
        self.assertEqual(receipt["usage"]["raw"], usage)
        self.assertGreater(receipt["prompt_bytes"], 0)
        self.assertGreater(receipt["output_bytes"], 0)
        self.assertGreaterEqual(receipt["latency_seconds"], 0)
        self.assertEqual(requests[0]["path"], "/v1/chat/completions")
        self.assertEqual(requests[0]["body"]["max_tokens"], 20)
        self.assertFalse(requests[0]["body"]["stream"])
        self.assertEqual(model.metadata["parameter_count"], 800000000)
        self.assertNotIn("Authorization", requests[0]["headers"])

    def test_missing_usage_stays_unknown(self):
        with local_server([completion()]) as (url, _):
            usage = LocalModel(url).step(MESSAGES)["usage"]
        self.assertFalse(usage["reported"])
        for field in ("prompt_tokens", "completion_tokens", "cached_tokens", "total_tokens"):
            self.assertIsNone(usage[field])

    def test_mixed_malformed_usage_preserves_valid_observations(self):
        raw = {"prompt_tokens": True, "completion_tokens": 4, "total_tokens": "4",
               "prompt_tokens_details": {"cached_tokens": -1}}
        with local_server([completion(usage=raw)]) as (url, _):
            usage = LocalModel(url).step(MESSAGES)["usage"]
        self.assertIsNone(usage["prompt_tokens"])
        self.assertEqual(usage["completion_tokens"], 4)
        self.assertIsNone(usage["cached_tokens"])
        self.assertEqual(usage["raw"], raw)
        self.assertEqual(len(usage["validation_errors"]), 3)
        inconsistent = normalize_usage({"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 8,
                                        "prompt_tokens_details": {"cached_tokens": 4}})
        self.assertIsNone(inconsistent["cached_tokens"])
        self.assertIsNone(inconsistent["total_tokens"])

    def test_invalid_action_receipts_preserve_output_and_usage(self):
        texts = [
            '```json\n{"tool":"finish","arguments":{}}\n```',
            '{"tool":"finish","arguments":{},"extra":true}',
            '{"tool":"finish","tool":"other","arguments":{}}',
            '{"tool":"finish","arguments":{"value":NaN}}',
            '{"tool":"finish","arguments":[]}',
        ]
        with local_server([completion(text, {"prompt_tokens": 5, "completion_tokens": 6}) for text in texts]) as (url, requests):
            model = LocalModel(url)
            for text in texts:
                receipt = model.step(MESSAGES)
                self.assertIsNone(receipt["action"])
                self.assertTrue(receipt["parse_error"])
                self.assertEqual(receipt["response_text"], text)
                self.assertEqual(receipt["usage"]["completion_tokens"], 6)
        self.assertEqual(len(requests), len(texts))

    def test_overflowing_exponents_in_actions_are_rejected_and_receipts_serialize(self):
        texts = [
            '{"tool":"finish","arguments":{"answer":1e999}}',
            '{"tool":"finish","arguments":{"nested":[{"answer":-1e999}]}}',
        ]
        for text in texts:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "non-finite JSON number"):
                parse_action(text)
        replies = [completion(text, {"prompt_tokens": 5, "completion_tokens": 6}) for text in texts]
        with local_server(replies) as (url, _):
            model = LocalModel(url)
            for text in texts:
                receipt = model.step(MESSAGES)
                self.assertIsNone(receipt["action"])
                self.assertIn("non-finite JSON number", receipt["parse_error"])
                self.assertEqual(receipt["response_text"], text)
                self.assertEqual(receipt["usage"]["prompt_tokens"], 5)
                self.assertEqual(receipt["usage"]["completion_tokens"], 6)
                json.dumps(receipt, allow_nan=False)

    def test_overflowing_exponent_in_outer_usage_is_a_preserved_failed_response(self):
        raw = (b'{"choices":[{"message":{"content":"{\\"tool\\":\\"finish\\",\\"arguments\\":{}}"}}],'
               b'"usage":{"prompt_tokens":1e999,"completion_tokens":2}}')
        with local_server([raw]) as (url, requests):
            with self.assertRaises(LocalModelError) as caught:
                LocalModel(url).step(MESSAGES)
        receipt = caught.exception.receipt
        self.assertIn("non-finite JSON number", str(caught.exception))
        self.assertEqual(receipt["raw_response_text"], raw.decode("utf-8"))
        self.assertIsNone(receipt["action"])
        self.assertIsNone(receipt["usage"]["prompt_tokens"])
        self.assertEqual(len(requests), 1)
        json.dumps(receipt, allow_nan=False)

    def test_unpaired_surrogate_values_and_keys_are_rejected_with_exportable_receipts(self):
        texts = [
            r'{"tool":"finish","arguments":{"evidence_refs":["\ud800"]}}',
            r'{"tool":"finish","arguments":{"nested":[{"\udfff":"value"}]}}',
        ]
        for text in texts:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "Unicode scalar"):
                parse_action(text)
        replies = [completion(text, {"prompt_tokens": 5, "completion_tokens": 6}) for text in texts]
        with local_server(replies) as (url, _):
            model = LocalModel(url)
            for text in texts:
                receipt = model.step(MESSAGES)
                self.assertIsNone(receipt["action"])
                self.assertIn("Unicode scalar", receipt["parse_error"])
                self.assertEqual(receipt["response_text"], text)
                self.assertEqual(receipt["usage"]["completion_tokens"], 6)
                json.dumps(receipt, ensure_ascii=False, allow_nan=False).encode("utf-8")

    def test_outer_surrogate_response_remains_preserved_as_escaped_json(self):
        raw = b'{"choices":[{"message":{"content":"\\ud800"}}],"usage":{"prompt_tokens":5}}'
        with local_server([raw]) as (url, _):
            with self.assertRaises(LocalModelError) as caught:
                LocalModel(url).step(MESSAGES)
        receipt = caught.exception.receipt
        self.assertIn("Unicode scalar", str(caught.exception))
        self.assertEqual(receipt["raw_response_text"], raw.decode("utf-8"))
        self.assertIsNone(receipt["action"])
        json.dumps(receipt, ensure_ascii=False, allow_nan=False).encode("utf-8")

    def test_valid_surrogate_pairs_are_decoded_as_unicode_scalars(self):
        text = r'{"tool":"finish","arguments":{"label":"\ud83d\ude00","\ud83d\ude01":"ok"}}'
        action = parse_action(text)
        self.assertEqual(action["arguments"]["label"], "\U0001f600")
        self.assertEqual(action["arguments"]["\U0001f601"], "ok")
        self.assertEqual(strict_json_loads(r'"\ud83d\ude00"'), "\U0001f600")
        json.dumps(action, ensure_ascii=False, allow_nan=False).encode("utf-8")

    def test_nonloopback_and_credential_urls_rejected(self):
        for url in ("https://api.example/v1", "http://10.0.0.1/v1", "http://0.0.0.0/v1",
                    "http://user:password@127.0.0.1/v1", "http://localhost/v1?key=x",
                    "http://localhost/v1#fragment", "http://2130706433/v1"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                LocalModel(url)
        self.assertEqual(LocalModel("http://localhost:8080/v1").metadata["connected_host"], "127.0.0.1")
        self.assertEqual(LocalModel("http://[::1]:8080/v1").metadata["connected_host"], "::1")

    def test_proxy_environment_is_ignored(self):
        proxy = {"HTTP_PROXY": "http://127.0.0.1:1", "HTTPS_PROXY": "http://127.0.0.1:1",
                 "ALL_PROXY": "http://127.0.0.1:1", "NO_PROXY": ""}
        with local_server([completion()]) as (url, _), patch.dict(os.environ, proxy):
            self.assertEqual(LocalModel(url).step(MESSAGES)["action"]["tool"], "finish")

    def test_redirect_is_rejected_without_following(self):
        with local_server([(302, b"redirect", {"Location": "https://api.example/v1"})]) as (url, requests):
            with self.assertRaises(LocalModelError) as caught:
                LocalModel(url).step(MESSAGES)
        self.assertEqual(len(requests), 1)
        self.assertEqual(caught.exception.receipt["http_status"], 302)
        self.assertIn("redirect", str(caught.exception).lower())

    def test_http_error_keeps_reported_usage(self):
        with local_server([(400, {"error": "context full", "usage": {"prompt_tokens": 70}}, {})]) as (url, requests):
            with self.assertRaises(LocalModelError) as caught:
                LocalModel(url).step(MESSAGES)
        self.assertEqual(caught.exception.receipt["usage"]["prompt_tokens"], 70)
        self.assertEqual(len(requests), 1)

    def test_context_and_response_byte_bounds(self):
        with local_server([completion()]) as (url, requests):
            with self.assertRaises(LocalModelError) as caught:
                LocalModel(url, max_context_bytes=1).step(MESSAGES)
            self.assertFalse(caught.exception.receipt["request_sent"])
            self.assertEqual(len(requests), 0)
            with self.assertRaises(LocalModelError) as caught:
                LocalModel(url, max_response_bytes=16).step(MESSAGES)
            self.assertGreater(caught.exception.receipt["response_bytes"], 16)

    def test_token_bounds_are_checked_when_reported(self):
        reply = completion(usage={"prompt_tokens": 20, "completion_tokens": 7})
        invalid = completion("invalid JSON", usage={"prompt_tokens": 20, "completion_tokens": 7})
        with local_server([reply, invalid, invalid]) as (url, _):
            with self.assertRaises(LocalModelError):
                LocalModel(url, max_tokens=5).step(MESSAGES)
            with self.assertRaises(LocalModelError):
                LocalModel(url, max_tokens=10, context_window_tokens=25).step(MESSAGES)
            with self.assertRaises(LocalModelError):
                LocalModel(url, max_tokens=5).step(MESSAGES)

    def test_timeout_and_runtime_exhaustion_have_receipts(self):
        with local_server([completion()], delay=0.1) as (url, _):
            with self.assertRaises(LocalModelError) as caught:
                LocalModel(url, timeout=0.02).step(MESSAGES)
            self.assertTrue(caught.exception.receipt["request_sent"])
        model = LocalModel(max_runtime_seconds=0.001)
        time.sleep(0.005)
        with self.assertRaises(LocalModelError) as caught:
            model.step(MESSAGES)
        self.assertFalse(caught.exception.receipt["request_sent"])

    def test_invalid_envelope_and_body_are_failed_attempts(self):
        for reply in (b"not JSON", {"choices": []}, {"choices": [{"message": {"content": None}}]}):
            with local_server([reply]) as (url, _):
                with self.assertRaises(LocalModelError) as caught:
                    LocalModel(url).step(MESSAGES)
                self.assertTrue(caught.exception.receipt["raw_response_text"])


if __name__ == "__main__":
    unittest.main()
