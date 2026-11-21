"""GDK-02 exact backend-counted preflight proxy for text-only chat completions.

Run locally between the Goose OpenAI-compatible provider and the actual
llama.cpp or vLLM server. The COMPLETE unchanged chat request is preprocessed
by that same backend before it can perform inference. Unsupported backend
token counters, missing output reservations and identity mismatches fail closed.

No external Python modules are required. The proxy intentionally forbids
streaming and non-text modalities until those wire forms have dedicated tests.
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import ipaddress
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .profiles import COUNTERS, ModelProfile, get_profile


class GateError(ValueError):
    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def check_local_url(value: str) -> str:
    """Require direct loopback HTTP, rejecting DNS aliases and URL tricks."""
    url = urllib.parse.urlsplit(value)
    try:
        host = url.hostname
        if not host or not ipaddress.ip_address(host).is_loopback:
            raise ValueError("backend/health endpoints must use loopback IP addresses")
        if url.scheme != "http" or url.username or url.password or url.query or url.fragment:
            raise ValueError("only plain loopback HTTP is supported")
        if url.path not in ("", "/", "/v1", "/v1/"):
            raise ValueError("backend URL must be a host origin or /v1")
        if not url.port:
            raise ValueError("backend URL needs an explicit port")
        origin_host = f"[{host}]" if ":" in host else host
        return f"http://{origin_host}:{url.port}"
    except (ValueError, TypeError) as error:
        raise ValueError(f"invalid loopback backend URL: {error}") from error


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise GateError("backend redirect refused", 503)


class BackendHTTP:
    """Only talks to the one fixed loopback inference backend."""

    def __init__(self, backend_url: str, api_key: str = "", timeout: float = 60.0):
        self.origin = check_local_url(backend_url)
        self.api_key = api_key
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def _send(self, method: str, path: str, data: bytes | None):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.origin + path, method=method, data=data, headers=headers)
        try:
            with self.opener.open(req, timeout=self.timeout) as resp:
                blob = resp.read(16 * 1024 * 1024 + 1)
                if len(blob) > 16 * 1024 * 1024:
                    raise GateError("backend response too large", 503)
                return resp.status, blob, resp.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as error:
            # Never echo backend bodies; they can contain portions of prompts.
            raise GateError(f"backend returned HTTP {error.code}", 503) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise GateError("backend unavailable or timed out", 503) from error

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        status, response, _ = self._send("POST", path, encoded)
        if status != 200:
            raise GateError("backend preprocessing unavailable", 503)
        try:
            result = json.loads(response)
            if isinstance(result, dict):
                return result
        except (ValueError, UnicodeError):
            pass
        raise GateError("invalid backend counter response", 503)

    def forward(self, body: bytes) -> tuple[int, bytes, str]:
        # The byte string from Goose is forwarded without reserialization.
        return self._send("POST", "/v1/chat/completions", body)

    def models(self) -> tuple[int, bytes, str]:
        return self._send("GET", "/v1/models", None)


@dataclass(frozen=True)
class GateReceipt:
    profile: str
    served_model_id: str
    counter: str
    request_sha256: str
    input_tokens: int | None
    output_reserve_tokens: int
    safety_reserve_tokens: int
    effective_window_tokens: int
    accepted: bool
    forwarded: bool
    reason: str | None = None


class ExactGate:
    def __init__(
        self, profile: ModelProfile, *, served_model_id: str, backend: Any,
        served_context_tokens: int, counter: str | None = None, receipts_path: Path | None = None,
    ):
        if not served_model_id.strip():
            raise ValueError("served model ID required for identity binding")
        self.profile = profile
        self.served_model_id = served_model_id
        self.backend = backend
        self.served_context_tokens = served_context_tokens
        self.input_limit = profile.input_limit(served_context_tokens)
        self.effective_window = min(profile.window_tokens, served_context_tokens)
        self.counter = counter or profile.default_counter
        if self.counter not in COUNTERS:
            raise ValueError(f"unsupported token counter: {self.counter}")
        self.receipts_path = receipts_path
        self._receipt_lock = threading.Lock()
        self.last_receipt: GateReceipt | None = None

    def count_full_request(self, payload: dict[str, Any]) -> int:
        if self.counter == "llama_cpp_input_tokens":
            result = self.backend.post_json("/v1/chat/completions/input_tokens", payload)
            count = result.get("input_tokens")
        elif self.counter == "vllm_render":
            result = self.backend.post_json("/v1/chat/completions/render", payload)
            ids = result.get("token_ids")
            count = len(ids) if isinstance(ids, list) and all(
                type(item) is int and item >= 0 for item in ids
            ) else None
        else:
            raise GateError("unsupported exact counter", 503)
        if type(count) is not int or count < 0:
            raise GateError("backend returned no valid exact input token count", 503)
        return count

    def _write_receipt(self, receipt: GateReceipt) -> None:
        with self._receipt_lock:
            self.last_receipt = receipt
            if self.receipts_path is None:
                return
            self.receipts_path.parent.mkdir(parents=True, exist_ok=True)
            info = {
                **receipt.__dict__,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            data = (json.dumps(info, sort_keys=True) + "\n").encode()
            fd = os.open(str(self.receipts_path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, data)
                os.fsync(fd)
            finally:
                os.close(fd)

    def preflight(self, payload: dict[str, Any], body: bytes) -> GateReceipt:
        digest = hashlib.sha256(body).hexdigest()
        input_tokens: int | None = None
        reason: str | None = None
        try:
            if not isinstance(payload, dict):
                raise GateError("chat completion body must be JSON object")
            if payload.get("model") != self.served_model_id:
                raise GateError("model identity does not match pinned served model")
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages or not all(
                isinstance(message, dict) and isinstance(message.get("role"), str)
                for message in messages
            ):
                raise GateError("chat completion requires nonempty messages")
            if payload.get("stream") is not False:
                raise GateError("only explicitly non-streaming chat is supported")
            if any(isinstance(msg.get("content"), list) for msg in messages):
                raise GateError("multimodal content requires a separate preflight contract")
            if "max_tokens" in payload and "max_completion_tokens" in payload:
                raise GateError("two output-token settings supplied")
            maximum = payload.get("max_tokens", payload.get("max_completion_tokens"))
            if type(maximum) is not int or maximum <= 0:
                raise GateError("positive explicit max_tokens required")
            if maximum > self.profile.output_reserve_tokens:
                raise GateError("output tokens exceed reserved envelope")
            if payload.get("n", 1) != 1:
                raise GateError("parallel generations require a separate budget")
            for field in ("truncate_prompt_tokens", "truncation", "prompt_token_ids"):
                if field in payload:
                    raise GateError(f"unsupported context-altering field: {field}")
            input_tokens = self.count_full_request(payload)
            if input_tokens > self.input_limit:
                raise GateError(
                    f"prompt {input_tokens} tokens exceeds exact input limit {self.input_limit}"
                )
        except GateError as error:
            reason = str(error)
            receipt = GateReceipt(
                profile=self.profile.name, served_model_id=self.served_model_id,
                counter=self.counter, request_sha256=digest, input_tokens=input_tokens,
                output_reserve_tokens=self.profile.output_reserve_tokens,
                safety_reserve_tokens=self.profile.safety_reserve_tokens,
                effective_window_tokens=self.effective_window,
                accepted=False, forwarded=False, reason=reason,
            )
            self._write_receipt(receipt)
            raise
        receipt = GateReceipt(
            profile=self.profile.name, served_model_id=self.served_model_id,
            counter=self.counter, request_sha256=digest, input_tokens=input_tokens,
            output_reserve_tokens=self.profile.output_reserve_tokens,
            safety_reserve_tokens=self.profile.safety_reserve_tokens,
            effective_window_tokens=self.effective_window,
            accepted=True, forwarded=False,
        )
        self._write_receipt(receipt)
        return receipt

    def forward(self, body: bytes, receipt: GateReceipt) -> tuple[int, bytes, str]:
        status, response, content_type = self.backend.forward(body)
        self._write_receipt(GateReceipt(**{**receipt.__dict__, "forwarded": True}))
        return status, response, content_type


def handler_for(gate: ExactGate, max_body_bytes: int = 2_000_000):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status: int, body: bytes, content_type: str = "application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def problem(self, error: GateError):
            blob = json.dumps({
                "error": {"type": "gdk_preflight", "message": str(error), "code": "gdk_gate"}
            }).encode()
            self.reply(error.status, blob)

        def do_GET(self):
            if self.path == "/gdk/health":
                info = {
                    "schema": "gdk.exact_gate.v1", "profile": gate.profile.name,
                    "served_model_id": gate.served_model_id, "counter": gate.counter,
                    "effective_window_tokens": gate.effective_window,
                    "input_limit_tokens": gate.input_limit,
                    "exact_backend_preflight": True,
                }
                self.reply(200, json.dumps(info).encode())
            elif self.path == "/v1/models":
                try:
                    self.reply(*gate.backend.models())
                except GateError as error:
                    self.problem(error)
            else:
                self.problem(GateError("route not available", 404))

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.problem(GateError("only chat completions are supported", 404))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= max_body_bytes:
                    raise GateError("missing or excessive Content-Length", 413)
                body = self.rfile.read(length)
                if len(body) != length:
                    raise GateError("incomplete request", 400)
                payload = json.loads(body)
                receipt = gate.preflight(payload, body)
                status, response, content_type = gate.forward(body, receipt)
                self.reply(status, response, content_type)
            except (ValueError, UnicodeError) as error:
                self.problem(error if isinstance(error, GateError) else GateError("invalid request", 400))
            except OSError:
                self.problem(GateError("backend forwarding failed", 503))

    return Handler


def probe(base_url: str, profile: str, served_model_id: str) -> dict[str, Any]:
    origin = check_local_url(base_url)
    req = urllib.request.Request(origin + "/gdk/health")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(req, timeout=5) as response:
            info = json.load(response)
    except (urllib.error.URLError, ValueError, TimeoutError) as error:
        raise GateError("exact gate health check failed", 503) from error
    if not isinstance(info, dict) or info.get("schema") != "gdk.exact_gate.v1":
        raise GateError("unexpected gate health response", 503)
    if info.get("profile") != profile or info.get("served_model_id") != served_model_id:
        raise GateError("gate profile or model mismatch", 503)
    if info.get("exact_backend_preflight") is not True or not (
        type(info.get("input_limit_tokens")) is int and info["input_limit_tokens"] > 0
    ):
        raise GateError("exact gate preflight is inactive", 503)
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--profile", choices=sorted(("inkling-2-mini", "bonsai-8b", "bonsai-27b")), required=True)
    serve.add_argument("--served-model-id", required=True)
    serve.add_argument("--backend-url", required=True)
    serve.add_argument("--backend-context-tokens", type=int, required=True)
    serve.add_argument("--counter", choices=sorted(COUNTERS))
    serve.add_argument("--port", type=int, default=8094)
    serve.add_argument("--receipt-file")
    serve.add_argument("--timeout", type=float, default=600.0)
    serve.add_argument("--backend-key-env", default="GDK_BACKEND_API_KEY")
    check = commands.add_parser("probe")
    check.add_argument("--base-url", required=True)
    check.add_argument("--profile", required=True)
    check.add_argument("--served-model-id", required=True)
    args = parser.parse_args()

    if args.command == "probe":
        try:
            print(json.dumps(probe(args.base_url, args.profile, args.served_model_id), sort_keys=True))
            return 0
        except (GateError, ValueError) as error:
            print(f"GDK probe rejected: {error}", file=sys.stderr)
            return 1

    try:
        if not 1 <= args.port <= 65535:
            raise ValueError("port must be 1..65535")
        gate = ExactGate(
            get_profile(args.profile),
            served_model_id=args.served_model_id,
            backend=BackendHTTP(
                args.backend_url, api_key=os.environ.get(args.backend_key_env, ""),
                timeout=args.timeout,
            ),
            served_context_tokens=args.backend_context_tokens,
            counter=args.counter,
            receipts_path=Path(args.receipt_file) if args.receipt_file else None,
        )
        server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(gate))
    except ValueError as error:
        print(f"GDK token gate setup rejected: {error}", file=sys.stderr)
        return 1
    print(json.dumps({
        "listening": f"http://127.0.0.1:{server.server_port}/v1",
        "profile": gate.profile.name, "model": gate.served_model_id,
        "counter": gate.counter, "input_limit_tokens": gate.input_limit,
    }), file=sys.stderr)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
