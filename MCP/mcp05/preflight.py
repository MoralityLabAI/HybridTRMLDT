"""Preflight the exact Goose provider request immediately before inference."""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from MCP.hubrun.mcp04 import PreflightError, WorkingSetError, exact_preflight, typed_packet
from MCP.hubrun.packet import compact_messages


def measure(forward, request, *, window: int, reserve: int):
    return exact_preflight(forward, request, context_window=window,
                           completion_reserve=reserve, safety_margin=64)


def prepare(forward, request, *, window: int, reserve: int, budget: int, memory: str):
    raw = measure(forward, request, window=window, reserve=reserve)
    if raw["fits"]:
        return request, {"preflight": raw, "compacted": False}
    archived = {}
    if memory != "raw":
        for recent, cap in ((2, budget), (1, min(budget, 900)), (1, min(budget, 500))):
            try:
                if memory == "packet":
                    messages, packet = compact_messages(request["messages"], threshold_tokens=0,
                                                        budget_tokens=cap, keep_recent=recent)
                else:
                    messages, packet = typed_packet(request["messages"], budget_tokens=cap,
                                                    keep_recent=recent, store=archived)
            except WorkingSetError:
                continue
            candidate = dict(request, messages=messages)
            checked = measure(forward, candidate, window=window, reserve=reserve)
            if checked["fits"]:
                return candidate, {"preflight": checked, "raw_prompt_tokens": raw["prompt_tokens"],
                                   **packet}
    raise PreflightError(f"context overflow: exact raw count {raw['prompt_tokens']} exceeds {raw['prompt_limit']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--window", required=True, type=int)
    parser.add_argument("--reserve", required=True, type=int)
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--memory", choices=("raw", "packet", "typed"), required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--timeout", type=int, required=True)
    args = parser.parse_args()
    upstream = urlsplit(args.upstream)
    if (upstream.scheme != "http" or upstream.hostname not in {"127.0.0.1", "localhost", "::1"}
            or upstream.username or upstream.password or upstream.query or upstream.fragment):
        parser.error("session model requires a loopback HTTP endpoint")
    base = f"{upstream.scheme}://{upstream.netloc}"
    chat_path = upstream.path.rstrip("/") + "/chat/completions"
    opener = build_opener(ProxyHandler({}))
    parity_failed = False

    def forward(method, path, raw):
        request = Request(base + path, data=raw, method=method, headers={"Content-Type": "application/json"})
        try:
            with opener.open(request, timeout=args.timeout) as response:
                return response.status, response.read(16 * 1024 * 1024)
        except HTTPError as error:
            return error.code, error.read(8192)

    def emit_trace(receipt):
        with args.trace.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, separators=(",", ":")) + "\n")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, status, payload):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            nonlocal parity_failed
            if self.path != "/v1/chat/completions":
                self.respond(404, b"{}")
                return
            started = time.monotonic()
            receipt = {}
            try:
                if parity_failed:
                    raise PreflightError("provider/preflight parity failed earlier in this session")
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 8 * 1024 * 1024:
                    raise PreflightError("request exceeds relay size limit")
                request = json.loads(self.rfile.read(size))
                if request.get("stream"):
                    raise PreflightError("session preflight requires non-streaming inference")
                measured, receipt = prepare(forward, request, window=args.window, reserve=args.reserve,
                                           budget=args.budget, memory=args.memory)
                receipt["preflight_seconds"] = time.monotonic() - started
                provider_started = time.monotonic()
                status, raw = forward("POST", chat_path, json.dumps(measured, ensure_ascii=False).encode())
                receipt["provider_seconds"] = time.monotonic() - provider_started
                if status == 200:
                    usage = json.loads(raw).get("usage") or {}
                    reported = usage.get("prompt_tokens")
                    parity_failed = reported != receipt["preflight"]["prompt_tokens"]
                    receipt.update(provider_prompt_tokens=reported,
                                   provider_completion_tokens=usage.get("completion_tokens"),
                                   provider_preflight_parity=not parity_failed)
                receipt["provider_status"] = status
            except (PreflightError, ValueError, OSError) as exc:
                receipt["error"] = str(exc)
                status, raw = 422, json.dumps({"error": {"message": str(exc)}}).encode()
            emit_trace(receipt)
            self.respond(status, raw)

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        print(server.server_port, flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
