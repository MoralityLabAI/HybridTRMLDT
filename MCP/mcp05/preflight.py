"""Preflight the exact Goose provider request immediately before inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from MCP.hubrun.mcp04 import PreflightError, WorkingSetError, exact_preflight, typed_packet
from MCP.hubrun.packet import compact_messages


class ProviderPrefix:
    """Reuse an episode's ordered, exactly matched completions after interruption."""

    def __init__(self, path: Path):
        self.path = path
        self.records = json.loads(path.read_bytes()) if path.exists() else []
        self.turn = 0

    def match(self, request_key: str):
        if self.turn < len(self.records):
            record = self.records[self.turn]
            if record["request_key"] == request_key:
                return record
        return None

    def record(self, request_key: str, response: dict, seconds: float):
        # Keep application-facing IDs unique even when a fresh worker restarts its counter.
        response["id"] = f"chatcmpl-mcp05-{self.turn}"
        for choice_index, choice in enumerate(response.get("choices", [])):
            calls = choice.get("message", {}).get("tool_calls") or []
            for call_index, call in enumerate(calls):
                call["id"] = f"call_mcp05_{self.turn}_{choice_index}_{call_index}"
        records = [*self.records[:self.turn],
                   {"request_key": request_key, "response": response, "provider_seconds": seconds}]
        pending = self.path.with_suffix(".tmp")
        pending.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        pending.replace(self.path)
        self.records = records


def measure(forward, request, *, window: int, reserve: int):
    return exact_preflight(forward, request, context_window=window,
                           completion_reserve=reserve, safety_margin=64)


def deduplicate_read_sources(messages: list[dict]) -> tuple[list[dict], int]:
    updated = list(messages)
    seen = set()
    removed = 0
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        content = message.get("content")
        if message.get("role") != "tool" or not isinstance(content, str):
            continue
        try:
            body = json.loads(content)
        except ValueError:
            continue
        if not isinstance(body, dict):
            continue
        identity = tuple(body.get(key) for key in ("handle", "source_digest", "revision", "source"))
        if not all(isinstance(value, str) and value for value in identity) or not identity[0].startswith("source:"):
            continue
        if identity in seen:
            del body["source"]
            body["source_in_later_result"] = True
            updated[index] = dict(message, content=json.dumps(body, ensure_ascii=False, separators=(",", ":")))
            removed += 1
        else:
            seen.add(identity)
    return updated, removed


def compact_create_history(messages: list[dict]) -> tuple[list[dict], int]:
    """Keep the completed request and tool receipt while dropping old authored bytes from the next prompt."""
    results = {m.get("tool_call_id"): m for m in messages if m.get("role") == "tool"}
    updated = []
    removed = 0
    for message in messages:
        calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            if function.get("name") == "create_file" and call.get("id") in results:
                try:
                    args = json.loads(function["arguments"])
                    source = args["content"]
                    if isinstance(source, str) and len(source) > 150:
                        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
                        args["content"] = f"[Prior proposal archived; SHA-256 {digest}. See the tool result for acceptance.]"
                        function = dict(function, arguments=json.dumps(args, ensure_ascii=False, separators=(",", ":")))
                        call = dict(call, function=function)
                        removed += 1
                except (KeyError, ValueError, TypeError):
                    pass
            calls.append(call)
        updated.append(dict(message, tool_calls=calls) if calls else message)
    return updated, removed


def prepare(forward, request, *, window: int, reserve: int, budget: int, memory: str, deduplicate_reads: bool = False,
            compact_writes: bool = False):
    raw = measure(forward, request, window=window, reserve=reserve)
    if compact_writes:
        messages, removed = compact_create_history(request["messages"])
        if removed:
            candidate = dict(request, messages=messages)
            checked = measure(forward, candidate, window=window, reserve=reserve)
            if checked["fits"] and checked["prompt_tokens"] < raw["prompt_tokens"]:
                return candidate, {"preflight": checked, "compacted": True, "method": "archive-create-content",
                                   "raw_prompt_tokens": raw["prompt_tokens"], "create_sources_archived": removed}
    if deduplicate_reads and memory == "typed":
        messages, removed = deduplicate_read_sources(request["messages"])
        if removed:
            candidate = dict(request, messages=messages)
            checked = measure(forward, candidate, window=window, reserve=reserve)
            if checked["fits"] and checked["prompt_tokens"] < raw["prompt_tokens"]:
                return candidate, {"preflight": checked, "compacted": True, "method": "duplicate-source-reads",
                                   "raw_prompt_tokens": raw["prompt_tokens"], "duplicate_sources_removed": removed}
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
    parser.add_argument("--deduplicate-reads", action="store_true")
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
    prefix_file = os.environ.get("MCP05_EPISODE_PREFIX_PATH")
    prefix = ProviderPrefix(Path(prefix_file)) if prefix_file else None
    wire_log = os.environ.get("MCP05_WIRE_LOG")
    fresh_prompt = os.environ.get("MCP05_FRESH_PROMPT") == "1"
    compact_writes = os.environ.get("MCP05_COMPACT_CREATE_HISTORY") == "1"

    def forward(method, path, raw):
        headers = {"Content-Type": "application/json"}
        if path == chat_path:
            # Leave time for an upstream deadline response before the relay closes the socket.
            headers["X-Inkling-Timeout-Seconds"] = str(max(args.timeout - 2, args.timeout / 2))
        request = Request(base + path, data=raw, method=method, headers=headers)
        try:
            with opener.open(request, timeout=args.timeout) as response:
                return response.status, response.read(16 * 1024 * 1024)
        except HTTPError as error:
            return error.code, error.read(8192)

    def emit_trace(receipt):
        with args.trace.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, separators=(",", ":")) + "\n")

    def emit_wire(event, body, status=None):
        if wire_log:
            with Path(wire_log).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"event": event, "status": status,
                                         "body": body.decode("utf-8", errors="replace")}) + "\n")

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
            provider_body = None
            try:
                if parity_failed:
                    raise PreflightError("provider/preflight parity failed earlier in this session")
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 8 * 1024 * 1024:
                    raise PreflightError("request exceeds relay size limit")
                request = json.loads(self.rfile.read(size))
                if request.get("stream"):
                    raise PreflightError("session preflight requires non-streaming inference")
                if fresh_prompt:
                    request["cache_prompt"] = False
                measured, receipt = prepare(forward, request, window=args.window, reserve=args.reserve,
                                           budget=args.budget, memory=args.memory,
                                           deduplicate_reads=args.deduplicate_reads, compact_writes=compact_writes)
                receipt["preflight_seconds"] = time.monotonic() - started
                body = json.dumps(measured, ensure_ascii=False).encode()
                emit_wire("request", body)
                saved = None
                if prefix is not None:
                    template = receipt["preflight"]["formatted_prompt_sha256"].encode()
                    request_key = hashlib.sha256(template + b"\0" + body).hexdigest()
                    saved = prefix.match(request_key)
                provider_started = time.monotonic()
                if saved is None:
                    status, raw = forward("POST", chat_path, body)
                    provider_body = raw
                else:
                    status, raw = 200, json.dumps(saved["response"], ensure_ascii=False).encode()
                    receipt.update(provider_replayed=True, original_provider_seconds=saved["provider_seconds"])
                receipt["provider_seconds"] = time.monotonic() - provider_started
                if status == 200:
                    response = json.loads(raw)
                    usage = response.get("usage") or {}
                    reported = usage.get("prompt_tokens")
                    parity_failed = reported != receipt["preflight"]["prompt_tokens"]
                    receipt.update(provider_prompt_tokens=reported,
                                   provider_completion_tokens=usage.get("completion_tokens"),
                                   provider_preflight_parity=not parity_failed)
                    if prefix is not None and not parity_failed:
                        if saved is None:
                            prefix.record(request_key, response, receipt["provider_seconds"])
                            raw = json.dumps(response, ensure_ascii=False).encode()
                        prefix.turn += 1
                receipt["provider_status"] = status
            except (PreflightError, ValueError, OSError) as exc:
                receipt["error"] = str(exc)
                status, raw = 422, json.dumps({"error": {"message": str(exc)}}).encode()
            emit_trace(receipt)
            emit_wire("response", raw if provider_body is None else provider_body, status)
            self.respond(status, raw)

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        print(server.server_port, flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
