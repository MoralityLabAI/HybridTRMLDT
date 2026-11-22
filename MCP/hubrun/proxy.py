"""OpenAI-compatible relay with a raw arm and a goblin packet arm.

Routes ``/<arm>/<tag>/v1/...`` to the upstream server. ``arm`` is ``raw``
(messages forwarded unchanged) or ``goblin`` (messages passed through
``compact_messages`` first). ``tag`` names the run in the call log. Every chat
call appends one JSON line to the log with the provider's exact token usage,
latency, finish reason, tool-call count and the compaction receipt.

The relay removes ``reasoning_content`` from replies before returning them, so
an environment and its later turns see the final answer only, as with a
hosted model that hides its reasoning. Its length is still logged. Both arms
are treated alike.

The upstream is reached either directly over HTTP or, from WSL, through the
Windows ``curl.exe`` (``--via-curl``): WSL in NAT mode cannot open Windows
loopback ports, but a Windows process started from WSL can.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from MCP.mcp02.textutil import digest
from .packet import compact_messages
from .mcp04 import PreflightError, WorkingSetError, adaptive_preflight, exact_preflight, typed_packet

ARMS = ("raw", "goblin", "mcp04", "mcp05")
STATUS_MARK = "\n__RELAY_STATUS__"


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Give every message string content and string tool-call arguments.

    Clients send ``content: null`` on assistant tool-call turns and ``null``
    for unset fields such as ``tool_calls`` or ``reasoning_content``, which
    llama-server's template parser rejects (HTTP 500). Null fields other than
    ``content`` are dropped. Applied to both arms.
    """
    out = []
    for message in messages:
        message = {k: v for k, v in message.items() if v is not None or k == "content"}
        if message.get("content") is None:
            message["content"] = ""
        calls = message.get("tool_calls")
        if calls:
            fixed = []
            for call in calls:
                call = dict(call)
                fn = dict(call.get("function") or {})
                if not isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.dumps(fn.get("arguments") or {})
                call["function"] = fn
                fixed.append(call)
            message["tool_calls"] = fixed
        out.append(message)
    return out


class Relay:
    def __init__(self, upstream: str, log_path: Path, via_curl: str | None, max_tokens_cap: int,
                 threshold: int, budget: int, keep_recent: int, per_message_cap: int, timeout: int,
                 strip_reasoning: bool = True, context_window: int = 32768,
                 completion_reserve: int = 4096, safety_margin: int = 1024):
        self.upstream = upstream.rstrip("/")
        self.strip_reasoning = strip_reasoning
        self.log_path = log_path
        self.via_curl = via_curl
        self.max_tokens_cap = max_tokens_cap
        self.compaction = {"threshold_tokens": threshold, "budget_tokens": budget, "keep_recent": keep_recent,
                           "per_message_cap": per_message_cap}
        self.timeout = timeout
        self.lock = threading.Lock()
        self.mcp04_context = {"context_window": context_window,
                              "completion_reserve": completion_reserve,
                              "safety_margin": safety_margin}
        self.mcp04_store: dict[str, dict[str, str]] = {}
        self.mcp04_parity_failure: str | None = None

    def forward(self, method: str, path: str, body: bytes | None) -> tuple[int, bytes]:
        url = self.upstream + path
        if self.via_curl:
            cmd = [self.via_curl, "-sS", "--max-time", str(self.timeout), "-X", method,
                   "-H", "Content-Type: application/json", "-w", STATUS_MARK + "%{http_code}", url]
            if body is not None:
                cmd[1:1] = ["--data-binary", "@-"]
            proc = subprocess.run(cmd, input=body or b"", capture_output=True, timeout=self.timeout + 30)
            raw = proc.stdout
            if STATUS_MARK.encode() not in raw:
                return 502, json.dumps({"error": {"message": "relay: " + proc.stderr.decode(errors="replace")[:500]}}).encode()
            payload, _, code = raw.rpartition(STATUS_MARK.encode())
            return int(code or b"502"), payload
        req = urllib.request.Request(url, data=body, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            return 502, json.dumps({"error": {"message": f"relay: {exc}"}}).encode()

    def chat(self, arm: str, tag: str, body: bytes) -> tuple[int, bytes]:
        if arm in ("mcp04", "mcp05") and self.mcp04_parity_failure:
            return 422, json.dumps({"error": {
                "type": "mcp04_template_parity_failed",
                "message": self.mcp04_parity_failure}}).encode()
        request = json.loads(body)
        messages = normalize_messages(request.get("messages", []))
        original_messages = messages
        request["messages"] = messages
        receipt = {"compacted": False}
        if arm == "goblin":
            messages, receipt = compact_messages(messages, **self.compaction)
            request["messages"] = messages
        elif arm in ("mcp04", "mcp05"):
            try:
                request, receipt = adaptive_preflight(
                    request, self.forward, **self.mcp04_context,
                    keep_recent=self.compaction["keep_recent"],
                    packet_budget=self.compaction["budget_tokens"],
                    store=self.mcp04_store.setdefault(tag, {}), policy=arm)
                messages = request["messages"]
            except (PreflightError, WorkingSetError, ValueError) as exc:
                err = {"error": {"type": f"{arm}_preflight_rejected", "message": str(exc)}}
                with self.lock, self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"arm": arm, "tag": tag, "status": 422,
                                         "error": str(exc), "forwarded": False}) + "\n")
                return 422, json.dumps(err).encode()
        cap = request.get("max_tokens") or request.get("max_completion_tokens")
        if not cap or cap > self.max_tokens_cap:
            request["max_tokens"] = self.max_tokens_cap
            request.pop("max_completion_tokens", None)
        sent = json.dumps(request).encode()
        start = time.time()
        status, payload = self.forward("POST", "/v1/chat/completions", sent)
        retry_failure = None
        if arm in ("mcp04", "mcp05") and status in (400, 413, 422):
            lower = payload.decode("utf-8", errors="replace").lower()
            if any(t in lower for t in ("context", "too many tokens", "exceed_context_size")):
                try:
                    shorter, compact_receipt = typed_packet(
                        original_messages, budget_tokens=min(self.compaction["budget_tokens"], 900),
                        keep_recent=1, store=self.mcp04_store.setdefault(tag, {}), policy=arm)
                    retry_request = dict(request, messages=shorter)
                    retry_preflight = exact_preflight(self.forward, retry_request, **self.mcp04_context)
                    if not retry_preflight["fits"]:
                        raise PreflightError("context retry still exceeds exact prompt limit")
                    request = retry_request
                    messages = shorter
                    receipt = dict(compact_receipt, preflight=retry_preflight,
                                   retried_after_provider_context_error=True)
                    sent = json.dumps(request).encode()
                    status, payload = self.forward("POST", "/v1/chat/completions", sent)
                except (PreflightError, WorkingSetError, ValueError) as exc:
                    retry_failure = str(exc)
        elapsed = time.time() - start
        record = {"ts": round(start, 3), "arm": arm, "tag": tag, "status": status, "latency_s": round(elapsed, 3),
                  "n_messages": len(messages), "request_bytes": len(sent), "request_digest": digest(request),
                  "tools": len(request.get("tools") or []), "compaction": receipt}
        if arm in ("mcp04", "mcp05"):
            record["preflight"] = receipt.get("preflight")
            record["retry_failure"] = retry_failure
            record["retried_after_context_error"] = receipt.get("retried_after_provider_context_error", False)
        try:
            data = json.loads(payload)
            usage = data.get("usage") or {}
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            if arm in ("mcp04", "mcp05") and isinstance(usage.get("prompt_tokens"), int):
                expected = receipt["preflight"]["prompt_tokens"]
                record["preflight_vs_provider_delta"] = usage["prompt_tokens"] - expected
                record["preflight_matches_provider"] = usage["prompt_tokens"] == expected
                if usage["prompt_tokens"] != expected:
                    # A completion already occurred, but prevent subsequent
                    # template-parity-unsafe inference in this relay process.
                    self.mcp04_parity_failure = (
                        f"provider prompt_tokens={usage['prompt_tokens']} versus "
                        f"applied template count={expected}; inspect pinned server chat formatting")
            record.update({"prompt_tokens": usage.get("prompt_tokens"),
                           "completion_tokens": usage.get("completion_tokens"),
                           "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                           "finish_reason": choice.get("finish_reason"),
                           "tool_calls": len(message.get("tool_calls") or []),
                           "reasoning_chars": len(message.get("reasoning_content") or ""),
                           "content_chars": len(message.get("content") or "")})
            if status >= 400:
                record["error"] = json.dumps(data)[:400]
            elif self.strip_reasoning:
                for item in data.get("choices") or []:
                    (item.get("message") or {}).pop("reasoning_content", None)
                payload = json.dumps(data).encode()
        except (json.JSONDecodeError, AttributeError):
            record["error"] = payload[:400].decode(errors="replace")
        with self.lock, self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return status, payload


def make_handler(relay: Relay):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _route(self) -> tuple[str, str, str] | None:
            parts = self.path.split("/", 3)
            if len(parts) < 4 or parts[1] not in ARMS:
                return None
            return parts[1], parts[2], "/" + parts[3]

        def _send(self, status: int, payload: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            route = self._route()
            if route is None:
                return self._send(404, b'{"error":"use /<raw|goblin|mcp04>/<tag>/v1/..."}')
            arm, _tag, rest = route
            if arm in ("mcp04", "mcp05") and rest.startswith("/v1/working-set/"):
                from urllib.parse import unquote
                handle = unquote(rest.removeprefix("/v1/working-set/"))
                body = relay.mcp04_store.get(_tag, {}).get(handle)
                if body is None:
                    return self._send(404, b'{"error":"unknown working-set handle"}')
                return self._send(200, json.dumps({"handle": handle, "content": body}).encode())
            self._send(*relay.forward("GET", rest, None))

        def do_POST(self) -> None:  # noqa: N802
            route = self._route()
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if route is None:
                return self._send(404, b'{"error":"use /<raw|goblin>/<tag>/v1/..."}')
            arm, tag, rest = route
            if rest.rstrip("/") == "/v1/chat/completions":
                return self._send(*relay.chat(arm, tag, body))
            self._send(*relay.forward("POST", rest, body))

        def log_message(self, fmt: str, *args) -> None:
            return

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=18130)
    ap.add_argument("--upstream", default="http://127.0.0.1:18127")
    ap.add_argument("--via-curl", default=None, help="path to Windows curl.exe when running in WSL")
    ap.add_argument("--log", required=True)
    ap.add_argument("--max-tokens-cap", type=int, default=4096)
    ap.add_argument("--threshold", type=int, default=6000)
    ap.add_argument("--budget", type=int, default=1800)
    ap.add_argument("--keep-recent", type=int, default=2)
    ap.add_argument("--per-message-cap", type=int, default=6000)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--context-window", type=int, default=32768)
    ap.add_argument("--completion-reserve", type=int, default=4096)
    ap.add_argument("--safety-margin", type=int, default=1024)
    ap.add_argument("--keep-reasoning", action="store_true",
                    help="return reasoning_content to the client (default: strip it; its length is still logged)")
    args = ap.parse_args()
    relay = Relay(args.upstream, Path(args.log), args.via_curl, args.max_tokens_cap, args.threshold, args.budget,
                  args.keep_recent, args.per_message_cap, args.timeout, strip_reasoning=not args.keep_reasoning,
                  context_window=args.context_window, completion_reserve=args.completion_reserve,
                  safety_margin=args.safety_margin)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(relay))
    print(json.dumps({"listening": args.port, "upstream": args.upstream, "via_curl": bool(args.via_curl),
                      "compaction": relay.compaction, "max_tokens_cap": args.max_tokens_cap}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
