"""End-to-end smoke for goblin-gdk with a scripted loopback OpenAI-compatible endpoint.

    python MCP/goose_gdk_app/ci_smoke.py PATH/TO/goblin-gdk

Starts a stdlib HTTP server that records each chat request and answers with a fixed message and
usage computed as ceil(request bytes / 4) (a stand-in, labelled as such). Runs the app twice on this
repository, with and without the compact packet, and checks that the packet reached the provider
request as a prompt part and that the receipts carry the goblin accounting.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUESTS: list[dict] = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        request = json.loads(body)
        REQUESTS.append(request)
        prompt = math.ceil(len(body) / 4)
        reply = {"id": "scripted", "object": "chat.completion", "model": request.get("model", "scripted"),
                 "choices": [{"index": 0, "finish_reason": "stop",
                              "message": {"role": "assistant", "content": "Edit the function named in the packet."}}],
                 "usage": {"prompt_tokens": prompt, "completion_tokens": 9, "total_tokens": prompt + 9}}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run(binary: str, port: int, *extra: str) -> dict:
    task = ("certify_and_apply_verified accepts a proposal that removes a candidate whose metadata is unknown; "
            "candidate_refine in MCP/mcp02/env.py should keep it.")
    cmd = [binary, "--task", task, "--base-url", f"http://127.0.0.1:{port}/v1/", "--model", "scripted",
           "--python", sys.executable, "--goblin-cwd", str(ROOT), *extra]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise SystemExit(f"goblin-gdk failed ({out.returncode}): {out.stderr[-2000:]}")
    return json.loads(out.stdout)


def main() -> int:
    binary = sys.argv[1]
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    with_packet = run(binary, port, "--repo", str(ROOT), "--budget", "1024")
    n_with = len(REQUESTS)
    without = run(binary, port, "--no-packet")
    sent = json.dumps(REQUESTS[n_with - 1])
    stats = with_packet["packet_stats"]
    checks = {
        "packet_reached_provider": "certify_and_apply_verified" in sent,
        "no_packet_run_has_no_packet": "retrieved_code_context" not in json.dumps(REQUESTS[-1]),
        "packet_within_budget": 0 < stats["packet_tokens"] <= 1024,
        "router_read_more_than_it_sent": stats["local_result_tokens"] > stats["packet_tokens"],
        "usage_recorded": with_packet["provider_input_tokens"] > without["provider_input_tokens"] > 0,
    }
    # Exercise the provider boundary with a long, durable two-turn history.
    # Only the inference view may omit those messages.
    with tempfile.TemporaryDirectory(prefix="gdk01-") as tmp:
        history = Path(tmp) / "history.json"
        history.write_text(json.dumps([
            {"role": "user", "text": "OLD_SENTINEL " + "archived content " * 240},
            {"role": "assistant", "text": "OLD_ANSWER " + "large result " * 240},
        ]), encoding="utf-8")
        unfiltered = run(binary, port, "--no-packet", "--history-file", str(history))
        unfiltered_request = json.dumps(REQUESTS[-1])
        filtered = run(binary, port, "--no-packet", "--history-file", str(history),
                       "--lean-context", "--lean-request-byte-limit", "6500",
                       "--lean-history-turns", "0")
        filtered_request = json.dumps(REQUESTS[-1])
        before_block = len(REQUESTS)
        # The Rust provider wrapper is expected to fail before any HTTP request.
        blocked = subprocess.run([
            binary, "--task", "tiny", "--no-packet",
            "--base-url", f"http://127.0.0.1:{port}/v1/", "--model", "scripted",
            "--lean-context", "--lean-request-byte-limit", "8",
        ], capture_output=True, text=True, timeout=600)
        checks.update({
            "baseline_preserves_old_history": "OLD_SENTINEL" in unfiltered_request,
            "lean_omits_prior_history": "OLD_SENTINEL" not in filtered_request
                                        and "OLD_ANSWER" not in filtered_request,
            "lean_retains_current_task": "certify_and_apply_verified" in filtered_request,
            "lean_reduces_provider_bytes": len(filtered_request) < len(unfiltered_request),
            "lean_durable_history_retained": filtered["session_message_count"] ==
                                            unfiltered["session_message_count"],
            "lean_receipt_reports_omission": filtered["lean_request"]["omitted_messages"] >= 2,
            "lean_receipt_within_byte_gate":
                filtered["lean_request"]["serialized_input_bytes"] <= 6500,
            "over_budget_blocks_http": blocked.returncode != 0
                                       and len(REQUESTS) == before_block,
        })
    server.shutdown()
    print(json.dumps({"checks": checks, "with_packet": with_packet, "without_packet": without,
                      "unfiltered": unfiltered, "lean": filtered}, indent=1))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
