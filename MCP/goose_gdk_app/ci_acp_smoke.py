"""End-to-end ACP smoke: drive `goblin-gdk --acp` with the frames OpenAB, TitanAgent's harness, sends.

    python MCP/goose_gdk_app/ci_acp_smoke.py PATH/TO/goblin-gdk

The `initialize`, `session/new`, `session/prompt` and `session/cancel` frames copy
`openab-core/src/acp/connection.rs` (openabdev/openab). A stdlib HTTP server stands in for the
OpenAI-compatible model: it records each chat request and answers with a fixed message and usage
ceil(request bytes / 4) (a stand-in, labelled as such). A request whose last message contains
`[slow]` is held until the client hangs up, so a cancel lands during inference.

Four turns run in one session on this repository: the first two check the packet tool call, the
answer, usage and multi-turn history; the third is cancelled during inference; the fourth checks the
cancelled turn left no trace in the history. Then unsupported requests, the audit log, stdout
framing and a clean exit on stdin EOF are checked.
"""
from __future__ import annotations

import json
import math
import queue
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPLY = "Edit the function named in the packet."
TASK = ("certify_and_apply_verified accepts a proposal that removes a candidate whose metadata is unknown; "
        "candidate_refine in MCP/mcp02/env.py should keep it.")
REQUESTS: list[dict] = []
SLOW_STARTED = threading.Event()
RELEASE = threading.Event()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        request = json.loads(body)
        REQUESTS.append(request)
        if "[slow]" in json.dumps(request["messages"][-1]):
            SLOW_STARTED.set()
            RELEASE.wait(60)
        prompt = math.ceil(len(body) / 4)
        reply = {"id": "scripted", "object": "chat.completion", "model": request.get("model", "scripted"),
                 "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": REPLY}}],
                 "usage": {"prompt_tokens": prompt, "completion_tokens": 9, "total_tokens": prompt + 9}}
        data = json.dumps(reply).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass  # the agent hung up after a cancel


class Agent:
    """Newline-delimited JSON-RPC over the agent's stdio, as OpenAB's AcpConnection speaks it."""

    def __init__(self, cmd: list[str], stderr):
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                     text=True, encoding="utf-8", bufsize=1)
        self.frames: queue.Queue = queue.Queue()
        self.bad_lines: list[str] = []
        self.next_id = 0
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                self.bad_lines.append(line)
                continue
            if frame.get("jsonrpc") != "2.0":
                self.bad_lines.append(line)
            self.frames.put(frame)

    def send(self, frame: dict):
        self.proc.stdin.write(json.dumps(frame) + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict) -> int:
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params})
        return self.next_id

    def wait(self, request_id: int, notes: list, timeout: float = 300, until=None) -> dict:
        """Collect notifications into `notes` until the response to `request_id` arrives.
        `until(notes)` runs after each notification and may send frames (a cancel)."""
        deadline = time.monotonic() + timeout
        while True:
            frame = self.frames.get(timeout=max(0.1, deadline - time.monotonic()))
            if frame.get("id") == request_id and "method" not in frame:
                return frame
            notes.append(frame)
            if until:
                until(notes)

    def call(self, method: str, params: dict, timeout: float = 300) -> tuple[dict, list]:
        notes: list = []
        return self.wait(self.request(method, params), notes, timeout), notes


def updates(notes: list, kind: str) -> list[dict]:
    return [n["params"]["update"] for n in notes
            if n.get("method") == "session/update" and n["params"]["update"]["sessionUpdate"] == kind]


def text(prompt: str) -> list[dict]:
    return [{"type": "text", "text": prompt}]


def main() -> int:
    binary = sys.argv[1]
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    work = Path(tempfile.mkdtemp(prefix="pocket-goblin-acp-"))
    audit = work / "audit.jsonl"
    stderr = open(work / "agent.stderr", "w", encoding="utf-8")
    cmd = [binary, "--acp", "--base-url", f"http://127.0.0.1:{server.server_address[1]}/v1/", "--model", "scripted",
           "--python", sys.executable, "--goblin-cwd", str(ROOT), "--budget", "1024",
           "--agent-name", "goblin", "--audit-log", str(audit), "--history-turns", "4"]
    agent = Agent(cmd, stderr)
    checks: dict[str, bool] = {}
    first: dict = {}
    try:
        init, _ = agent.call("initialize", {"protocolVersion": 1, "clientCapabilities": {},
                                            "clientInfo": {"name": "openab", "version": "0.1.0"}}, timeout=30)
        result = init["result"]
        checks["initialize_protocol_1"] = result["protocolVersion"] == 1 and result["agentInfo"]["name"] == "goblin"
        checks["no_session_loading"] = result["agentCapabilities"]["loadSession"] is False

        new, _ = agent.call("session/new", {"cwd": str(ROOT), "mcpServers": []}, timeout=30)
        sid = new["result"]["sessionId"]

        first, notes = agent.call("session/prompt", {"sessionId": sid, "prompt": text(TASK)})
        started, finished = updates(notes, "tool_call"), updates(notes, "tool_call_update")
        chunks = updates(notes, "agent_message_chunk")
        receipt = first["result"]["_meta"]["pocketGoblin"]
        checks["packet_reported_as_search_tool_call"] = (
            len(started) == 1 and started[0]["kind"] == "search" and len(finished) == 1
            and finished[0]["status"] == "completed" and finished[0]["toolCallId"] == started[0]["toolCallId"]
            and 0 < finished[0]["rawOutput"]["packet_tokens"] <= 1024)
        checks["updates_carry_the_session_id"] = all(n["params"]["sessionId"] == sid for n in notes)
        checks["answer_streamed_before_result"] = [c["content"]["text"] for c in chunks] == [REPLY]
        checks["end_turn_with_provider_usage"] = (
            first["result"]["stopReason"] == "end_turn" and first["result"]["usage"]["outputTokens"] == 9
            and first["result"]["usage"]["inputTokens"] == receipt["provider_input_tokens"] > 0)
        checks["packet_reached_provider"] = "certify_and_apply_verified" in json.dumps(REQUESTS[-1])

        second, _ = agent.call("session/prompt", {"sessionId": sid, "prompt": text("Which test covers this?")})
        sent = json.dumps(REQUESTS[-1]["messages"])
        checks["second_turn_sees_first_turn"] = (
            second["result"]["_meta"]["pocketGoblin"]["history_messages"] == 2 and "candidate_refine" in sent
            and REPLY in sent)

        cancel_sent = []

        def cancel_once_inference_starts(notes: list) -> None:
            if updates(notes, "tool_call_update") and not cancel_sent and SLOW_STARTED.wait(120):
                agent.send({"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": sid}})
                cancel_sent.append(True)

        notes = []
        slow = agent.request("session/prompt", {"sessionId": sid, "prompt": text("[slow] keep digging")})
        began = time.monotonic()
        cancelled = agent.wait(slow, notes, until=cancel_once_inference_starts)
        RELEASE.set()
        checks["cancel_during_inference_stops_the_turn"] = (
            SLOW_STARTED.is_set() and cancelled["result"]["stopReason"] == "cancelled"
            and not updates(notes, "agent_message_chunk") and time.monotonic() - began < 120)

        fourth, _ = agent.call("session/prompt", {"sessionId": sid, "prompt": text("Summarize the fix.")})
        sent = json.dumps(REQUESTS[-1]["messages"])
        checks["cancelled_turn_left_no_history"] = (
            fourth["result"]["_meta"]["pocketGoblin"]["history_messages"] == 4 and "[slow]" not in sent
            and "Which test covers this?" in sent)

        load, _ = agent.call("session/load", {"sessionId": sid, "cwd": str(ROOT), "mcpServers": []}, timeout=30)
        stray, _ = agent.call("session/prompt", {"sessionId": "nope", "prompt": text("hi")}, timeout=30)
        checks["typed_errors"] = (load["error"]["code"] == -32601
                                  and stray["error"]["data"]["category"] == "session_not_found")

        agent.proc.stdin.close()
        checks["clean_exit_on_stdin_eof"] = agent.proc.wait(timeout=60) == 0
        checks["stdout_is_only_jsonrpc"] = not agent.bad_lines

        lines = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
        ops = [(line["op"], line["status"]) for line in lines]
        checks["audit_log_records_each_operation"] = (
            ops == [("retrieve", "ok"), ("infer", "ok")] * 2 + [("retrieve", "ok"), ("infer", "cancelled"),
                                                                ("retrieve", "ok"), ("infer", "ok")]
            and all(line["agent"] == "goblin" and line["session"] == sid for line in lines))
    finally:
        RELEASE.set()
        if agent.proc.poll() is None:
            agent.proc.kill()
        server.shutdown()
        stderr.close()
    print(json.dumps({"checks": checks, "first_turn": first.get("result")}, indent=1))
    if not all(checks.values()):
        print((work / "agent.stderr").read_text(encoding="utf-8")[-4000:], file=sys.stderr)
    return 0 if checks and all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
