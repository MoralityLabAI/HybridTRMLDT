"""End-to-end ACP smoke: drive `goblin-gdk --acp` with the frames OpenAB, TitanAgent's harness, sends.

    python MCP/goose_gdk_app/ci_acp_smoke.py PATH/TO/goblin-gdk

The `initialize`, `session/new`, `session/prompt` and `session/cancel` frames, and the answers to
the agent's `session/request_permission` requests, copy `crates/openab-core/src/acp/connection.rs`
(openabdev/openab). A stdlib HTTP server stands in for the
OpenAI-compatible model: it records each chat request and answers with a fixed message and usage
ceil(request bytes / 4) (a stand-in, labelled as such). A request whose last message contains
`[slow]` is held until the client hangs up, so a cancel lands during inference.

Four turns run in one session on this repository: the first two check the packet tool call, the
answer, usage and multi-turn history; the third is cancelled during inference; the fourth checks the
cancelled turn left no trace in the history. Then unsupported requests, the audit log, stdout
framing and a clean exit on stdin EOF are checked.

A second agent runs with the repair-session flags against a verifier manifest on a fresh copy of
the MCP-05 gym fixture (`fall` subtracts gravity; a frozen test expects it to add). The scripted
model searches, reads, proposes the fix twice and runs the verifier, through the MCP-05 preflight
relay (the stand-in serves `/apply-template` and `/tokenize` so the relay's exact count matches the
reported usage). The client answers `session/request_permission` as OpenAB does, except that it
rejects the first write. Checked: each repair tool call is a `tool_call`/`tool_call_update` pair of
the right kind, permission is asked once per edit, the rejected edit is a tool error, the approved
edit is applied and verified, the audit log has an `op: write` line with the path per edit, and a
second repair prompt on the now-dirty worktree is a typed `repair_error`.
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
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPLY = "Edit the function named in the packet."
TASK = ("certify_and_apply_verified accepts a proposal that removes a candidate whose metadata is unknown; "
        "candidate_refine in MCP/mcp02/env.py should keep it.")
REQUESTS: list[dict] = []
SLOW_STARTED = threading.Event()
RELEASE = threading.Event()
REPAIR_MODEL = "scripted-repair"
REPAIR_TASK = "The player falls upward. Correct vertical gravity in physics."


def template(request: dict) -> str:
    """The stand-in chat template the MCP-05 relay counts tokens with: messages and tools as
    canonical JSON at four characters per token, as in MCP/mcp05/test_mcp05.py."""
    return json.dumps({"messages": request["messages"], "tools": request.get("tools", [])}, sort_keys=True)


def repair_reply(request: dict) -> dict:
    """The scripted repair model: search, read `fall`, propose the fix (the client rejects this
    first proposal), propose it again, run the verifier. Usage is the stand-in template count, so
    the relay sees provider/preflight parity."""
    results = [m for m in request["messages"] if m["role"] == "tool"]
    step = len(results)
    if step == 0:
        call = ("context_query", {"query": "fall gravity"})
    elif step == 1:
        call = ("context_read", {"uri": "engine/physics.py::fall"})
    elif step in (2, 3):
        handle = json.loads(results[1]["content"])["handle"]
        call = ("apply_edit", {"handle": handle, "old": "return speed - gravity", "new": "return speed + gravity"})
    elif step == 4:
        call = ("run_verifier", {})
    else:
        call = None
    message = {"role": "assistant", "content": "Done." if call is None else None}
    if call:
        message["tool_calls"] = [{"id": f"repair_{step}", "type": "function",
                                  "function": {"name": call[0], "arguments": json.dumps(call[1])}}]
    prompt = len(template(request)) // 4
    return {"id": f"repair_{step}", "object": "chat.completion", "model": REPAIR_MODEL,
            "choices": [{"index": 0, "finish_reason": "tool_calls" if call else "stop", "message": message}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": 23, "total_tokens": prompt + 23}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        request = json.loads(body)
        if self.path == "/apply-template":
            return self.reply({"prompt": template(request)})
        if self.path == "/tokenize":
            return self.reply({"tokens": list(range(len(request["content"]) // 4))})
        REQUESTS.append(request)
        if request.get("model") == REPAIR_MODEL:
            return self.reply(repair_reply(request))
        if "[slow]" in json.dumps(request["messages"][-1]):
            SLOW_STARTED.set()
            RELEASE.wait(60)
        prompt = math.ceil(len(body) / 4)
        self.reply({"id": "scripted", "object": "chat.completion", "model": request.get("model", "scripted"),
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": REPLY}}],
                    "usage": {"prompt_tokens": prompt, "completion_tokens": 9, "total_tokens": prompt + 9}})

    def reply(self, value: dict) -> None:
        data = json.dumps(value).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass  # the agent hung up after a cancel


def openab_answer(params: dict | None) -> dict:
    """OpenAB's answer to session/request_permission (build_permission_response in
    crates/openab-core/src/acp/connection.rs): the most permissive allow option, else the first
    non-reject option, else cancelled."""
    options = (params or {}).get("options")
    if options is None:
        return {"outcome": {"outcome": "selected", "optionId": "allow_always"}}
    for kind in ("allow_always", "allow_once"):
        for option in options:
            if option.get("kind") == kind:
                return {"outcome": {"outcome": "selected", "optionId": option["optionId"]}}
    for option in options:
        if option.get("kind") not in ("reject_once", "reject_always"):
            return {"outcome": {"outcome": "selected", "optionId": option["optionId"]}}
    return {"outcome": {"outcome": "cancelled"}}


def openab_reply(request: dict) -> dict:
    """OpenAB's reply to an agent-to-client request (run_reader_loop): permission requests are
    answered, anything else gets -32601."""
    if request["method"] == "session/request_permission":
        return {"jsonrpc": "2.0", "id": request["id"], "result": openab_answer(request.get("params"))}
    return {"jsonrpc": "2.0", "id": request["id"],
            "error": {"code": -32601, "message": f"unsupported agent request method `{request['method']}`"}}


class Agent:
    """Newline-delimited JSON-RPC over the agent's stdio, as OpenAB's AcpConnection speaks it.
    `on_request` answers the agent's own requests; the default replies as OpenAB does."""

    def __init__(self, cmd: list[str], stderr, on_request=openab_reply):
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                     text=True, encoding="utf-8", bufsize=1)
        self.frames: queue.Queue = queue.Queue()
        self.bad_lines: list[str] = []
        self.next_id = 0
        self.on_request = on_request
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
        """Collect notifications and agent requests into `notes` until the response to
        `request_id` arrives, answering each agent request with `on_request`.
        `until(notes)` runs after each frame and may send frames (a cancel)."""
        deadline = time.monotonic() + timeout
        while True:
            frame = self.frames.get(timeout=max(0.1, deadline - time.monotonic()))
            if frame.get("id") == request_id and "method" not in frame:
                return frame
            if "method" in frame and "id" in frame:
                self.send(self.on_request(frame))
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


INITIALIZE = {"protocolVersion": 1, "clientCapabilities": {}, "clientInfo": {"name": "openab", "version": "0.1.0"}}


def one_shot_phase(binary: str, port: int, work: Path, checks: dict) -> dict:
    """Four packet-and-inference turns in one session on this repository."""
    audit = work / "audit.jsonl"
    stderr = open(work / "agent.stderr", "w", encoding="utf-8")
    cmd = [binary, "--acp", "--base-url", f"http://127.0.0.1:{port}/v1/", "--model", "scripted",
           "--python", sys.executable, "--goblin-cwd", str(ROOT), "--budget", "1024",
           "--agent-name", "goblin", "--audit-log", str(audit), "--history-turns", "4"]
    agent = Agent(cmd, stderr)
    first: dict = {}
    try:
        init, _ = agent.call("initialize", INITIALIZE, timeout=30)
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
        stderr.close()
    return first.get("result") or {}


def commit_fixture(root: Path) -> None:
    """The MCP-05 gym fixture of MCP/mcp05/test_mcp05.py, committed: `fall` subtracts gravity and a
    frozen test expects it to add."""
    (root / "engine").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "engine/physics.py").write_bytes(b"def fall(speed, gravity):\n    return speed - gravity\n")
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


def repair_phase(binary: str, port: int, work: Path, checks: dict) -> dict:
    """One repair turn that edits and verifies, then a second prompt on the dirty worktree."""
    sys.path.insert(0, str(ROOT))
    from MCP.mcp05.gym import verifier_manifest

    fixture = work / "fixture"
    commit_fixture(fixture)
    manifest = work / "verifier.json"
    manifest.write_text(json.dumps(verifier_manifest(sys.executable)), encoding="utf-8")
    audit = work / "repair-audit.jsonl"
    asks: list[dict] = []

    def review(request: dict) -> dict:
        """Reject the first write, then answer as OpenAB does."""
        if request["method"] != "session/request_permission":
            return openab_reply(request)
        asks.append(request)
        if len(asks) > 1:
            return openab_reply(request)
        reject = next(o["optionId"] for o in request["params"]["options"] if o["kind"] == "reject_once")
        return {"jsonrpc": "2.0", "id": request["id"], "result": {"outcome": {"outcome": "selected", "optionId": reject}}}

    stderr = open(work / "repair.stderr", "w", encoding="utf-8")
    cmd = [binary, "--acp", "--agent-name", "fixer", "--audit-log", str(audit),
           "--base-url", f"http://127.0.0.1:{port}/v1/", "--model", REPAIR_MODEL,
           "--python", sys.executable, "--goblin-cwd", str(ROOT), "--no-packet",
           "--max-agent-iterations", "8", "--memory-mode", "typed", "--max-tool-calls", "10",
           "--verifier", str(manifest)]
    agent = Agent(cmd, stderr, on_request=review)
    summary: dict = {}
    try:
        agent.call("initialize", INITIALIZE, timeout=30)
        new, _ = agent.call("session/new", {"cwd": str(fixture), "mcpServers": []}, timeout=30)
        sid = new["result"]["sessionId"]

        repair, notes = agent.call("session/prompt", {"sessionId": sid, "prompt": text(REPAIR_TASK)})
        result = repair["result"]
        receipt = result["_meta"]["pocketGoblin"]
        started, finished = updates(notes, "tool_call"), updates(notes, "tool_call_update")
        edits = [update["toolCallId"] for update in started if update["kind"] == "edit"]
        summary = {"stopReason": result["stopReason"], "usage": result["usage"], "answer": receipt["answer"],
                   "verified": receipt["state"]["verified"], "inference_calls": receipt["inference_calls"],
                   "tool_calls": [(update["kind"], update["title"]) for update in started],
                   "statuses": [update["status"] for update in finished]}
        checks["repair_tool_calls_paired_with_kinds"] = (
            [update["kind"] for update in started] == ["search", "read", "edit", "edit", "execute"]
            and [update["toolCallId"] for update in finished] == [update["toolCallId"] for update in started]
            and len(set(edits)) == 2 and all(note["params"]["sessionId"] == sid for note in notes))
        checks["permission_asked_once_per_edit"] = (
            [ask["params"]["toolCall"]["toolCallId"] for ask in asks] == edits
            and all(ask["params"]["toolCall"]["locations"][0]["path"].endswith("physics.py")
                    and ask["params"]["toolCall"]["content"][0]["type"] == "diff" for ask in asks))
        tool_results = [m["content"] for m in REQUESTS[-1]["messages"] if m["role"] == "tool"]
        checks["rejected_edit_is_a_tool_error"] = (
            [update["status"] for update in finished] == ["completed", "completed", "failed", "completed", "completed"]
            and "rejected" in finished[2]["rawOutput"]["error"] and "rejected" in tool_results[2])
        physics = (fixture / "engine/physics.py").read_text(encoding="utf-8")
        checks["approved_edit_applied_and_verified"] = (
            "return speed + gravity" in physics and receipt["state"]["verified"] is True
            and receipt["state"]["verifier_calls"] == 1 and finished[4]["rawOutput"]["passed"] is True)
        chunks = [chunk["content"]["text"] for chunk in updates(notes, "agent_message_chunk")]
        checks["repair_answer_and_provider_usage"] = (
            result["stopReason"] == "end_turn" and chunks == [receipt["answer"]] and receipt["answer"]
            and result["usage"]["inputTokens"] == receipt["provider_input_tokens"] > 0
            and all(call["provider_preflight_parity"] for call in receipt["preflight"]))

        again, _ = agent.call("session/prompt", {"sessionId": sid, "prompt": text("Also clamp fall at zero.")},
                              timeout=120)
        checks["dirty_worktree_prompt_is_a_typed_error"] = again.get("error", {}).get("data", {}).get(
            "category") == "repair_error"

        agent.proc.stdin.close()
        checks["repair_agent_exits_cleanly_with_only_jsonrpc"] = agent.proc.wait(timeout=60) == 0 and not agent.bad_lines
        lines = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
        checks["audit_log_has_a_write_line_with_the_path_per_edit"] = (
            [(line["op"], line["status"]) for line in lines]
            == [("write", "rejected"), ("write", "ok"), ("verify", "ok"), ("repair", "ok"), ("repair", "failed")]
            and all(line["path"] == "engine/physics.py" for line in lines if line["op"] == "write")
            and all(line["agent"] == "fixer" and line["session"] == sid for line in lines))
    finally:
        if agent.proc.poll() is None:
            agent.proc.kill()
        stderr.close()
    return summary


def main() -> int:
    binary = sys.argv[1]
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    work = Path(tempfile.mkdtemp(prefix="pocket-goblin-acp-"))
    checks: dict[str, bool] = {}
    first: dict = {}
    repair: dict = {}
    try:
        first = one_shot_phase(binary, port, work, checks)
        repair = repair_phase(binary, port, work, checks)
    except Exception:  # an unexpected frame shape: report it with the agents' stderr below
        traceback.print_exc()
        checks["ran_to_completion"] = False
    finally:
        RELEASE.set()
        server.shutdown()
    print(json.dumps({"checks": checks, "first_turn": first, "repair_turn": repair}, indent=1))
    if not checks or not all(checks.values()):
        for log in ("agent.stderr", "repair.stderr"):
            if (work / log).exists():
                print(f"--- {log}\n" + (work / log).read_text(encoding="utf-8")[-4000:], file=sys.stderr)
    return 0 if checks and all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
