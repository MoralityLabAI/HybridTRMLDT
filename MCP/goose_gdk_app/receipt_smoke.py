"""End-to-end smoke for receipt-goblin with a scripted loopback OpenAI-compatible endpoint.

    python MCP/goose_gdk_app/receipt_smoke.py PATH/TO/receipt-goblin

Writes a two-arm fixture (receipts JSON and a Markdown table whose goblin row carries the raw arm's
accepted count) and runs the app over every claim in the table, with values shown and withheld.
The scripted model links each claim to the packet leaf whose last path segment names the claim's
column and whose cell label shares the most words with the claim; its first answer of the run is
prose, so the certify step's repair path runs once. Usage is ceil(request bytes / 4), a stand-in.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUESTS: list[dict] = []
RECEIPTS = {"runs": [
    {"arm": "raw", "policy": "bm25", "accepted": 12, "tool_calls_mean": 31.5},
    {"arm": "goblin", "policy": "looped", "accepted": 27, "tool_calls_mean": 6.25},
]}
DOC = ("# Fixture\n\nAccepted tasks and mean tool calls per arm.\n\n| Arm | Accepted | Tool calls |\n| --- | --- | --- |\n"
       "| raw | 12 | 31.5 |\n| goblin | 12 | 6.25 |\n")
EXPECTED = {("raw", "12"): "supported", ("raw", "31.5"): "supported",
            ("goblin", "12"): "contradicted", ("goblin", "6.25"): "supported"}
WORD = re.compile(r"[a-z]+")


def text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def role_text(request: dict, role: str) -> list[str]:
    return [text_of(m) for m in request["messages"] if m.get("role") == role]


def scripted_answer(request: dict) -> str:
    if len(REQUESTS) == 1:
        return "I am not sure which line it is."
    system = role_text(request, "system")[0]
    user = role_text(request, "user")[0]
    claim = user.split("\n")[0]
    column = WORD.findall(claim.split("⟦")[0].rsplit(";", 1)[-1].rsplit("|", 1)[-1].lower())
    words = set(WORD.findall(claim.lower()))
    best, best_score = "NONE", -1
    for line in system.splitlines():
        match = re.match(r"(L\d+) (\S+) :: (\S+(?: \S+)*?)(?: = \S+)?$", line)
        if not match:
            continue
        path = match.group(3)
        last = path.rsplit("/", 1)[-1].lower()
        if not any(word in last for word in column):
            continue
        score = len(set(WORD.findall(path.lower())) & words)
        if score > best_score:
            best, best_score = match.group(1), score
    return json.dumps({"leaf": best})


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
                              "message": {"role": "assistant", "content": scripted_answer(request)}}],
                 "usage": {"prompt_tokens": prompt, "completion_tokens": 7, "total_tokens": prompt + 7}}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run(binary: str, port: int, work: Path, out: Path, *extra: str) -> list[dict]:
    cmd = [binary, "--doc", str(work / "doc.md"), "--receipts", str(work / "receipts"), "--out", str(out),
           "--base-url", f"http://127.0.0.1:{port}/v1/", "--model", "scripted", "--python", sys.executable,
           "--server-cwd", str(ROOT), *extra]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if done.returncode != 0:
        raise SystemExit(f"receipt-goblin failed ({done.returncode}): {done.stderr[-2000:]}")
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    binary = sys.argv[1]
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "receipts").mkdir()
        (work / "receipts" / "summary.json").write_text(json.dumps(RECEIPTS), encoding="utf-8")
        (work / "doc.md").write_text(DOC, encoding="utf-8")
        try:
            shown = run(binary, port, work, work / "shown.jsonl", "--arm", "shown")
            n_shown = len(REQUESTS)
            blind = run(binary, port, work, work / "blind.jsonl", "--arm", "blind", "--blind")
            resumed = run(binary, port, work, work / "blind.jsonl", "--arm", "blind", "--blind")
        finally:
            server.shutdown()
    by_row = lambda rows: {("goblin" if ":8:" in r["id"] else "raw", r["target"]): r for r in rows}
    shown_rows, blind_rows = by_row(shown), by_row(blind)
    first_system = role_text(REQUESTS[0], "system")[0]
    blind_system = role_text(REQUESTS[n_shown], "system")[0]
    checks = {
        "four_claims_each_arm": len(shown_rows) == 4 and len(blind_rows) == 4,
        "packet_reached_provider_as_prompt_part": "arm=goblin" in first_system and "\nL01 " in first_system,
        "verdicts_shown": {k: shown_rows[k]["verdict"] for k in EXPECTED} == EXPECTED,
        "verdicts_blind": {k: blind_rows[k]["verdict"] for k in EXPECTED} == EXPECTED,
        "blind_packet_withholds_values": " = " not in "\n".join(l for l in blind_system.splitlines() if l.startswith("L")),
        "repair_ran_once": sum(r["repairs"] for r in shown) == 1 and sum(r["inference_calls"] for r in shown) == 5,
        "usage_recorded": all(r["provider_input_tokens"] > 0 for r in shown + blind),
        "resume_skips_done_claims": len(resumed) == 4 and len(REQUESTS) == n_shown + 4,
    }
    print(json.dumps({"checks": checks, "shown": list(shown_rows.values())[:1]}, indent=1, default=str))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
