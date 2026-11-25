"""Three-profile GDK-02 integration smoke, using scripted backend counters.

The Rust app must probe GDK, retrieve MCP-03 context, persist the captured
packet and send its chat request through the exact gate for each model profile.
No actual model weights are loaded by this test.
"""
from __future__ import annotations

import http.server
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from MCP.GDK.token_gate import probe
from MCP.GDK.working_set import WorkingSetStore


class Backend(http.server.BaseHTTPRequestHandler):
    calls: list[tuple[str, dict]] = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        req = json.loads(body)
        self.calls.append((self.path, req))
        if self.path == "/v1/chat/completions/input_tokens":
            answer = {"input_tokens": 320}
        elif self.path == "/v1/chat/completions/render":
            answer = {"token_ids": list(range(320))}
        elif self.path == "/v1/chat/completions":
            answer = {
                "id": "mock", "object": "chat.completion", "model": req["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "Recorded"}}],
                "usage": {"prompt_tokens": 320, "completion_tokens": 1, "total_tokens": 321},
            }
        else:
            self.send_error(404)
            return
        buf = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(buf)))
        self.end_headers()
        self.wfile.write(buf)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run(binary: str, profile: str, backend_port: int, work: str):
    port = free_port()
    proxy = subprocess.Popen([
        sys.executable, "-m", "MCP.GDK.token_gate", "serve",
        "--profile", profile, "--served-model-id", "scripted",
        "--backend-url", f"http://127.0.0.1:{backend_port}",
        "--backend-context-tokens", "12288", "--port", str(port),
        "--receipt-file", str(Path(work) / f"{profile}.jsonl"),
    ], cwd=ROOT, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    try:
        origin = f"http://127.0.0.1:{port}/v1/"
        ready = False
        for _ in range(100):
            if proxy.poll() is not None:
                raise AssertionError(f"{profile} token proxy exited: {proxy.stderr.read()}")
            try:
                probe(origin, profile, "scripted")
                ready = True
                break
            except Exception:
                time.sleep(0.05)
        if not ready:
            raise AssertionError(f"{profile} gate did not become healthy")
        req = [
            binary, "--model", "scripted", "--task", "Locate candidate_refine",
            "--repo", str(ROOT), "--budget", "512", "--max-tokens", "64",
            "--goblin-cwd", str(ROOT), "--python", sys.executable,
            "--base-url", origin, "--lean-profile", profile,
            "--session-id", f"gdk02-{profile}", "--working-set-root", work,
        ]
        output = subprocess.run(req, cwd=ROOT, text=True, capture_output=True, timeout=300)
        if output.returncode:
            raise AssertionError(f"{profile} Rust GDK failed: {output.stderr[-3000:]}")
        result = json.loads(output.stdout)
        assert result["exact_gate"]["profile"] == profile
        assert result["working_set"]["handle"].startswith("sha256:")
        assert result["provider_input_tokens"] == 320, json.dumps({
            "profile": profile, "result": result,
            "gate_receipts": Path(work, f"{profile}.jsonl").read_text()
        }, indent=2)
        assert result["lean_request"]["visible_messages"] == 1
        stored = WorkingSetStore(work).load(
            session_id=f"gdk02-{profile}",
            catalog_epoch=result["working_set"]["catalog_epoch"],
        )
        assert stored["capture"]["packet"], "missing durable packet"
        assert result["packet_stats"]["worktree_revision"] == stored["index"]["catalog_epoch"]
        bad = subprocess.run(req + ["--max-tokens", "4096"], cwd=ROOT,
                             text=True, capture_output=True, timeout=30)
        # Parsing occurs before any backend or MCP process.
        assert bad.returncode != 0 and "max-tokens" in bad.stderr
        return {
            "profile": profile,
            "input_tokens": result["provider_input_tokens"],
            "packet_handle": result["working_set"]["handle"],
            "gate": result["exact_gate"]["counter"],
            "session_messages": result["session_message_count"],
        }
    finally:
        proxy.terminate()
        try:
            proxy.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            proxy.kill()
            proxy.communicate()


def main():
    binary = str(Path(sys.argv[1]).resolve())
    backend = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    threading.Thread(target=backend.serve_forever, daemon=True).start()
    reports = []
    try:
        with tempfile.TemporaryDirectory(prefix="gdk02-") as tmp:
            for profile in ("inkling-2-mini", "bonsai-8b", "bonsai-27b"):
                before = len(Backend.calls)
                reports.append(run(binary, profile, backend.server_port, tmp))
                paths = [name for name, _ in Backend.calls[before:]]
                expected = "/v1/chat/completions/render" if profile == "inkling-2-mini" else (
                    "/v1/chat/completions/input_tokens"
                )
                assert paths == [expected, "/v1/chat/completions"], (profile, paths)
    finally:
        backend.shutdown()
        backend.server_close()
    print(json.dumps({"GDK-02 scripted profile smoke": reports}, indent=2))


if __name__ == "__main__":
    main()
