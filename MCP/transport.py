"""A bounded stdio client for this gym's deliberately small MCP subset."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Sequence

from .local_model import strict_json_loads

PROTOCOL_VERSION = "2025-06-18"


class TransportError(RuntimeError):
    pass


class JsonRpcError(TransportError):
    def __init__(self, error: dict[str, Any]):
        super().__init__(f"JSON-RPC {error.get('code')}: {error.get('message')}")
        self.code = error.get("code")
        self.error = error


def validate_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        raise TransportError("tools/list must return a tools array")
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str) or not tool["name"]:
            raise TransportError("tool must have a nonempty string name")
        if tool["name"] in names:
            raise TransportError("duplicate tool name")
        names.add(tool["name"])
        schema = tool.get("inputSchema")
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise TransportError("tool inputSchema must declare object type")
        if "description" in tool and not isinstance(tool["description"], str):
            raise TransportError("tool description must be a string")
    return tools


def validate_call_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict) or not isinstance(result.get("content"), list):
        raise TransportError("tool call result must contain a content array")
    for content in result["content"]:
        if not isinstance(content, dict) or content.get("type") != "text" or not isinstance(content.get("text"), str):
            raise TransportError("this gym supports text content blocks only")
    if "structuredContent" in result and not isinstance(result["structuredContent"], dict):
        raise TransportError("structuredContent must be an object")
    if "isError" in result and not isinstance(result["isError"], bool):
        raise TransportError("isError must be a boolean")
    return result


def append_wire(path: Path | None, direction: str, wire: bytes) -> None:
    if path is None:
        return
    entry = {"direction": direction, "bytes": len(wire), "monotonic_seconds": time.monotonic()}
    try:
        entry["body"] = wire.decode("utf-8")
    except UnicodeError:
        entry["body_base64"] = base64.b64encode(wire).decode("ascii")
    with path.open("a", encoding="utf-8", newline="\n") as journal:
        journal.write(json.dumps(entry, ensure_ascii=False) + "\n")


class StdioClient:
    def __init__(
        self,
        architecture: str = "search",
        catalog_size: int = 32,
        depth: int = 3,
        split: str = "test",
        seed: int = 7,
        case_index: int = 0,
        n: int = 24,
        *,
        command: Sequence[str] | None = None,
        cwd: str | Path | None = None,
        timeout: float = 10.0,
        max_message_bytes: int = 1048576,
        wire_log: Path | None = None,
    ):
        if timeout <= 0 or max_message_bytes <= 0:
            raise ValueError("timeout and max_message_bytes must be positive")
        if case_index < 0 or n <= case_index:
            raise ValueError("case_index must be within n cases")
        self.command = list(command) if command is not None else [
            sys.executable, "-u", "-m", "MCP.serve", "--architecture", architecture,
            "--catalog-size", str(catalog_size), "--depth", str(depth), "--split", split,
            "--seed", str(seed), "--case-index", str(case_index), "--n", str(n),
            "--max-message-bytes", str(max_message_bytes),
        ]
        self.cwd = str(cwd or Path(__file__).resolve().parent.parent)
        self.timeout = float(timeout)
        self.max_message_bytes = max_message_bytes
        self.wire_log = wire_log
        self.process: subprocess.Popen[bytes] | None = None
        self._queue: queue.Queue[bytes | Exception | None] = queue.Queue(maxsize=8)
        self._stderr = bytearray()
        self._next_id = 0
        self._initialized = False
        self._closed = False
        self.initialize_result: dict[str, Any] | None = None
        self.last_exchange: dict[str, Any] | None = None

    def __enter__(self) -> "StdioClient":
        self._start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _start(self) -> None:
        if self._closed:
            raise TransportError("client is closed")
        if self.process is not None:
            return
        self.process = subprocess.Popen(self.command, cwd=self.cwd, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

        def read_stdout() -> None:
            assert self.process is not None and self.process.stdout is not None
            try:
                while True:
                    line = self.process.stdout.readline(self.max_message_bytes + 1)
                    if not line:
                        self._queue.put(None)
                        return
                    if len(line) > self.max_message_bytes or not line.endswith(b"\n"):
                        self._queue.put(TransportError("server message violates bounded newline framing"))
                        return
                    self._queue.put(line)
            except (OSError, ValueError) as exc:
                self._queue.put(TransportError(f"server stdout failed: {exc}"))

        def read_stderr() -> None:
            assert self.process is not None and self.process.stderr is not None
            try:
                while True:
                    chunk = self.process.stderr.read(4096)
                    if not chunk:
                        return
                    self._stderr.extend(chunk)
                    if len(self._stderr) > 65536:
                        del self._stderr[:-65536]
            except (OSError, ValueError):
                return

        threading.Thread(target=read_stdout, daemon=True).start()
        threading.Thread(target=read_stderr, daemon=True).start()

    @property
    def stderr(self) -> str:
        return self._stderr.decode("utf-8", errors="replace")

    def _send(self, message: dict[str, Any], deadline: float | None = None) -> int:
        self._start()
        assert self.process is not None and self.process.stdin is not None
        wire = json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
        if len(wire) > self.max_message_bytes:
            raise TransportError("client message exceeds max_message_bytes")
        deadline = deadline if deadline is not None else time.monotonic() + self.timeout
        done: queue.Queue[Exception | None] = queue.Queue(maxsize=1)
        pipe = self.process.stdin

        def write_stdin() -> None:
            try:
                offset = 0
                while offset < len(wire):
                    written = pipe.write(wire[offset:])
                    if not written:
                        raise OSError("zero-byte pipe write")
                    offset += written
                pipe.flush()
                done.put(None)
            except (OSError, ValueError) as exc:
                done.put(exc)

        threading.Thread(target=write_stdin, daemon=True).start()
        try:
            error = done.get(timeout=max(0.001, deadline - time.monotonic()))
        except queue.Empty as exc:
            self.close(abort=True)
            raise TransportError(f"MCP request timed out writing after {self.timeout:g}s") from exc
        if error is not None:
            self.close()
            raise TransportError(f"server input closed: {error}") from error
        append_wire(self.wire_log, "client_to_mcp", wire)
        return len(wire)

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Expose raw requests for protocol tests; callers normally use methods below."""
        request_id = self._next_id
        self._next_id += 1
        message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        started = time.monotonic()
        deadline = started + self.timeout
        sent = self._send(message, deadline)
        try:
            line = self._queue.get(timeout=max(0.001, deadline - time.monotonic()))
        except queue.Empty as exc:
            self.close()
            raise TransportError(f"MCP request timed out after {self.timeout:g}s") from exc
        if line is None or isinstance(line, Exception):
            self.close()
            raise TransportError(str(line) if line else "MCP server exited before responding")
        append_wire(self.wire_log, "mcp_to_client", line)
        self.last_exchange = {"request_bytes": sent, "response_bytes": len(line),
                              "latency_seconds": time.monotonic() - started, "id": request_id}
        try:
            response = strict_json_loads(line.decode("utf-8"))
        except (ValueError, UnicodeError) as exc:
            self.close()
            raise TransportError(f"invalid JSON-RPC response: {exc}") from exc
        if not isinstance(response, dict) or response.get("jsonrpc") != "2.0":
            self.close()
            raise TransportError("invalid JSON-RPC response envelope")
        if type(response.get("id")) is not int or response["id"] != request_id:
            self.close()
            raise TransportError("JSON-RPC response ID mismatch")
        if ("result" in response) == ("error" in response):
            self.close()
            raise TransportError("response must contain exactly one of result and error")
        if "error" in response:
            if not isinstance(response["error"], dict):
                raise TransportError("malformed JSON-RPC error")
            raise JsonRpcError(response["error"])
        return response["result"]

    def initialize(self) -> dict[str, Any]:
        if self._initialized:
            assert self.initialize_result is not None
            return self.initialize_result
        result = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "hybrid-mcp-gym-client", "version": "0.1.0"},
        })
        if not isinstance(result, dict) or result.get("protocolVersion") != PROTOCOL_VERSION:
            self.close()
            raise TransportError("server selected an unsupported protocol version")
        if not isinstance(result.get("capabilities"), dict) or "tools" not in result["capabilities"]:
            self.close()
            raise TransportError("server lacks tools capability")
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self._initialized = True
        self.initialize_result = result
        return result

    def tools(self) -> list[dict[str, Any]]:
        if not self._initialized:
            raise TransportError("initialize must complete before tools/list")
        result = self.request("tools/list")
        if not isinstance(result, dict) or result.get("nextCursor"):
            raise TransportError("this gym client requires a complete unpaginated tools list")
        return validate_tools(result.get("tools"))

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self._initialized:
            raise TransportError("initialize must complete before tools/call")
        return validate_call_result(self.request("tools/call", {"name": name, "arguments": arguments}))

    def close(self, *, abort: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        self._initialized = False
        process = self.process
        if process is None:
            return
        # On Windows, closing stdin can wait for a blocked pipe writer.
        if abort and process.poll() is None:
            process.terminate()
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=0.3)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()
