"""Serve one synthetic gym case over a pinned, minimal MCP stdio subset.

Implemented: initialize, notifications/initialized, tools/list, tools/call.
Only UTF-8 newline-delimited JSON-RPC and text tool results are supported.
This is a gym transport, not a claim of full MCP specification conformance.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, BinaryIO

from .local_model import strict_json_loads
from .transport import PROTOCOL_VERSION, validate_call_result, validate_tools


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _schema_error(value: Any, schema: dict[str, Any], path: str = "arguments") -> str | None:
    """Validate the JSON-schema vocabulary used by this synthetic environment."""
    kind = schema.get("type")
    matches = {
        "object": isinstance(value, dict), "array": isinstance(value, list),
        "string": isinstance(value, str), "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "null": value is None,
    }
    if isinstance(kind, str) and kind in matches and not matches[kind]:
        return f"{path} must be {kind}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path} is outside the allowed enum"
    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                return f"{path}.{name} is required"
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            return f"{path} has unexpected fields"
        for name, child in properties.items():
            if name in value:
                error = _schema_error(value[name], child, f"{path}.{name}")
                if error:
                    return error
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            return f"{path} has too few items"
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return f"{path} has too many items"
        if isinstance(schema.get("items"), dict):
            for index, item in enumerate(value):
                error = _schema_error(item, schema["items"], f"{path}[{index}]")
                if error:
                    return error
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path} is below minimum"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path} is above maximum"
    return None


class Session:
    def __init__(self, env: Any):
        self.env = env
        self.catalog = validate_tools(env.tools())
        self.by_name = {tool["name"]: tool for tool in self.catalog}
        self.initialize_requested = False
        self.ready = False

    def dispatch(self, message: Any) -> dict[str, Any] | None:
        if not isinstance(message, dict):
            return _error(None, -32600, "Request must be an object")
        has_id = "id" in message
        request_id = message.get("id")
        if has_id and (not isinstance(request_id, (str, int)) or isinstance(request_id, bool)):
            return _error(None, -32600, "Invalid request ID")
        method = message.get("method")
        if message.get("jsonrpc") != "2.0" or not isinstance(method, str):
            return _error(request_id, -32600, "Invalid JSON-RPC request")
        if not has_id:
            if method == "notifications/initialized" and self.initialize_requested:
                self.ready = True
            return None
        params = message.get("params", {})
        if not isinstance(params, dict):
            return _error(request_id, -32602, "params must be an object")
        if method == "initialize":
            if self.initialize_requested:
                return _error(request_id, -32600, "Session was already initialized")
            if (not isinstance(params.get("protocolVersion"), str)
                    or not isinstance(params.get("capabilities"), dict)
                    or not isinstance(params.get("clientInfo"), dict)):
                return _error(request_id, -32602, "Invalid initialization parameters")
            client_info = params["clientInfo"]
            if not isinstance(client_info.get("name"), str) or not isinstance(client_info.get("version"), str):
                return _error(request_id, -32602, "clientInfo requires name and version")
            self.initialize_requested = True
            result = {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "hybrid-mcp-gym", "version": "0.1.0"},
                "_meta": {"publicTask": self.env.task, "syntheticGym": True},
            }
        elif not self.ready:
            return _error(request_id, -32002, "Initialize handshake must complete first")
        elif method == "tools/list":
            if params.get("cursor") is not None:
                return _error(request_id, -32602, "Pagination is unsupported in this gym")
            result = {"tools": self.catalog}
        elif method == "tools/call":
            name, arguments = params.get("name"), params.get("arguments", {})
            if not isinstance(name, str) or name not in self.by_name:
                return _error(request_id, -32602, "Unknown tool name")
            if not isinstance(arguments, dict):
                return _error(request_id, -32602, "arguments must be an object")
            error = _schema_error(arguments, self.by_name[name]["inputSchema"])
            if error:
                return _error(request_id, -32602, error)
            try:
                result = validate_call_result(self.env.call(name, arguments))
            except Exception:
                result = {"content": [{"type": "text", "text": "Internal tool execution error"}], "isError": True}
        else:
            return _error(request_id, -32601, "Method not found")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}


def serve(env: Any, source: BinaryIO, destination: BinaryIO, max_message_bytes: int = 1048576) -> None:
    session = Session(env)
    while True:
        line = source.readline(max_message_bytes + 1)
        if not line:
            return
        if len(line) > max_message_bytes or not line.endswith(b"\n"):
            while line and not line.endswith(b"\n"):
                line = source.readline(max_message_bytes + 1)
            response = _error(None, -32700, "Invalid bounded newline framing")
        else:
            try:
                response = session.dispatch(strict_json_loads(line.decode("utf-8")))
            except (ValueError, UnicodeError):
                response = _error(None, -32700, "Parse error")
        if response is not None:
            wire = json.dumps(response, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
            if len(wire) > max_message_bytes:
                wire = json.dumps(_error(response.get("id"), -32603, "Response exceeds message byte limit"), separators=(",", ":")).encode("utf-8") + b"\n"
            destination.write(wire)
            destination.flush()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("flat", "search", "layered", "layered_ldt"), default="search")
    parser.add_argument("--catalog-size", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--max-message-bytes", type=int, default=1048576)
    args = parser.parse_args(argv)
    if args.case_index < 0 or args.case_index >= args.n:
        parser.error("case-index must be within n cases")
    if args.max_message_bytes < 256:
        parser.error("max-message-bytes must be at least 256")
    from .environment import RetrievalEnv, make_cases
    cases = make_cases(split=args.split, seed=args.seed, n=args.n, catalog_size=args.catalog_size, depth=args.depth)
    env = RetrievalEnv(cases[args.case_index], architecture=args.architecture, catalog_size=args.catalog_size, depth=args.depth)
    serve(env, sys.stdin.buffer, sys.stdout.buffer, args.max_message_bytes)


if __name__ == "__main__":
    main()
