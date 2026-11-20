"""Data Goblin MCP server: compact code-context packets over stdio (for Goose and GDK apps).

Tools:

* ``compact_context(task, repo_path, budget_tokens=2048, max_tool_calls=24, arm="hybrid")`` runs a
  local router over the repository's code MCP surface and returns the token-budgeted packet with
  its accounting: tokens in the packet, tool calls, tokens the router read locally, router FLOPs.
* ``read_symbol(repo_path, uri)`` returns one symbol's source, for follow-up reads by the agent.

The router is the BM25 control by default. Set ``MCP03_ROUTER_CHECKPOINT`` to a trained MCP-02
router checkpoint (needs torch) to use a learned looped/TRM/HRM router instead.

    python -m MCP.mcp03.goblin_server            # Goose: --with-extension "python -m MCP.mcp03.goblin_server"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from MCP.serve import serve

from .agent import make_code_control, run_code_episode
from .env import CodeEnv
from .repo import snapshot_from_dir
from .tokens import describe as tokenizer_name

_CACHE: dict[str, tuple] = {}


def _router():
    path = os.environ.get("MCP03_ROUTER_CHECKPOINT")
    if not path:
        return make_code_control("lexical_bm25")
    from MCP.mcp02.models import RouterPolicy, file_sha256, load
    from .run import LearnedCodePolicy
    return LearnedCodePolicy(RouterPolicy(Path(path).stem, load(path, "cpu"), checkpoint=file_sha256(path)))


class GoblinService:
    """Duck-typed environment for MCP.serve: ``task``, ``tools()``, ``call()``."""

    task = {"service": "data-goblin", "purpose": "compact code context packets"}

    def __init__(self):
        self.router = _router()

    def tools(self) -> list[dict]:
        def tool(name, desc, props, req):
            return {"name": name, "description": desc, "annotations": {"readOnlyHint": True},
                    "inputSchema": {"type": "object", "properties": props, "required": req,
                                    "additionalProperties": False}}
        return [
            tool("compact_context", "Explore a Python repository with a local router and return a compact, "
                 "token-budgeted context packet (ranked files plus source snippets) for the task.",
                 {"task": {"type": "string"}, "repo_path": {"type": "string"},
                  "budget_tokens": {"type": "integer", "minimum": 128, "maximum": 32768},
                  "max_tool_calls": {"type": "integer", "minimum": 1, "maximum": 200},
                  "arm": {"type": "string", "enum": ["search", "layered", "hybrid", "layered_ldt"]},
                  "compose": {"type": "boolean"}},
                 ["task", "repo_path"]),
            tool("read_symbol", "Read one symbol's source by path::qualname.",
                 {"repo_path": {"type": "string"}, "uri": {"type": "string"}}, ["repo_path", "uri"]),
        ]

    def _snapshot(self, repo_path: str):
        root = Path(repo_path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("repo_path is not a directory")
        key = str(root)
        if key not in _CACHE:
            snap = snapshot_from_dir(root)
            env = CodeEnv(snap, {}, {"files": [], "symbols": []}, "hybrid")
            _CACHE[key] = (snap, env.index)
        return _CACHE[key]

    def call(self, name: str, arguments: dict) -> dict:
        import json
        try:
            if name == "compact_context":
                snap, index = self._snapshot(arguments["repo_path"])
                arm = arguments.get("arm", "hybrid")
                budget = int(arguments.get("budget_tokens", 2048))
                env = CodeEnv(snap, {}, {"files": [], "symbols": []}, arm, index=index)
                compose = bool(arguments.get("compose", os.environ.get("MCP03_COMPOSED") == "1"))
                ep = run_code_episode(env, arguments["task"], policy=self.router,
                                      max_tool_calls=int(arguments.get("max_tool_calls", 24)),
                                      budgets=(budget,), return_text=True, compose=compose)
                packet = ep["packets"][("c" if compose else "") + str(budget)]
                payload = {"packet": packet["text"], "packet_tokens": packet["tokens"],
                           "tokenizer": tokenizer_name(), "tool_calls": ep["tool_calls"],
                           "local_result_tokens": ep["compute"]["result_tokens"],
                           "router": getattr(self.router, "name", "router"), "router_flops": ep["compute"]["flops"],
                           "files_indexed": len(snap.files), "reads": ep["reads"]}
            elif name == "read_symbol":
                snap, index = self._snapshot(arguments["repo_path"])
                env = CodeEnv(snap, {}, {"files": [], "symbols": []}, "hybrid", index=index)
                payload = env.call("read_symbol", {"uri": arguments["uri"]})["structuredContent"]
            else:
                raise ValueError("unknown tool")
            return {"content": [{"type": "text", "text": json.dumps(payload)}], "structuredContent": payload,
                    "isError": False}
        except (ValueError, KeyError, OSError) as error:
            payload = {"error": str(error)}
            return {"content": [{"type": "text", "text": json.dumps(payload)}], "structuredContent": payload,
                    "isError": True}


def main() -> None:
    serve(GoblinService(), sys.stdin.buffer, sys.stdout.buffer, max_message_bytes=8 * 1024 * 1024)


if __name__ == "__main__":
    main()
