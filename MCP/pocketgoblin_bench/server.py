"""PocketGoblin Bench trainer tools as an MCP stdio server (newline JSON-RPC 2.0, protocol 2025-06-18).

    python -m MCP.pocketgoblin_bench.server --config MCP/pocketgoblin_bench/configs/smoke.json \
        --mcp-seed 0 --receipt RUN/episode.json

The server holds one episode. It writes the episode receipt when ``submit`` completes or stdin closes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from MCP.serve import serve

from .episode import Episode, load_config


def _result(payload: dict, error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(payload, sort_keys=True)}],
            "structuredContent": payload, "isError": error}


class BenchService:
    """Duck-typed environment for ``MCP.serve``: ``task``, ``tools()``, ``call()``."""

    def __init__(self, episode: Episode):
        self.episode = episode
        self.task = episode.public_task()

    def tools(self) -> list[dict]:
        return self.episode.tools()

    def call(self, name: str, arguments: dict) -> dict:
        payload = self.episode.call(name, arguments)
        return _result(payload, error="error" in payload and "code" in payload)


def write_receipt(episode: Episode, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(episode.receipt(), indent=1, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--mcp-seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--receipt", type=Path)
    args = p.parse_args(argv)
    episode = Episode(load_config(args.config), args.mcp_seed, threads=args.threads)
    try:
        serve(BenchService(episode), sys.stdin.buffer, sys.stdout.buffer, max_message_bytes=4 * 1024 * 1024)
    finally:
        write_receipt(episode, args.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
