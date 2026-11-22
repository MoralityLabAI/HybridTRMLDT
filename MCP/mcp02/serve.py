"""Serve one MCP-02 world over the pinned MCP stdio subset used by MCP-01.

    python -m MCP.mcp02.serve --arm layered_ldt --layout PSCB --split test --seed 3303 --index 0 --catalog-size 256
"""
from __future__ import annotations

import argparse
import sys

from MCP.serve import serve

from .env import ARMS, Env02
from .world import make_world


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--layout", default="SCB")
    p.add_argument("--split", choices=("train", "dev", "test"), default="test")
    p.add_argument("--seed", type=int, default=3303)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--catalog-size", type=int, default=256)
    p.add_argument("--max-message-bytes", type=int, default=4194304)
    args = p.parse_args(argv)
    case = make_world(args.split, args.seed, args.index, args.catalog_size)
    env = Env02(case, args.arm, args.layout if args.arm.startswith("layered") else "")
    serve(env, sys.stdin.buffer, sys.stdout.buffer, max_message_bytes=args.max_message_bytes)


if __name__ == "__main__":
    main()
