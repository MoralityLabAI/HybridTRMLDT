"""CPU-only probe for the pinned local llama-server chat-template/tokenizer contract.

Does not issue any model inference. Use a direct upstream port or the WSL
Windows curl bridge. A 422/unsupported template intentionally blocks MCP-04.

    python -m MCP.hubrun.preflight_probe --upstream http://127.0.0.1:18127 \
       --via-curl /mnt/c/Windows/System32/curl.exe --out /tmp/mcp04_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from .mcp04 import PreflightError, exact_preflight
from .proxy import Relay


def probe(forward, *, context_window: int = 32768,
          completion_reserve: int = 4096, safety_margin: int = 1024) -> dict:
    messages = [
        {"role": "system", "content": "You are a tool-using assistant."},
        {"role": "user", "content": "Read a local source symbol and report the line."},
    ]
    schema = {"type": "function", "function": {
        "name": "read_symbol", "description": "Read one source symbol by URI.",
        "parameters": {"type": "object", "properties": {
            "uri": {"type": "string"}}, "required": ["uri"]}}}
    test = {"messages": messages, "tools": [schema]}
    empty = exact_preflight(forward, dict(test, tools=[]),
                            context_window=context_window,
                            completion_reserve=completion_reserve,
                            safety_margin=safety_margin)
    with_tools = exact_preflight(forward, test, context_window=context_window,
                                 completion_reserve=completion_reserve,
                                 safety_margin=safety_margin)
    if with_tools["prompt_tokens"] <= empty["prompt_tokens"]:
        raise PreflightError("tool schema did not increase applied-template token count")
    if not with_tools["fits"]:
        raise PreflightError("tool-aware prompt exceeds the reserved model context")
    return {"status": "compatible_sample", "inference_calls": 0,
            "tool_free_prompt_tokens": empty["prompt_tokens"],
            "one_tool_prompt_tokens": with_tools["prompt_tokens"],
            "tool_schema_delta": with_tools["prompt_tokens"] - empty["prompt_tokens"],
            "within_reserved_context": with_tools["fits"],
            "preflight": with_tools,
            "requires_real_server_provider_usage_parity": True}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", default="http://127.0.0.1:18127")
    p.add_argument("--via-curl", help="Windows curl.exe path from WSL")
    p.add_argument("--context-window", type=int, default=32768)
    p.add_argument("--completion-reserve", type=int, default=4096)
    p.add_argument("--safety-margin", type=int, default=1024)
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    with TemporaryDirectory() as directory:
        relay = Relay(args.upstream, Path(directory) / "probe.jsonl", args.via_curl,
                      4096, 6000, 1800, 2, 6000, args.timeout)
        try:
            result = probe(relay.forward, context_window=args.context_window,
                           completion_reserve=args.completion_reserve,
                           safety_margin=args.safety_margin)
        except (PreflightError, ValueError) as error:
            result = {"status": "rejected", "inference_calls": 0,
                      "reason": str(error)}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "compatible_sample" else 2


if __name__ == "__main__":
    sys.exit(main())
