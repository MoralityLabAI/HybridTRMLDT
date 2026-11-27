"""Command line: ``python -m MCP.pocketgoblin_bench {reference,controls,llm}``."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import reference
from .episode import Episode, load_config

REFERENCES = ("default", "random")


def _run_reference(args) -> int:
    config = load_config(args.config)
    first = True
    for trainer in args.trainers:
        for seed in range(args.mcp_seeds):
            started = time.perf_counter()
            receipt_path = args.out / f"{trainer}-mcp{seed}.json" if args.out else None
            if args.transport == "stdio":
                client = reference.StdioToolClient(args.config, seed, args.threads, receipt_path)
            else:
                client = reference.InProcessClient(config, seed, args.threads)
            try:
                if trainer == "default":
                    reference.run_default(client)
                else:
                    reference.run_random(client, config["reference"]["random_search_seed"] + seed,
                                         config["reference"]["random_search_max_draws"])
                receipt = client.receipt()
            finally:
                client.close()
            if args.out and args.transport != "stdio":
                args.out.mkdir(parents=True, exist_ok=True)
                receipt_path.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            m = receipt["test_metrics"]
            print(json.dumps({"trainer": trainer, "mcp_seed": seed, "score": receipt["score"],
                              "normalized_packet_recall": receipt["normalized_packet_recall"],
                              "packet_recall": round(m["packet_recall"], 4), "accepted": round(m["accepted"], 4),
                              "packet_tokens": round(m["packet_tokens"], 1), "tool_calls": round(m["tool_calls"], 2),
                              "units_spent": receipt["budget"]["spent"], "units_total": receipt["budget"]["total"],
                              "fits": receipt["budget"]["fits_used"], "family": receipt["candidate"]["family"],
                              "wall_seconds": round(time.perf_counter() - started, 1)}), flush=True)
            if first:
                first = False
                print(json.dumps({"controls_test_mcp0": receipt["controls_test"]}, sort_keys=True), flush=True)
    return 0


def _run_controls(args) -> int:
    episode = Episode(load_config(args.config), args.mcp_seed)
    out = {"mcp_seed": args.mcp_seed}
    for split in args.splits:
        out[split] = episode._controls_for(split)
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference", help="run the scripted reference trainers")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--trainers", nargs="+", choices=REFERENCES, default=list(REFERENCES))
    r.add_argument("--mcp-seeds", type=int, default=1, help="episodes 0..N-1")
    r.add_argument("--transport", choices=("inprocess", "stdio"), default="inprocess")
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--out", type=Path)
    c = sub.add_parser("controls", help="control and oracle metrics per split (standard library only)")
    c.add_argument("--config", type=Path, required=True)
    c.add_argument("--mcp-seed", type=int, default=0)
    c.add_argument("--splits", nargs="+", default=["dev", "test"])
    m = sub.add_parser("llm", help="drive the tool surface with a loopback OpenAI-compatible model")
    m.add_argument("--config", type=Path, required=True)
    m.add_argument("--base-url", required=True)
    m.add_argument("--model", required=True)
    m.add_argument("--mcp-seeds", type=int, default=1)
    m.add_argument("--max-turns", type=int, default=40)
    m.add_argument("--threads", type=int, default=4)
    m.add_argument("--out", type=Path, required=True, help="episode receipt JSONL")
    args = p.parse_args(argv)
    if args.cmd == "reference":
        return _run_reference(args)
    if args.cmd == "controls":
        return _run_controls(args)
    from .llm_runner import run_llm
    return run_llm(args)


if __name__ == "__main__":
    sys.exit(main())
