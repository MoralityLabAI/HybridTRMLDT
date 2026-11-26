"""Receipt Goblin inside the Goose agent harness: benchmark claims through `goose run` with the server as an
extension and an OpenAI-compatible local model (Inky on squid5080 by default).

    python MCP/receipt_goblin/scripts/goose_demo.py --out DIR [--per-variant 4] [--goose PATH]

Goose gets only the Receipt Goblin extension (--no-profile) and is told to call claim_packet, pick a leaf
and call certify. Each run's JSON transcript is parsed for its tool requests and the last certify response.
A run that never gets a certify response has outcome `no_certify` (or `no_tool_call` when the model called
no tool at all); the scorer counts those as flagged, so the summary also reports how many runs certified.
Writes goose_cli.jsonl (one row per claim), the raw transcripts and goose_cli_summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from MCP.receipt_goblin.bench import load_bench  # noqa: E402
from MCP.receipt_goblin.score import summarize  # noqa: E402

TASK = """Check one numeric claim from a research write-up against the run receipts it was written from.

Call the receipts tool claim_packet with exactly these arguments:
{args}

It returns a packet of receipt leaves (id, file :: path = value). Choose the ONE leaf that records the
quantity the claim's number {target} describes: the same metric, for the same run, arm and setting. Decide by
meaning, not by whether the value matches; values get copied from the wrong cell. Then call certify with the
packet_id and that leaf id (or NONE if no leaf records it). Finish by stating certify's verdict."""


def pick(items: list[dict], per_variant: int) -> list[dict]:
    rng = random.Random(0)
    out = []
    for variant in ("clean", "decoy", "off_receipt"):
        pool = sorted((i for i in items if i["variant"] == variant), key=lambda i: i["id"])
        out.extend(rng.sample(pool, min(per_variant, len(pool))))
    return out


def transcript(stdout: str) -> dict | None:
    """Goose prints a banner, then the JSON transcript."""
    start = stdout.find("{\n")
    if start < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(stdout[start:])[0]
    except json.JSONDecodeError:
        return None


def _find(obj, key):
    """First value stored under ``key`` anywhere inside ``obj``."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        obj = list(obj.values())
    if isinstance(obj, list):
        for value in obj:
            found = _find(value, key)
            if found is not None:
                return found
    return None


def _payloads(obj):
    """Dicts inside a tool response, including JSON objects serialized in its text parts."""
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _payloads(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _payloads(value)
    elif isinstance(obj, str) and obj.lstrip().startswith("{"):
        try:
            yield from _payloads(json.loads(obj))
        except json.JSONDecodeError:
            pass


def read_run(doc: dict | None) -> dict:
    calls, certified, final_text = [], None, ""
    for message in (doc or {}).get("messages", []):
        for part in message.get("content", []):
            kind = part.get("type", "")
            if kind == "toolRequest":
                calls.append({"name": str(_find(part, "name") or ""), "arguments": _find(part, "arguments")})
            elif kind == "toolResponse":
                for payload in _payloads(part):
                    if "verdict" in payload:
                        certified = payload
                        break
            elif kind == "text" and message.get("role") == "assistant":
                final_text = part.get("text", "")
    usage = (doc or {}).get("metadata", {})
    return {"calls": calls, "certified": certified, "final_text": final_text,
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-variant", type=int, default=4)
    ap.add_argument("--goose", default=str(Path.home() / ".local" / "bin" / ("goose.exe" if os.name == "nt" else "goose")))
    ap.add_argument("--host", default="http://127.0.0.1:18090")
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GOOSE_PROVIDER="openai", GOOSE_MODEL=args.model, OPENAI_HOST=args.host,
               OPENAI_BASE_PATH="v1/chat/completions", OPENAI_API_KEY="local-no-key", GOOSE_MODE="auto",
               GOOSE_CONTEXT_LIMIT="16384", GOOSE_TEMPERATURE="0")
    extension = f"receipts:PYTHONPATH={ROOT} {sys.executable} -m MCP.receipt_goblin.server"
    rows = []
    for item in pick(load_bench(), args.per_variant):
        call = {"text": item["text"], "start": item["start"], "end": item["end"],
                "receipts_root": str(ROOT / item["receipts"])}
        task = TASK.format(args=json.dumps(call, ensure_ascii=False), target=item["target"])
        cmd = [args.goose, "run", "--no-session", "--no-profile", "--with-extension", extension,
               "--output-format", "json", "--max-turns", str(args.max_turns), "-t", task]
        started = time.monotonic()
        try:
            done = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=args.timeout)
            stdout, stderr, code = done.stdout, done.stderr, done.returncode
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout.decode("utf-8", "replace") if isinstance(error.stdout, bytes) else error.stdout or ""
            stderr = error.stderr.decode("utf-8", "replace") if isinstance(error.stderr, bytes) else error.stderr or ""
            code = "timeout"
        seconds = time.monotonic() - started
        slug = re.sub(r"[^A-Za-z0-9]+", "_", item["id"])[-80:]
        (out / "raw" / f"{slug}.json").write_text(stdout, encoding="utf-8")
        (out / "raw" / f"{slug}.stderr.txt").write_text(stderr, encoding="utf-8")
        run = read_run(transcript(stdout))
        names = [c["name"] for c in run["calls"]]
        cert = run["certified"] or {}
        outcome = "certified" if cert else ("no_tool_call" if not names else "no_certify")
        row = {"arm": "goose_cli", "id": item["id"], "variant": item["variant"], "label": item["label"],
               "target": item["target"], "gold_leaves": item.get("gold_leaves", []),
               "verdict": cert.get("verdict", outcome), "outcome": outcome, "leaf_id": cert.get("leaf_id"),
               "linked_gold": (cert.get("leaf") in item["gold_leaves"]) if item.get("gold_leaves") else None,
               "leaf": cert.get("leaf"), "leaf_value": cert.get("leaf_value"),
               "tool_calls": names, "final_text": run["final_text"][-400:],
               "input_tokens": run["input_tokens"], "output_tokens": run["output_tokens"],
               "exit": code, "seconds": round(seconds, 1)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (out / "goose_cli.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                         encoding="utf-8")
    summary = summarize(rows)
    for variant, stats in summary.get("goose_cli", {}).items():
        if isinstance(stats, dict):
            mine = [r for r in rows if r["variant"] == variant]
            stats["certified"] = sum(r["outcome"] == "certified" for r in mine)
    (out / "goose_cli_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
