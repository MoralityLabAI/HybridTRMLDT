"""Publish hub runs into the repository: summary, public-environment traces, relay call logs and a manifest.

Traces of private environments (tags in ``summarize.PRIVATE_PREFIXES``) are not
copied; their call-log lines are, because those hold token counts and request
hashes only. Smoke runs are left out.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .summarize import PRIVATE_PREFIXES, _latest

SKIP = ("smoke",)


def _venv_info(venv: Path) -> dict:
    py = venv / "bin" / "python"
    freeze = subprocess.run([str(py), "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    if not freeze.strip():
        freeze = subprocess.run(["uv", "pip", "freeze", "--python", str(py)], capture_output=True, text=True).stdout
    pins = {line.split("==")[0].lower(): line.split("==")[1] for line in freeze.splitlines() if "==" in line}
    return {"verifiers": pins.get("verifiers"), "freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
            "packages": len(pins)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--calls", required=True, nargs="+")
    ap.add_argument("--venvs", required=True)
    ap.add_argument("--queue-log", required=True)
    ap.add_argument("--server-config", required=True, help="JSON file describing the model server")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "traces").mkdir(parents=True, exist_ok=True)

    subprocess.run([sys.executable, "-m", "MCP.hubrun.summarize", "--runs", args.runs, "--calls", *args.calls,
                    "--out", str(out)], check=True, stdout=subprocess.DEVNULL)

    copied = {}
    for run_dir in sorted(Path(args.runs).iterdir()):
        tag = run_dir.name
        if not run_dir.is_dir() or tag.startswith(SKIP) or tag.startswith(PRIVATE_PREFIXES):
            continue
        src = _latest(list(run_dir.rglob("traces.jsonl"))) or _latest(list(run_dir.rglob("results.jsonl")))
        if src is None:
            continue
        dest = out / "traces" / f"{tag}.{src.name}.gz"
        with src.open("rb") as fin, open(dest, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fout:
            shutil.copyfileobj(fin, fout)
        copied[tag] = {"file": dest.name, "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest()}

    lines = []
    for log in args.calls:
        for line in Path(log).read_text(encoding="utf-8").splitlines():
            if not json.loads(line)["tag"].startswith(SKIP):
                lines.append(line)
    with gzip.open(out / "calls.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    venvs = {p.name: _venv_info(p) for p in sorted(Path(args.venvs).iterdir()) if (p / "bin" / "python").exists()}
    manifest = {"server": json.loads(Path(args.server_config).read_text(encoding="utf-8")), "environments": venvs,
                "traces": copied, "call_lines": len(lines),
                "private_runs_aggregate_only": sorted(t.name for t in Path(args.runs).iterdir()
                                                      if t.name.startswith(PRIVATE_PREFIXES))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    shutil.copyfile(args.queue_log, out / "queue.log")
    print(json.dumps({"traces": len(copied), "call_lines": len(lines), "venvs": len(venvs)}))


if __name__ == "__main__":
    main()
