"""Summarise the Stage A reproduction of MOLT-GOBLIN-01 Study v2 (upstream code, CPU)."""
from __future__ import annotations

import argparse
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path


def summarise(study: dict) -> str:
    cells = defaultdict(list)
    for r in study["results"]:
        cells[(r["arm"], r["panel"], r["loss_schedule"])].append(r)
    lines = [f"Upstream Study v2, torch {study['torch']}, device {study['device']}, threads {study['threads']}, "
             f"train loops {study['train_loops']}, status {study['status']}.", "",
             "| arm | panel | loss | seeds | params | updates | fwd block calls | len8 seq acc @train loops | len16 seq acc @16 loops | train CPU s | append-only parity max abs |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    L = study["train_loops"]
    for (arm, panel, loss), rs in sorted(cells.items()):
        def acc(n, r_, field="sequence_accuracy"):
            vals = []
            for r in rs:
                for e in r["evaluations"]:
                    if e["length"] == n and e["requested_loops"] == r_:
                        vals.append(e[field])
            return stats.mean(vals) if vals else float("nan")
        par = [r["append_only_parity"]["max_abs_vs_sequential"] for r in rs if r.get("append_only_parity")]
        lines.append(f"| {arm} | {panel} | {loss} | {len(rs)} | {rs[0]['parameters']} | {rs[0]['optimizer_updates']} | "
                     f"{rs[0]['forward_block_calls']} | {acc(8, L):.3f} (tok {acc(8, L, 'token_accuracy'):.3f}) | "
                     f"{acc(16, 16):.3f} (tok {acc(16, 16, 'token_accuracy'):.3f}) | "
                     f"{stats.mean(r['train_process_cpu_s'] for r in rs):.1f} | {max(par) if par else float('nan'):.2e} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", type=Path, required=True)
    a = ap.parse_args()
    print(summarise(json.loads(a.study.read_text(encoding="utf-8-sig"))))


if __name__ == "__main__":
    main()
