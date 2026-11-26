"""Score Receipt Goblin verdicts against benchmark labels.

A claim is *passed* when the verdict is ``supported`` and *flagged* otherwise. Clean items should
pass; ``off_receipt`` and ``decoy`` items should be flagged.

    python -m MCP.receipt_goblin.score RECEIPTS.jsonl [...]      # GDK app receipts, one row per item
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def _correct(row: dict) -> bool:
    flagged = row["verdict"] != "supported"
    return flagged if row["label"] == "contradicted" else not flagged


def summarize(rows: list[dict]) -> dict:
    cells: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cells[row["arm"]][row["variant"]].append(row)
    out = {}
    for arm, variants in sorted(cells.items()):
        arm_out = {}
        for variant, items in sorted(variants.items()):
            n = len(items)
            cell = {"n": n, "correct": sum(_correct(r) for r in items)}
            cell["rate"] = round(cell["correct"] / n, 4) if n else None
            linked = [r["linked_gold"] for r in items if r.get("linked_gold") is not None]
            if linked:
                cell["linked_gold"] = round(sum(linked) / len(linked), 4)
            recall = [r["packet_gold_recall"] for r in items if r.get("packet_gold_recall") is not None]
            if recall:
                cell["packet_gold_recall"] = round(sum(recall) / len(recall), 4)
            for key in ("packet_tokens", "provider_input_tokens", "provider_output_tokens"):
                values = [r[key] for r in items if isinstance(r.get(key), (int, float))]
                if values:
                    cell[key + "_mean"] = round(sum(values) / len(values), 1)
            repairs = [r.get("repairs", 0) for r in items if "repairs" in r]
            if repairs:
                cell["repairs"] = sum(repairs)
            arm_out[variant] = cell
        rates = [arm_out[v]["rate"] for v in ("clean", "off_receipt", "decoy") if v in arm_out]
        arm_out["balanced"] = round(sum(rates) / len(rates), 4) if rates else None
        out[arm] = arm_out
    return out


def main() -> None:
    rows = []
    for path in sys.argv[1:]:
        rows.extend(json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    print(json.dumps(summarize(rows), indent=1))


if __name__ == "__main__":
    main()
