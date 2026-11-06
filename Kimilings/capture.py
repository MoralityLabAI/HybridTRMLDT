"""Audit routed Kimi expert activation metadata and rank pilot targets."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from Kimilings.index import manifest_for_task, read_jsonl


def _is_sha(value: object, length: int = 64) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        c in "0123456789abcdef" for c in value
    )


def audit_capture(manifest: dict, captures: list[dict], *, min_tokens: int = 100) -> dict:
    """Require traceable real full-Kimi forwards before reasoning claims."""
    tasks = manifest_for_task(manifest)
    seen: set[tuple] = set()
    tokens: dict[tuple[int, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    task_counts: dict[tuple[int, int], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in captures:
        fields = ("task_id", "split", "source_model_revision",
                  "activation_sha256", "router_receipt", "reference_forward_receipt",
                  "layer", "expert_index", "token_count", "capture_id")
        if any(key not in row for key in fields):
            raise ValueError("Incomplete routed activation receipt")
        tid = row["task_id"]
        if tid not in tasks or tasks[tid]["split"] != row["split"]:
            raise ValueError("Capture task is unregistered or split-mismatched")
        if row["split"] == "final":
            raise PermissionError("Final evaluation tasks cannot supply VPD training inputs")
        if row.get("activation_source") != "routed_kimi":
            raise ValueError("Only actual routed inputs belong in reasoning captures")
        if not _is_sha(row["source_model_revision"], length=40):
            raise ValueError("Pin original full Kimi model revision, not slice commit")
        if any(not _is_sha(row[k]) for k in (
            "activation_sha256", "router_receipt", "reference_forward_receipt"
        )):
            raise ValueError("Content-addressed activation, router and forward receipts required")
        if any(type(row[k]) is not int for k in ("layer", "expert_index", "token_count")):
            raise ValueError("Invalid layer, expert index or routed-token count")
        if not 0 <= row["layer"] < 92 or not 0 <= row["expert_index"] <= 15:
            raise ValueError("Target must be one of the 16 exported index stacks")
        if row["token_count"] <= 0 or not row["capture_id"]:
            raise ValueError("Empty capture")
        key = (tid, row["capture_id"], row["layer"], row["expert_index"])
        if key in seen:
            raise ValueError("Duplicate capture block")
        seen.add(key)
        site = (row["layer"], row["expert_index"])
        family = tasks[tid]["family"]
        tokens[site][family] += row["token_count"]
        task_counts[site][family].add(tid)
    candidates = []
    for (layer, expert), by_family in tokens.items():
        real_families = [f for f in ("logic", "arc_agi_2") if by_family.get(f, 0) >= min_tokens]
        candidates.append({
            "layer": layer, "expert_index": expert,
            "token_counts_by_family": dict(sorted(by_family.items())),
            "task_counts_by_family": {
                f: len(ids) for f, ids in sorted(task_counts[(layer, expert)].items())
            },
            "covered_primary_families": len(real_families),
            "candidate_ready": len(real_families) == 2,
            "min_primary_family_tokens": min(
                (by_family.get(f, 0) for f in ("logic", "arc_agi_2")), default=0
            ),
        })
    candidates.sort(key=lambda r: (-r["covered_primary_families"],
                                    -r["min_primary_family_tokens"],
                                    r["layer"], r["expert_index"]))
    return {
        "protocol": "KIMILINGS-01",
        "status": "real_capture_metadata_audited" if captures else "activation_capture_missing",
        "capture_blocks": len(captures),
        "candidate_sites": candidates,
        "ready_sites": sum(r["candidate_ready"] for r in candidates),
    }


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--min-tokens", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.min_tokens < 1:
        parser.error("--min-tokens must be positive")
    m = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = audit_capture(m, read_jsonl(args.captures), min_tokens=args.min_tokens)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Capture blocks: {result['capture_blocks']}; ready sites: {result['ready_sites']}")


if __name__ == "__main__":
    main()
