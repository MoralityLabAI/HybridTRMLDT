"""CLI for deterministic Kimilings inventories, plans and offline evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from Kimilings.index import ARC_REVISION, audit_index, build_index, read_jsonl
from Kimilings.router import VALID_POLICIES, route
from Kimilings.score import score_arc, score_logic_verdicts
from Kimilings.vpd import causal_transfer, summarize_vpd


def _save(path: Path, obj: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KIMILINGS-01 offline eval contracts")
    commands = parser.add_subparsers(dest="command", required=True)

    idx = commands.add_parser("index", help="Index task IDs and hashes; seal ARC public eval")
    idx.add_argument("--arc-training", type=Path, required=True)
    idx.add_argument("--arc-evaluation", type=Path)
    idx.add_argument("--logic-metadata", type=Path)
    idx.add_argument("--arc-revision", default=ARC_REVISION)
    idx.add_argument("--seed", type=int, default=11)
    idx.add_argument("--out", type=Path, required=True)

    audit = commands.add_parser("audit", help="Check split separation from index metadata")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--out", type=Path, required=True)

    plan = commands.add_parser("route", help="Write router plans; does not execute Kimi")
    plan.add_argument("--records", type=Path, required=True)
    plan.add_argument("--policy", choices=VALID_POLICIES, required=True)
    plan.add_argument("--out", type=Path, required=True)

    lg = commands.add_parser("logic", help="Aggregate task-verifier receipts")
    lg.add_argument("--manifest", required=True)
    lg.add_argument("--verdicts", type=Path, required=True)
    lg.add_argument("--split", choices=("vpd_train", "calibration", "validation"), required=True)
    lg.add_argument("--out", type=Path, required=True)

    arc = commands.add_parser("arc", help="Grade actual ARC task predictions")
    arc.add_argument("--manifest", required=True)
    arc.add_argument("--task-dir", type=Path, required=True)
    arc.add_argument("--predictions", type=Path, required=True)
    arc.add_argument("--split", choices=("vpd_train", "calibration", "validation", "final"),
                     default="validation")
    arc.add_argument("--unlock-final", action="store_true")
    arc.add_argument("--ledger", type=Path)
    arc.add_argument("--out", type=Path, required=True)

    vp = commands.add_parser("vpd", help="Score VPD reconstruction/sparsity sweeps")
    vp.add_argument("--records", type=Path, required=True)
    vp.add_argument("--out", type=Path, required=True)

    tr = commands.add_parser("transfer", help="Score matched causal ablations")
    tr.add_argument("--records", type=Path, required=True)
    tr.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "index":
        result = build_index(args.arc_training, args.arc_evaluation,
                             args.logic_metadata, arc_revision=args.arc_revision,
                             seed=args.seed)
        result["audit"] = audit_index(result)
    elif args.command == "audit":
        result = audit_index(_load(args.manifest))
    elif args.command == "route":
        result = [route(r, args.policy) for r in read_jsonl(args.records)]
    elif args.command == "logic":
        result = score_logic_verdicts(_load(args.manifest),
                                      read_jsonl(args.verdicts), split=args.split)
    elif args.command == "arc":
        if args.split == "final" and args.ledger is None:
            parser.error("Final ARC requires --ledger PATH")
        result = score_arc(_load(args.manifest), args.task_dir, args.predictions,
                           split=args.split, unlock_final=args.unlock_final,
                           ledger_path=args.ledger)
    elif args.command == "vpd":
        result = summarize_vpd(read_jsonl(args.records))
    else:
        result = causal_transfer(read_jsonl(args.records))
    _save(args.out, result)
    print(json.dumps({"output": str(args.out), "status": "written"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
